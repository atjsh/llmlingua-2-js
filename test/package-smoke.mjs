import assert from "node:assert/strict";
import { access, mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";

const root = resolve(import.meta.dirname, "..");
const temporaryDirectory = await mkdtemp(join(tmpdir(), "llmlingua-package-"));

function run(command, args, cwd = root) {
  const result = spawnSync(command, args, {
    cwd,
    encoding: "utf8",
    env: { ...process.env, npm_config_audit: "false", npm_config_fund: "false" },
  });

  assert.equal(
    result.status,
    0,
    `${command} ${args.join(" ")} failed:\n${result.stdout}\n${result.stderr}`
  );
  return result.stdout;
}

try {
  const packResult = JSON.parse(
    run("npm", ["pack", "--dry-run", "--json", "--pack-destination", temporaryDirectory])
  )[0];
  const packedManifest = JSON.parse(
    await readFile(join(root, "package.json"), "utf8")
  );

  assert.doesNotMatch(JSON.stringify(packedManifest), /@tensorflow\/tfjs/);

  for (const file of packResult.files) {
    const contents = await readFile(join(root, file.path), "utf8");
    assert.doesNotMatch(contents, /@tensorflow\/tfjs/);
  }

  const tarballName = run("npm", [
    "pack",
    "--json",
    "--pack-destination",
    temporaryDirectory,
  ]);
  const tarballPath = join(
    temporaryDirectory,
    JSON.parse(tarballName)[0].filename
  );

  run("npm", ["init", "--yes"], temporaryDirectory);
  run(
    "npm",
    [
      "install",
      "--ignore-scripts",
      tarballPath,
      "@huggingface/transformers@3.5.2",
      "js-tiktoken@1.0.20",
    ],
    temporaryDirectory
  );
  run(
    process.execPath,
    [
      "--input-type=module",
      "--eval",
      "import('@atjsh/llmlingua-2').then((m) => { if (!m.LLMLingua2) process.exit(1) })",
    ],
    temporaryDirectory
  );

  const installedManifest = JSON.parse(
    await readFile(
      join(temporaryDirectory, "node_modules", "@atjsh", "llmlingua-2", "package.json"),
      "utf8"
    )
  );
  assert.doesNotMatch(JSON.stringify(installedManifest), /@tensorflow\/tfjs/);
  await assert.rejects(
    access(join(temporaryDirectory, "node_modules", "@tensorflow", "tfjs"))
  );
  console.log("Packed package installs and imports without TensorFlow.js.");
} finally {
  await rm(temporaryDirectory, { recursive: true, force: true });
}
