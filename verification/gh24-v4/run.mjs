import assert from "node:assert/strict";
import { cp, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../..");

function run(command, args, cwd = ROOT, environment = {}) {
  const result = spawnSync(command, args, {
    cwd,
    encoding: "utf8",
    env: {
      ...process.env,
      npm_config_audit: "false",
      npm_config_fund: "false",
      ...environment,
    },
  });
  assert.equal(
    result.status,
    0,
    `${command} ${args.join(" ")} failed:\n${result.stdout}\n${result.stderr}`
  );
  process.stdout.write(result.stdout);
  process.stderr.write(result.stderr);
  return result.stdout;
}

run("corepack", ["yarn", "build"]);
run(process.execPath, ["--test", join(HERE, "core.test.mjs")]);

if (process.env.LLMLINGUA_RUN_MODEL_E2E === "1") {
  run(process.execPath, ["--test", join(HERE, "tinybert-node.test.mjs")]);
}

const packDirectory = await mkdtemp(join(tmpdir(), "llmlingua-gh24-v4-pack-"));
const consumer = await mkdtemp(join(tmpdir(), "llmlingua-gh24-v4-consumer-"));
try {
  const packed = spawnSync(
    "npm",
    ["pack", "--json", "--pack-destination", packDirectory],
    { cwd: ROOT, encoding: "utf8" }
  );
  assert.equal(packed.status, 0, packed.stderr);
  const [{ filename }] = JSON.parse(packed.stdout);
  const tarball = join(packDirectory, filename);

  await writeFile(
    join(consumer, "package.json"),
    JSON.stringify({ private: true, type: "module" })
  );
  run(
    "npm",
    [
      "install",
      "--ignore-scripts",
      tarball,
      "@huggingface/transformers@4.2.0",
      "js-tiktoken@1.0.20",
    ],
    consumer
  );
  await cp(join(HERE, "runtime-import.mjs"), join(consumer, "runtime-import.mjs"));
  await cp(
    join(HERE, "consumer-smoke.ts.txt"),
    join(consumer, "consumer-smoke.ts")
  );

  const manifest = JSON.parse(
    await readFile(
      join(consumer, "node_modules/@atjsh/llmlingua-2/package.json"),
      "utf8"
    )
  );
  assert.equal(
    manifest.peerDependencies["@huggingface/transformers"],
    "^4.2.0"
  );

  for (const major of [22, 24, 26]) {
    run("npx", ["--yes", `node@${major}`, "runtime-import.mjs"], consumer);
  }
  run(
    process.execPath,
    [
      join(ROOT, "node_modules/typescript/bin/tsc"),
      "--noEmit",
      "--strict",
      "--target",
      "ES2022",
      "--module",
      "NodeNext",
      "--moduleResolution",
      "NodeNext",
      "--skipLibCheck",
      "consumer-smoke.ts",
    ],
    consumer
  );
} finally {
  await rm(consumer, { recursive: true, force: true });
  await rm(packDirectory, { recursive: true, force: true });
}
