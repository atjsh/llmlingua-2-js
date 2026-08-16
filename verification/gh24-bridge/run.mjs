import assert from "node:assert/strict";
import { cp, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../..");
const VERSIONS = process.argv.slice(2);
if (VERSIONS.length === 0) VERSIONS.push("3.5.2", "4.0.0", "4.2.0");

function run(command, args, cwd = ROOT, extraEnvironment = {}) {
  const result = spawnSync(command, args, {
    cwd,
    encoding: "utf8",
    stdio: "inherit",
    env: {
      ...process.env,
      npm_config_audit: "false",
      npm_config_fund: "false",
      ...extraEnvironment,
    },
  });
  assert.equal(result.status, 0, `${command} ${args.join(" ")} failed`);
}

run("corepack", ["yarn", "build"]);

const packDirectory = await mkdtemp(join(tmpdir(), "llmlingua-gh24-pack-"));
try {
  const pack = spawnSync(
    "npm",
    ["pack", "--json", "--pack-destination", packDirectory],
    { cwd: ROOT, encoding: "utf8" }
  );
  assert.equal(pack.status, 0, pack.stderr);
  const [{ filename }] = JSON.parse(pack.stdout);
  const tarball = join(packDirectory, filename);

  for (const version of VERSIONS) {
    const consumer = await mkdtemp(
      join(tmpdir(), `llmlingua-gh24-hf-${version}-`)
    );
    try {
      assert.ok(consumer.startsWith(join(tmpdir(), "llmlingua-gh24-hf-")));
      assert.notEqual(resolve(consumer), ROOT);
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
          `@huggingface/transformers@${version}`,
          "js-tiktoken@1.0.20",
        ],
        consumer
      );
      await cp(join(HERE, "consumer.test.mjs"), join(consumer, "consumer.test.mjs"));
      await cp(
        join(HERE, "consumer-smoke.ts.txt"),
        join(consumer, "consumer-smoke.ts")
      );

      const installedManifest = JSON.parse(
        await readFile(
          join(
            consumer,
            "node_modules/@atjsh/llmlingua-2/package.json"
          ),
          "utf8"
        )
      );
      assert.equal(
        installedManifest.peerDependencies["@huggingface/transformers"],
        "^3.5.2 || ^4.0.0"
      );

      run(
        process.execPath,
        ["--test", "consumer.test.mjs"],
        consumer,
        { LLMLINGUA_REPOSITORY_ROOT: ROOT }
      );
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
    }
  }
} finally {
  await rm(packDirectory, { recursive: true, force: true });
}
