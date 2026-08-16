# GH-24 v4 verification

Temporary, history-only verification for the v4-only 3.0 implementation. Remove
this directory in the cleanup commit after its passing result is committed.

Run the deterministic mock, package, type, and Node 22/24/26 checks:

```sh
node verification/gh24-v4/run.mjs
```

Run the pinned TinyBERT CPU/fp32 check using `src/e2e/long-texts.ts`:

```sh
LLMLINGUA_RUN_MODEL_E2E=1 node verification/gh24-v4/run.mjs
```

Run the Chromium WASM/fp32 golden after making Playwright 1.62.1 importable:

```sh
PLAYWRIGHT_MODULE_PATH=/absolute/path/to/playwright/index.mjs \
  LLMLINGUA_RUN_BROWSER_E2E=1 \
  node verification/gh24-v4/browser/run.mjs
```

On a machine with a real Chromium WebGPU adapter, run the opt-in accelerated
golden in a headed browser:

```sh
PLAYWRIGHT_MODULE_PATH=/absolute/path/to/playwright/index.mjs \
  LLMLINGUA_RUN_BROWSER_WEBGPU=1 \
  node verification/gh24-v4/browser/run.mjs
```

The runner builds and packs the repository, installs the tarball into an OS
temporary directory, runs imports with Node 22, 24, and 26, and deletes the
temporary files. It does not alter the root manifest or lockfile.

The real-model goldens are pinned to TinyBERT revision
`a9af82841d3f815c9c492b13791c6517154791d3`. They are captured once while
creating this migration evidence. A later mismatch blocks the change; never
update a golden merely to make the refactor pass.

Run the non-gating CPU/RAM benchmark for the 3.0 candidate:

```sh
node --expose-gc verification/gh24-v4/benchmark.mjs \
  --label 3.0.0-candidate
```

For the recorded 2.0.5 comparison, install the published baseline and HF 4.2.0
into an OS temporary directory, then pass its package directory:

```sh
llmlingua_benchmark_dir=$(mktemp -d /tmp/llmlingua-gh24-benchmark.XXXXXX)
npm install --prefix "$llmlingua_benchmark_dir" --ignore-scripts \
  --no-audit --no-fund @atjsh/llmlingua-2@2.0.5 \
  @huggingface/transformers@4.2.0 js-tiktoken@1.0.20
node --expose-gc verification/gh24-v4/benchmark.mjs \
  --label 2.0.5-baseline \
  --package-root "$llmlingua_benchmark_dir/node_modules/@atjsh/llmlingua-2"
```

The benchmark excludes installation and model loading, verifies committed
output hashes, and reports wall time, process CPU time, and memory. Results and
host provenance are in `benchmark-results.md`.
