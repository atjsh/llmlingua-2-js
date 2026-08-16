import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { softmax } from "@huggingface/transformers";
import { chromium } from "playwright";

const SHAPE = [50, 512, 2];
const WARMUPS = 5;
const SAMPLES = 11;
const baseline = JSON.parse(
  await readFile(
    resolve(dirname(fileURLToPath(import.meta.url)), "benchmark-postprocessing.baseline.json"),
    "utf8",
  ),
);

assert.deepEqual(baseline.shape, SHAPE);
assert.equal(baseline.warmups, WARMUPS);
assert.equal(baseline.samples, SAMPLES);

function makeLogits(length) {
  const logits = new Float32Array(length);
  let state = 0x12345678;
  for (let i = 0; i < length; ++i) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    logits[i] = (state / 2 ** 32) * 16 - 8;
  }
  return logits;
}

function score(logits, batches, sequenceLength, classes) {
  let checksum = 0;
  for (let batch = 0; batch < batches; ++batch) {
    const probabilities = Array.from({ length: sequenceLength }, (_, token) => {
      const offset = (batch * sequenceLength + token) * classes;
      return softmax(logits.subarray(offset, offset + classes))[1];
    });
    for (const probability of probabilities) checksum += probability;
  }
  return checksum;
}

function median(values) {
  return [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
}

function benchmarkNode() {
  const rows = SHAPE[0] * SHAPE[1];
  const logits = makeLogits(rows * SHAPE[2]);
  for (let i = 0; i < WARMUPS; ++i) score(logits, ...SHAPE);

  const timings = [];
  let checksum;
  for (let i = 0; i < SAMPLES; ++i) {
    const start = performance.now();
    checksum = score(logits, ...SHAPE);
    timings.push(performance.now() - start);
  }
  return { checksum, medianMs: median(timings) };
}

const browserBundle = resolve(
  dirname(fileURLToPath(import.meta.resolve("@huggingface/transformers"))),
  "transformers.js",
);
const browserModule = await readFile(browserBundle);
const browser = await chromium.launch();

try {
  const page = await browser.newPage();
  await page.route("http://benchmark.test/**", async (route) => {
    if (new URL(route.request().url()).pathname === "/transformers.js") {
      await route.fulfill({ body: browserModule, contentType: "text/javascript" });
      return;
    }
    await route.fulfill({
      body: `<!doctype html><script type="module">
        import { softmax } from "/transformers.js";
        globalThis.runBenchmark = ({ shape, warmups, samples }) => {
          const rows = shape[0] * shape[1];
          const logits = new Float32Array(rows * shape[2]);
          let state = 0x12345678;
          for (let i = 0; i < logits.length; ++i) {
            state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
            logits[i] = (state / 2 ** 32) * 16 - 8;
          }
          const score = () => {
            let checksum = 0;
            for (let batch = 0; batch < shape[0]; ++batch) {
              const probabilities = Array.from({ length: shape[1] }, (_, token) => {
                const offset = (batch * shape[1] + token) * shape[2];
                return softmax(logits.subarray(offset, offset + shape[2]))[1];
              });
              for (const probability of probabilities) checksum += probability;
            }
            return checksum;
          };
          for (let i = 0; i < warmups; ++i) score();
          const timings = [];
          let checksum;
          for (let i = 0; i < samples; ++i) {
            const start = performance.now();
            checksum = score();
            timings.push(performance.now() - start);
          }
          timings.sort((a, b) => a - b);
          return { checksum, medianMs: timings[Math.floor(timings.length / 2)] };
        };
        globalThis.benchmarkReady = true;
      </script>`,
      contentType: "text/html",
    });
  });
  await page.goto("http://benchmark.test/");
  await page.waitForFunction(() => globalThis.benchmarkReady === true);

  const node = benchmarkNode();
  const chromiumResult = await page.evaluate(
    ({ shape, warmups, samples }) =>
      globalThis.runBenchmark({ shape, warmups, samples }),
    { shape: SHAPE, warmups: WARMUPS, samples: SAMPLES },
  );

  assert.ok(Number.isFinite(node.checksum));
  assert.ok(Math.abs(node.checksum - chromiumResult.checksum) < 1e-6);

  console.log(`HF softmax postprocessing [${SHAPE}] (${WARMUPS} warmups, ${SAMPLES} samples)`);
  console.log(`Node ${process.version}: ${node.medianMs.toFixed(3)} ms`);
  console.log(`Chromium ${browser.version()}: ${chromiumResult.medianMs.toFixed(3)} ms`);
  console.log(
    `Pre-change TFJS: Node ${baseline.node.medianMs.toFixed(3)} ms, ` +
      `Chromium ${baseline.chromium.medianMs.toFixed(3)} ms`,
  );
  console.log(
    `Observed speedup: Node ${(baseline.node.medianMs / node.medianMs).toFixed(2)}x, ` +
      `Chromium ${(baseline.chromium.medianMs / chromiumResult.medianMs).toFixed(2)}x ` +
      "(informational only)",
  );
  console.log(`checksum: ${node.checksum.toFixed(6)}`);
} finally {
  await browser.close();
}
