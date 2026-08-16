import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../../..");
const DIST = resolve(ROOT, "dist");
const modulePath = process.env.PLAYWRIGHT_MODULE_PATH;
assert.ok(modulePath, "PLAYWRIGHT_MODULE_PATH is required");
const { chromium } = await import(pathToFileURL(resolve(modulePath)).href);
const runModel = process.env.LLMLINGUA_RUN_BROWSER_E2E === "1";
const runWebGpu = process.env.LLMLINGUA_RUN_BROWSER_WEBGPU === "1";
const golden = JSON.parse(
  await readFile(resolve(HERE, "../tinybert-golden.json"), "utf8")
);

const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
};

function inside(base, pathname) {
  const candidate = resolve(base, `.${pathname}`);
  return candidate === base || candidate.startsWith(`${base}${sep}`)
    ? candidate
    : null;
}

function html() {
  return `<!doctype html><meta charset="utf-8"><script type="importmap">${JSON.stringify(
    {
      imports: {
        "llmlingua-under-test": "/__dist/index.js",
        "llmlingua-e2e": "/__dist/e2e/long-texts.js",
        "@huggingface/transformers":
          "/node_modules/@huggingface/transformers/dist/transformers.web.js",
        "onnxruntime-web/webgpu":
          "/node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs",
        "onnxruntime-common":
          "/node_modules/onnxruntime-common/dist/esm/index.js",
        "es-toolkit/array": "/node_modules/es-toolkit/dist/array/index.mjs",
      },
    },
    null,
    2
  )}</script>`;
}

const server = createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(
      new URL(request.url, "http://localhost").pathname
    );
    if (pathname === "/") {
      response.setHeader("content-type", types[".html"]);
      response.end(html());
      return;
    }
    const file = pathname.startsWith("/__dist/")
      ? inside(DIST, pathname.slice("/__dist".length))
      : inside(ROOT, pathname);
    if (!file || !(await stat(file)).isFile()) throw new Error("Not found");
    response.setHeader(
      "content-type",
      types[extname(file)] ?? "application/octet-stream"
    );
    response.setHeader("cross-origin-resource-policy", "cross-origin");
    response.end(await readFile(file));
  } catch {
    response.statusCode = 404;
    response.end("Not found");
  }
});

await new Promise((resolveListen) => server.listen(0, "127.0.0.1", resolveListen));
const origin = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({ headless: !runWebGpu });
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.stack ?? error.message));
  await page.goto(origin);
  const mocked = await page.evaluate(async () => {
    const { runMock } = await import("/verification/gh24-v4/browser/cases.mjs");
    return runMock();
  });
  assert.equal(mocked, "ordinary");
  assert.deepEqual(errors, []);
  console.log("chromium: mocked v4 compressor passed");

  if (runModel) {
    const actual = await page.evaluate(async (inputIndex) => {
      const { runTinyBert } = await import(
        "/verification/gh24-v4/browser/cases.mjs"
      );
      return runTinyBert(inputIndex, "wasm");
    }, golden.expected.inputIndex);
    assert.deepEqual(actual, golden.expected.chromium);
    assert.deepEqual(errors, []);
    console.log("chromium: TinyBERT WASM/fp32 golden passed");
  }

  if (runWebGpu) {
    const hasAdapter = await page.evaluate(async () =>
      Boolean(navigator.gpu && (await navigator.gpu.requestAdapter()))
    );
    assert.equal(hasAdapter, true, "Chromium WebGPU adapter is required");
    const actual = await page.evaluate(async (inputIndex) => {
      const { runTinyBert } = await import(
        "/verification/gh24-v4/browser/cases.mjs"
      );
      return runTinyBert(inputIndex, "webgpu");
    }, golden.expected.inputIndex);
    assert.deepEqual(actual, golden.expected.webgpu);
    assert.deepEqual(errors, []);
    console.log("chromium: TinyBERT WebGPU/fp32 golden passed");
  }
} finally {
  await browser.close();
  await new Promise((resolveClose, rejectClose) =>
    server.close((error) => (error ? rejectClose(error) : resolveClose()))
  );
}
