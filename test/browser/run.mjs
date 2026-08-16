import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const distRoot = resolve(process.env.LLMLINGUA_DIST_ROOT ?? resolve(root, "dist"));
const tfjsRoot = process.env.LLMLINGUA_TFJS_ROOT
  ? resolve(process.env.LLMLINGUA_TFJS_ROOT)
  : null;
const captureGolden = process.env.CAPTURE_PRECHANGE_BROWSER_GOLDEN === "1";
const runModel = process.env.RUN_BROWSER_MODEL_E2E === "1" || captureGolden;

const playwrightSpecifier = process.env.PLAYWRIGHT_MODULE_PATH
  ? pathToFileURL(resolve(process.env.PLAYWRIGHT_MODULE_PATH)).href
  : "playwright";
const { chromium, firefox, webkit } = await import(playwrightSpecifier);
const browserTypes = { chromium, firefox, webkit };
const browserNames = (process.env.BROWSER_ENGINES ?? "chromium,firefox,webkit")
  .split(",")
  .map((name) => name.trim())
  .filter(Boolean);

const contentTypes = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
};

function inside(base, path) {
  const candidate = resolve(base, `.${path}`);
  return candidate === base || candidate.startsWith(`${base}${sep}`)
    ? candidate
    : null;
}

function importMapHtml() {
  return `<!doctype html>
<meta charset="utf-8">
<script type="importmap">
${JSON.stringify({
  imports: {
    "llmlingua-under-test": "/__dist/index.js",
    "@huggingface/transformers":
      "/node_modules/@huggingface/transformers/dist/transformers.web.js",
    "onnxruntime-common":
      "/node_modules/onnxruntime-web/node_modules/onnxruntime-common/dist/esm/index.js",
    "onnxruntime-web": "/node_modules/onnxruntime-web/dist/ort.bundle.min.mjs",
    "es-toolkit/array": "/node_modules/es-toolkit/dist/array/index.mjs",
    "@tensorflow/tfjs": "/__tfjs-wrapper.mjs",
  },
})}
</script>`;
}

const server = createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
    if (pathname === "/") {
      response.setHeader("content-type", contentTypes[".html"]);
      response.end(importMapHtml());
      return;
    }
    if (pathname === "/__tfjs-wrapper.mjs") {
      if (!tfjsRoot) throw new Error("The pre-change TFJS root was not provided");
      response.setHeader("content-type", contentTypes[".mjs"]);
      response.end(`await import("/__tfjs/dist/tf.min.js");
const { softmax, tensor3d } = globalThis.tf;
export { softmax, tensor3d };`);
      return;
    }

    let file;
    if (pathname.startsWith("/__dist/")) {
      file = inside(distRoot, pathname.slice("/__dist".length));
    } else if (pathname.startsWith("/__tfjs/")) {
      if (!tfjsRoot) throw new Error("The pre-change TFJS root was not provided");
      file = inside(tfjsRoot, pathname.slice("/__tfjs".length));
    } else {
      file = inside(root, pathname);
    }
    if (!file || !(await stat(file)).isFile()) throw new Error("Not found");

    response.setHeader(
      "content-type",
      contentTypes[extname(file)] ?? "application/octet-stream"
    );
    response.setHeader("cross-origin-resource-policy", "cross-origin");
    response.end(await readFile(file));
  } catch {
    response.statusCode = 404;
    response.end("Not found");
  }
});

await new Promise((resolveListen) => server.listen(0, "127.0.0.1", resolveListen));
const address = server.address();
const origin = `http://127.0.0.1:${address.port}`;

async function inBrowser(browserType, callback) {
  const browser = await browserType.launch();
  try {
    const page = await browser.newPage();
    const browserErrors = [];
    page.on("console", (message) => {
      if (message.type() === "error") browserErrors.push(message.text());
    });
    page.on("pageerror", (error) => browserErrors.push(error.stack ?? error.message));
    await page.goto(origin);
    const result = await callback(page);
    assert.deepEqual(browserErrors, []);
    return result;
  } finally {
    await browser.close();
  }
}

try {
  for (const name of browserNames) {
    const browserType = browserTypes[name];
    assert.ok(browserType, `Unknown browser engine: ${name}`);
    const result = await inBrowser(browserType, (page) =>
      page.evaluate(async () => {
        const { runMockCompressor } = await import("/test/browser/cases.mjs");
        return runMockCompressor();
      })
    );
    assert.deepEqual(result, ["gamma", "delta"]);
    console.log(`${name}: mocked compressor passed`);
  }

  if (runModel) {
    const actual = await inBrowser(chromium, (page) =>
      page.evaluate(async () => {
        const { runTinyBertWasm } = await import("/test/browser/cases.mjs");
        return runTinyBertWasm();
      })
    );
    if (captureGolden) {
      assert.notEqual(
        process.env.LLMLINGUA_DIST_ROOT,
        undefined,
        "Golden capture is restricted to an explicit pre-change dist"
      );
      console.log(JSON.stringify(actual, null, 2));
    } else {
      const golden = JSON.parse(
        await readFile(resolve(root, "test/browser/tinybert-wasm.golden.json"), "utf8")
      );
      assert.deepEqual(actual, golden.expected);
      console.log("chromium: TinyBERT WASM/fp32 golden passed");
    }
  }
} finally {
  await new Promise((resolveClose, rejectClose) =>
    server.close((error) => (error ? rejectClose(error) : resolveClose()))
  );
}
