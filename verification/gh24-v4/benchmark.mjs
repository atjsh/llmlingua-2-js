import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { performance } from "node:perf_hooks";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const SOURCE_ROOT = resolve(HERE, "../..");
const args = {};
for (let index = 2; index < process.argv.length; index++) {
  const key = process.argv[index];
  assert.match(key, /^--/);
  const next = process.argv[index + 1];
  args[key.slice(2)] = !next || next.startsWith("--") ? true : next;
  if (args[key.slice(2)] !== true) index++;
}
const packageRoot = resolve(String(args["package-root"] ?? SOURCE_ROOT));
const label = String(args.label ?? "working-tree");
const mockWarmups = Number(args["mock-warmups"] ?? 5);
const mockSamples = Number(args["mock-samples"] ?? 20);
const modelWarmups = Number(args["model-warmups"] ?? 2);
const modelSamples = Number(args["model-samples"] ?? 5);
const runModel = args["skip-model"] !== true;
const goldens = JSON.parse(
  readFileSync(join(HERE, "benchmark-golden.json"), "utf8")
);

const requireFromPackage = createRequire(join(packageRoot, "package.json"));
const transformersEntry = requireFromPackage.resolve("@huggingface/transformers");
const [{ LLMLingua2 }, transformers, { EXAMPLES }] = await Promise.all([
  import(pathToFileURL(join(packageRoot, "dist/index.js"))),
  import(pathToFileURL(transformersEntry)),
  import(pathToFileURL(join(SOURCE_ROOT, "dist/e2e/long-texts.js"))),
]);
const { AutoConfig, AutoModelForTokenClassification, AutoTokenizer, Tensor } =
  transformers;

function packageManifestFrom(entry, expectedName) {
  let directory = dirname(entry);
  while (directory !== dirname(directory)) {
    try {
      const manifest = JSON.parse(readFileSync(join(directory, "package.json"), "utf8"));
      if (manifest.name === expectedName) return manifest;
    } catch {}
    directory = dirname(directory);
  }
  throw new Error(`Could not locate ${expectedName}'s package.json`);
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function quantile(values, fraction) {
  const ordered = [...values].sort((a, b) => a - b);
  return ordered[Math.min(ordered.length - 1, Math.ceil(ordered.length * fraction) - 1)];
}

function memorySnapshot() {
  const memory = process.memoryUsage();
  return {
    rssMiB: memory.rss / 1024 / 1024,
    heapUsedMiB: memory.heapUsed / 1024 / 1024,
    externalMiB: memory.external / 1024 / 1024,
    maxRssMiB: process.resourceUsage().maxRSS / 1024,
  };
}

async function measure(name, warmups, samples, operation) {
  let expectedOutput;
  for (let index = 0; index < warmups; index++) {
    expectedOutput ??= await operation();
  }
  globalThis.gc?.();
  const memoryBefore = memorySnapshot();
  const wallMs = [];
  const cpuMs = [];

  for (let index = 0; index < samples; index++) {
    const cpuStart = process.cpuUsage();
    const wallStart = performance.now();
    const output = await operation();
    wallMs.push(performance.now() - wallStart);
    const cpu = process.cpuUsage(cpuStart);
    cpuMs.push((cpu.user + cpu.system) / 1000);
    assert.equal(output, expectedOutput, `${name} output changed at sample ${index}`);
  }

  globalThis.gc?.();
  const memoryAfter = memorySnapshot();
  return {
    warmups,
    samples,
    outputBytes: Buffer.byteLength(expectedOutput),
    outputSha256: sha256(expectedOutput),
    wallMs: {
      median: quantile(wallMs, 0.5),
      p95: quantile(wallMs, 0.95),
    },
    cpuMs: {
      median: quantile(cpuMs, 0.5),
      p95: quantile(cpuMs, 0.95),
    },
    memoryMiB: {
      rssBefore: memoryBefore.rssMiB,
      rssAfter: memoryAfter.rssMiB,
      rssDelta: memoryAfter.rssMiB - memoryBefore.rssMiB,
      heapUsedDelta: memoryAfter.heapUsedMiB - memoryBefore.heapUsedMiB,
      externalDelta: memoryAfter.externalMiB - memoryBefore.externalMiB,
      processMaxRss: memoryAfter.maxRssMiB,
    },
  };
}

function createMockCompressor() {
  const tokenToId = new Map([["[PAD]", 0], ["[CLS]", 101], ["[SEP]", 102]]);
  const idToToken = new Map([...tokenToId].map(([token, id]) => [id, token]));
  const words = Array.from({ length: 510 }, (_, index) => `w${index}`);
  words.forEach((word, index) => {
    tokenToId.set(word, 1000 + index);
    idToToken.set(1000 + index, word);
  });
  const contexts = Array(50).fill(words.join(" "));
  const tokenizer = async (values) => {
    const rows = values.map((value) => tokenizer.encode(value));
    const width = Math.max(...rows.map((row) => row.length));
    assert.deepEqual([rows.length, width], [50, 512]);
    return {
      input_ids: new Tensor(
        "int64",
        BigInt64Array.from(rows.flat(), BigInt),
        [rows.length, width]
      ),
      attention_mask: new Tensor(
        "int64",
        BigInt64Array.from(Array(rows.length * width).fill(1), BigInt),
        [rows.length, width]
      ),
    };
  };
  tokenizer.tokenize = (text, { add_special_tokens = false } = {}) => {
    const tokens = text.trim().split(/\s+/).filter(Boolean);
    return add_special_tokens ? ["[CLS]", ...tokens, "[SEP]"] : tokens;
  };
  tokenizer.encode = (text, { add_special_tokens = true } = {}) => {
    const ids = tokenizer.tokenize(text).map((token) => tokenToId.get(token));
    assert.ok(ids.every(Number.isInteger));
    return add_special_tokens ? [101, ...ids, 102] : ids;
  };
  tokenizer.decode = (ids, { skip_special_tokens = true } = {}) =>
    Array.from(ids, Number)
      .map((id) => idToToken.get(id))
      .filter(Boolean)
      .filter((token) => !skip_special_tokens || !["[PAD]", "[CLS]", "[SEP]"].includes(token))
      .join(" ");
  tokenizer.all_special_ids = [0, 101, 102];

  const model = async ({ input_ids }) => {
    assert.deepEqual(input_ids.dims, [50, 512]);
    const logits = Array.from(input_ids.data, Number).flatMap((id) => {
      const score = id >= 1000 ? ((id - 1000) / 509) * 8 - 4 : -8;
      return [0, score];
    });
    return {
      logits: new Tensor("float32", Float32Array.from(logits), [50, 512, 2]),
    };
  };
  const compressor = new LLMLingua2.PromptCompressor(
    model,
    tokenizer,
    LLMLingua2.get_pure_tokens_bert_base_multilingual_cased,
    LLMLingua2.is_begin_of_new_word_bert_base_multilingual_cased,
    { encode: () => [1] },
    undefined,
    () => {}
  );
  const options = {
    reduce_rate: 0.5,
    token_to_word: "mean",
    force_tokens: [],
    token_map: {},
    force_token_id_sequences: [],
    force_token_sequences: [],
    force_reserve_digit: false,
    drop_consecutive: false,
  };

  return async () => {
    const directContexts = packageRoot === SOURCE_ROOT
      ? contexts.map((text) => ({ text, separatorBefore: "" }))
      : contexts;
    const output = await compressor.compressContexts(directContexts, options);
    assert.equal(output.length, 50);
    return JSON.stringify(output);
  };
}

async function createTinyBertCompressor() {
  const modelName = "atjsh/llmlingua-2-js-tinybert-meetingbank";
  const revision = "a9af82841d3f815c9c492b13791c6517154791d3";
  const options = { revision };
  const config = await AutoConfig.from_pretrained(modelName, options);
  const tokenizer = await AutoTokenizer.from_pretrained(modelName, {
    ...options,
    config,
  });
  const model = await AutoModelForTokenClassification.from_pretrained(modelName, {
    ...options,
    config,
    device: "cpu",
    dtype: "fp32",
    subfolder: "",
  });
  const compressor = new LLMLingua2.PromptCompressor(
    model,
    tokenizer,
    LLMLingua2.get_pure_tokens_bert_base_multilingual_cased,
    LLMLingua2.is_begin_of_new_word_bert_base_multilingual_cased,
    { encode: () => [1] },
    undefined,
    () => {}
  );

  return {
    dispose: () => model.dispose?.(),
    operation: () => compressor.compress(EXAMPLES[1], { rate: 0.5 }),
    provenance: { modelName, revision, input: "src/e2e/long-texts.ts#sorted EXAMPLES[1]" },
  };
}

const manifest = requireFromPackage("./package.json");
const result = {
  label,
  packageVersion: manifest.version,
  transformersVersion: packageManifestFrom(
    transformersEntry,
    "@huggingface/transformers"
  ).version,
  runtime: {
    node: process.version,
    platform: process.platform,
    arch: process.arch,
    cpu: process.report.getReport().header.cpus[0]?.model,
  },
  mock: {
    shape: [50, 512, 2],
    ...(await measure("mock", mockWarmups, mockSamples, createMockCompressor())),
  },
};

function verifyGolden(name, measurement) {
  const expected = goldens[label]?.[name];
  if (!expected) return;
  assert.equal(measurement.outputSha256, expected.outputSha256);
  assert.equal(measurement.outputBytes, expected.outputBytes);
  measurement.goldenVerified = true;
}

verifyGolden("mock", result.mock);

if (runModel) {
  const tinyBert = await createTinyBertCompressor();
  try {
    result.tinyBert = {
      device: "cpu",
      dtype: "fp32",
      ...tinyBert.provenance,
      ...(await measure("TinyBERT", modelWarmups, modelSamples, tinyBert.operation)),
    };
    verifyGolden("tinyBert", result.tinyBert);
  } finally {
    await tinyBert.dispose();
  }
}

process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
