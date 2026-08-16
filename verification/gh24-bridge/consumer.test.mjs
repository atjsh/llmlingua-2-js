import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";

import { LLMLingua2 } from "@atjsh/llmlingua-2";
import {
  AutoConfig,
  AutoModelForTokenClassification,
  AutoTokenizer,
  Tensor,
} from "@huggingface/transformers";
import { Tiktoken } from "js-tiktoken/lite";
import o200kBase from "js-tiktoken/ranks/o200k_base";

const ROOT = process.env.LLMLINGUA_REPOSITORY_ROOT;
assert.ok(ROOT, "LLMLINGUA_REPOSITORY_ROOT is required");

const golden = JSON.parse(
  await readFile(
    resolve(ROOT, "verification/gh24-bridge/tinybert-golden.json"),
    "utf8"
  )
);
const xlmGolden = JSON.parse(
  await readFile(
    resolve(ROOT, "verification/gh24-bridge/xlm-tokenizer-golden.json"),
    "utf8"
  )
);

function tensor(type, values, dims) {
  const data =
    type === "int64"
      ? BigInt64Array.from(values, BigInt)
      : Float32Array.from(values);
  return new Tensor(type, data, dims);
}

function createPublicTokenizer() {
  const tokenToId = new Map([
    ["[PAD]", 0],
    ["[CLS]", 101],
    ["[SEP]", 102],
    ["alpha", 11],
    ["beta", 12],
    ["[END]", 13],
  ]);
  const idToToken = new Map(
    [...tokenToId].map(([token, id]) => [id, token])
  );
  const split = (text) => text.trim().split(/\s+/).filter(Boolean);

  const tokenizer = async (contexts) => {
    const values = Array.isArray(contexts) ? contexts : [contexts];
    const rows = values.map((value) => tokenizer.encode(value));
    const width = Math.max(...rows.map((row) => row.length));
    const ids = [];
    const masks = [];

    for (const row of rows) {
      ids.push(...row, ...Array(width - row.length).fill(0));
      masks.push(...Array(row.length).fill(1), ...Array(width - row.length).fill(0));
    }

    return {
      input_ids: tensor("int64", ids, [rows.length, width]),
      attention_mask: tensor("int64", masks, [rows.length, width]),
    };
  };

  tokenizer.tokenize = (text, { add_special_tokens = false } = {}) => {
    const tokens = split(text);
    return add_special_tokens ? ["[CLS]", ...tokens, "[SEP]"] : tokens;
  };
  tokenizer.encode = (text, { add_special_tokens = true } = {}) => {
    const ids = split(text).map((token) => {
      const id = tokenToId.get(token);
      if (id === undefined) throw new Error(`Unknown mock token: ${token}`);
      return id;
    });
    return add_special_tokens ? [101, ...ids, 102] : ids;
  };
  tokenizer.decode = (ids, options = {}) => {
    const skipSpecialTokens = options.skip_special_tokens ?? true;
    return Array.from(ids, Number)
      .map((id) => idToToken.get(id))
      .filter(Boolean)
      .filter(
        (token) =>
          token !== "[END]" &&
          (!skipSpecialTokens || !["[PAD]", "[CLS]", "[SEP]"].includes(token))
      )
      .join(" ");
  };
  tokenizer.all_special_ids = [0, 101, 102];

  assert.equal("decoder" in tokenizer, false);
  assert.equal("model" in tokenizer, false);
  assert.equal("special_tokens" in tokenizer, false);
  return tokenizer;
}

function createModel() {
  return async ({ input_ids }) => {
    const logits = Array.from(input_ids.data, Number).flatMap((id) => {
      if (id === 11) return [0, 3];
      if (id === 12) return [0, -3];
      return [0, 0];
    });
    return {
      logits: tensor("float32", logits, [...input_ids.dims, 2]),
    };
  };
}

function createPinnedXLMMockModel() {
  // These preceding-token scores deliberately account for the 2.0.5 ID-0
  // probability shift and retain: Hello, café, 東京, omega.
  const highScoreIds = new Set([0, 4, 6, 292, 51703]);
  return async ({ input_ids }) => ({
    logits: tensor(
      "float32",
      Array.from(input_ids.data, Number).flatMap((id) => [
        0,
        highScoreIds.has(id) ? 4 : -4,
      ]),
      [...input_ids.dims, 2]
    ),
  });
}

function createXLMTokenizer() {
  const tokenToId = new Map([
    ["<s>", 0],
    ["</s>", 2],
    ["alpha", 11],
    ["beta", 12],
    ["[END]", 13],
  ]);
  const idToToken = new Map(
    [...tokenToId].map(([token, id]) => [id, token])
  );
  const split = (text) => text.trim().split(/\s+/).filter(Boolean);

  const tokenizer = async (contexts) => {
    const values = Array.isArray(contexts) ? contexts : [contexts];
    const rows = values.map((value) => tokenizer.encode(value));
    const width = Math.max(...rows.map((row) => row.length));
    const ids = [];
    const masks = [];
    for (const row of rows) {
      ids.push(...row, ...Array(width - row.length).fill(2));
      masks.push(...Array(row.length).fill(1), ...Array(width - row.length).fill(0));
    }
    return {
      input_ids: tensor("int64", ids, [rows.length, width]),
      attention_mask: tensor("int64", masks, [rows.length, width]),
    };
  };

  tokenizer.tokenize = (text, { add_special_tokens = false } = {}) => {
    const tokens = split(text).map((token) => `▁${token}`);
    return add_special_tokens ? ["<s>", ...tokens, "</s>"] : tokens;
  };
  tokenizer.encode = (text, { add_special_tokens = true } = {}) => {
    const ids = split(text).map((token) => tokenToId.get(token));
    assert.ok(ids.every((id) => id !== undefined));
    return add_special_tokens ? [0, ...ids, 2] : ids;
  };
  tokenizer.decode = (ids) =>
    Array.from(ids, Number)
      .map((id) => idToToken.get(id))
      .filter((token) => token && !["<s>", "</s>", "[END]"].includes(token))
      .join(" ");
  tokenizer.all_special_ids = [0, 2];
  return tokenizer;
}

function createCompressor(tokenizer = createPublicTokenizer(), logger = () => {}) {
  return new LLMLingua2.PromptCompressor(
    createModel(),
    tokenizer,
    LLMLingua2.get_pure_tokens_bert_base_multilingual_cased,
    LLMLingua2.is_begin_of_new_word_bert_base_multilingual_cased,
    { encode: () => [1] },
    undefined,
    logger
  );
}

test("compresses with tokenizer public APIs only", async () => {
  const output = await createCompressor().compress("alpha beta [END]", {
    rate: 0.5,
    chunkEndTokens: [],
  });
  assert.equal(output, "alpha");
});

test("preserves the legacy terminal-token drop", async () => {
  const output = await createCompressor().compress("alpha beta", {
    rate: 1,
    chunkEndTokens: [],
  });
  assert.equal(output, "alpha");
});

test("preserves the legacy XLM active-ID-0 probability shift", async () => {
  const compressor = new LLMLingua2.PromptCompressor(
    createModel(),
    createXLMTokenizer(),
    LLMLingua2.get_pure_tokens_xlm_roberta_large,
    LLMLingua2.is_begin_of_new_word_xlm_roberta_large,
    { encode: () => [1] },
    undefined,
    () => {}
  );
  const output = await compressor.compress("alpha beta [END]", {
    rate: 0.5,
    chunkEndTokens: [],
  });

  // BOS id 0 is removed without removing its probability: beta receives
  // alpha's high score. This intentionally remains until the 3.0 correction.
  assert.equal(output, " beta");
});

test("factory forwards the supplied logger into the compressor", async () => {
  const originalTokenizerLoader = AutoTokenizer.from_pretrained;
  const originalModelLoader = AutoModelForTokenClassification.from_pretrained;
  const originalConsoleLog = console.log;
  const suppliedLogs = [];
  const defaultLogs = [];

  AutoTokenizer.from_pretrained = async () => createPublicTokenizer();
  AutoModelForTokenClassification.from_pretrained = async () => createModel();
  console.log = (...message) => defaultLogs.push(message);

  try {
    const { promptCompressor } = await LLMLingua2.WithBERTMultilingual(
      "mock-model",
      {
        transformerJSConfig: { device: "cpu", dtype: "fp32" },
        oaiTokenizer: { encode: () => [1] },
        pretrainedConfig: {},
        logger: (...message) => suppliedLogs.push(message),
      }
    );
    await promptCompressor.compress("alpha beta [END]", {
      rate: 0.5,
      chunkEndTokens: [],
    });
  } finally {
    AutoTokenizer.from_pretrained = originalTokenizerLoader;
    AutoModelForTokenClassification.from_pretrained = originalModelLoader;
    console.log = originalConsoleLog;
  }

  assert.equal(defaultLogs.length, 0, "compressor leaked logs to console.log");
  assert.ok(
    suppliedLogs.some(([message]) => message === "original token length: appx. "),
    "supplied logger did not receive compressor messages"
  );
  assert.ok(
    suppliedLogs.some(([message]) => message === "model inference finished"),
    "supplied logger did not receive inference messages"
  );
});

test("pinned TinyBERT tokenizer supports the public bridge contract", async () => {
  const { model, revision } = golden.expected;
  const tokenizer = await AutoTokenizer.from_pretrained(model, { revision });
  const input = "alpha beta";
  const tokens = tokenizer.tokenize(input);
  const ids = tokenizer.encode(input, { add_special_tokens: false });

  assert.ok(tokens.length > 0);
  assert.equal(ids.length, tokens.length);
  assert.equal(
    tokenizer.decode(ids, {
      skip_special_tokens: true,
      clean_up_tokenization_spaces: false,
    }),
    input
  );
});

test("pinned XLM tokenizer and mock logits preserve exact leading-space bytes", async () => {
  const { model, revision, input, output, length, byteLength, utf8Hex, sha256 } =
    xlmGolden.expected;
  const tokenizer = await AutoTokenizer.from_pretrained(model, { revision });
  const compressor = new LLMLingua2.PromptCompressor(
    createPinnedXLMMockModel(),
    tokenizer,
    LLMLingua2.get_pure_tokens_xlm_roberta_large,
    LLMLingua2.is_begin_of_new_word_xlm_roberta_large,
    { encode: () => [1] },
    undefined,
    () => {}
  );
  const [actual] = await compressor.compressContexts([input], {
    reduce_rate: 0.5,
    token_to_word: "mean",
    force_tokens: [],
    token_map: {},
    force_reserve_digit: false,
    drop_consecutive: false,
  });

  assert.equal(actual, output);
  assert.equal(actual.length, length);
  assert.equal(Buffer.byteLength(actual), byteLength);
  assert.equal(Buffer.from(actual).toString("hex"), utf8Hex);
  assert.equal(createHash("sha256").update(actual).digest("hex"), sha256);
});

test(
  "TinyBERT CPU/fp32 output matches the frozen 2.0.4 golden",
  { skip: process.env.LLMLINGUA_RUN_MODEL_E2E !== "1", timeout: 300_000 },
  async () => {
    const {
      model: modelName,
      revision,
      device,
      dtype,
      length,
      sha256,
    } = golden.expected;
    const { EXAMPLES } = await import(
      pathToFileURL(resolve(ROOT, "dist/e2e/long-texts.js"))
    );
    const loadOptions = { revision };
    const config = await AutoConfig.from_pretrained(modelName, loadOptions);
    const tokenizer = await AutoTokenizer.from_pretrained(modelName, {
      ...loadOptions,
      config,
    });
    const model = await AutoModelForTokenClassification.from_pretrained(
      modelName,
      {
        ...loadOptions,
        config,
        device,
        dtype,
        subfolder: "",
      }
    );

    try {
      const compressor = new LLMLingua2.PromptCompressor(
        model,
        tokenizer,
        LLMLingua2.get_pure_tokens_bert_base_multilingual_cased,
        LLMLingua2.is_begin_of_new_word_bert_base_multilingual_cased,
        new Tiktoken(o200kBase),
        undefined,
        () => {}
      );
      const output = await compressor.compress(EXAMPLES[5], { rate: 0.5 });

      assert.equal(output.length, length);
      assert.equal(createHash("sha256").update(output).digest("hex"), sha256);
    } finally {
      await model.dispose?.();
    }
  }
);
