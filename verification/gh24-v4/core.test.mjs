import assert from "node:assert/strict";
import test from "node:test";

import {
  AutoModelForTokenClassification,
  AutoTokenizer,
  Tensor,
} from "@huggingface/transformers";

import { LLMLingua2 } from "../../dist/index.js";
import { percentile } from "../../dist/lib/llmlingua-2/utils.js";

const SCORE = {
  low: [0, -3, 0],
  middle: [0, 0, 0],
  high: [0, 2, 0],
  highest: [0, 4, 0],
  trap: [0, 2, 10],
};

test("percentile keeps its inclusive interpolation and range contract", () => {
  const values = [7, 15, 36, 39, 40, 41];
  assert.equal(percentile(values, 25), 20.25);
  assert.equal(percentile([], 50), 0);
  assert.throws(() => percentile([], -1), RangeError);
  assert.throws(() => percentile(values, Number.NaN), RangeError);
  assert.deepEqual(values, [7, 15, 36, 39, 40, 41]);
});

function trackedTensor(type, values, dims, records, name) {
  const data =
    type === "int64"
      ? BigInt64Array.from(values, BigInt)
      : type === "float16"
        ? Uint16Array.from(values)
        : Float32Array.from(values);
  const tensor = new Tensor(type, data, dims);
  const originalDispose = tensor.dispose.bind(tensor);
  const record = { name, disposeCalls: 0, tensor };
  records.push(record);
  tensor.dispose = () => {
    record.disposeCalls++;
    if (record.disposeCalls === 1) originalDispose();
  };
  return tensor;
}

function createTokenizer({
  records = [],
  xlm = false,
  throwOnRowWrapper = false,
} = {}) {
  const specials = xlm
    ? { pad: "<pad>", cls: "<s>", sep: "</s>", padId: 1, clsId: 0, sepId: 2 }
    : { pad: "[PAD]", cls: "[CLS]", sep: "[SEP]", padId: 0, clsId: 101, sepId: 102 };
  const tokenToId = new Map([
    [specials.pad, specials.padId],
    [specials.cls, specials.clsId],
    [specials.sep, specials.sepId],
  ]);
  const idToToken = new Map([...tokenToId].map(([token, id]) => [id, token]));
  let nextId = 1000;
  const calls = [];

  const split = (text) => text.trim().split(/\s+/).filter(Boolean);
  const idFor = (token) => {
    if (!tokenToId.has(token)) {
      while (idToToken.has(nextId)) nextId++;
      tokenToId.set(token, nextId);
      idToToken.set(nextId, token);
      nextId++;
    }
    return tokenToId.get(token);
  };

  const tokenizer = async (contexts, options = {}) => {
    const values = Array.isArray(contexts) ? contexts : [contexts];
    calls.push({ contexts: [...values], options });
    const rows = values.map((value) => tokenizer.encode(value));
    const width = Math.max(...rows.map((row) => row.length));
    const ids = [];
    const masks = [];
    const types = [];
    for (const row of rows) {
      const padding = width - row.length;
      ids.push(...row, ...Array(padding).fill(specials.padId));
      masks.push(...Array(row.length).fill(1), ...Array(padding).fill(0));
      types.push(...Array(width).fill(0));
    }
    const inputIds = trackedTensor(
      "int64",
      ids,
      [rows.length, width],
      records,
      "input_ids"
    );
    const attentionMask = trackedTensor(
      "int64",
      masks,
      [rows.length, width],
      records,
      "attention_mask"
    );
    if (throwOnRowWrapper) {
      attentionMask._getitem = () => {
        throw new Error("attention-mask row wrapper was allocated");
      };
    }
    return {
      input_ids: inputIds,
      attention_mask: attentionMask,
      token_type_ids: trackedTensor(
        "int64",
        types,
        [rows.length, width],
        records,
        "token_type_ids"
      ),
    };
  };

  tokenizer.tokenize = (text, { add_special_tokens = false } = {}) => {
    const tokens = split(text);
    const visible = xlm ? tokens.map((token) => `▁${token}`) : tokens;
    return add_special_tokens
      ? [specials.cls, ...visible, specials.sep]
      : visible;
  };
  tokenizer.encode = (text, { add_special_tokens = true } = {}) => {
    const ids = tokenizer
      .tokenize(text)
      .map((token) => idFor(token));
    return add_special_tokens
      ? [specials.clsId, ...ids, specials.sepId]
      : ids;
  };
  tokenizer.decode = (ids, { skip_special_tokens = true } = {}) =>
    Array.from(ids, Number)
      .map((id) => idToToken.get(id))
      .filter(Boolean)
      .filter(
        (token) =>
          !skip_special_tokens ||
          ![specials.pad, specials.cls, specials.sep].includes(token)
      )
      .map((token) => (xlm ? token.replace(/^▁/, "") : token))
      .join(" ");
  tokenizer.all_special_ids = [specials.padId, specials.clsId, specials.sepId];

  return { calls, idFor, idToToken, specials, tokenizer };
}

function createHarness({
  logitsByToken = {},
  oaiTokenizer = { encode: () => [1] },
  tokenizerOptions,
  float16 = false,
  logger,
} = {}) {
  const records = [];
  const tokenizerHarness = createTokenizer({ records, ...tokenizerOptions });
  const modelCalls = [];
  const model = async (inputs) => {
    const [batch, length] = inputs.input_ids.dims;
    modelCalls.push({
      keys: Object.keys(inputs),
      inputIds: Array.from(inputs.input_ids.data, Number),
      attentionMask: Array.from(inputs.attention_mask.data, Number),
      dims: [batch, length],
    });
    const logits = Array.from(inputs.input_ids.data, Number).flatMap((id) => {
      const token = tokenizerHarness.idToToken.get(id);
      return logitsByToken[token] ?? SCORE.middle;
    });

    if (!float16) {
      return {
        logits: trackedTensor(
          "float32",
          logits,
          [batch, length, 3],
          records,
          "logits"
        ),
      };
    }

    const half = trackedTensor(
      "float16",
      Array(logits.length).fill(0),
      [batch, length, 3],
      records,
      "logits-fp16"
    );
    let toCalls = 0;
    half.to = (type) => {
      toCalls++;
      assert.equal(type, "float32");
      return trackedTensor(
        "float32",
        logits,
        [batch, length, 3],
        records,
        "logits-converted"
      );
    };
    half.getToCalls = () => toCalls;
    return {
      logits: half,
      auxiliary: trackedTensor(
        "float32",
        [1],
        [1],
        records,
        "output-auxiliary"
      ),
    };
  };

  const compressor = new LLMLingua2.PromptCompressor(
    model,
    tokenizerHarness.tokenizer,
    tokenizerOptions?.xlm
      ? LLMLingua2.get_pure_tokens_xlm_roberta_large
      : LLMLingua2.get_pure_tokens_bert_base_multilingual_cased,
    tokenizerOptions?.xlm
      ? LLMLingua2.is_begin_of_new_word_xlm_roberta_large
      : LLMLingua2.is_begin_of_new_word_bert_base_multilingual_cased,
    oaiTokenizer,
    undefined,
    logger
  );

  return {
    compressor,
    modelCalls,
    records,
    ...tokenizerHarness,
  };
}

async function compress(logitsByToken, context, options = {}, harnessOptions = {}) {
  const harness = createHarness({ logitsByToken, ...harnessOptions });
  const output = await harness.compressor.compress(context, {
    rate: 0.5,
    chunkEndTokens: [],
    ...options,
  });
  return { ...harness, output };
}

test("chunks at 510 content tokens and retains every terminal token", async () => {
  for (const [length, expected] of [
    [1, [1]],
    [509, [509]],
    [510, [510]],
    [511, [510, 1]],
    [1020, [510, 510]],
  ]) {
    const words = Array.from({ length }, (_, index) => `t${index}`);
    const harness = createHarness({
      logitsByToken: { [`t${length - 1}`]: SCORE.highest },
    });
    const output = await harness.compressor.compress(words.join(" "), {
      rate: 0.5,
      forceTokens: [`t${length - 1}`],
      chunkEndTokens: [],
    });
    const chunks = harness.calls.flatMap(({ contexts }) => contexts);
    assert.deepEqual(
      chunks.map((chunk) => harness.tokenizer.tokenize(chunk).length),
      expected
    );
    assert.equal(
      harness.tokenizer.tokenize(chunks.at(-1)).at(-1),
      `t${length - 1}`
    );
    assert.ok(
      harness.calls.every(({ options }) => options.max_length === 512),
      "tokenizer max_length was not pinned to the model limit"
    );
    assert.doesNotMatch(output, /\n/);
    assert.match(output, new RegExp(`t${length - 1}$`));
  }

  const words = Array.from({ length: 600 }, (_, index) => `word${index}`);
  const preserved = await createHarness().compressor.compress(words.join(" "), {
    rate: 0,
    forceReserveDigit: true,
    chunkEndTokens: [],
  });
  assert.equal(preserved, words.join(" "));
});

test("keeps XLM token ID 0 aligned with its probability", async () => {
  const { output } = await compress(
    {
      "<s>": SCORE.highest,
      "▁alpha": SCORE.high,
      "▁beta": SCORE.trap,
      "</s>": SCORE.middle,
    },
    "alpha beta",
    {},
    { tokenizerOptions: { xlm: true } }
  );
  assert.equal(output, "alpha");
});

test("uses the full class row before selecting class 1", async () => {
  const { output } = await compress(
    { trap: SCORE.trap, ordinary: SCORE.middle },
    "trap ordinary"
  );
  assert.equal(output, "ordinary");
});

test("uses padding masks and flat offsets across a chunk batch", async () => {
  const words = Array.from({ length: 511 }, (_, index) => `w${index}`);
  const harness = createHarness({
    logitsByToken: {
      w509: SCORE.high,
      w510: SCORE.highest,
    },
    tokenizerOptions: { throwOnRowWrapper: true },
  });
  const output = await harness.compressor.compress(words.join(" "), {
    rate: 0.5,
    forceTokens: ["w510"],
    chunkEndTokens: [],
  });
  assert.match(output, /w509/);
  assert.match(output, /w510/);
  const { attentionMask, dims, keys } = harness.modelCalls[0];
  assert.deepEqual(dims, [2, 512]);
  assert.ok(keys.includes("token_type_ids"));
  const first = attentionMask.slice(0, 512);
  const second = attentionMask.slice(512);
  assert.ok(first.every((value) => value === 1));
  assert.deepEqual(second.slice(0, 3), [1, 1, 1]);
  assert.ok(second.slice(3).every((value) => value === 0));
});

test("uses the OAI tokenizer for target-token rate calculation", async () => {
  const oaiTokenizer = {
    encode: (text) => (text.includes(" ") ? Array(8).fill(1) : [1]),
  };
  const { output } = await compress(
    {
      one: SCORE.low,
      two: SCORE.middle,
      three: SCORE.high,
      four: SCORE.highest,
    },
    "one two three four",
    { targetToken: 2 },
    { oaiTokenizer }
  );
  assert.equal(output, "four");
});

test("forces literal and overlapping token-ID phrases", async () => {
  const literal = await compress(
    {},
    "drop C++ guide omit",
    { rate: 0, forceTokens: ["C++ guide"] }
  );
  assert.equal(literal.output, "C++ guide");

  const overlap = await compress(
    {},
    "alpha beta gamma tail",
    { rate: 0, forceTokens: ["alpha beta", "beta gamma"] }
  );
  assert.equal(overlap.output, "alpha beta gamma");
});

test("does not split a forced phrase at a 510-token boundary", async () => {
  const fillers = Array.from({ length: 509 }, (_, index) => `f${index}`);
  const { output } = await compress(
    {},
    [...fillers, "C++", "guide", "tail"].join(" "),
    { rate: 0, forceTokens: ["C++ guide"] }
  );
  assert.equal(output, "C++ guide");
});

test("rejects a single model word longer than 510 tokens", () => {
  const harness = createHarness();
  const pieces = ["giant", ...Array(510).fill("##piece")];
  harness.tokenizer.tokenize = () => pieces;
  harness.tokenizer.encode = () => pieces.map((_, index) => 2000 + index);
  assert.throws(
    () => harness.compressor.chunkContext("giant", new Set(), []),
    /single model word/
  );
});

test("reserves digits anywhere in a word and drops repeated forced words", async () => {
  const digit = await compress({}, "plain abc1 tail", {
    rate: 0,
    forceReserveDigit: true,
  });
  assert.equal(digit.output, "abc1");

  const kept = await compress({}, "keep keep other keep keep", {
    rate: 0,
    forceTokens: ["keep"],
  });
  const dropped = await compress({}, "keep keep other keep keep", {
    rate: 0,
    forceTokens: ["keep"],
    dropConsecutive: true,
  });
  assert.equal(kept.output, "keep keep keep keep");
  assert.equal(dropped.output, "keep");

  const phrase = await compress({}, "C++ guide C++ guide", {
    rate: 0,
    forceTokens: ["C++ guide"],
    dropConsecutive: true,
  });
  assert.equal(phrase.output, "C++ guide");

  const percentileGap = await compress(
    { low: SCORE.low, high: SCORE.high, gap: SCORE.middle },
    [
      ...Array(51).fill("low"),
      ...Array(47).fill("high"),
      "keep",
      "gap",
      "keep",
    ].join(" "),
    { rate: 0.5, forceTokens: ["keep"], dropConsecutive: true }
  );
  assert.equal(
    percentileGap.output.split(/\s+/).filter((word) => word === "keep").length,
    1
  );
});

test("supports canonical and deprecated snake-case token aggregation", async () => {
  const fixture = {
    alpha: SCORE.highest,
    "##tail": SCORE.low,
    beta: SCORE.high,
  };
  const mean = await compress(fixture, "alpha ##tail beta", {
    tokenToWord: "mean",
  });
  const first = await compress(fixture, "alpha ##tail beta", {
    tokenToWord: "first",
  });
  const harness = createHarness({ logitsByToken: fixture });
  const canonical = await harness.compressor.compress_prompt(
    "alpha ##tail beta",
    { rate: 0.5, token_to_word: "first", chunk_end_tokens: [] }
  );
  const deprecated = await harness.compressor.compress_prompt(
    "alpha ##tail beta",
    { rate: 0.5, token_to_Word: "first", chunk_end_tokens: [] }
  );
  const canonicalWins = await harness.compressor.compress_prompt(
    "alpha ##tail beta",
    {
      rate: 0.5,
      token_to_word: "mean",
      token_to_Word: "first",
      chunk_end_tokens: [],
    }
  );
  assert.notEqual(mean.output, first.output);
  assert.equal(canonical, first.output);
  assert.equal(deprecated, first.output);
  assert.equal(canonicalWins, mean.output);
});

test("matches official percentile ties and safe rate endpoints", async () => {
  const logits = {
    low: SCORE.low,
    tiedA: SCORE.middle,
    tiedB: SCORE.middle,
    high: SCORE.high,
  };
  const tied = await compress(logits, "low tiedA tiedB high", { rate: 0.5 });
  assert.equal(tied.output, "high");

  const none = await compress(logits, "low tiedA tiedB high", { rate: 0 });
  assert.equal(none.output, "");

  const all = await compress(logits, "low tiedA tiedB high", { rate: 1 });
  assert.equal(all.output, "low tiedA tiedB high");

  const forced = await compress(logits, "low tiedA", {
    rate: 0,
    forceTokens: ["low"],
  });
  assert.equal(forced.output, "low");
});

test("converts fp16 logits and disposes owned tensors exactly once", async () => {
  const harness = createHarness({
    logitsByToken: { low: SCORE.low, high: SCORE.high },
    float16: true,
    tokenizerOptions: { throwOnRowWrapper: true },
  });
  const output = await harness.compressor.compress("low high", {
    rate: 0.5,
    chunkEndTokens: [],
  });
  assert.equal(output, "high");
  assert.equal(
    harness.records.find(({ name }) => name === "logits-fp16").tensor.getToCalls(),
    1
  );
  for (const record of harness.records) {
    assert.equal(record.disposeCalls, 1, `${record.name} disposal count`);
  }
});

test("is silent by default and honors an explicit logger", async () => {
  const originalTokenizerLoader = AutoTokenizer.from_pretrained;
  const originalModelLoader = AutoModelForTokenClassification.from_pretrained;
  const originalConsoleLog = console.log;
  const consoleMessages = [];
  const suppliedMessages = [];
  const quietHarness = createHarness();

  AutoTokenizer.from_pretrained = async () => quietHarness.tokenizer;
  AutoModelForTokenClassification.from_pretrained = async () => async (inputs) => ({
    logits: new Tensor(
      "float32",
      new Float32Array(inputs.input_ids.size * 3),
      [...inputs.input_ids.dims, 3]
    ),
  });
  console.log = (...message) => consoleMessages.push(message);

  try {
    const quiet = await LLMLingua2.WithBERTMultilingual("mock", {
      transformerJSConfig: { device: "cpu", dtype: "fp32" },
      oaiTokenizer: { encode: () => [1] },
      pretrainedConfig: {},
    });
    await quiet.promptCompressor.compress("alpha beta", {
      rate: 0.5,
      chunkEndTokens: [],
    });

    const loud = await LLMLingua2.WithBERTMultilingual("mock", {
      transformerJSConfig: { device: "cpu", dtype: "fp32" },
      oaiTokenizer: { encode: () => [1] },
      pretrainedConfig: {},
      logger: (...message) => suppliedMessages.push(message),
    });
    await loud.promptCompressor.compress("alpha beta", {
      rate: 0.5,
      chunkEndTokens: [],
    });
  } finally {
    AutoTokenizer.from_pretrained = originalTokenizerLoader;
    AutoModelForTokenClassification.from_pretrained = originalModelLoader;
    console.log = originalConsoleLog;
  }

  assert.equal(consoleMessages.length, 0);
  assert.ok(suppliedMessages.length > 0);
});
