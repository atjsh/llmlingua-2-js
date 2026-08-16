import assert from "node:assert/strict";
import test from "node:test";

import { softmax, Tensor } from "@huggingface/transformers";

import { LLMLingua2 } from "../dist/index.js";

const PYTORCH_SOFTMAX_VECTORS = [
  {
    logits: [0, 1, 0],
    probabilities: [
      0.21194157004356384, 0.5761169195175171, 0.21194157004356384,
    ],
  },
  {
    logits: [0, 2, 10],
    probabilities: [
      0.00004538264329312369, 0.0003353348874952644, 0.9996192455291748,
    ],
  },
  {
    logits: [-3.25, 0.75, 1.5],
    probabilities: [
      0.005841720383614302, 0.31894710659980774, 0.6752110719680786,
    ],
  },
];

const LOGITS = {
  low: [0, 0, 2],
  middle: [0, 1, 0],
  high: [0, 2, 0],
  trap: [0, 2, 10],
};

function createHarness(logitsByToken) {
  const tokenToId = new Map([
    ["[PAD]", 0],
    ["[CLS]", 1],
    ["[SEP]", 2],
  ]);
  const idToToken = new Map([...tokenToId].map(([token, id]) => [id, token]));
  const modelShapes = [];
  const attentionMasks = [];
  const selections = [];
  let inferenceStarted = false;

  const split = (text) => text.trim().split(/\s+/).filter(Boolean);
  const idFor = (token) => {
    if (!tokenToId.has(token)) {
      const id = tokenToId.size;
      tokenToId.set(token, id);
      idToToken.set(id, token);
    }
    return tokenToId.get(token);
  };

  const tokenizer = Object.assign(
    async (contexts) => {
      const rows = contexts.map((context) => [
        idFor("[CLS]"),
        ...split(context).map(idFor),
        idFor("[SEP]"),
      ]);
      const sequenceLength = Math.max(...rows.map((row) => row.length));
      const ids = [];
      const masks = [];

      for (const row of rows) {
        const padding = sequenceLength - row.length;
        ids.push(...row, ...Array(padding).fill(0));
        masks.push(...Array(row.length).fill(1), ...Array(padding).fill(0));
      }

      return {
        input_ids: new Tensor(
          "int64",
          BigInt64Array.from(ids, BigInt),
          [rows.length, sequenceLength]
        ),
        attention_mask: new Tensor(
          "int64",
          BigInt64Array.from(masks, BigInt),
          [rows.length, sequenceLength]
        ),
      };
    },
    {
      special_tokens: {
        cls_token: "[CLS]",
        sep_token: "[SEP]",
        pad_token: "[PAD]",
      },
      tokenize: split,
      decoder: {
        decode(tokens) {
          const visible = tokens.filter(
            (token) => !["[PAD]", "[CLS]", "[SEP]", "[END]"].includes(token)
          );
          if (inferenceStarted) selections.push(visible);
          return visible.join(" ");
        },
      },
      model: {
        convert_ids_to_tokens(ids) {
          return ids.map((id) => idToToken.get(Number(id)));
        },
      },
    }
  );

  const model = async ({ input_ids, attention_mask }) => {
    inferenceStarted = true;
    const [batchSize, sequenceLength] = input_ids.dims;
    modelShapes.push([batchSize, sequenceLength, 3]);
    attentionMasks.push(attention_mask.tolist().map((row) => row.map(Number)));

    const logits = Array.from(input_ids.data).flatMap((id) => {
      const token = idToToken.get(Number(id));
      return logitsByToken[token] ?? [0, 0, 0];
    });

    return {
      logits: new Tensor(
        "float32",
        Float32Array.from(logits),
        [batchSize, sequenceLength, 3]
      ),
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

  return { attentionMasks, compressor, modelShapes, selections };
}

async function compressFixture(logitsByToken, context, options = {}) {
  const harness = createHarness(logitsByToken);
  const output = await harness.compressor.compress(context, {
    rate: 0.5,
    chunkEndTokens: [],
    ...options,
  });
  return { ...harness, output };
}

function selectedWordIndices(selections, sourceWords) {
  return selections.map((selected, row) =>
    selected.map((word) => sourceWords[row].indexOf(word))
  );
}

test("Transformers.js softmax matches committed PyTorch fp32 vectors", () => {
  for (const { logits, probabilities } of PYTORCH_SOFTMAX_VECTORS) {
    const actual = softmax(Float32Array.from(logits));
    actual.forEach((probability, index) => {
      assert.ok(
        Math.abs(probability - probabilities[index]) <= 1e-6,
        `${logits}: class ${index} was ${probability}, expected ${probabilities[index]}`
      );
    });
  }
});

test("compressor applies full three-class softmax and selects class 1", async () => {
  const { modelShapes, output, selections } = await compressFixture(
    { trap: LOGITS.trap, ordinary: [0, 0, 0] },
    "trap ordinary [END]"
  );

  assert.deepEqual(modelShapes, [[1, 4, 3]]);
  assert.deepEqual(selections, [["ordinary"]]);
  assert.deepEqual(
    selectedWordIndices(selections, [["trap", "ordinary"]]),
    [[1]]
  );
  assert.equal(output, "ordinary");
});

test("compressor preserves mean and first subtoken aggregation", async () => {
  const fixture = {
    alpha: LOGITS.high,
    "##tail": LOGITS.low,
    beta: LOGITS.middle,
  };
  const mean = await compressFixture(fixture, "alpha ##tail beta [END]", {
    tokenToWord: "mean",
  });
  const first = await compressFixture(fixture, "alpha ##tail beta [END]", {
    tokenToWord: "first",
  });

  assert.deepEqual(mean.selections, [["beta"]]);
  assert.deepEqual(
    selectedWordIndices(mean.selections, [["alphatail", "beta"]]),
    [[1]]
  );
  assert.equal(mean.output, "beta");
  assert.deepEqual(first.selections, [["alphatail"]]);
  assert.deepEqual(
    selectedWordIndices(first.selections, [["alphatail", "beta"]]),
    [[0]]
  );
  assert.equal(first.output, "alphatail");
});

test("compressor preserves strict percentile ties and forced score 1", async () => {
  const tied = await compressFixture(
    {
      low: LOGITS.low,
      tiedA: LOGITS.middle,
      tiedB: LOGITS.middle,
      high: LOGITS.high,
    },
    "low tiedA tiedB high [END]"
  );
  assert.deepEqual(tied.selections, [["high"]]);
  assert.deepEqual(
    selectedWordIndices(tied.selections, [["low", "tiedA", "tiedB", "high"]]),
    [[3]]
  );
  assert.equal(tied.output, "high");

  const forced = await compressFixture(
    { forced: LOGITS.low, other: LOGITS.high },
    "forced other [END]",
    { forceTokens: ["forced"], rate: 0 }
  );
  assert.deepEqual(forced.selections, [["forced"]]);
  assert.deepEqual(
    selectedWordIndices(forced.selections, [["forced", "other"]]),
    [[0]]
  );
  assert.equal(forced.output, "forced");
});

test("compressor handles padded logits across multiple batches", async () => {
  const fixture = {
    a: LOGITS.low,
    b: LOGITS.high,
    ".": LOGITS.middle,
    c: LOGITS.low,
    d: LOGITS.middle,
    e: LOGITS.high,
    f: LOGITS.trap,
    g: LOGITS.middle,
    h: LOGITS.middle,
    i: LOGITS.high,
  };
  const harness = createHarness(fixture);
  harness.compressor.llmlingua2Config.max_seq_length = 5;
  harness.compressor.llmlingua2Config.max_batch_size = 2;

  const output = await harness.compressor.compress(
    "a b . c d e f g h i [END]",
    { rate: 0.5, chunkEndTokens: ["."] }
  );

  assert.deepEqual(harness.modelShapes, [
    [2, 6, 3],
    [1, 5, 3],
  ]);
  assert.deepEqual(harness.attentionMasks, [
    [
      [1, 1, 1, 1, 1, 0],
      [1, 1, 1, 1, 1, 1],
    ],
    [[1, 1, 1, 1, 1]],
  ]);
  assert.deepEqual(harness.selections, [["b"], ["d", "e"], ["i"]]);
  assert.deepEqual(
    selectedWordIndices(harness.selections, [
      ["a", "b", "."],
      ["c", "d", "e", "f"],
      ["g", "h", "i"],
    ]),
    [[1], [1, 2], [2]]
  );
  assert.equal(output, "b\nd e\ni");
});
