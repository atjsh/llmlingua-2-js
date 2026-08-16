import {
  AutoConfig,
  AutoModelForTokenClassification,
  AutoTokenizer,
  env,
  Tensor,
} from "@huggingface/transformers";
import { LLMLingua2 } from "llmlingua-under-test";
import { EXAMPLES } from "llmlingua-e2e";

const MODEL = "atjsh/llmlingua-2-js-tinybert-meetingbank";
const REVISION = "a9af82841d3f815c9c492b13791c6517154791d3";

function tensor(type, values, dims) {
  return new Tensor(
    type,
    type === "int64"
      ? BigInt64Array.from(values, BigInt)
      : Float32Array.from(values),
    dims
  );
}

function createMockTokenizer() {
  const tokenToId = new Map([
    ["[PAD]", 0],
    ["[CLS]", 101],
    ["[SEP]", 102],
    ["trap", 11],
    ["ordinary", 12],
  ]);
  const idToToken = new Map([...tokenToId].map(([token, id]) => [id, token]));
  const split = (text) => text.trim().split(/\s+/).filter(Boolean);
  const tokenizer = async (contexts) => {
    const rows = contexts.map((context) => tokenizer.encode(context));
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
    const ids = split(text).map((token) => tokenToId.get(token));
    return add_special_tokens ? [101, ...ids, 102] : ids;
  };
  tokenizer.decode = (ids) =>
    Array.from(ids, Number)
      .map((id) => idToToken.get(id))
      .filter((token) => token && !["[PAD]", "[CLS]", "[SEP]"].includes(token))
      .join(" ");
  tokenizer.all_special_ids = [0, 101, 102];
  return tokenizer;
}

export async function runMock() {
  const tokenizer = createMockTokenizer();
  const model = async ({ input_ids }) => ({
    logits: tensor(
      "float32",
      Array.from(input_ids.data, Number).flatMap((id) => {
        if (id === 11) return [0, 2, 10];
        return [0, 0, 0];
      }),
      [...input_ids.dims, 3]
    ),
  });
  const compressor = new LLMLingua2.PromptCompressor(
    model,
    tokenizer,
    LLMLingua2.get_pure_tokens_bert_base_multilingual_cased,
    LLMLingua2.is_begin_of_new_word_bert_base_multilingual_cased,
    { encode: () => [1] }
  );
  return compressor.compress("trap ordinary", {
    rate: 0.5,
    chunkEndTokens: [],
  });
}

async function sha256(value) {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(value)
  );
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0")
  ).join("");
}

export async function runTinyBert(inputIndex, device) {
  env.allowLocalModels = false;
  env.allowRemoteModels = true;
  env.backends.onnx.wasm.numThreads = 1;
  env.backends.onnx.wasm.wasmPaths = "/node_modules/onnxruntime-web/dist/";

  const options = { revision: REVISION };
  const config = await AutoConfig.from_pretrained(MODEL, options);
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL, {
    ...options,
    config,
  });
  const model = await AutoModelForTokenClassification.from_pretrained(MODEL, {
    ...options,
    config,
    device,
    dtype: "fp32",
    subfolder: "",
  });
  try {
    const compressor = new LLMLingua2.PromptCompressor(
      model,
      tokenizer,
      LLMLingua2.get_pure_tokens_bert_base_multilingual_cased,
      LLMLingua2.is_begin_of_new_word_bert_base_multilingual_cased,
      { encode: () => [1] }
    );
    const output = await compressor.compress(EXAMPLES[inputIndex], { rate: 0.5 });
    return {
      device,
      dtype: "fp32",
      output,
      sha256: await sha256(output),
    };
  } finally {
    await model.dispose?.();
  }
}
