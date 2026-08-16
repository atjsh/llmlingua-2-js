import {
  AutoConfig,
  env,
  Tensor,
} from "@huggingface/transformers";
import { LLMLingua2 } from "llmlingua-under-test";

const TINYBERT_MODEL = "atjsh/llmlingua-2-js-tinybert-meetingbank";
const TINYBERT_REVISION = "a9af82841d3f815c9c492b13791c6517154791d3";
const TINYBERT_INPUT = `John: So, um, I've been thinking about the project, you know, and I believe we need to, uh, make some changes. I mean, we want the project to succeed, right? So, like, I think we should consider maybe revising the timeline.
Sarah: I totally agree, John. I mean, we have to be realistic, you know. The timeline is, like, too tight. You know what I mean? We should definitely extend it.`;

function tensor(type, values, dims) {
  const data = type === "int64"
    ? BigInt64Array.from(values, BigInt)
    : Float32Array.from(values);
  return new Tensor(type, data, dims);
}

function logitsForClass1(class1Logit) {
  return [0, class1Logit, 0];
}

function createMockTokenizer() {
  const tokens = new Map([
    [101n, "[CLS]"],
    [102n, "[SEP]"],
    [11n, "alpha"],
    [12n, "##beta"],
    [13n, "gamma"],
    [21n, "delta"],
    [22n, "epsilon"],
  ]);

  const tokenizer = async (contexts) => {
    if (contexts.length !== 2) {
      throw new Error(`Expected two mock contexts, received ${contexts.length}`);
    }
    return {
      input_ids: tensor(
        "int64",
        [101, 11, 12, 13, 102, 101, 21, 22, 102, 0],
        [2, 5]
      ),
      attention_mask: tensor(
        "int64",
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [2, 5]
      ),
    };
  };

  tokenizer.special_tokens = {
    cls_token: "[CLS]",
    sep_token: "[SEP]",
    pad_token: "[PAD]",
  };
  tokenizer.model = {
    convert_ids_to_tokens: (ids) => ids.map((id) => tokens.get(BigInt(id))),
  };
  tokenizer.decoder = { decode: (items) => items.join(" ") };

  return tokenizer;
}

export async function runMockCompressor() {
  const logits = [
    ...logitsForClass1(0),
    ...logitsForClass1(-2),
    ...logitsForClass1(-1),
    ...logitsForClass1(3),
    ...logitsForClass1(0),
    ...logitsForClass1(0),
    ...logitsForClass1(2),
    ...logitsForClass1(-3),
    ...logitsForClass1(0),
    ...logitsForClass1(0),
  ];
  const model = async () => ({ logits: tensor("float32", logits, [2, 5, 3]) });
  const tokenizer = createMockTokenizer();
  const compressor = new LLMLingua2.PromptCompressor(
    model,
    tokenizer,
    (token) => token.replace(/^##/, ""),
    (token) => !token.startsWith("##"),
    { encode: () => [0] },
    undefined,
    () => {}
  );

  return compressor.compressContexts(["first", "second"], {
    reduce_rate: 0.5,
    token_to_word: "mean",
    force_tokens: [],
    token_map: {},
    force_reserve_digit: false,
    drop_consecutive: false,
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

export async function runTinyBertWasm() {
  env.allowLocalModels = false;
  env.allowRemoteModels = true;
  env.backends.onnx.wasm.numThreads = 1;
  env.backends.onnx.wasm.wasmPaths = "/node_modules/onnxruntime-web/dist/";

  const pretrainedConfig = await AutoConfig.from_pretrained(TINYBERT_MODEL, {
    revision: TINYBERT_REVISION,
  });
  const { promptCompressor } = await LLMLingua2.WithBERTMultilingual(
    TINYBERT_MODEL,
    {
      transformerJSConfig: { device: "wasm", dtype: "fp32" },
      oaiTokenizer: { encode: () => [0] },
      pretrainedConfig,
      pretrainedTokenizerOptions: { revision: TINYBERT_REVISION },
      modelSpecificOptions: { revision: TINYBERT_REVISION },
      logger: () => {},
    }
  );
  const output = await promptCompressor.compress(TINYBERT_INPUT, { rate: 0.5 });

  return {
    model: TINYBERT_MODEL,
    revision: TINYBERT_REVISION,
    device: "wasm",
    dtype: "fp32",
    input: TINYBERT_INPUT,
    output,
    sha256: await sha256(output),
  };
}
