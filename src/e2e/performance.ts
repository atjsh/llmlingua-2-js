import {
  AutoConfig,
  AutoModelForTokenClassification,
  AutoTokenizer,
  TransformersJSConfig,
} from "@huggingface/transformers";
import { Tiktoken } from "js-tiktoken/lite";
import o200k_base from "js-tiktoken/ranks/o200k_base";

import { EXAMPLES } from "./long-texts.js";
import { LLMLingua2 } from "../index.js";
import {
  get_pure_tokens_xlm_roberta_large,
  is_begin_of_new_word_xlm_roberta_large,
} from "../lib/llmlingua-2/utils.js";

const oai_tokenizer = new Tiktoken(o200k_base);

const modelName = "atjsh/llmlingua-2-js-xlm-roberta-large-meetingbank";

const modelOptions: TransformersJSConfig = {
  device: "gpu",
  dtype: "q4",
};

const config = await AutoConfig.from_pretrained(modelName);

console.log({
  modelOptions,
  config,
});

const tokenizer = await AutoTokenizer.from_pretrained(modelName, {
  config: {
    ...config,
    "transformers.js_config": modelOptions,
  },
});

const model = await AutoModelForTokenClassification.from_pretrained(modelName, {
  config: {
    ...config,
    "transformers.js_config": modelOptions,
  },
});

const promptCompressor = new LLMLingua2.PromptCompressor(
  model,
  tokenizer,
  get_pure_tokens_xlm_roberta_large,
  is_begin_of_new_word_xlm_roberta_large,
  oai_tokenizer
);

const start = performance.now();

const result = await promptCompressor.compress_prompt(
  EXAMPLES[EXAMPLES.length - 1],
  {
    rate: 0.5,
  }
);

const end = performance.now();

// console.log("Compression result:", result.slice(0, 200));
console.log("Compressed result: \n", result);

console.log("Time taken for compression:", end - start, "ms");
console.log(
  "Time taken for compression (human-readable):",
  ((end - start) / 1000).toFixed(2),
  "seconds"
);
