// SPDX-License-Identifier: MIT

import { Tiktoken } from "js-tiktoken/lite";
import o200k_base from "js-tiktoken/ranks/o200k_base";

import { LLMLingua2 } from "../../index.js";
import { EXAMPLES } from "../long-texts.js";

const modelName = "atjsh/llmlingua-2-js-xlm-roberta-large-meetingbank";
const oai_tokenizer = new Tiktoken(o200k_base);

const { promptCompressor } = await LLMLingua2.WithXLMRoBERTa(modelName, {
  transformerJSConfig: {
    device: "auto",
    dtype: "fp32",
  },
  oaiTokenizer: oai_tokenizer,
  modelSpecificOptions: {
    use_external_data_format: true,
  },
});

const start = performance.now();

const result = await promptCompressor.compress_prompt(
  EXAMPLES[EXAMPLES.length - 1],
  {
    rate: 0.5,
  }
);

const end = performance.now();

console.log({ result });

console.log("Time taken for compression:", end - start, "ms");
console.log(
  "Time taken for compression (human-readable):",
  ((end - start) / 1000).toFixed(2),
  "seconds"
);
