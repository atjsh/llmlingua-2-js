// SPDX-License-Identifier: MIT

import { LLMLingua2 } from "../../index.js";
import { EXAMPLES } from "../long-texts.js";

const modelName = "Arcoldd/llmlingua4j-bert-base-onnx";
const countTokens = await LLMLingua2.loadTokenCounter("Xenova/gpt-4o");

const { promptCompressor } = await LLMLingua2.WithBERTMultilingual(modelName, {
  transformerJSConfig: {
    device: "auto",
    dtype: "fp32",
  },
  countTokens,
  modelSpecificOptions: {
    subfolder: "",
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
