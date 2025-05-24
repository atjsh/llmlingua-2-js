import { EXAMPLES } from "./long-texts.js";
import { PromptCompressorLLMLingua2 } from "../prompt-compressor.js";

const promptCompressor = new PromptCompressorLLMLingua2(
  "atjsh/llmlingua-2-js-xlm-roberta-large-meetingbank",
  {
    // dtype: "uint8",
    // dtype: "fp16",
    // device: "gpu",
  }
);

await promptCompressor.init();

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
