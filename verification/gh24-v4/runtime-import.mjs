import assert from "node:assert/strict";

const { LLMLingua2 } = await import("@atjsh/llmlingua-2");
assert.equal(typeof LLMLingua2.PromptCompressor, "function");
assert.equal(typeof LLMLingua2.WithBERTMultilingual, "function");
console.log(`${process.version}: packed package import passed`);
