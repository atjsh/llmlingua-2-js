import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  AutoConfig,
  AutoModelForTokenClassification,
  AutoTokenizer,
  Tensor,
} from "@huggingface/transformers";

import { EXAMPLES } from "../../dist/e2e/long-texts.js";
import { LLMLingua2 } from "../../dist/index.js";

const golden = JSON.parse(
  await readFile(new URL("./tinybert-golden.json", import.meta.url), "utf8")
);

test("pinned TinyBERT CPU/fp32 output matches the 3.0 golden", {
  timeout: 300_000,
}, async () => {
  const { model: modelName, revision, inputIndex, node } = golden.expected;
  const options = { revision };
  const config = await AutoConfig.from_pretrained(modelName, options);
  const tokenizer = await AutoTokenizer.from_pretrained(modelName, {
    ...options,
    config,
  });
  const model = await AutoModelForTokenClassification.from_pretrained(
    modelName,
    {
      ...options,
      config,
      device: "cpu",
      dtype: "fp32",
      subfolder: "",
    }
  );

  try {
    const compressor = new LLMLingua2.PromptCompressor(
      model,
      tokenizer,
      LLMLingua2.get_pure_tokens_bert_base_multilingual_cased,
      LLMLingua2.is_begin_of_new_word_bert_base_multilingual_cased,
      { encode: () => [1] }
    );

    const boundaryInput = [
      "hello",
      ...Array(100).fill("electroencephalographically"),
    ].join(" ");
    const originalIds = tokenizer.encode(boundaryInput, {
      add_special_tokens: false,
    });
    const chunks = compressor.chunkContext(boundaryInput, new Set(), []);
    const roundTrippedIds = chunks.flatMap(({ text }) =>
      tokenizer.encode(text, { add_special_tokens: false })
    );
    assert.deepEqual(roundTrippedIds, originalIds);
    assert.ok(
      chunks.every(
        ({ text }) =>
          tokenizer.encode(text, { add_special_tokens: false }).length <= 510
      )
    );

    const zeroModel = async ({ input_ids }) => ({
      logits: new Tensor(
        "float32",
        new Float32Array(input_ids.size * 2),
        [...input_ids.dims, 2]
      ),
    });
    const literalCompressor = new LLMLingua2.PromptCompressor(
      zeroModel,
      tokenizer,
      LLMLingua2.get_pure_tokens_bert_base_multilingual_cased,
      LLMLingua2.is_begin_of_new_word_bert_base_multilingual_cased,
      { encode: () => [1] }
    );
    for (const literal of ["C++ guide", "MiXeD Case", "foo.bar"]) {
      assert.equal(
        await literalCompressor.compress(`drop ${literal} omit`, {
          rate: 0,
          forceTokens: [literal],
        }),
        literal
      );
    }
    for (const { context, forceTokens, expected } of [
      {
        context: "drop MiXeD Case Foo omit",
        forceTokens: ["MiXeD Case", "Case Foo"],
        expected: "MiXeD Case Foo",
      },
      {
        context: "drop C++ Guide Foo.Bar omit",
        forceTokens: ["C++ Guide", "Guide Foo.Bar"],
        expected: "C++ Guide Foo.Bar",
      },
    ]) {
      assert.equal(
        await literalCompressor.compress(context, {
          rate: 0,
          forceTokens,
        }),
        expected
      );
    }
    await assert.rejects(
      literalCompressor.compress("drop 🦄 omit", {
        rate: 0,
        forceTokens: ["🦄"],
      }),
      /special|unknown/i
    );
    await assert.rejects(
      literalCompressor.compress("drop foo bar Foo omit", {
        rate: 0,
        forceTokens: ["foo bar Foo", "foo"],
      }),
      /overlap|literal/i
    );

    const output = await compressor.compress(EXAMPLES[inputIndex], {
      rate: 0.5,
    });
    assert.equal(output, node.output);
    assert.equal(
      createHash("sha256").update(output).digest("hex"),
      node.sha256
    );
  } finally {
    await model.dispose?.();
  }
});
