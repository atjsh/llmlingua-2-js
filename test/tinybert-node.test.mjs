import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import test from "node:test";

import {
  AutoConfig,
  AutoModelForTokenClassification,
  AutoTokenizer,
} from "@huggingface/transformers";
import { Tiktoken } from "js-tiktoken/lite";
import o200kBase from "js-tiktoken/ranks/o200k_base";

import { EXAMPLES } from "../dist/e2e/long-texts.js";
import { LLMLingua2 } from "../dist/index.js";

const golden = JSON.parse(
  await readFile(new URL("./tinybert-node.golden.json", import.meta.url), "utf8")
);
const {
  model: MODEL,
  revision: REVISION,
  device: DEVICE,
  dtype: DTYPE,
  length: EXPECTED_LENGTH,
  sha256: EXPECTED_SHA256,
} = golden.expected;

test(
  "TinyBERT CPU/fp32 output matches the frozen pre-change golden",
  { skip: process.env.LLMLINGUA_RUN_MODEL_E2E !== "1" },
  async () => {
    const cacheDirectory = process.env.LLMLINGUA_MODEL_CACHE;
    const pretrainedOptions = {
      revision: REVISION,
      ...(cacheDirectory ? { cache_dir: resolve(cacheDirectory) } : {}),
    };
    const config = await AutoConfig.from_pretrained(MODEL, pretrainedOptions);
    const tokenizer = await AutoTokenizer.from_pretrained(MODEL, {
      ...pretrainedOptions,
      config,
    });
    const model = await AutoModelForTokenClassification.from_pretrained(MODEL, {
      ...pretrainedOptions,
      config,
      device: DEVICE,
      dtype: DTYPE,
      subfolder: "",
    });

    try {
      const compressor = new LLMLingua2.PromptCompressor(
        model,
        tokenizer,
        LLMLingua2.get_pure_tokens_bert_base_multilingual_cased,
        LLMLingua2.is_begin_of_new_word_bert_base_multilingual_cased,
        new Tiktoken(o200kBase),
        undefined,
        () => {}
      );
      const output = await compressor.compress(EXAMPLES[5], { rate: 0.5 });

      assert.equal(output.length, EXPECTED_LENGTH);
      assert.equal(
        createHash("sha256").update(output).digest("hex"),
        EXPECTED_SHA256
      );
    } finally {
      await model.dispose?.();
    }
  }
);
