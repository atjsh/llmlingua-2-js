// SPDX-License-Identifier: MIT

import {
  AutoConfig,
  AutoModelForTokenClassification,
  AutoTokenizer,
  PretrainedConfig,
  TransformersJSConfig,
} from "@huggingface/transformers";
import { Tiktoken } from "js-tiktoken";

import { PromptCompressorLLMLingua2 } from "./prompt-compressor.js";
import {
  get_pure_tokens_bert_base_multilingual_cased,
  get_pure_tokens_xlm_roberta_large,
  is_begin_of_new_word_bert_base_multilingual_cased,
  is_begin_of_new_word_xlm_roberta_large,
} from "./utils.js";

type PreTrainedTokenizerOptions = Parameters<
  typeof AutoTokenizer.from_pretrained
>[1];
type PretrainedModelOptions = Parameters<
  typeof AutoModelForTokenClassification.from_pretrained
>[1];

async function prepareDependencies(
  modelName: string,
  transformerJSConfig: TransformersJSConfig,

  pretrainedConfig?: PretrainedConfig | null,
  pretrainedTokenizerOptions?: PreTrainedTokenizerOptions | null,
  modelSpecificOptions?: PretrainedModelOptions | null
) {
  const config =
    pretrainedConfig ?? (await AutoConfig.from_pretrained(modelName));
  console.debug({ config });

  const tokenizerConfig = {
    config: {
      ...config,
      ...(transformerJSConfig
        ? { "transformers.js_config": transformerJSConfig }
        : {}),
    },
    ...pretrainedTokenizerOptions,
  };
  console.debug({ tokenizerConfig });

  const tokenizer = await AutoTokenizer.from_pretrained(
    modelName,
    tokenizerConfig
  );
  console.debug({ tokenizer });

  const modelConfig = {
    config: {
      ...config,
      ...(transformerJSConfig
        ? { "transformers.js_config": transformerJSConfig }
        : {}),
    },
    ...modelSpecificOptions,
  };
  console.debug({ modelConfig });

  const model = await AutoModelForTokenClassification.from_pretrained(
    modelName,
    modelConfig
  );
  console.debug({ model });

  return { model, tokenizer, config };
}

/**
 * Factory functions to create instances of LLMLingua-2 PromptCompressor
 * with XLM-RoBERTa model.
 */
export async function WithXLMRoBERTa(
  modelName: string,
  transformerJSConfig: TransformersJSConfig,
  oaiTokenizer: Tiktoken,

  {
    pretrainedConfig,
    pretrainedTokenizerOptions,
    modelSpecificOptions,
  }: {
    pretrainedConfig?: PretrainedConfig | null;
    pretrainedTokenizerOptions?: PreTrainedTokenizerOptions | null;
    modelSpecificOptions?: PretrainedModelOptions | null;
  }
) {
  const { model, tokenizer, config } = await prepareDependencies(
    modelName,
    transformerJSConfig,
    pretrainedConfig,
    pretrainedTokenizerOptions,
    modelSpecificOptions
  );

  const promptCompressor = new PromptCompressorLLMLingua2(
    model,
    tokenizer,
    get_pure_tokens_xlm_roberta_large,
    is_begin_of_new_word_xlm_roberta_large,
    oaiTokenizer
  );

  console.debug({ promptCompressor });

  return {
    promptCompressor,
    model,
    tokenizer,
    config,
  };
}

/**
 * Factory functions to create instances of LLMLingua-2 PromptCompressor
 * with BERT Multilingual model.
 */
export async function WithBERTMultilingual(
  modelName: string,
  modelOptions: TransformersJSConfig,
  oaiTokenizer: Tiktoken,

  {
    pretrainedConfig,
    pretrainedTokenizerOptions,
    modelSpecificOptions,
  }: {
    pretrainedConfig?: PretrainedConfig | null;
    pretrainedTokenizerOptions?: PreTrainedTokenizerOptions | null;
    modelSpecificOptions?: PretrainedModelOptions | null;
  }
) {
  const { model, tokenizer, config } = await prepareDependencies(
    modelName,
    modelOptions,
    pretrainedConfig,
    pretrainedTokenizerOptions,
    modelSpecificOptions
  );

  const promptCompressor = new PromptCompressorLLMLingua2(
    model,
    tokenizer,
    get_pure_tokens_bert_base_multilingual_cased,
    is_begin_of_new_word_bert_base_multilingual_cased,
    oaiTokenizer
  );

  console.debug({ promptCompressor });

  return {
    promptCompressor,
    model,
    tokenizer,
    config,
  };
}
