// SPDX-License-Identifier: MIT

/**
 * @categoryDescription Factory
A collection of utility functions and types for model-specific token handling.
 *
 * @showCategories
 */

import {
  AutoConfig,
  AutoModelForTokenClassification,
  AutoTokenizer,
  type PretrainedConfig,
  type PretrainedModelOptions,
  type PretrainedTokenizerOptions,
  type PreTrainedModel,
  type PreTrainedTokenizer,
} from "@huggingface/transformers";

import { PromptCompressorLLMLingua2 } from "./prompt-compressor.js";
import {
  get_pure_tokens_bert_base_multilingual_cased,
  get_pure_tokens_xlm_roberta_large,
  is_begin_of_new_word_bert_base_multilingual_cased,
  is_begin_of_new_word_xlm_roberta_large,
  type CountTokensFunction,
  type Logger,
  type TokenCountingTokenizer,
} from "./utils.js";

type TransformersJSConfig = PretrainedConfig["transformers.js_config"];
const silentLogger: Logger = () => undefined;

function selectTokenCounter(
  options: LLMLingua2FactoryOptions
): CountTokensFunction | TokenCountingTokenizer {
  const counter = options.countTokens ?? options.oaiTokenizer;
  if (!counter) {
    throw new TypeError(
      "LLMLingua2 factory requires a token counter: pass `countTokens` (see `loadTokenCounter`). " +
        "The deprecated `oaiTokenizer` option is also accepted."
    );
  }
  return counter;
}

async function prepareDependencies(
  modelName: string,
  transformerJSConfig: TransformersJSConfig,
  logger: Logger,

  pretrainedConfig?: PretrainedConfig | null,
  pretrainedTokenizerOptions?: PretrainedTokenizerOptions | null,
  modelSpecificOptions?: PretrainedModelOptions | null
) {
  const config =
    pretrainedConfig ?? (await AutoConfig.from_pretrained(modelName));
  logger({ config });

  const tokenizerConfig = {
    config: {
      ...config,
      "transformers.js_config": transformerJSConfig,
    },
    ...pretrainedTokenizerOptions,
  };
  logger({ tokenizerConfig });

  const tokenizer = await AutoTokenizer.from_pretrained(
    modelName,
    tokenizerConfig
  );
  logger({ tokenizer });

  const modelConfig = {
    config: {
      ...config,
      "transformers.js_config": transformerJSConfig,
    },
    ...modelSpecificOptions,
  };
  logger({ modelConfig });

  const model = await AutoModelForTokenClassification.from_pretrained(
    modelName,
    modelConfig
  );
  logger({ model });

  return { model, tokenizer, config };
}

/**
 * Options for the LLMLingua-2 factory functions.
 *
 * @category Factory
 */
export interface LLMLingua2FactoryOptions {
  /**
   * Configuration for Transformers.js.
   */
  transformerJSConfig: TransformersJSConfig;

  /**
   * Counts tokens in the tokenizer of the LLM the compressed prompt is destined
   * for, which is what sizes the compression budget.
   *
   * Takes precedence over {@link LLMLingua2FactoryOptions.oaiTokenizer}. Build
   * one with `loadTokenCounter`, or supply any `(text: string) => number`.
   */
  countTokens?: CountTokensFunction;

  /**
   * The tokenizer to use calculating the compression rate.
   *
   * @deprecated Pass {@link LLMLingua2FactoryOptions.countTokens} instead. This
   *   option is retained so existing `js-tiktoken` callers keep working, and
   *   will be removed in the next major release.
   */
  oaiTokenizer?: TokenCountingTokenizer;

  /**
   * Optional pretrained configuration.
   */
  pretrainedConfig?: PretrainedConfig | null;

  /**
   * Optional pretrained tokenizer options.
   */
  pretrainedTokenizerOptions?: PretrainedTokenizerOptions | null;

  /**
   * Optional model-specific options passed to Transformers.js unchanged.
   */
  modelSpecificOptions?: PretrainedModelOptions | null;

  /**
   * Optional logger function. Logging is disabled when omitted.
   */
  logger?: Logger;
}

/**
 * Return type for the LLMLingua-2 factory functions. Use `promptCompressor` to compress prompts.
 *
 * @category Factory
 */
export interface LLMLingua2FactoryReturn {
  /**
   * Instance of LLMLingua-2 PromptCompressor.
   *
   * @see {@link PromptCompressorLLMLingua2}
   */
  promptCompressor: PromptCompressorLLMLingua2;

  /**
   * The model used for token classification.
   */
  model: PreTrainedModel;

  /**
   * The tokenizer used for tokenization.
   */
  tokenizer: PreTrainedTokenizer;

  /**
   * The configuration used for the model.
   */
  config: PretrainedConfig;
}

/**
 * Factory functions to create instances of LLMLingua-2 PromptCompressor
 * with XLM-RoBERTa model.
 *
 * @category Factory
 * 
 * @example
 * ```ts
import { LLMLingua2 } from "@atjsh/llmlingua-2";

const modelName = "atjsh/llmlingua-2-js-xlm-roberta-large-meetingbank";

// "Xenova/gpt-4o" is tiktoken's o200k_base. Use "Xenova/gpt-3.5-turbo" for
// cl100k_base, which is what the original LLMLingua counts with.
const countTokens = await LLMLingua2.loadTokenCounter("Xenova/gpt-4o");

const { promptCompressor } = await LLMLingua2.WithXLMRoBERTa(modelName,
  {
    transformerJSConfig: {
      device: "cpu",
      dtype: "fp32",
    },
    countTokens,
    modelSpecificOptions: {
      use_external_data_format: { "model.onnx": 1 },
    },
  }
);

const compressedText: string = await promptCompressor.compress_prompt(
  "LLMLingua-2, a small-size yet powerful prompt compression method trained via data distillation from GPT-4 for token classification with a BERT-level encoder, excels in task-agnostic compression. It surpasses LLMLingua in handling out-of-domain data, offering 3x-6x faster performance.",
  { rate: 0.8 }
);

console.log({ compressedText });
```
 */
export async function WithXLMRoBERTa(
  modelName: string,
  options: LLMLingua2FactoryOptions
): Promise<LLMLingua2FactoryReturn> {
  const {
    transformerJSConfig,
    pretrainedConfig,
    pretrainedTokenizerOptions,
    modelSpecificOptions,
    logger = silentLogger,
  } = options;
  const countTokens = selectTokenCounter(options);

  const { model, tokenizer, config } = await prepareDependencies(
    modelName,
    transformerJSConfig,
    logger,
    pretrainedConfig,
    pretrainedTokenizerOptions,
    modelSpecificOptions
  );

  const promptCompressor = new PromptCompressorLLMLingua2(
    model,
    tokenizer,
    get_pure_tokens_xlm_roberta_large,
    is_begin_of_new_word_xlm_roberta_large,
    countTokens,
    undefined,
    logger
  );

  logger({ promptCompressor });

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
 *
 * @category Factory
 * 
 * @example
 * ```ts
import { LLMLingua2 } from "@atjsh/llmlingua-2";

const modelName = "Arcoldd/llmlingua4j-bert-base-onnx";

// "Xenova/gpt-4o" is tiktoken's o200k_base. Use "Xenova/gpt-3.5-turbo" for
// cl100k_base, which is what the original LLMLingua counts with.
const countTokens = await LLMLingua2.loadTokenCounter("Xenova/gpt-4o");

const { promptCompressor } = await LLMLingua2.WithBERTMultilingual(modelName,
  {
    transformerJSConfig: {
      device: "cpu",
      dtype: "fp32",
    },
    countTokens,
    modelSpecificOptions: {
      subfolder: "",
    },
  }
);

const compressedText: string = await promptCompressor.compress_prompt(
  "LLMLingua-2, a small-size yet powerful prompt compression method trained via data distillation from GPT-4 for token classification with a BERT-level encoder, excels in task-agnostic compression. It surpasses LLMLingua in handling out-of-domain data, offering 3x-6x faster performance.",
  { rate: 0.8 }
);

console.log({ compressedText });
```
 */
export async function WithBERTMultilingual(
  modelName: string,
  options: LLMLingua2FactoryOptions
): Promise<LLMLingua2FactoryReturn> {
  const {
    transformerJSConfig,
    pretrainedConfig,
    pretrainedTokenizerOptions,
    modelSpecificOptions,
    logger = silentLogger,
  } = options;
  const countTokens = selectTokenCounter(options);

  const { model, tokenizer, config } = await prepareDependencies(
    modelName,
    transformerJSConfig,
    logger,
    pretrainedConfig,
    pretrainedTokenizerOptions,
    modelSpecificOptions
  );

  const promptCompressor = new PromptCompressorLLMLingua2(
    model,
    tokenizer,
    get_pure_tokens_bert_base_multilingual_cased,
    is_begin_of_new_word_bert_base_multilingual_cased,
    countTokens,
    undefined,
    logger
  );

  logger({ promptCompressor });

  return {
    promptCompressor,
    model,
    tokenizer,
    config,
  };
}
