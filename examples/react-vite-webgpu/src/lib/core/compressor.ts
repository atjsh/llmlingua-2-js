import { LLMLingua2 } from "@atjsh/llmlingua-2";
import {
  type DataType,
  type DeviceType,
  type PretrainedModelOptions,
  AutoConfig,
  AutoTokenizer,
  BertForTokenClassification,
  env,
  PreTrainedModel,
} from "@huggingface/transformers";
import { Tiktoken } from "js-tiktoken/lite";
import o200k_base from "js-tiktoken/ranks/o200k_base";

import { MobileBertForTokenClassification } from "@/lib/transformers-js/mobileBertForTokenClassification";

const oaiTokenizer = new Tiktoken(o200k_base);
type TransformersJSConfig =
  LLMLingua2.FactoryOptions["transformerJSConfig"];
type BrowserBackend = Extract<DeviceType, "webgpu" | "wasm">;

export const LLMLingua2CompressorModelName = {
  TINYBERT: "TINYBERT",
  MOBILEBERT: "MOBILEBERT",
  BERT: "BERT",
  ROBERTA: "ROBERTA",
} as const;
export type LLMLingua2CompressorModelName =
  keyof typeof LLMLingua2CompressorModelName;

export const LLMLingua2CompressorModels = {
  TINYBERT: {
    key: "TINYBERT",
    modelName: "atjsh/llmlingua-2-js-tinybert-meetingbank-onnx-v4",
    revision: "e49e9637ec1e9a88defd52b6422cf2d40e96d539",
    defaultDevice: "webgpu",
    defaultModelDataTypes: { webgpu: "bnb4", wasm: "uint8" },
    maxBatchSize: 50,
    maxForceTokens: 100,
    maxSequenceLength: 512,

    pretrainedModel: BertForTokenClassification,
    tokenUtils: {
      getPureTokens: LLMLingua2.get_pure_tokens_bert_base_multilingual_cased,
      isBeginOfNewWord:
        LLMLingua2.is_begin_of_new_word_bert_base_multilingual_cased,
    },
  },
  MOBILEBERT: {
    key: "MOBILEBERT",
    modelName: "atjsh/llmlingua-2-js-mobilebert-meetingbank-onnx-v4",
    revision: "3e0b45f7bafc20bb5f83a288ff02decf3facf900",
    defaultDevice: "webgpu",
    defaultModelDataTypes: { webgpu: "bnb4", wasm: "bnb4" },
    maxBatchSize: 50,
    maxForceTokens: 100,
    maxSequenceLength: 512,

    pretrainedModel: MobileBertForTokenClassification,
    tokenUtils: {
      getPureTokens: LLMLingua2.get_pure_tokens_bert_base_multilingual_cased,
      isBeginOfNewWord:
        LLMLingua2.is_begin_of_new_word_bert_base_multilingual_cased,
    },
  },
  BERT: {
    key: "BERT",
    modelName:
      "atjsh/llmlingua-2-js-bert-base-multilingual-cased-meetingbank-onnx-v4",
    revision: "db67b6283d60e7190b32a6b8a8a87c87a6c1375a",
    defaultDevice: "webgpu",
    defaultModelDataTypes: { webgpu: "bnb4", wasm: "uint8" },
    maxBatchSize: 50,
    maxForceTokens: 100,
    maxSequenceLength: 512,
    factory: LLMLingua2.WithBERTMultilingual,
  },
  ROBERTA: {
    key: "ROBERTA",
    modelName:
      "atjsh/llmlingua-2-js-xlm-roberta-large-meetingbank-onnx-v4",
    revision: "844e52fefd284e6479b027b3ccb47c1b9954d640",
    defaultDevice: "webgpu",
    defaultModelDataTypes: { webgpu: "fp32", wasm: "fp32" },
    maxBatchSize: 50,
    maxForceTokens: 100,
    maxSequenceLength: 512,

    pretrainedModelOptions: {
      use_external_data_format: { "model.onnx": 1 },
    },
    factory: LLMLingua2.WithXLMRoBERTa,
  },
} as const satisfies Record<string, LLMLingua2ModelConfig>;

export interface LLMLingua2ModelConfig {
  key: LLMLingua2CompressorModelName;
  modelName: string;
  revision: string;
  defaultDevice: BrowserBackend;
  defaultModelDataTypes: Record<BrowserBackend, DataType>;
  maxBatchSize: number;
  maxForceTokens: number;
  maxSequenceLength: number;
  factory?:
    | typeof LLMLingua2.WithBERTMultilingual
    | typeof LLMLingua2.WithXLMRoBERTa;
  pretrainedModel?: typeof PreTrainedModel;
  pretrainedModelOptions?: PretrainedModelOptions;
  tokenUtils?: {
    getPureTokens?: LLMLingua2.GetPureTokenFunction;
    isBeginOfNewWord?: LLMLingua2.IsBeginOfNewWordFunction;
  };
}

interface LLMLingua2CompressorConfig {
  modelSelection: LLMLingua2CompressorModelName | LLMLingua2ModelConfig;
  transformersJSConfig: {
    device: BrowserBackend;
  };
}

export interface LLMLingua2CompressorOptions {
  keepingTokens: string[];
  pruningTokens: string[];
  keepDigits: boolean;
  chunkEndTokens: string[];
  rate: number;
}

export interface CompressorConfig {
  llmlingua2Config: LLMLingua2CompressorConfig;
}

export function isModelSelectionKey(
  key: LLMLingua2CompressorModelName | LLMLingua2ModelConfig
): key is LLMLingua2CompressorModelName {
  return typeof key === "string" && key in LLMLingua2CompressorModels;
}

async function LLMLingua2CompressorFactory(options: {
  llmlingua2Config: LLMLingua2CompressorConfig;
  environment: {
    isWebGPUAvailable: boolean;
  };
}): Promise<LLMLingua2.PromptCompressor> {
  const { llmlingua2Config } = options;
  const { modelSelection, transformersJSConfig: providedTransformersJSConfig } =
    llmlingua2Config;

  const model: LLMLingua2ModelConfig = isModelSelectionKey(modelSelection)
    ? LLMLingua2CompressorModels[modelSelection]
    : modelSelection;

  const device =
    options.environment.isWebGPUAvailable === false &&
    providedTransformersJSConfig.device === "webgpu"
      ? "wasm"
      : providedTransformersJSConfig.device;
  const transformersJSConfig: TransformersJSConfig = {
    device,
    dtype: model.defaultModelDataTypes[device],
  };
  env.remotePathTemplate = `{model}/resolve/${model.revision}/`;
  const config = await AutoConfig.from_pretrained(model.modelName, {
    revision: model.revision,
  });
  const runtimeConfig = {
    ...config,
    "transformers.js_config": transformersJSConfig,
  };

  if (model.factory) {
    const { promptCompressor } = await model.factory(model.modelName, {
      transformerJSConfig: transformersJSConfig,
      oaiTokenizer,
      pretrainedConfig: runtimeConfig,
      pretrainedTokenizerOptions: { revision: model.revision },
      modelSpecificOptions: {
        ...model.pretrainedModelOptions,
        revision: model.revision,
      },
    });

    return promptCompressor;
  }

  if (
    model.pretrainedModel &&
    model.tokenUtils?.getPureTokens &&
    model.tokenUtils.isBeginOfNewWord
  ) {
    const tokenizer = await AutoTokenizer.from_pretrained(model.modelName, {
      revision: model.revision,
      config: runtimeConfig,
    });

    const pretrainedModel = await model.pretrainedModel.from_pretrained(
      model.modelName,
      {
        ...model.pretrainedModelOptions,
        revision: model.revision,
        config: runtimeConfig,
      }
    );

    const promptCompressor = new LLMLingua2.PromptCompressor(
      pretrainedModel,
      tokenizer,
      model.tokenUtils.getPureTokens,
      model.tokenUtils.isBeginOfNewWord,
      oaiTokenizer,
      {
        max_batch_size: model.maxBatchSize,
        max_force_token: model.maxForceTokens,
        max_seq_length: model.maxSequenceLength,
      }
    );

    return promptCompressor;
  }

  throw new Error(
    "Invalid LLMLingua2 model configuration. Please check the model settings."
  );
}

export class LossyTextCompressor {
  #config: CompressorConfig;
  #llmlingua2Compressor?: LLMLingua2.PromptCompressor;

  get #compressor(): LLMLingua2.PromptCompressor {
    if (!this.#llmlingua2Compressor) {
      throw new Error("Compressor is not initialized. Call init() first.");
    }
    return this.#llmlingua2Compressor;
  }

  constructor(config: CompressorConfig) {
    this.#config = config;
  }

  async #checkIfWebGPUAvailable(): Promise<boolean> {
    try {
      if (navigator.gpu) {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          return true;
        }
      }
      return false;
    } catch (error) {
      console.warn("WebGPU is not available:", error);
      return false;
    }
  }

  public async init() {
    const isWebGPUAvailable = await this.#checkIfWebGPUAvailable();
    this.#llmlingua2Compressor = await LLMLingua2CompressorFactory({
      llmlingua2Config: this.#config.llmlingua2Config,
      environment: {
        isWebGPUAvailable,
      },
    });
    return true;
  }

  public async compress(text: string, options: LLMLingua2CompressorOptions) {
    return await this.#compressor.compress(text, {
      rate: options.rate,
      forceTokens: options.keepingTokens,
      forceReserveDigit: options.keepDigits,
      chunkEndTokens: options.chunkEndTokens,
    });
  }
}
