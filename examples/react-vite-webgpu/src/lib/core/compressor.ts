import { LLMLingua2 } from "@atjsh/llmlingua-2";
import {
  type DataType,
  type DeviceType,
  type PretrainedModelOptions,
  type TransformersJSConfig,
  AutoConfig,
  AutoTokenizer,
  BertForTokenClassification,
  PreTrainedModel,
} from "@huggingface/transformers";
import { Tiktoken } from "js-tiktoken/lite";
import o200k_base from "js-tiktoken/ranks/o200k_base";

import { MobileBertForTokenClassification } from "@/lib/transformers-js/mobileBertForTokenClassification";

const oaiTokenizer = new Tiktoken(o200k_base);

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
    modelName: "atjsh/llmlingua-2-js-tinybert-meetingbank",
    defaultDevice: "auto",
    defaultModelDataType: "fp32",
    maxBatchSize: 50,
    maxForceTokens: 100,
    maxSequenceLength: 312,

    pretrainedModel: BertForTokenClassification,
    tokenUtils: {
      getPureTokens: LLMLingua2.get_pure_tokens_bert_base_multilingual_cased,
      isBeginOfNewWord:
        LLMLingua2.is_begin_of_new_word_bert_base_multilingual_cased,
    },
  },
  MOBILEBERT: {
    key: "MOBILEBERT",
    modelName: "atjsh/llmlingua-2-js-mobilebert-meetingbank",
    defaultDevice: "auto",
    defaultModelDataType: "fp32",
    maxBatchSize: 50,
    maxForceTokens: 100,
    maxSequenceLength: 128,

    pretrainedModel: MobileBertForTokenClassification,
    tokenUtils: {
      getPureTokens: LLMLingua2.get_pure_tokens_bert_base_multilingual_cased,
      isBeginOfNewWord:
        LLMLingua2.is_begin_of_new_word_bert_base_multilingual_cased,
    },
  },
  BERT: {
    key: "BERT",
    modelName: "Arcoldd/llmlingua4j-bert-base-onnx",
    defaultDevice: "auto",
    defaultModelDataType: "fp32",
    maxBatchSize: 50,
    maxForceTokens: 100,
    maxSequenceLength: 512,

    pretrainedModelOptions: {
      subfolder: "",
    },
    factory: LLMLingua2.WithBERTMultilingual,
  },
  ROBERTA: {
    key: "ROBERTA",
    modelName: "atjsh/llmlingua-2-js-xlm-roberta-large-meetingbank",
    defaultDevice: "gpu",
    defaultModelDataType: "int8",
    maxBatchSize: 50,
    maxForceTokens: 100,
    maxSequenceLength: 512,

    pretrainedModelOptions: {
      use_external_data_format: true,
    },
    factory: LLMLingua2.WithXLMRoBERTa,
  },
} as const satisfies Record<string, LLMLingua2ModelConfig>;

export interface LLMLingua2ModelConfig {
  key: LLMLingua2CompressorModelName;
  modelName: string;
  defaultDevice: DeviceType;
  defaultModelDataType: DataType;
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
    device: DeviceType;
    modelDataType: DataType;
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
      ? "auto"
      : providedTransformersJSConfig.device;
  const transformersJSConfig: TransformersJSConfig = {
    device,
    dtype: providedTransformersJSConfig.modelDataType,
  };

  if (model.factory) {
    const { promptCompressor } = await model.factory(model.modelName, {
      transformerJSConfig: transformersJSConfig,
      oaiTokenizer,
      modelSpecificOptions: model.pretrainedModelOptions,
    });

    return promptCompressor;
  }

  if (
    model.pretrainedModel &&
    model.tokenUtils?.getPureTokens &&
    model.tokenUtils.isBeginOfNewWord
  ) {
    const config = await AutoConfig.from_pretrained(model.modelName);
    const tokenizer = await AutoTokenizer.from_pretrained(model.modelName, {
      config: {
        ...config,
        "transformers.js_config": transformersJSConfig,
      },
    });

    const pretrainedModel = await model.pretrainedModel.from_pretrained(
      model.modelName,
      {
        config: {
          ...config,
          "transformers.js_config": transformersJSConfig,
        },
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
