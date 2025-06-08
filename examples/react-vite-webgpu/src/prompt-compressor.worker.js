import {
  AutoConfig,
  AutoTokenizer,
  InterruptableStoppingCriteria,
  MobileBertPreTrainedModel,
  TokenClassifierOutput,
} from "@huggingface/transformers";
import { Tiktoken } from "js-tiktoken/lite";
import o200k_base from "js-tiktoken/ranks/o200k_base";

import { LLMLingua2 } from "@atjsh/llmlingua-2";

const oai_tokenizer = new Tiktoken(o200k_base);
const stopping_criteria = new InterruptableStoppingCriteria();

/**
 * @type {LLMLingua2.PromptCompressor}
 */
let promptCompressor = null;
let webGPUAvailable = false;

export class MobileBertForTokenClassification extends MobileBertPreTrainedModel {
  /**
   * Calls the model on new inputs.
   *
   * @param {Object} model_inputs The inputs to the model.
   * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for token classification.
   */
  async _call(model_inputs) {
    return new TokenClassifierOutput(await super._call(model_inputs));
  }
}

async function check() {
  try {
    if (navigator.gpu) {
      const adapter = await navigator.gpu?.requestAdapter();
      if (adapter) {
        webGPUAvailable = true;

        return;
      }
    }

    webGPUAvailable = false;
  } catch (e) {
    self.postMessage({
      status: "error",
      data: e.toString(),
    });
  }
}

async function load(loadConfig) {
  self.postMessage({
    status: "loading",
    data: "Loading model...",
  });

  const { dtype, modelName, modelKind } = loadConfig;

  try {
    if (modelKind === "bert") {
      promptCompressor = (
        await LLMLingua2.WithBERTMultilingual(modelName, {
          transformerJSConfig: {
            device: webGPUAvailable ? "webgpu" : "auto",
            dtype: dtype,
          },
          oaiTokenizer: oai_tokenizer,
          modelSpecificOptions: {
            subfolder: "",
          },
        })
      ).promptCompressor;
    } else if (modelKind === "roberta") {
      promptCompressor = (
        await LLMLingua2.WithXLMRoBERTa(modelName, {
          transformerJSConfig: {
            device: webGPUAvailable ? "webgpu" : "auto",
            dtype: dtype,
          },
          oaiTokenizer: oai_tokenizer,
        })
      ).promptCompressor;
    } else if (modelKind === "mobilebert") {
      const transformerJSConfig = {
        device: webGPUAvailable ? "webgpu" : "auto",
        dtype: dtype,
      };

      const config = await AutoConfig.from_pretrained(modelName);
      const tokenizer = await AutoTokenizer.from_pretrained(modelName, {
        config: {
          ...config,
          "transformers.js_config": transformerJSConfig,
        },
      });
      const model = await MobileBertForTokenClassification.from_pretrained(
        modelName,
        {
          config: {
            ...config,
            "transformers.js_config": transformerJSConfig,
          },
        }
      );
      promptCompressor = new LLMLingua2.PromptCompressor(
        model,
        tokenizer,
        LLMLingua2.get_pure_tokens_bert_base_multilingual_cased,
        LLMLingua2.is_begin_of_new_word_bert_base_multilingual_cased,
        oai_tokenizer
      );
    } else {
      promptCompressor = (
        await LLMLingua2.WithBERTMultilingual(modelName, {
          transformerJSConfig: {
            device: webGPUAvailable ? "webgpu" : "auto",
            dtype: dtype,
          },
          oaiTokenizer: oai_tokenizer,
        })
      ).promptCompressor;
    }
  } catch (error) {
    console.error(error);

    self.postMessage({
      status: "error",
      data: error.toString(),
    });
    return;
  }

  self.postMessage({
    status: "ready",
    device: webGPUAvailable ? "webgpu" : "auto",
  });
}

async function generate(messages) {
  self.postMessage({ status: "start" });

  console.log({ messages });

  const input = messages[messages.length - 1];
  const compressionRate = (input.compressionRate / 100).toFixed(5);
  const inputText = input.content;
  const inputLength = inputText.length;

  const start = performance.now();

  const result = await promptCompressor.compress_prompt(inputText, {
    rate: compressionRate,
  });

  const end = performance.now();
  const time = end - start;

  const compressedLength = result.length;

  console.log({ result });
  self.postMessage({
    status: "update",
    output: {
      result: result,
      inputLength: inputLength,
      compressedLength: compressedLength,
      time: time,
    },
  });

  // Send the output back to the main thread
  self.postMessage({
    status: "complete",
    output: {
      result: result,
      inputLength: inputLength,
      compressedLength: compressedLength,
    },
  });
}

// Listen for messages from the main thread
self.addEventListener("message", async (e) => {
  const { type, data } = e.data;

  switch (type) {
    case "check":
      check();
      break;

    case "load":
      load(data);
      break;

    case "generate":
      stopping_criteria.reset();
      generate(data);
      break;

    case "interrupt":
      stopping_criteria.interrupt();
      break;

    case "reset":
      // past_key_values_cache = null;
      stopping_criteria.reset();
      break;
  }
});
