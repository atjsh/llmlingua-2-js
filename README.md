# JavaScript/TypeScript Implementation of LLMLingua-2 (Experimental)

[[한국어]](README.ko-kr.md)

[![NPM Version](https://img.shields.io/npm/v/%40atjsh%2Fllmlingua-2)](https://www.npmjs.com/package/@atjsh/llmlingua-2)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Run [LLMLingua-2](https://github.com/microsoft/LLMLingua) in your web browser or Node.js.

## What is LLMLingua-2?

LLMLingua-2, a small-size yet powerful prompt compression method trained via data distillation from GPT-4 for token classification with a BERT-level encoder, excels in task-agnostic compression. It surpasses LLMLingua in handling out-of-domain data, offering 3x-6x faster performance. [^llmlingua-2]

[^llmlingua-2]: [LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression](https://aclanthology.org/2024.findings-acl.57/)

## Status

**This library is under active development and the API is subject to change.**

# Demo

## Try Now (No Installation Required)

You can test the library [here](https://atjsh.github.io/llmlingua-2-js). WebGPU-enabled browsers are required (Google Chrome is sufficient).

## Source Code

The demo is available in the `examples` directory. You can run it using the following command:

```sh
cd examples/react-vite-webgpu
npm install
```

[Learn More](/examples/react-vite-webgpu/README.md)

# Installation

## Prerequisites

This implementation depends on the following libraries:

- [**@huggingface/transformers**](https://github.com/huggingface/transformers.js)
- [**@tensorflow/tfjs**](https://github.com/tensorflow/tfjs)
- [**js-tiktoken**](https://www.npmjs.com/package/js-tiktoken)

Especially, the `@huggingface/transformers` library utilizes various computational optimizations to achieve high performance. Please consult if the running environment supports the minimum requirements from these libraries.

## NPM

First, install the dependencies:

```sh
npm install @huggingface/transformers @tensorflow/tfjs js-tiktoken
```

Then, install the library:

```sh
npm install @atjsh/llmlingua-2
```

# Usage

## Model Selection

To get started, you can choose between models based on your needs.

1. **XLM-RoBERTa**
   - Pros: High accuracy
   - Cons: Slower, and slightly larger in size
   - Public Model: **[atjsh/llmlingua-2-js-xlm-roberta-large-meetingbank](https://huggingface.co/atjsh/llmlingua-2-js-xlm-roberta-large-meetingbank)**
2. **BERT**
   - Pros: Faster, smaller in size
   - Cons: Lower accuracy compared to XLM-RoBERTa
   - Public Model: **[Arcoldd/llmlingua4j-bert-base-onnx](https://huggingface.co/Arcoldd/llmlingua4j-bert-base-onnx)**

[Learn More](https://llmlingua.com/llmlingua2.html#:~:text=our%20classification%20model.-,Performance,-We%20evaluate%20LLMLingua) about the performance of each model. (Actual performance may vary.)

## With XLM-RoBERTa (The Large Model)

```typescript
import { LLMLingua2 } from "@atjsh/llmlingua-2";

import { Tiktoken } from "js-tiktoken/lite";
import o200k_base from "js-tiktoken/ranks/o200k_base";

const modelName = "atjsh/llmlingua-2-js-xlm-roberta-large-meetingbank";
const oai_tokenizer = new Tiktoken(o200k_base);

const { promptCompressor } = await LLMLingua2.WithXLMRoBERTa(
  modelName,
  {
    device: "auto",
    dtype: "fp32",
  },
  oai_tokenizer,
  {
    modelSpecificOptions: {
      use_external_data_format: true,
    },
  }
);

const compressedText: string = await promptCompressor.compress_prompt(
  "LLMLingua-2, a small-size yet powerful prompt compression method trained via data distillation from GPT-4 for token classification with a BERT-level encoder, excels in task-agnostic compression. It surpasses LLMLingua in handling out-of-domain data, offering 3x-6x faster performance.",
  { rate: 0.8 }
);

console.log({ compressedText });
```

## With BERT (The Small Model)

```typescript
import { LLMLingua2 } from "@atjsh/llmlingua-2";

import { Tiktoken } from "js-tiktoken/lite";
import o200k_base from "js-tiktoken/ranks/o200k_base";

const modelName = "Arcoldd/llmlingua4j-bert-base-onnx";
const oai_tokenizer = new Tiktoken(o200k_base);

const { promptCompressor } = await LLMLingua2.WithBERTMultilingual(
  modelName,
  {
    device: "auto",
    dtype: "fp32",
  },
  oai_tokenizer,
  {
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

## Customization

You can instantiate the `PromptCompressor` by yourself.

```typescript
const promptCompressor = new LLMLingua2.PromptCompressor(
  model,
  tokenizer,
  get_pure_tokens,
  is_begin_of_new_word,
  oai_tokenizer
);
```

## API Reference

> Not available at the moment. Please refer to the type definitions of the `LLMLingua2` object for more details.

# Testing

## Unit Tests

> Not available at the moment.

## E2E Tests

E2E tests are partially available in following directories:

- `src/e2e`
- `examples/**`

# License

See [LICENSE](LICENSE) for details.

# Credits

This software includes other software related under the following licenses:

- LLMLingua (https://github.com/microsoft/LLMLingua), Copyright (c) Microsoft Corporation. (The original logic and implementation was licensed under the MIT license. For licensing, see: https://github.com/microsoft/LLMLingua/blob/main/LICENSE )
