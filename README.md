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

## Test Now

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

```typescript
import { PromptCompressorLLMLingua2 } from "@atjsh/llmlingua-2";

const modelName = "YOUR_MODEL_NAME"; // e.g., "atjsh/llmlingua-2-js-xlm-roberta-large-meetingbank"

const compressor = new PromptCompressorLLMLingua2(modelName, { dtype: "int8" });
await compressor.init();

const compressedText: string = await compressor.compress_prompt(
  "LLMLingua-2, a small-size yet powerful prompt compression method trained via data distillation from GPT-4 for token classification with a BERT-level encoder, excels in task-agnostic compression. It surpasses LLMLingua in handling out-of-domain data, offering 3x-6x faster performance.",
  { rate: 0.8 }
);
```

# License

See [LICENSE](LICENSE) for details.

# Credits

This software includes other software related under the following licenses:

- LLMLingua (https://github.com/microsoft/LLMLingua), Copyright (c) Microsoft Corporation. (The original logic and implementation was licensed under the MIT license. For licensing, see: https://github.com/microsoft/LLMLingua/blob/main/LICENSE )
