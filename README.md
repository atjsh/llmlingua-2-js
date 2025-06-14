# JavaScript/TypeScript Implementation of LLMLingua-2 (Experimental)

[![NPM Version](https://img.shields.io/npm/v/%40atjsh%2Fllmlingua-2)](https://www.npmjs.com/package/@atjsh/llmlingua-2)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**LLMLingua-2**, Originally developed and implemented in Python by [Microsoft](https://github.com/microsoft/LLMLingua), is a small-size yet powerful prompt compression method.

- **Efficient**: Compresses context prompts with BERT sized models.
- **Accurate**: Achieves high accuracy then other methods while requiring less computational resources.

**llmlingua-2-js**, ported by [atjsh](https://github.com/atjsh), is a pure JavaScript/TypeScript implementation of LLMLingua-2, designed to run in web browsers and Node.js environments.

- **Performance**: Everything can be done in the browser. If your environment supports WebGPU, you can use it. Server-side processing is not required by default.
- **Correctness**: The original logic will be ported to TypeScript as accurately as possible.

# Try Demo Online (No Installation Required)

[You can try it on the GitHub Pages](https://atjsh.github.io/llmlingua-2-js) without any installation.

The source code for demo is available in the `examples` directory. You can run it locally using the following command:

```sh
cd examples/react-vite-webgpu
yarn install
```

[Learn More: Demo source code](/examples/react-vite-webgpu/README.md)

# Getting Started

This implementation depends on the following libraries:

- [**@huggingface/transformers**](https://github.com/huggingface/transformers.js)
- [**@tensorflow/tfjs**](https://github.com/tensorflow/tfjs)
- [**js-tiktoken**](https://www.npmjs.com/package/js-tiktoken)

Especially, the `@huggingface/transformers` library utilizes various computational optimizations to achieve high performance. Please consult if the running environment supports the minimum requirements from these libraries.

## Installation

You can use the library by downloading the library from [npm](https://www.npmjs.com/package/@atjsh/llmlingua-2) and importing it into your JavaScript/TypeScript project.

First, install the dependencies:

```sh
npm install @huggingface/transformers @tensorflow/tfjs js-tiktoken
```

Then, install the library:

```sh
npm install @atjsh/llmlingua-2
```

## Model Selection

You can choose between models based on your needs.

1. **XLM-RoBERTa**
   - Pros: High accuracy
   - Cons: Slower, and slightly larger in size
   - Public Model: **[atjsh/llmlingua-2-js-xlm-roberta-large-meetingbank](https://huggingface.co/atjsh/llmlingua-2-js-xlm-roberta-large-meetingbank)**
2. **BERT**
   - Pros: Faster, smaller in size
   - Cons: Lower accuracy compared to XLM-RoBERTa
   - Public Model: **[Arcoldd/llmlingua4j-bert-base-onnx](https://huggingface.co/Arcoldd/llmlingua4j-bert-base-onnx)**

[Learn More](https://llmlingua.com/llmlingua2.html#:~:text=our%20classification%20model.-,Performance,-We%20evaluate%20LLMLingua) about the performance of each model (actual performance may vary).

The model files will be downloaded automatically by default.

## [API Reference](https://llmlingua-2-js-typedoc.vercel.app/modules/LLMLingua2.html)

For more details on how to use the library, please refer to the [API reference documentation](https://llmlingua-2-js-typedoc.vercel.app/modules/LLMLingua2.html).

# Testing

> Unit tests are not available at the moment.

E2E tests are partially available in following directories:

- `src/e2e`
- `examples/**`

# License

See [LICENSE](LICENSE) for details.

# Credits

This software includes other software related under the following licenses:

- LLMLingua (https://github.com/microsoft/LLMLingua), Copyright (c) Microsoft Corporation. (The original logic and implementation was licensed under the MIT license. For licensing, see: https://github.com/microsoft/LLMLingua/blob/main/LICENSE )
