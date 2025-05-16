# JavaScript/TypeScript Implementation of LLMLingua-2

[![NPM Version](https://img.shields.io/npm/v/%40atjsh%2Fllmlingua-2)](https://www.npmjs.com/package/@atjsh/llmlingua-2)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Run [LLMLingua-2](https://github.com/microsoft/LLMLingua) in your web browser.

## What is LLMLingua-2?

LLMLingua-2, a small-size yet powerful prompt compression method trained via data distillation from GPT-4 for token classification with a BERT-level encoder, excels in task-agnostic compression. It surpasses LLMLingua in handling out-of-domain data, offering 3x-6x faster performance. [^llmlingua-2]

[^llmlingua-2]: [LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression](https://aclanthology.org/2024.findings-acl.57/)

## Status

**This library is under active development and the API is subject to change.**

## Installation

### NPM

First, install the dependencies:

```sh
npm install @huggingface/transformers @tensorflow/tfjs tiktoken
```

Then, install the library:

```sh
npm install @atjsh/llmlingua-2
```

# License

See [LICENSE](LICENSE) for details.

# Credits

This software includes other software related under the following licenses:

- LLMLingua (https://github.com/microsoft/LLMLingua), Copyright (c) Microsoft Corporation. (The original logic and implementation was licensed under the MIT license. For licensing, see: https://github.com/microsoft/LLMLingua/blob/main/LICENSE )
