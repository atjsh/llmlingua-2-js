# LLMLingua-2의 JavaScript/TypeScript 구현체 (실험)

[[English]](README.md)

[![NPM 버전](https://img.shields.io/npm/v/%40atjsh%2Fllmlingua-2)](https://www.npmjs.com/package/@atjsh/llmlingua-2)
[![라이센스](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

웹 브라우저 및 Node.js에서 [LLMLingua-2](https://github.com/microsoft/LLMLingua-2)를 실행하기 위한 구현체.

## LLMLingua-2란?

LLMLingua-2는 BERT 수준의 인코더를 사용하여 GPT-4로부터 데이터 증류를 통해 훈련된 작고 강력한 프롬프트 압축 방법으로, 태스크에 구애받지 않는 압축을 지원한다. LLMLingua보다 도메인 외 데이터 처리에서 우수하며, 3배에서 6배 더 빠른 성능을 제공한다. [^llmlingua-2]

[^llmlingua-2]: [LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression](https://aclanthology.org/2024.findings-acl.57/)

## 상태

**이 라이브러리는 활발히 개발 중이며 API는 변경될 수 있다.**

# 데모

## 지금 테스트하기

> **참고:** 데모를 웹에 배포하지 않은 상태이다. 그동안 **로컬에서 데모를 실행할 수 있다.**

## 소스 코드

데모는 `examples` 디렉토리에서 사용할 수 있다. 다음 명령어로 실행할 수 있다:

```sh
cd examples/react-vite-webgpu
npm install
```

[자세히 알아보기](/examples/react-vite-webgpu/README.md)

# 설치

## 필수 조건

이 구현체는 다음 라이브러리에 의존한다:

- [**@huggingface/transformers**](https://github.com/huggingface/transformers.js)
- [**@tensorflow/tfjs**](https://github.com/tensorflow/tfjs)

특히, `@huggingface/transformers` 라이브러리는 높은 성능을 달성하기 위해 다양한 계산 최적화를 활용한다. 실행 환경이 이러한 라이브러리의 최소 요구 사항을 지원하는지 확인하라.

## Browser

브라우저에서 라이브러리를 사용하려면 CDN을 통해 포함할 수 있다:

```html
<script src="https://cdn.jsdelivr.net/npm/@atjsh/llmlingua-2@latest/dist/llmlingua-2.min.js"></script>
```

## NPM

먼저, 종속성을 설치해야 한다:

```sh
npm install @huggingface/transformers @tensorflow/tfjs tiktoken
```

그런 다음, 라이브러리를 설치한다:

```sh
npm install @atjsh/llmlingua-2
```

# 사용법

```typescript
import { PromptCompressorLLMLingua2 } from "@atjsh/llmlingua-2";

const modelName = "YOUR_MODEL_NAME"; // e.g., "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"

const compressor = new PromptCompressorLLMLingua2(modelName, { dtype: "int8" });
await compressor.init();

const compressedText: string = await compressor.compress_prompt(
  "LLMLingua-2, a small-size yet powerful prompt compression method trained via data distillation from GPT-4 for token classification with a BERT-level encoder, excels in task-agnostic compression. It surpasses LLMLingua in handling out-of-domain data, offering 3x-6x faster performance.",
  { rate: 0.8 }
);
```

# 라이센스 (License)

[LICENSE](LICENSE)

# 크레딧 (Credits)

This software includes other software related under the following licenses:

- LLMLingua (https://github.com/microsoft/LLMLingua), Copyright (c) Microsoft Corporation. (The original logic and implementation was licensed under the MIT license. For licensing, see: https://github.com/microsoft/LLMLingua/blob/main/LICENSE )
