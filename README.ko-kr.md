# LLMLingua-2의 JavaScript/TypeScript 구현체 (실험)

[[English]](README.md)

[![NPM 버전](https://img.shields.io/npm/v/%40atjsh%2Fllmlingua-2)](https://www.npmjs.com/package/@atjsh/llmlingua-2)
[![라이센스](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

웹 브라우저 및 Node.js에서 [LLMLingua-2](https://github.com/microsoft/LLMLingua-2)를 실행하기 위한 구현체.

## LLMLingua-2란?

LLMLingua-2는 BERT 수준의 인코더를 사용하여 GPT-4로부터 데이터 증류를 통해 훈련된 작고 강력한 프롬프트 압축 방법으로, 태스크에 구애받지 않는 압축을 지원한다. LLMLingua보다 도메인 외 데이터 처리에서 우수하며, 3배에서 6배 더 빠른 성능을 제공한다. [^llmlingua-2]

[^llmlingua-2]: [LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression](https://aclanthology.org/2024.findings-acl.57/)

## 상태

**이 라이브러리는 지속적으로 개발되고 있는 중이며 API는 변경될 수 있다.**

# 데모

## 지금 테스트하기

라이브러리를 [여기](https://atjsh.github.io/llmlingua-2-js)에서 테스트해볼 수 있다. WebGPU 지원 브라우저가 필요하다 (Google Chrome 등).

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
- [**js-tiktoken**](https://www.npmjs.com/package/js-tiktoken)

특히, `@huggingface/transformers` 라이브러리는 높은 성능을 달성하기 위해 다양한 계산 최적화를 활용한다. 실행 환경이 이러한 라이브러리의 최소 요구 사항을 지원하는지 확인하라.

## NPM

먼저, 종속성을 설치해야 한다:

```sh
npm install @huggingface/transformers @tensorflow/tfjs js-tiktoken
```

그런 다음, 라이브러리를 설치한다:

```sh
npm install @atjsh/llmlingua-2
```

# 사용법

## 모델 선택

시작하려면, 사용할 모델을 선택해야 한다. 현재 지원되는 모델은 다음과 같다:

1. **XLM-RoBERTa**
   - 장점: 높은 정확도
   - 단점: 느리고, 크기가 약간 큼
   - 공개 모델: **[atjsh/llmlingua-2-js-xlm-roberta-large-meetingbank](https://huggingface.co/atjsh/llmlingua-2-js-xlm-roberta-large-meetingbank)**
2. **BERT**
   - 장점: 빠르고, 작은 크기
   - 단점: 정확도가 약간 낮음
   - 공개 모델: **[Arcoldd/llmlingua4j-bert-base-onnx](https://huggingface.co/Arcoldd/llmlingua4j-bert-base-onnx)**

## XLM-RoBERTa 사용 예제

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

## BERT 사용 예제

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

## 커스터마이징

`PromptCompressor`를 직접 인스턴스화할 수 있다. 예를 들어, 다음과 같이 할 수 있다:

```typescript
const promptCompressor = new LLMLingua2.PromptCompressor(
  model,
  tokenizer,
  get_pure_tokens,
  is_begin_of_new_word,
  oai_tokenizer
);
```

# API 참조

> 현재는 제공되지 않음. 대신, 타입 정의를 참조하라.

# 테스팅

## 단위 테스트

> 현재는 제공되지 않음.

## 통합 테스트

현재, 통합 테스트는 부분적으로 제공된다. 아래 디렉토리에서 확인할 수 있다:

- `src/e2e`
- `examples/**`

# 라이센스 (License)

[LICENSE](LICENSE)

# 크레딧 (Credits)

This software includes other software related under the following licenses:

- LLMLingua (https://github.com/microsoft/LLMLingua), Copyright (c) Microsoft Corporation. (The original logic and implementation was licensed under the MIT license. For licensing, see: https://github.com/microsoft/LLMLingua/blob/main/LICENSE )
