// SPDX-License-Identifier: MIT

/**
 * @categoryDescription Token Counting
 * Helpers for supplying the token counter that sizes the compression budget.
 */

import type { Tokenizer } from "@huggingface/tokenizers";

import type { CountTokensFunction, TokenCountingTokenizer } from "./utils.js";

const DEFAULT_ENDPOINT = "https://huggingface.co";
const DEFAULT_REVISION = "main";

/**
 * Adapts a `@huggingface/tokenizers` `Tokenizer` into a {@link CountTokensFunction}.
 *
 * Special tokens are excluded so the count matches what the target LLM bills for
 * the text itself. Tokenizers that append a BOS/CLS token during post-processing
 * would otherwise inflate every count by a constant.
 *
 * @category Token Counting
 *
 * @example
 * ```ts
import { LLMLingua2 } from "@atjsh/llmlingua-2";
import { Tokenizer } from "@huggingface/tokenizers";

const tokenizer = new Tokenizer(tokenizerJson, tokenizerConfigJson);
const countTokens = LLMLingua2.createTokenCounter(tokenizer);
```
 */
export function createTokenCounter(tokenizer: Tokenizer): CountTokensFunction {
  return (text) =>
    tokenizer.encode(text, { add_special_tokens: false }).ids.length;
}

/**
 * Options for {@link loadTokenCounter}.
 *
 * @category Token Counting
 */
export interface LoadTokenCounterOptions {
  /**
   * Git revision to read the tokenizer files from. Defaults to `"main"`.
   */
  revision?: string;

  /**
   * Base URL of the Hugging Face Hub, for mirrors or self-hosted endpoints.
   * Defaults to `"https://huggingface.co"`.
   */
  endpoint?: string;
}

async function fetchJson(url: string, modelId: string): Promise<unknown> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Failed to load tokenizer files for "${modelId}": ${response.status} ${response.statusText} (${url})`
    );
  }
  return await response.json();
}

/**
 * Downloads `tokenizer.json` and `tokenizer_config.json` for a Hugging Face Hub
 * model and returns a {@link CountTokensFunction} built from them.
 *
 * Pick the model to match the LLM the compressed prompt is sent to. Any Hub
 * tokenizer works; for OpenAI-compatible counting:
 *
 * | Model | tiktoken encoding |
 * |---|---|
 * | `"Xenova/gpt-4o"` | `o200k_base` |
 * | `"Xenova/gpt-3.5-turbo"` | `cl100k_base`, which is what the original LLMLingua counts with |
 *
 * Requires the optional peer dependency `@huggingface/tokenizers`, which is
 * imported dynamically so callers who supply their own counter never load it.
 *
 * @category Token Counting
 *
 * @example
 * ```ts
import { LLMLingua2 } from "@atjsh/llmlingua-2";

const countTokens = await LLMLingua2.loadTokenCounter("Xenova/gpt-4o");
```
 */
export async function loadTokenCounter(
  modelId: string,
  options: LoadTokenCounterOptions = {}
): Promise<CountTokensFunction> {
  const { revision = DEFAULT_REVISION, endpoint = DEFAULT_ENDPOINT } = options;

  const { Tokenizer } = await import("@huggingface/tokenizers");

  const base = `${endpoint.replace(/\/+$/, "")}/${modelId}/resolve/${revision}`;
  const [tokenizerJson, tokenizerConfigJson] = await Promise.all([
    fetchJson(`${base}/tokenizer.json`, modelId),
    fetchJson(`${base}/tokenizer_config.json`, modelId),
  ]);

  return createTokenCounter(
    new Tokenizer(tokenizerJson as object, tokenizerConfigJson as object)
  );
}

/**
 * Normalizes either accepted form of a token counter into a
 * {@link CountTokensFunction}.
 *
 * @internal
 */
export function resolveTokenCounter(
  source: CountTokensFunction | TokenCountingTokenizer
): CountTokensFunction {
  return typeof source === "function"
    ? source
    : (text) => source.encode(text).length;
}
