// SPDX-License-Identifier: MIT

/**
 * @categoryDescription Adaptors
 * A collection of utility functions and types for model-specific token handling.
 *
 * @categoryDescription Utils
 * A collection of utility functions.
 */

/**
 * Type definition for a logger function.
 *
 * @category Utils
 */
export type Logger = (...message: unknown[]) => void;

// Equivalent to Python's string.punctuation
const PUNCTUATION: string = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

/**
 * Implementation on `get_pure_token` function of original LLMLingua implementation.
 * @see [Original Implementation](https://github.com/microsoft/LLMLingua/blob/e4e172afb42d8ae3c0b6cb271a3f5d6a812846a0/llmlingua/utils.py#L108)
 *
 * @category Adaptors
 */
export type GetPureTokenFunction = (token: string | null | undefined) => string;

/**
 * Implementation of `GetPureTokenFunction` for "XLM-RoBERTa Large" model.
 *
 * @category Adaptors
 */
export const get_pure_tokens_xlm_roberta_large: GetPureTokenFunction = (
  token
) => {
  return token ? token.replace(/^▁/, "") : "";
};

/**
 * Implementation of `GetPureTokenFunction` for "BERT Base Multilingual Cased" model.
 *
 * @category Adaptors
 */
export const get_pure_tokens_bert_base_multilingual_cased: GetPureTokenFunction =
  (token) => {
    return token ? token.replace(/^##/, "") : "";
  };

/**
 * Implementation on `is_begin_of_new_word` function of original LLMLingua implementation.
 * @see [Original Implementation](https://github.com/microsoft/LLMLingua/blob/e4e172afb42d8ae3c0b6cb271a3f5d6a812846a0/llmlingua/utils.py#L81)
 *
 * @category Adaptors
 */
export type IsBeginOfNewWordFunction = (
  token: string | null | undefined,
  force_tokens?: string[],
  token_map?: Record<string, string>
) => boolean;

/**
 * Implementation of `IsBeginOfNewWordFunction` for "XLM-RoBERTa Large" model.
 *
 * @category Adaptors
 */
export const is_begin_of_new_word_xlm_roberta_large: IsBeginOfNewWordFunction =
  (token, force_tokens = [], token_map = {}) => {
    if (
      token &&
      (PUNCTUATION.includes(token) ||
        force_tokens.includes(token) ||
        Object.values(token_map).includes(token))
    ) {
      return true;
    }
    return token?.startsWith("▁") || false;
  };

/**
 * Implementation of `IsBeginOfNewWordFunction` for "BERT Base Multilingual Cased" model.
 *
 * @category Adaptors
 */
export const is_begin_of_new_word_bert_base_multilingual_cased: IsBeginOfNewWordFunction =
  (token, force_tokens = [], token_map = {}) => {
    if (
      force_tokens.includes(token ? token.replace(/^##/, "") : "") ||
      Object.values(token_map).includes(token ? token.replace(/^##/, "") : "")
    ) {
      return true;
    }
    return !token?.startsWith("##");
  };

/**
 * Implementation on `replace_added_token` function of original LLMLingua implementation.
 * @see [Original Implementation](https://github.com/microsoft/LLMLingua/blob/e4e172afb42d8ae3c0b6cb271a3f5d6a812846a0/llmlingua/utils.py#L102)
 *
 * @category Utils
 */
export function replace_added_token(
  token: string,
  token_map: Record<string, string>
) {
  let t = token;
  for (const [ori, added] of Object.entries(token_map)) {
    t = t.replaceAll(added, ori);
  }

  return t;
}

/**
 * Calculate the **p-th percentile** of a numeric array.
 *
 * The function follows the “inclusive” linear-interpolation rule used by Excel’s
 * `PERCENTILE.INC` and NumPy’s default percentile implementation:
 *
 * 1. The input array is **copied and sorted** (ascending) so the original order
 *    is preserved.
 * 2. An index `k = (n − 1) × (p / 100)` is computed, where `n` is the array’s
 *    length.
 * 3. If `k` is an integer, the element at that index is the percentile.
 *    Otherwise, the result is the linear interpolation between the two nearest
 *    ranks (`⌊k⌋` and `⌈k⌉`).
 *
 * @param {number[]} arr – Source data. The function does **not** mutate it.
 * @param {number} p     – Desired percentile (0 ≤ `p` ≤ 100).
 * @returns {number}     The computed percentile value. If the array is empty,
 *                       the function returns `0`.
 *
 * @throws {RangeError}  If `p` is outside the 0–100 range.
 *
 * @example
 * const data = [7, 15, 36, 39, 40, 41];
 * percentile(data, 25); // → 15   (1st quartile)
 * percentile(data, 50); // → 37.5 (median with interpolation)
 * percentile(data, 90); // → 40.5
 *
 * @category Utils
 */
export function percentile(arr: number[], p: number): number {
  if (arr.length === 0) return 0;
  const sortedArr = [...arr].sort((a, b) => a - b);
  const k = (sortedArr.length - 1) * (p / 100);
  const f = Math.floor(k);
  const c = Math.ceil(k);
  if (f === c) {
    return sortedArr[f];
  }
  const d0 = sortedArr[f] * (c - k);
  const d1 = sortedArr[c] * (k - f);
  return d0 + d1;
}
