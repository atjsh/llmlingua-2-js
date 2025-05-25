// SPDX-License-Identifier: MIT

// Equivalent to Python's string.punctuation
const PUNCTUATION: string = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

export type GetPureTokenFunction = (token: string | null | undefined) => string;

export const get_pure_tokens_xlm_roberta_large: GetPureTokenFunction = (
  token: string | null | undefined
): string => {
  return token ? token.replace(/^▁/, "") : "";
};

export const get_pure_tokens_bert_base_multilingual_cased: GetPureTokenFunction =
  (token: string | null | undefined): string => {
    return token ? token.replace(/^##/, "") : "";
  };

export type IsBeginOfNewWordFunction = (
  token: string | null | undefined,
  force_tokens?: string[],
  token_map?: Record<string, string>
) => boolean;

export const is_begin_of_new_word_xlm_roberta_large: IsBeginOfNewWordFunction =
  (
    token: string | null | undefined,
    force_tokens: string[] = [],
    token_map: Record<string, string> = {}
  ): boolean => {
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

export const is_begin_of_new_word_bert_base_multilingual_cased: IsBeginOfNewWordFunction =
  (
    token: string | null | undefined,
    force_tokens: string[] = [],
    token_map: Record<string, string> = {}
  ): boolean => {
    if (
      force_tokens.includes(token ? token.replace(/^##/, "") : "") ||
      Object.values(token_map).includes(token ? token.replace(/^##/, "") : "")
    ) {
      return true;
    }
    return !token?.startsWith("##");
  };

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
