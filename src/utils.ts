// SPDX-License-Identifier: MIT

// Equivalent to Python's string.punctuation
const PUNCTUATION: string = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

export function get_pure_token(
  token: string | null | undefined,
  model_name: string
): string {
  if (model_name.includes("bert-base-multilingual-cased")) {
    return token ? token.replace(/^##/, "") : "";
  } else if (model_name.includes("xlm-roberta-large")) {
    return token ? token.replace(/^▁/, "") : "";
  }
  throw new Error(
    `get_pure_token: Model name ${model_name} not explicitly handled.`
  );
}

export function is_begin_of_new_word(
  token: string | null | undefined,
  model_name: string,
  force_tokens: string[] = [],
  token_map: Record<string, string> = {}
): boolean {
  if (model_name.includes("bert-base-multilingual-cased")) {
    if (
      force_tokens.includes(token ? token.replace(/^##/, "") : "") ||
      Object.values(token_map).includes(token ? token.replace(/^##/, "") : "")
    ) {
      return true;
    }
    return !token?.startsWith("##");
  } else if (model_name.includes("xlm-roberta-large")) {
    if (
      token &&
      (PUNCTUATION.includes(token) ||
        force_tokens.includes(token) ||
        Object.values(token_map).includes(token))
    ) {
      return true;
    }
    return token?.startsWith("▁") || false;
  }
  throw new Error(
    `is_begin_of_new_word: Model name ${model_name} not explicitly handled.`
  );
}

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
