// SPDX-License-Identifier: MIT

/**
 * Migration harness for GH-33: proves that counting with `@huggingface/tokenizers`
 * (`Xenova/gpt-4o`) matches counting with `js-tiktoken` (`o200k_base`).
 *
 * The compression threshold is a percentile over per-word token counts, so any
 * divergence here silently changes compressed output. Run this before releasing:
 *
 *     node dist/e2e/token-counter-parity.js
 *
 * `js-tiktoken` is a dev-only dependency kept solely for this comparison and can
 * be dropped once parity is confirmed.
 */

import { Tiktoken } from "js-tiktoken/lite";
import o200k_base from "js-tiktoken/ranks/o200k_base";

import { LLMLingua2 } from "../index.js";
import { EXAMPLES } from "./long-texts.js";

const tiktoken = new Tiktoken(o200k_base);
const countWithTiktoken = (text: string) => tiktoken.encode(text).length;
const countWithHuggingFace = await LLMLingua2.loadTokenCounter("Xenova/gpt-4o");

let checked = 0;
const mismatches: { text: string; tiktoken: number; huggingface: number }[] = [];

function compare(text: string): void {
  checked++;
  const expected = countWithTiktoken(text);
  const actual = countWithHuggingFace(text);
  if (expected !== actual) {
    mismatches.push({ text, tiktoken: expected, huggingface: actual });
  }
}

for (const example of EXAMPLES) {
  // Whole contexts: what sizes a `targetToken` budget.
  compare(example);

  // Individual words: what the per-word probability weighting actually feeds,
  // and therefore what decides the compression threshold.
  for (const word of example.split(/\s+/)) {
    if (word.length > 0) {
      compare(word);
    }
  }
}

console.log(`Compared ${checked.toLocaleString()} strings.`);

if (mismatches.length > 0) {
  console.error(`\n${mismatches.length.toLocaleString()} mismatch(es):`);
  for (const { text, tiktoken: expected, huggingface: actual } of mismatches) {
    const preview = text.length > 80 ? `${text.slice(0, 80)}…` : text;
    console.error(
      `  js-tiktoken=${expected} huggingface=${actual} ${JSON.stringify(preview)}`
    );
  }
  process.exit(1);
}

console.log("Token counts match. Safe to drop js-tiktoken.");
