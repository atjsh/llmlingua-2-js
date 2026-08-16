// SPDX-License-Identifier: MIT

/**
 * @categoryDescription Core
 * Class & functions for customized use of prompt compression
 */

import {
  PreTrainedModel,
  PreTrainedTokenizer,
  softmax,
  Tensor,
} from "@huggingface/transformers";
import { chunk } from "es-toolkit/array";

import { resolveTokenCounter } from "./token-counter.js";
import {
  CountTokensFunction,
  GetPureTokenFunction,
  IsBeginOfNewWordFunction,
  Logger,
  percentile,
  TokenCountingTokenizer,
} from "./utils.js";

type TokenSpan = readonly [start: number, end: number];
interface ContextChunk {
  text: string;
  separatorBefore: string;
}
interface ForcedLiteralSpan {
  start: number;
  end: number;
  literal: string;
}
interface ForcedTokenSequence {
  key: string;
  ids: number[];
  literal: string;
}
interface ForcedTokenOccurrence {
  key: string;
  literal: string;
  start: number;
  end: number;
}
interface ForcedWordOccurrence extends ForcedTokenOccurrence {
  sourceIndex: number;
}

function collectTensors(
  value: unknown,
  tensors: Set<Tensor>,
  seen = new Set<object>()
): void {
  if (value instanceof Tensor) {
    tensors.add(value);
  } else if (value !== null && typeof value === "object" && !seen.has(value)) {
    seen.add(value);
    for (const child of Object.values(value)) {
      collectTensors(child, tensors, seen);
    }
  }
}

/**
 * Options for compressing prompts.
 *
 * @category Core
 */
export interface CompressPromptOptions {
  /**
   * Float value between 0 and 1 indicating the rate of compression.
   * 0.1 means 10% of the original tokens will be kept
   */
  rate: number;

  /**
   * Target number of tokens to keep after compression.
   * If set, this will override the `rate` option.
   *
   * @defaultValue `-1` (no target)
   */
  targetToken?: number;

  /**
   * How to convert token probabilities to word probabilities.
   * "mean" will average the probabilities of tokens in a word,
   * "first" will take the probability of the first token in a word.
   *
   * @defaultValue `"mean"`
   */
  tokenToWord?: "mean" | "first";

  /**
   * List of tokens that must be kept in the compressed prompt.
   * These tokens will not be removed regardless of their probability.
   *
   * @defaultValue `[]`
   */
  forceTokens?: string[];

  /**
   * If true, reserve a digit for forced tokens.
   *
   * @defaultValue `false`
   */
  forceReserveDigit?: boolean;

  /**
   * If true, drop consecutive tokens that are forced.
   * This is useful to avoid keeping too many forced tokens in a row.
   *
   * @alpha
   * @defaultValue `false`
   */
  dropConsecutive?: boolean;

  /**
   * List of tokens that indicate the end of a chunk.
   * The context will be split into chunks at these tokens.
   * @defaultValue `[".", "\n"]`
   */
  chunkEndTokens?: string[];
}

/**
 * Options for compressing prompts.
 *
 * @category Core
 */
export interface CompressPromptOptionsSnakeCase {
  /**
   * Float value between 0 and 1 indicating the rate of compression.
   * 0.1 means 10% of the original tokens will be kept
   *
   * @group Events
   */
  rate: number;

  /**
   * Target number of tokens to keep after compression.
   * If set, this will override the `rate` option.
   *
   * @defaultValue `-1` (no target)
   */
  target_token?: number;

  /**
   * How to convert token probabilities to word probabilities.
   * "mean" will average the probabilities of tokens in a word,
   * "first" will take the probability of the first token in a word.
   *
   * @defaultValue `"mean"`
   */
  token_to_word?: "mean" | "first";

  /**
   * @deprecated Use `token_to_word` instead.
   */
  token_to_Word?: "mean" | "first";

  /**
   * List of tokens that must be kept in the compressed prompt.
   * These tokens will not be removed regardless of their probability.
   *
   * @defaultValue `[]`
   */
  force_tokens?: string[];

  /**
   * If true, reserve a digit for forced tokens.
   *
   * @defaultValue `false`
   */
  force_reserve_digit?: boolean;

  /**
   * If true, drop consecutive tokens that are forced.
   * This is useful to avoid keeping too many forced tokens in a row.
   *
   * @alpha
   * @defaultValue `false`
   */
  drop_consecutive?: boolean;

  /**
   * List of tokens that indicate the end of a chunk.
   * The context will be split into chunks at these tokens.
   * @defaultValue `[".", "\n"]`
   */
  chunk_end_tokens?: string[];
}

interface CompressSingleContextOptions {
  context: string;
  rate: number;
  target_token: number;
  token_to_word: "mean" | "first";
  force_tokens: string[];
  force_reserve_digit: boolean;
  drop_consecutive: boolean;
  chunk_end_tokens: string[];
}

/**
 * The TypeScript implementation on original `PromptCompressor`, which is a class for compressing prompts using a language model.
 *
 * @see [Original Implementation](https://github.com/microsoft/LLMLingua/blob/e0e9d99beb94098bbd924aa53c2c112eac41c758/llmlingua/prompt_compressor.py)
 * @category Core
 */
export class PromptCompressorLLMLingua2 {
  private specialTokenIds: Set<number>;

  /**
   * Normalized form of the `countTokens` constructor argument.
   */
  private readonly countTokens: CountTokensFunction;

  constructor(
    /**
     * The pre-trained model to use for compression.
     */
    private readonly model: PreTrainedModel,

    /**
     * The pre-trained tokenizer to use for compression.
     */
    private readonly tokenizer: PreTrainedTokenizer,

    /**
     * Function to get the pure token from a token.
     * This is used to normalize tokens before processing.
     */
    private readonly getPureToken: GetPureTokenFunction,

    /**
     * Function to check if a token is the beginning of a new word.
     * This is used to determine how to merge tokens into words.
     */
    private readonly isBeginOfNewWord: IsBeginOfNewWordFunction,

    /**
     * Counts tokens in the tokenizer of the LLM this prompt is destined for,
     * which is what sizes the compression budget.
     *
     * Passing a tokenizer object is deprecated; pass a
     * {@link CountTokensFunction} such as the one returned by
     * `loadTokenCounter` instead.
     */
    countTokens: CountTokensFunction | TokenCountingTokenizer,

    /**
     * Configuration for LLMLingua2.
     */
    private readonly llmlingua2Config = {
      /**
       * Maximum batch size for processing prompts.
       * This is used to limit the number of prompts processed in a single batch.
       */
      max_batch_size: 50,

      /**
       * Maximum number of tokens to force in the compression.
       * This is used to ensure that certain tokens are always included in the compressed prompt.
       */
      max_force_token: 100,

      /**
       * Maximum sequence length for the model.
       * This is used to limit the length of the input sequences to the model.
       */
      max_seq_length: 512,
    },

    /**
     * Logger function to log messages.
     */
    private readonly logger: Logger = () => {}
  ) {
    this.countTokens = resolveTokenCounter(countTokens);
    this.specialTokenIds = new Set(this.tokenizer.all_special_ids);
  }

  /**
   * Compresses a prompt based on the given options.
   */
  public async compress(
    context: string,
    {
      rate,
      targetToken = -1,
      tokenToWord = "mean",
      forceTokens = [],
      forceReserveDigit = false,
      dropConsecutive = false,
      chunkEndTokens = [".", "\n"],
    }: CompressPromptOptions
  ): Promise<string> {
    return this.compressSingleContext({
      context,
      rate,
      target_token: targetToken,
      token_to_word: tokenToWord,
      force_tokens: forceTokens,
      force_reserve_digit: forceReserveDigit,
      drop_consecutive: dropConsecutive,
      chunk_end_tokens: chunkEndTokens,
    });
  }

  /**
   * Compresses a prompt based on the given options. Alias for `compress`, but uses snake_case for options.
   *
   * @alias compress
   */
  public async compress_prompt(
    context: string,
    options: CompressPromptOptionsSnakeCase
  ) {
    return this.compress(context, {
      rate: options.rate,
      targetToken: options.target_token,
      tokenToWord: options.token_to_word ?? options.token_to_Word,
      forceTokens: options.force_tokens,
      forceReserveDigit: options.force_reserve_digit,
      dropConsecutive: options.drop_consecutive,
      chunkEndTokens: options.chunk_end_tokens,
    });
  }

  private async compressSingleContext(options: CompressSingleContextOptions) {
    const { context } = options;
    const {
      rate,
      target_token,
      token_to_word,
      force_tokens,
      force_reserve_digit,
      drop_consecutive,
      chunk_end_tokens,
    } = options;

    if (!Number.isFinite(rate) || rate < 0 || rate > 1) {
      throw new RangeError("rate must be between 0 and 1");
    }
    if (force_tokens.length > this.llmlingua2Config.max_force_token) {
      throw new RangeError(
        `forceTokens cannot contain more than ${this.llmlingua2Config.max_force_token} entries`
      );
    }

    const n_original_token = this.countTokens(context);

    this.logger(
      "original token length: appx. ",
      n_original_token.toLocaleString()
    );

    let final_reduce_rate = 1.0 - rate;

    if (target_token > 0 && n_original_token > 0) {
      const rate_to_keep_for_token_level = Math.min(
        target_token / n_original_token,
        1.0
      );
      final_reduce_rate = 1.0 - rate_to_keep_for_token_level;
    }

    if (final_reduce_rate <= 0) {
      this.logger("compression finished");
      return context;
    }

    const chunkEndTokenSet = new Set(chunk_end_tokens);
    const forceTokenSequenceMap = new Map<string, ForcedTokenSequence>();
    for (const literal of force_tokens) {
      const ids = this.tokenizer.encode(literal, {
        add_special_tokens: false,
      });
      if (
        ids.length === 0 ||
        ids.some((id) => this.specialTokenIds.has(id))
      ) {
        throw new RangeError(
          "Each forceToken must encode to one or more non-special model tokens"
        );
      }
      const key = ids.join(",");
      if (!forceTokenSequenceMap.has(key)) {
        forceTokenSequenceMap.set(key, { key, ids, literal });
      }
    }
    const forceTokenSequences = [...forceTokenSequenceMap.values()];
    const maxContentLength = this.llmlingua2Config.max_seq_length - 2;

    if (
      forceTokenSequences.some(({ ids }) => ids.length > maxContentLength)
    ) {
      throw new RangeError(
        `A forced token sequence cannot exceed ${maxContentLength} model tokens`
      );
    }

    const chunkedContexts = this.chunkContext(
      context,
      chunkEndTokenSet,
      forceTokenSequences
    );

    this.logger(
      "chunking finished. chunk count: ",
      chunkedContexts.length.toLocaleString()
    );

    const compressed_context_strs = await this.compressContexts(
      chunkedContexts,
      {
        reduce_rate: Math.max(0, final_reduce_rate),
        token_to_word,
        force_token_sequences: forceTokenSequences,
        force_reserve_digit,
        drop_consecutive,
      }
    );

    this.logger("compression finished");

    let finalCompressedContext = "";
    for (let i = 0; i < compressed_context_strs.length; i++) {
      const compressedChunk = compressed_context_strs[i];
      if (compressedChunk.length === 0) continue;
      if (finalCompressedContext.length > 0) {
        finalCompressedContext += chunkedContexts[i].separatorBefore;
      }
      finalCompressedContext += compressedChunk;
    }
    return finalCompressedContext;
  }

  private chunkContext(
    originText: string,
    chunkEndTokens: Set<string>,
    forceTokenSequences: ForcedTokenSequence[]
  ): ContextChunk[] {
    const maxLenTokens = this.llmlingua2Config.max_seq_length - 2;
    const chunkTokenSpans: TokenSpan[] = [];
    const origin_tokens = this.tokenizer.tokenize(originText);
    const origin_token_ids = this.tokenizer.encode(originText, {
      add_special_tokens: false,
    });
    const n = origin_tokens.length;

    if (origin_token_ids.length !== n) {
      throw new Error("Tokenizer tokens and IDs are not aligned");
    }

    const forcedOccurrences = this.findForcedTokenOccurrences(
      origin_token_ids,
      forceTokenSequences
    );
    const isWordBoundary = (boundary: number) =>
      boundary === n || this.isBeginOfNewWord(origin_tokens[boundary]);
    const isSafeBoundary = (boundary: number) =>
      isWordBoundary(boundary) &&
      !forcedOccurrences.some(
        ({ start, end }) => start < boundary && boundary < end
      );
    let st = 0;

    while (st < n) {
      let end = Math.min(st + maxLenTokens, n);

      if (end < n) {
        let chunkEnd: number | undefined;

        for (let i = end - 1; i > st; i--) {
          if (chunkEndTokens.has(origin_tokens[i]) && isSafeBoundary(i + 1)) {
            chunkEnd = i + 1;
            break;
          }
        }
        end = chunkEnd ?? end;

        while (end > st && !isSafeBoundary(end)) {
          end--;
        }
        if (end === st) {
          const hasWordBoundary = Array.from(
            { length: Math.min(st + maxLenTokens, n) - st },
            (_, offset) => st + offset + 1
          ).some(isWordBoundary);
          throw new RangeError(
            hasWordBoundary
              ? `Overlapping forced token sequences cannot fit within ${maxLenTokens} model tokens`
              : `A single model word cannot exceed ${maxLenTokens} tokens`
          );
        }
      }

      chunkTokenSpans.push([st, end]);
      st = end;
    }

    return chunkTokenSpans.map(([start, end], index) => ({
      text: this.decodeTokenIds(origin_token_ids.slice(start, end)),
      separatorBefore:
        index === 0
          ? ""
          : this.decodeBoundarySeparator(
              origin_token_ids.slice(...chunkTokenSpans[index - 1]),
              origin_token_ids.slice(start, end)
            ),
    }));
  }

  private findForcedTokenOccurrences(
    tokenIds: number[],
    sequences: ForcedTokenSequence[]
  ): ForcedTokenOccurrence[] {
    const occurrences: ForcedTokenOccurrence[] = [];

    for (const { key, ids, literal } of sequences) {
      for (let start = 0; start <= tokenIds.length - ids.length; start++) {
        if (
          ids.every(
            (tokenId, offset) => tokenIds[start + offset] === tokenId
          )
        ) {
          occurrences.push({ key, literal, start, end: start + ids.length });
        }
      }
    }

    return occurrences.sort(
      (left, right) =>
        left.start - right.start ||
        left.end - right.end ||
        left.key.localeCompare(right.key)
    );
  }

  private forcedTokenMask(
    tokenCount: number,
    occurrences: ForcedTokenOccurrence[]
  ): boolean[] {
    const forced = Array<boolean>(tokenCount).fill(false);

    for (const { start, end } of occurrences) {
      for (let i = start; i < end; i++) {
        forced[i] = true;
      }
    }

    return forced;
  }

  private decodeTokenIds(tokenIds: number[]): string {
    if (tokenIds.length === 0) return "";

    return this.tokenizer.decode(tokenIds, {
      clean_up_tokenization_spaces: false,
    });
  }

  private decodeBoundarySeparator(
    leftTokenIds: number[],
    rightTokenIds: number[]
  ): string {
    const left = this.decodeTokenIds(leftTokenIds);
    const right = this.decodeTokenIds(rightTokenIds);
    const combined = this.decodeTokenIds([...leftTokenIds, ...rightTokenIds]);

    if (combined.startsWith(left) && combined.endsWith(right)) {
      return combined.slice(left.length, combined.length - right.length);
    }
    if (combined.startsWith(left)) {
      const increment = combined.slice(left.length);
      if (increment.endsWith(right)) {
        return increment.slice(0, increment.length - right.length);
      }
    }
    return "";
  }

  private literalMatchesTokenPosition(
    container: ForcedLiteralSpan,
    occurrence: ForcedTokenOccurrence,
    tokenIds: number[]
  ): boolean {
    if (
      occurrence.start < container.start ||
      occurrence.end > container.end
    ) {
      return false;
    }

    const expectedPrefix = tokenIds.slice(container.start, occurrence.start);
    const expectedThrough = tokenIds.slice(container.start, occurrence.end);
    const matches = (actual: number[], expected: number[]) =>
      actual.length === expected.length &&
      actual.every((id, index) => id === expected[index]);

    for (
      let start = 0;
      start + occurrence.literal.length <= container.literal.length;
      start++
    ) {
      if (
        container.literal.slice(start, start + occurrence.literal.length) !==
        occurrence.literal
      ) {
        continue;
      }
      const prefix = this.tokenizer.encode(container.literal.slice(0, start), {
        add_special_tokens: false,
      });
      if (!matches(prefix, expectedPrefix)) continue;

      const through = this.tokenizer.encode(
        container.literal.slice(0, start + occurrence.literal.length),
        { add_special_tokens: false }
      );
      if (matches(through, expectedThrough)) return true;
    }

    return false;
  }

  private renderKeptTokens(
    tokenIds: number[],
    keptTokenMask: boolean[],
    forcedOccurrences: ForcedTokenOccurrence[],
    suppressedOccurrences: Set<number>
  ): string {
    const candidates = forcedOccurrences
      .map((occurrence, index) => ({ occurrence, index }))
      .filter(
        ({ occurrence, index }) =>
          !suppressedOccurrences.has(index) &&
          keptTokenMask
            .slice(occurrence.start, occurrence.end)
            .every(Boolean)
      )
      .sort(
        (left, right) =>
          left.occurrence.start - right.occurrence.start ||
          right.occurrence.end - left.occurrence.end ||
          left.occurrence.key.localeCompare(right.occurrence.key)
      );
    const literalOccurrences: ForcedLiteralSpan[] = [];

    for (const { occurrence } of candidates) {
      const previous = literalOccurrences.at(-1);
      if (!previous || occurrence.start >= previous.end) {
        literalOccurrences.push({ ...occurrence });
        continue;
      }

      if (occurrence.end <= previous.end) {
        if (
          !this.literalMatchesTokenPosition(previous, occurrence, tokenIds)
        ) {
          throw new RangeError(
            "Overlapping forceTokens contain inconsistent literal text"
          );
        }
        continue;
      }

      const sharedTokenIds = tokenIds.slice(occurrence.start, previous.end);
      let sharedLiteralLength = -1;
      for (
        let length = Math.min(
          previous.literal.length,
          occurrence.literal.length
        );
        length > 0;
        length--
      ) {
        const sharedLiteral = occurrence.literal.slice(0, length);
        if (!previous.literal.endsWith(sharedLiteral)) continue;
        const encoded = this.tokenizer.encode(sharedLiteral, {
          add_special_tokens: false,
        });
        if (
          encoded.length === sharedTokenIds.length &&
          encoded.every((id, index) => id === sharedTokenIds[index])
        ) {
          sharedLiteralLength = length;
          break;
        }
      }
      if (sharedLiteralLength < 0) {
        throw new RangeError(
          "Overlapping forceTokens contain inconsistent literal text"
        );
      }
      previous.literal += occurrence.literal.slice(sharedLiteralLength);
      previous.end = occurrence.end;
    }

    const prefixIds: number[] = [];
    let rendered = "";

    const appendDecoded = (ids: number[]) => {
      if (ids.length === 0) return "";
      const before = this.decodeTokenIds(prefixIds);
      const combined = this.decodeTokenIds([...prefixIds, ...ids]);
      const increment = combined.startsWith(before)
        ? combined.slice(before.length)
        : this.decodeTokenIds(ids);
      prefixIds.push(...ids);
      return increment;
    };
    const appendNormalRange = (start: number, end: number) => {
      const ids: number[] = [];
      for (let i = start; i < end; i++) {
        if (keptTokenMask[i]) ids.push(tokenIds[i]);
      }
      rendered += appendDecoded(ids);
    };

    let position = 0;
    for (const occurrence of literalOccurrences) {
      appendNormalRange(position, occurrence.start);
      const ids = tokenIds.slice(occurrence.start, occurrence.end);
      const increment = appendDecoded(ids);
      const decodedOccurrence = this.decodeTokenIds(ids);
      const separator =
        decodedOccurrence.length > 0 && increment.endsWith(decodedOccurrence)
          ? increment.slice(0, -decodedOccurrence.length)
          : increment.slice(0, increment.length - increment.trimStart().length);
      rendered += separator + occurrence.literal;
      position = occurrence.end;
    }
    appendNormalRange(position, tokenIds.length);

    return rendered;
  }

  private mergeTokenToWord(
    tokens: string[],
    token_ids: number[],
    token_probs: number[],
    forced_token_mask: boolean[],
    force_reserve_digit: boolean
  ): {
    words: string[];
    word_token_spans: TokenSpan[];
    word_probs_with_force_logic: number[][];
  } {
    const words: string[] = [];
    const word_token_spans: TokenSpan[] = [];
    const word_probs_with_force_logic: number[][] = [];

    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];
      const token_id = token_ids[i];
      const forced = forced_token_mask[i];
      const prob =
        forced || (force_reserve_digit && /\d/.test(token))
          ? 1.0
          : token_probs[i];

      if (this.specialTokenIds.has(token_id)) {
        continue;
      }

      if (words.length === 0 || this.isBeginOfNewWord(token)) {
        words.push(token);
        word_token_spans.push([i, i + 1]);
        word_probs_with_force_logic.push([prob]);
      } else {
        const pure_token = this.getPureToken(token);
        const last = words.length - 1;

        words[last] += pure_token;
        word_token_spans[last] = [word_token_spans[last][0], i + 1];
        word_probs_with_force_logic[last].push(prob);
      }
    }

    return {
      words,
      word_token_spans,
      word_probs_with_force_logic,
    };
  }

  private tokenProbToWordProb(
    tokenProbsPerWord: number[][],
    convertMode: "mean" | "first" = "mean"
  ): number[] {
    if (convertMode === "mean") {
      return tokenProbsPerWord.map(
        (probs) => probs.reduce((sum, prob) => sum + prob, 0) / probs.length
      );
    } else if (convertMode === "first") {
      return tokenProbsPerWord.map((probs) => probs[0]);
    }
    throw new Error(`Unknown convertMode: ${convertMode}`);
  }

  private async compressContexts(
    contexts: ContextChunk[],
    options: {
      reduce_rate: number;
      token_to_word: "mean" | "first";
      force_token_sequences: ForcedTokenSequence[];
      force_reserve_digit: boolean;
      drop_consecutive: boolean;
    }
  ): Promise<string[]> {
    const {
      reduce_rate,
      token_to_word,
      force_token_sequences,
      force_reserve_digit,
      drop_consecutive,
    } = options;

    if (reduce_rate <= 0) {
      return contexts.map(({ text }) => text);
    } else if (contexts.length === 0) {
      return [];
    }

    const compressed_chunk_strings_flat: string[] = [];

    const chunked_contexts = chunk(
      contexts,
      this.llmlingua2Config.max_batch_size
    );

    for (const contextChunks of chunked_contexts) {
      const contextTexts = contextChunks.map(({ text }) => text);
      const inputs = await this.tokenizer(contextTexts, {
        padding: true,
        truncation: true,
        max_length: this.llmlingua2Config.max_seq_length,
      });
      const ownedTensors = new Set<Tensor>();
      collectTensors(inputs, ownedTensors);

      try {
        this.logger("input tokenization finished");

        const { input_ids, attention_mask } = inputs;
        const outputs = (await this.model(inputs)) as {
          logits: Tensor;
          [key: string]: unknown;
        };
        collectTensors(outputs, ownedTensors);

        this.logger("model inference finished");

        const [batch_size, seq_len, num_classes] = outputs.logits.dims;
        const [input_batch_size, input_ids_seq_len] = input_ids.dims;
        const [mask_batch_size, mask_seq_len] = attention_mask.dims;

        if (
          batch_size !== input_batch_size ||
          batch_size !== mask_batch_size ||
          seq_len !== input_ids_seq_len ||
          seq_len !== mask_seq_len ||
          num_classes < 2
        ) {
          throw new Error("Model logits and tokenizer inputs are not aligned");
        }

        this.logger("logits shape:", outputs.logits.dims);

        const floatLogits = outputs.logits.to("float32");
        if (floatLogits !== outputs.logits) {
          ownedTensors.add(floatLogits);
        }
        const logits = floatLogits.data as Float32Array;

        for (let batchIndex = 0; batchIndex < batch_size; batchIndex++) {
          const active_ids: number[] = [];
          const active_probs: number[] = [];
          const sequenceOffset = batchIndex * seq_len;

          for (let tokenIndex = 0; tokenIndex < seq_len; tokenIndex++) {
            if (Number(attention_mask.data[sequenceOffset + tokenIndex]) <= 0) {
              continue;
            }

            active_ids.push(
              Number(input_ids.data[sequenceOffset + tokenIndex])
            );
            const logitsOffset =
              (sequenceOffset + tokenIndex) * num_classes;
            active_probs.push(
              softmax(
                logits.subarray(logitsOffset, logitsOffset + num_classes)
              )[1]
            );
          }

          const active_tokens = this.tokenizer.tokenize(
            contextTexts[batchIndex],
            { add_special_tokens: true }
          );

          if (active_tokens.length !== active_ids.length) {
            throw new Error("Tokenizer tokens and IDs are not aligned");
          }

          const forcedTokenOccurrences = this.findForcedTokenOccurrences(
            active_ids,
            force_token_sequences
          );
          const forcedTokenMask = this.forcedTokenMask(
            active_ids.length,
            forcedTokenOccurrences
          );

          const {
            words,
            word_token_spans,
            word_probs_with_force_logic,
          } = this.mergeTokenToWord(
            active_tokens,
            active_ids,
            active_probs,
            forcedTokenMask,
            force_reserve_digit
          );

          const word_probs = this.tokenProbToWordProb(
            word_probs_with_force_logic,
            token_to_word
          );

          let forcedWordOccurrences: ForcedWordOccurrence[] = [];
          const droppedOccurrences = new Set<number>();
          const postSelectionDrops = new Set<number>();

          if (drop_consecutive) {
            forcedWordOccurrences = forcedTokenOccurrences.flatMap(
              (
                { key, literal, start: tokenStart, end: tokenEnd },
                sourceIndex
              ) => {
                const start = word_token_spans.findIndex(
                  ([wordStart, wordEnd]) =>
                    wordStart < tokenEnd && tokenStart < wordEnd
                );
                if (start < 0) return [];

                let end = start + 1;
                while (
                  end < word_token_spans.length &&
                  word_token_spans[end][0] < tokenEnd
                ) {
                  end++;
                }
                return [{ key, literal, start, end, sourceIndex }];
              }
            );
            const dropThreshold = percentile(
              word_probs,
              Math.min(100, Math.floor(100 * reduce_rate))
            );
            const forcedWords = Array<boolean>(words.length).fill(false);
            for (const { start, end } of forcedWordOccurrences) {
              forcedWords.fill(true, start, end);
            }

            let previous: ForcedTokenOccurrence | undefined;

            for (let i = 0; i < forcedWordOccurrences.length; i++) {
              const occurrence = forcedWordOccurrences[i];
              if (previous) {
                let hasSelectedWordBetween = false;
                for (
                  let word = previous.end;
                  word < occurrence.start;
                  word++
                ) {
                  if (!forcedWords[word] && word_probs[word] > dropThreshold) {
                    hasSelectedWordBetween = true;
                    break;
                  }
                }
                if (
                  !hasSelectedWordBetween &&
                  occurrence.key === previous.key
                ) {
                  droppedOccurrences.add(occurrence.sourceIndex);
                }
              }
              previous = occurrence;
            }

            const keptForcedWords = Array<boolean>(words.length).fill(false);
            const droppedForcedWords = Array<boolean>(words.length).fill(false);
            for (let i = 0; i < forcedWordOccurrences.length; i++) {
              const { sourceIndex, start, end } = forcedWordOccurrences[i];
              (droppedOccurrences.has(sourceIndex)
                ? droppedForcedWords
                : keptForcedWords
              ).fill(true, start, end);
            }
            for (let i = 0; i < words.length; i++) {
              if (droppedForcedWords[i] && !keptForcedWords[i]) {
                word_probs[i] = 0.0;
              }
            }
          }

          const new_token_probs: number[] = [];
          for (let i = 0; i < words.length; i++) {
            const tokenCount = this.countTokens(words[i]);
            new_token_probs.push(...Array(tokenCount).fill(word_probs[i]));
          }

          const threshold = percentile(
            new_token_probs,
            Math.min(100, Math.floor(100 * reduce_rate + 1))
          );

          const keepWords = word_probs.map(
            (wordProb) =>
              wordProb > threshold ||
              (threshold === 1.0 && wordProb === threshold)
          );

          if (drop_consecutive) {
            const occurrenceCoverage = Array.from(
              { length: words.length },
              () => [] as number[]
            );
            for (let i = 0; i < forcedWordOccurrences.length; i++) {
              const { sourceIndex, start, end } = forcedWordOccurrences[i];
              if (droppedOccurrences.has(sourceIndex)) continue;
              for (let word = start; word < end; word++) {
                occurrenceCoverage[word].push(sourceIndex);
              }
            }
            let previous: ForcedTokenOccurrence | undefined;

            for (let i = 0; i < forcedWordOccurrences.length; i++) {
              const occurrence = forcedWordOccurrences[i];
              if (droppedOccurrences.has(occurrence.sourceIndex)) continue;
              if (
                !keepWords
                  .slice(occurrence.start, occurrence.end)
                  .some(Boolean)
              ) {
                continue;
              }

              let hasKeptWordBetween = false;
              if (previous) {
                for (
                  let word = previous.end;
                  word < occurrence.start;
                  word++
                ) {
                  if (
                    keepWords[word] &&
                    (occurrenceCoverage[word].length === 0 ||
                      occurrenceCoverage[word].some(
                        (index) => !postSelectionDrops.has(index)
                      ))
                  ) {
                    hasKeptWordBetween = true;
                    break;
                  }
                }
              }
              if (
                previous &&
                !hasKeptWordBetween &&
                occurrence.key === previous.key
              ) {
                postSelectionDrops.add(occurrence.sourceIndex);
                continue;
              }
              previous = occurrence;
            }

            const retainedForcedWords = Array<boolean>(words.length).fill(
              false
            );
            const droppedForcedWords = Array<boolean>(words.length).fill(
              false
            );
            for (let i = 0; i < forcedWordOccurrences.length; i++) {
              const { sourceIndex, start, end } = forcedWordOccurrences[i];
              if (droppedOccurrences.has(sourceIndex)) continue;
              (postSelectionDrops.has(sourceIndex)
                ? droppedForcedWords
                : retainedForcedWords
              ).fill(true, start, end);
            }
            for (let i = 0; i < words.length; i++) {
              if (droppedForcedWords[i] && !retainedForcedWords[i]) {
                keepWords[i] = false;
              }
            }
          }

          const keptTokenMask = Array<boolean>(active_ids.length).fill(false);
          for (let i = 0; i < words.length; i++) {
            if (!keepWords[i]) continue;
            keptTokenMask.fill(
              true,
              word_token_spans[i][0],
              word_token_spans[i][1]
            );
          }

          compressed_chunk_strings_flat.push(
            this.renderKeptTokens(
              active_ids,
              keptTokenMask,
              forcedTokenOccurrences,
              new Set([...droppedOccurrences, ...postSelectionDrops])
            )
          );
        }
      } finally {
        for (const tensor of ownedTensors) {
          tensor.dispose();
        }
      }
    }

    return compressed_chunk_strings_flat;
  }
}
