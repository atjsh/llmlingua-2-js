// SPDX-License-Identifier: MIT

/**
 * @categoryDescription Core
 * Class & functions for customized use of prompt compression
 */

import {
  PreTrainedModel,
  PreTrainedTokenizer,
  Tensor,
  TokenClassifierOutput,
} from "@huggingface/transformers";
import { softmax, tensor3d } from "@tensorflow/tfjs";
import { chunk } from "es-toolkit/array";
import { Tiktoken } from "js-tiktoken/lite";

import {
  GetPureTokenFunction,
  IsBeginOfNewWordFunction,
  Logger,
  percentile,
  replace_added_token,
} from "./utils.js";

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
 * @see [Original Implementation](https://github.com/microsoft/LLMLingua/blob/e4e172afb42d8ae3c0b6cb271a3f5d6a812846a0/llmlingua/prompt_compressor.py)
 * @category Core
 */
export class PromptCompressorLLMLingua2 {
  private addedTokens: string[] = [];
  private specialTokens: Set<string>;

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
     * The tokenizer to use calculating the compression rate.
     */
    private readonly oaiTokenizer: Tiktoken,

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
    private readonly logger: Logger = console.log
  ) {
    for (let i = 0; i < this.llmlingua2Config.max_force_token; i++) {
      this.addedTokens.push(`[NEW${i}]`);
    }

    const specialTokensMap = this.tokenizer.special_tokens || {};

    this.specialTokens = new Set<string>();

    for (const [key, value] of Object.entries(specialTokensMap)) {
      if (key !== "additional_special_tokens") {
        this.specialTokens.add(value);
      }
    }
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
      tokenToWord: options.token_to_Word,
      forceTokens: options.force_tokens,
      forceReserveDigit: options.force_reserve_digit,
      dropConsecutive: options.drop_consecutive,
      chunkEndTokens: options.chunk_end_tokens,
    });
  }

  private async compressSingleContext(options: CompressSingleContextOptions) {
    let { context } = options;
    const {
      rate,
      target_token,
      token_to_word,
      force_tokens,
      force_reserve_digit,
      drop_consecutive,
      chunk_end_tokens,
    } = options;

    let token_map: Record<string, string> = {};

    for (let i = 0; i < force_tokens.length; i++) {
      const token = force_tokens[i];
      const tokenized = this.tokenizer.tokenize(token);
      if (tokenized.length !== 1) {
        token_map[token] = this.addedTokens[i];
      }
    }

    const chunkEndTokenSet = new Set(chunk_end_tokens);
    chunk_end_tokens.forEach((token) => {
      if (token_map[token]) {
        chunkEndTokenSet.add(token_map[token]);
      }
    });

    const n_original_token = this.getTokenLength(context);

    this.logger(
      "original token length: appx. ",
      n_original_token.toLocaleString()
    );

    for (const [original, newToken] of Object.entries(token_map)) {
      context = context.replace(new RegExp(original, "g"), newToken);
    }

    const chunkedContexts = this.chunkContext(context, chunkEndTokenSet);

    this.logger(
      "chunking finished. chunk count: ",
      chunkedContexts.length.toLocaleString()
    );

    let final_reduce_rate = 1.0 - rate;

    if (target_token > 0 && n_original_token > 0) {
      const rate_to_keep_for_token_level = Math.min(
        target_token / n_original_token,
        1.0
      );
      final_reduce_rate = 1.0 - rate_to_keep_for_token_level;
    }

    const compressed_context_strs = await this.compressContexts(
      chunkedContexts,
      {
        reduce_rate: Math.max(0, final_reduce_rate),
        token_to_word,
        force_tokens,
        token_map,
        force_reserve_digit,
        drop_consecutive,
      }
    );

    this.logger("compression finished");

    const final_compressed_context = compressed_context_strs.join("\n");
    return final_compressed_context;
  }

  private chunkContext(
    originText: string,
    chunkEndTokens: Set<string>
  ): string[] {
    const maxLenTokens = this.llmlingua2Config.max_seq_length - 2;
    const origin_list: string[] = [];
    const origin_tokens = this.tokenizer.tokenize(originText);
    const n = origin_tokens.length;
    let st = 0;

    while (st < n) {
      if (st + maxLenTokens > n - 1) {
        const chunk = this.tokenizer.decoder.decode(
          origin_tokens.slice(st, n - 1)
        );
        origin_list.push(chunk);
        break;
      } else {
        let ed = st + maxLenTokens;

        for (let j = 0; j < ed - st; j++) {
          if (chunkEndTokens.has(origin_tokens[ed - j])) {
            ed = ed - j;
            break;
          }
        }

        const chunk = this.tokenizer.decoder.decode(
          origin_tokens.slice(st, ed + 1)
        );

        origin_list.push(chunk);
        st = ed + 1;
      }
    }

    return origin_list;
  }

  private getTokenLength(text: string): number {
    return this.tokenizer.tokenize(text).length;
  }

  private mergeTokenToWord(
    tokens: string[],
    token_probs: number[],
    force_tokens_original: string[],
    token_map: Record<string, string>,
    force_reserve_digit: boolean
  ): {
    words: string[];
    word_probs_with_force_logic: number[][];
    valid_token_probs_no_force: number[][];
  } {
    const words: string[] = [];
    const word_probs_with_force_logic: number[][] = [];
    const valid_token_probs_no_force: number[][] = [];

    for (let i = 0; i < tokens.length; i++) {
      let token = tokens[i];
      let prob = token_probs[i];

      if (this.specialTokens.has(token)) {
      } else if (
        this.isBeginOfNewWord(token, force_tokens_original, token_map)
      ) {
        const pure_token = this.getPureToken(token);
        const prob_no_force = prob;

        if (
          force_tokens_original.includes(pure_token) ||
          Object.values(token_map).includes(pure_token)
        ) {
          prob = 1.0;
        }
        token = replace_added_token(token, token_map);
        words.push(token);
        word_probs_with_force_logic.push([
          force_reserve_digit && token.match(/^\d/) ? 1.0 : prob,
        ]);
        valid_token_probs_no_force.push([prob_no_force]);
      } else {
        const pure_token = this.getPureToken(token);

        words[words.length - 1] += pure_token;

        if (word_probs_with_force_logic.length === 0) {
          word_probs_with_force_logic.push([
            force_reserve_digit && token.match(/^\d/) ? 1.0 : prob,
          ]);
        } else {
          word_probs_with_force_logic[
            word_probs_with_force_logic.length - 1
          ].push(force_reserve_digit && token.match(/^\d/) ? 1.0 : prob);
        }

        if (valid_token_probs_no_force.length === 0) {
          valid_token_probs_no_force.push([prob]);
        } else {
          valid_token_probs_no_force[
            valid_token_probs_no_force.length - 1
          ].push(prob);
        }
      }
    }

    return {
      words,
      word_probs_with_force_logic,
      valid_token_probs_no_force,
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
    contexts: string[],
    options: {
      reduce_rate: number;
      token_to_word: "mean" | "first";
      force_tokens: string[];
      token_map: Record<string, string>;
      force_reserve_digit: boolean;
      drop_consecutive: boolean;
    }
  ): Promise<string[]> {
    const {
      reduce_rate,
      token_to_word,
      force_tokens,
      token_map,
      force_reserve_digit,
      drop_consecutive,
    } = options;

    if (reduce_rate <= 0) {
      return contexts;
    } else if (contexts.length === 0) {
      return [];
    }

    const compressed_chunk_strings_flat: string[] = [];

    const chunked_contexts = chunk(
      contexts,
      this.llmlingua2Config.max_batch_size
    );

    for (const context of chunked_contexts) {
      const { input_ids, attention_mask } = await this.tokenizer(context, {
        padding: true,
        truncation: true,
      });

      this.logger("input tokenization finished");

      const input_ids_dims = input_ids.dims;

      const outputs: TokenClassifierOutput = await this.model({
        input_ids,
        attention_mask,
      });

      this.logger("model inference finished");

      const [batch_size, seq_len, num_classes] = outputs.logits.dims;

      this.logger("logits shape:", outputs.logits.dims);

      const logits = tensor3d(
        outputs.logits.data,
        [batch_size, seq_len, num_classes],
        "float32"
      );

      this.logger("logits tensor created with shape:", logits.shape);

      const probs = softmax(logits, -1);

      for (let j = 0; j < batch_size; j++) {
        const chunk_probs_class1 = probs.slice([j, 0, 1], [1, -1, 1]);
        const chunk_ids = input_ids[j] as Tensor;
        const chunk_mask = attention_mask[j] as Tensor;

        const chunk_mask_number_array = Array.from(chunk_mask.data, (v) =>
          Number(v)
        );

        const active_probs = chunk_probs_class1
          .dataSync()
          .filter((_, i) => chunk_mask_number_array[i] > 0);

        const active_ids = chunk_ids.data
          .filter((_, i) => chunk_mask_number_array[i] > 0n)
          .filter((v) => v !== 0n);

        if (active_ids.length === 0) {
          compressed_chunk_strings_flat.push("");
          continue;
        }

        const token_list = this.tokenizer.model.convert_ids_to_tokens(
          new Tensor("int64", active_ids, [active_ids.length]).tolist()
        );

        const token_prob_list = Array.from(active_probs);

        const { words, word_probs_with_force_logic } = this.mergeTokenToWord(
          token_list,
          token_prob_list,
          force_tokens,
          token_map,
          force_reserve_digit
        );

        const word_probs = this.tokenProbToWordProb(
          word_probs_with_force_logic,
          token_to_word
        );

        const new_token_probs: number[] = [];
        for (let i = 0; i < words.length; i++) {
          const word = words[i];
          const word_prob = word_probs[i];

          const new_token = this.oaiTokenizer.encode(word);

          new_token_probs.push(...Array(new_token.length).fill(word_prob));
        }

        const threshold = percentile(new_token_probs, 100 * reduce_rate);

        const keep_words: string[] = [];

        for (let i = 0; i < words.length; i++) {
          const word = words[i];
          const word_prob = word_probs[i];

          if (
            word_prob > threshold ||
            (threshold === 1.0 && word_prob == threshold)
          ) {
            keep_words.push(word);
          }
        }

        const keep_str = replace_added_token(
          this.tokenizer.decoder.decode(keep_words),
          token_map
        );

        compressed_chunk_strings_flat.push(keep_str);
      }
    }

    return compressed_chunk_strings_flat;
  }
}
