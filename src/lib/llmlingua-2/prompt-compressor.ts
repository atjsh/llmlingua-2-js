// SPDX-License-Identifier: MIT

import {
  BertForTokenClassification,
  PreTrainedTokenizer,
  Tensor,
  TokenClassifierOutput,
  XLMRobertaForTokenClassification,
} from "@huggingface/transformers";
import { softmax, tensor3d } from "@tensorflow/tfjs";
import { chunk } from "es-toolkit/array";
import { Tiktoken } from "js-tiktoken/lite";

import {
  GetPureTokenFunction,
  IsBeginOfNewWordFunction,
  percentile,
  replace_added_token,
} from "./utils.js";

export class PromptCompressorLLMLingua2 {
  private addedTokens: string[] = [];
  private specialTokens: Set<string>;

  constructor(
    private readonly model:
      | XLMRobertaForTokenClassification
      | BertForTokenClassification,
    private readonly tokenizer: PreTrainedTokenizer,
    private readonly getPureToken: GetPureTokenFunction,
    private readonly isBeginOfNewWord: IsBeginOfNewWordFunction,
    private readonly oaiTokenizer: Tiktoken,
    private readonly llmlingua2Config = {
      max_batch_size: 50,
      max_force_token: 100,
      max_seq_length: 512,
    }
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

  public async compress_prompt(
    context: string,
    {
      rate,
      target_token = -1,
      token_to_word = "mean",
      force_tokens = [],
      force_reserve_digit = false,
      drop_consecutive = false,
      chunk_end_tokens = [".", "\n"],
    }: {
      rate: number;
      target_token?: number;
      token_to_word?: "mean" | "first";
      force_tokens?: string[];
      force_reserve_digit?: boolean;
      drop_consecutive?: boolean;
      chunk_end_tokens?: string[];
    }
  ) {
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

    console.log(
      "original token length: appx. ",
      n_original_token.toLocaleString()
    );

    for (const [original, newToken] of Object.entries(token_map)) {
      context = context.replace(new RegExp(original, "g"), newToken);
    }

    const chunkedContexts = this.chunkContext(context, chunkEndTokenSet);

    console.log(
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

    const compressed_context_strs = await this.compress(chunkedContexts, {
      reduce_rate: Math.max(0, final_reduce_rate),
      token_to_word,
      force_tokens,
      token_map,
      force_reserve_digit,
      drop_consecutive,
    });

    console.log("compression finished");

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

  private async compress(
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

      console.log("input tokenization finished");

      const input_ids_dims = input_ids.dims;

      const outputs: TokenClassifierOutput = await this.model({
        input_ids,
        attention_mask,
      });

      console.log("model inference finished");

      const [batch_size, seq_len, num_classes] = outputs.logits.dims;

      console.log("logits shape:", outputs.logits.dims);

      const logits = tensor3d(
        outputs.logits.data,
        [batch_size, seq_len, num_classes],
        "float32"
      );

      console.log("logits tensor created with shape:", logits.shape);

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
