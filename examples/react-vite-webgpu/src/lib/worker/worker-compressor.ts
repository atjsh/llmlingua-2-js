import * as Comlink from "comlink";

import {
  LossyTextCompressor,
  type CompressorConfig,
  type LLMLingua2CompressorOptions
} from "@/lib/core/compressor";

let compressor: LossyTextCompressor | null = null;

async function initCompressor(config: CompressorConfig): Promise<true> {
  if (compressor) {
    return true;
  }

  compressor = new LossyTextCompressor(config);
  await compressor.init();
  return true;
}

function compressText(
  text: string,
  options: LLMLingua2CompressorOptions
): Promise<string> {
  if (!compressor) {
    throw new Error(
      "Compressor is not initialized. Call initCompressor first."
    );
  }
  return compressor.compress(text, options);
}

Comlink.expose({
  initCompressor,
  compressText
});

export interface ComlinkCompressorWorker {
  initCompressor: typeof initCompressor;
  compressText: typeof compressText;
}
