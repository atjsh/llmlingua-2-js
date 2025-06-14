import { useEffect, useRef, useState } from "react";

import * as Comlink from "comlink";
import { ChevronsUpDown } from "lucide-react";

import {
  LLMLingua2CompressorModels,
  type LLMLingua2ModelConfig
} from "@/lib/core/compressor";
import type { ComlinkCompressorWorker } from "@/lib/worker/worker-compressor";
import CompressorWorker from "@/lib/worker/worker-compressor?worker";

import { Button } from "@/components/ui/button";
import * as Card from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";

const CompressorReadyStatus = {
  READY: "READY",
  NOT_READY: "NOT_READY",
  ERROR: "ERROR"
} as const;
type CompressorReadyStatus =
  (typeof CompressorReadyStatus)[keyof typeof CompressorReadyStatus];

const CompressionProgressStatus = {
  NOT_STARTED: "NOT_STARTED",
  IN_PROGRESS: "IN_PROGRESS",
  COMPLETED: "COMPLETED",
  ERROR: "ERROR"
} as const;
type CompressionProgressStatus =
  (typeof CompressionProgressStatus)[keyof typeof CompressionProgressStatus];

export const CompressorDashboard: React.FC<{
  compressionTarget: string;
  onCompressed: (compressedText: string) => void;
}> = ({ compressionTarget, onCompressed }) => {
  const [compressionStatus, setCompressionStatus] =
    useState<CompressorReadyStatus>(CompressorReadyStatus.NOT_READY);
  const [compressionProgress, setCompressionProgress] =
    useState<CompressionProgressStatus>(CompressionProgressStatus.NOT_STARTED);
  const [model, setModel] = useState<LLMLingua2ModelConfig>(
    LLMLingua2CompressorModels.TINYBERT
  );
  const [device, setDevice] = useState(model.defaultDevice);
  const [modelDataType, setModelDataType] = useState(
    model.defaultModelDataType
  );
  const [keepNewLine, setKeepNewLine] = useState(true);
  const [keepingTokens, setKeepingTokens] = useState<string[]>([".", ";"]);
  const [keepDigits, setKeepDigits] = useState(true);
  const [chunkEndTokens, setChunkEndTokens] = useState<string[]>(["\n", ";"]);
  const [pruningTokens, setPruningTokens] = useState<string[]>([]);
  const [rate, setRate] = useState(0.5);

  const compressorWorker =
    useRef<Comlink.Remote<ComlinkCompressorWorker> | null>(null);

  useEffect(() => {
    const initCompressor = async () => {
      const worker = new CompressorWorker();
      compressorWorker.current = Comlink.wrap(worker);
      const compressor = await compressorWorker.current.initCompressor({
        llmlingua2Config: {
          modelSelection: model.key,
          transformersJSConfig: {
            device,
            modelDataType
          }
        }
      });
      return compressor;
    };

    initCompressor()
      .then(() => {
        setCompressionStatus(CompressorReadyStatus.READY);
      })
      .catch((error) => {
        console.error("Failed to initialize compressor:", error);
        setCompressionStatus(CompressorReadyStatus.ERROR);
      });

    return () => {
      if (compressorWorker.current) {
        compressorWorker.current[Comlink.releaseProxy]();
      }
    };
  }, [model, device, modelDataType]);

  const handleCompression = async () => {
    if (compressionStatus !== CompressorReadyStatus.READY) {
      console.warn("Compressor is not ready.");
      return;
    }
    if (!compressorWorker.current) {
      console.error("Compressor worker is not initialized.");
      return;
    }

    setCompressionProgress(CompressionProgressStatus.IN_PROGRESS);

    try {
      const compressedText = await compressorWorker.current.compressText(
        compressionTarget,
        {
          keepingTokens,
          pruningTokens,
          keepDigits,
          chunkEndTokens,
          rate
        }
      );
      console.log("Compressed text:", compressedText);
      onCompressed(compressedText);
      setCompressionProgress(CompressionProgressStatus.COMPLETED);
    } catch (error) {
      console.error("Compression failed:", error);
      setCompressionProgress(CompressionProgressStatus.ERROR);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <Card.Card>
        <Card.CardHeader>
          <Card.CardTitle>Compress</Card.CardTitle>
          <Card.CardAction>
            <Button
              size={"lg"}
              disabled={
                compressionStatus !== CompressorReadyStatus.READY ||
                compressionProgress === CompressionProgressStatus.IN_PROGRESS
              }
              onClick={handleCompression}
              className="font-bold cursor-pointer"
            >
              {compressionStatus === CompressorReadyStatus.NOT_READY && (
                <span>Loading...</span>
              )}
              {compressionStatus === CompressorReadyStatus.ERROR && (
                <span>Error</span>
              )}
              {compressionStatus === CompressorReadyStatus.READY && (
                <span>Start</span>
              )}
            </Button>
          </Card.CardAction>
          <Card.CardDescription>
            {compressionProgress === CompressionProgressStatus.IN_PROGRESS && (
              <span className="text-yellow-500">
                Compression in progress...
              </span>
            )}
            {compressionProgress === CompressionProgressStatus.COMPLETED && (
              <span className="text-green-500">Compression completed!</span>
            )}
            {compressionProgress === CompressionProgressStatus.ERROR && (
              <span className="text-red-500">Compression failed.</span>
            )}
          </Card.CardDescription>
        </Card.CardHeader>
        <Card.CardContent>
          <div className="mb-4">
            <label className="block mb-2">Model</label>
            <select
              value={model.key}
              onChange={(e) => {
                setCompressionStatus(CompressorReadyStatus.NOT_READY);
                const selectedModel =
                  LLMLingua2CompressorModels[
                    e.target.value as keyof typeof LLMLingua2CompressorModels
                  ];
                setModel(selectedModel);
                setDevice(selectedModel.defaultDevice);
                setModelDataType(selectedModel.defaultModelDataType);
              }}
              className="w-full p-2 border rounded"
            >
              {Object.values(LLMLingua2CompressorModels).map((m) => (
                <option key={m.modelName} value={m.key}>
                  {m.modelName}
                </option>
              ))}
            </select>
          </div>
          <div className="mb-4">
            <Label className="block mb-2" htmlFor="compression-rate-slider">
              Rate
            </Label>
            <Slider
              id="compression-rate-slider"
              value={[rate]}
              onValueChange={(value) => setRate(value[0])}
              min={0}
              max={1}
              step={0.01}
            />
            <div className="mt-2 text-sm text-gray-600">
              Compression Rate: {(rate * 100).toFixed(0)}%
            </div>
          </div>
          <hr className="my-4 hidden" />

          <Collapsible className="hidden">
            <CollapsibleTrigger className=" mb-3" asChild>
              <Button variant="secondary" className="w-full justify-between">
                More Settings
                <ChevronsUpDown />
                <span className="sr-only">Toggle</span>
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="mb-4">
                <Label className="block mb-2">Keeping Tokens</Label>
                <Input
                  value={keepingTokens.join(",")}
                  onChange={(e) =>
                    setKeepingTokens(
                      e.target.value.split(",").map((s) => s.trim())
                    )
                  }
                  placeholder="e.g. ., ;"
                />
                <div className="mt-2 text-sm text-gray-600">
                  These tokens will be preserved in the compression process.
                </div>
              </div>
              <div className="mb-4">
                <Label className="block mb-2">Keep Digits (Numbers)</Label>
                <Checkbox
                  checked={keepDigits}
                  onCheckedChange={(checked) =>
                    setKeepDigits(checked.valueOf().toString() === "true")
                  }
                />
                <div className="mt-2 text-sm text-gray-600">
                  {keepDigits
                    ? "Always keep digits in the text."
                    : "Digits may be removed."}
                </div>
              </div>
              <div className="mb-4">
                <Label className="block mb-2">Keep New Line</Label>
                <Checkbox
                  checked={keepNewLine}
                  onCheckedChange={(checked) =>
                    setKeepNewLine(checked.valueOf().toString() === "true")
                  }
                />
                <div className="mt-2 text-sm text-gray-600">
                  {keepNewLine
                    ? "Always keep new line characters in the text."
                    : "New line characters may be removed."}
                </div>
              </div>
              <div className="mb-4">
                <Label className="block mb-2">Chunk End Tokens</Label>
                <Input
                  value={chunkEndTokens.join(", ")}
                  onChange={(e) =>
                    setChunkEndTokens(
                      e.target.value.split(",").map((s) => s.trim())
                    )
                  }
                  placeholder="e.g. \n, ;"
                />
                <div className="mt-2 text-sm text-gray-600">
                  These tokens will be used to determine the end of chunks in
                  the compression process.
                </div>
              </div>
              <div className="mb-4">
                <Label className="block mb-2">Pruning Tokens</Label>
                <Input
                  value={pruningTokens.join(", ")}
                  onChange={(e) =>
                    setPruningTokens(
                      e.target.value.split(",").map((s) => s.trim())
                    )
                  }
                  placeholder="e.g. a, b, c"
                />
                <div className="mt-2 text-sm text-gray-600">
                  Always remove these tokens.
                </div>
              </div>
            </CollapsibleContent>
          </Collapsible>
        </Card.CardContent>
      </Card.Card>
    </div>
  );
};
