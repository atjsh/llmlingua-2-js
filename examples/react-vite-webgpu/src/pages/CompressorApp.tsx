import { useState } from "react";

import { CompressorDashboard } from "@/components/CompressorPanel";
import { ModeToggle } from "@/components/mode-toggle";
import { Button } from "@/components/ui/button";
import * as Card from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";

export function CompressorApp() {
  const [compressionTarget, setCompressionTarget] = useState(
    "LLMLingua-2, a small-size yet powerful prompt compression method trained via data distillation from GPT-4 for token classification with a BERT-level encoder, excels in task-agnostic compression. It surpasses LLMLingua in handling out-of-domain data, offering 3x-6x faster performance."
  );
  const [compressedText, setCompressedText] = useState("");

  const handleCompression = (compressed: string) => {
    setCompressedText(compressed);
  };

  return (
    <div className=" text-left p-3 lg:p-6 flex flex-col gap-6">
      <div className="flex lg:flex-row lg:items-center justify-between gap-4 flex-col">
        <h1 className="scroll-m-20 text-4xl font-extrabold tracking-tight text-balance">
          🤖🗜️ llmlingua-2-js
        </h1>

        <ul className="flex flex-wrap items-center gap-4">
          <li>
            <a href="https://github.com/atjsh/llmlingua-2-js">
              <Button variant={"link"} className=" cursor-pointer">
                GitHub
              </Button>
            </a>
          </li>
          <li>
            <a href="https://www.npmjs.com/package/@atjsh/llmlingua-2">
              <Button variant={"link"} className=" cursor-pointer">
                NPM
              </Button>
            </a>
          </li>
          <li>
            <a href="https://llmlingua-2-js-typedoc.vercel.app/">
              <Button variant={"link"} className=" cursor-pointer">
                API Reference
              </Button>
            </a>
          </li>
          <li>
            <ModeToggle />
          </li>
        </ul>
      </div>

      <Separator />

      <div className="w-full lg:w-1/2 mx-auto px-4 flex flex-col gap-4 mb-10">
        <h2>Text Compressor For LLMs; Right in Your Browser!</h2>
        <div className="flex flex-col gap-2 text-sm">
          <p>
            🗜️ Long documents, Markdown, or Source codes... Compress them before
            sending to LLMs, to save costs and improve performance.
          </p>
          <p>
            ⚙️ To get started, configure the compression settings below and
            enter the text you want to compress.
          </p>
          <p>
            🔒 Every text you enter stays in your browser, and is not sent to
            any server.
          </p>
        </div>
      </div>

      <Card.Card className="">
        <Card.CardContent className="flex flex-col lg:flex-row gap-4">
          <div className="w-full lg:w-1/3">
            <CompressorDashboard
              compressionTarget={compressionTarget}
              onCompressed={handleCompression}
            />
          </div>

          <div className="flex flex-col w-full lg:w-2/3 gap-4">
            <Label
              className="text-lg font-semibold"
              htmlFor="compression-target"
            >
              Compression Target
            </Label>
            <Textarea
              id="compression-target"
              value={compressionTarget}
              onChange={(e) => setCompressionTarget(e.target.value)}
              placeholder="Enter text to compress"
              className="h-40"
            />

            <Label className="text-lg font-semibold" htmlFor="compressed-text">
              Compressed Result
            </Label>
            <Textarea
              id="compressed-text"
              value={compressedText}
              readOnly
              placeholder="Compressed text will appear here"
              className="h-40 bg-gray-100"
            />
          </div>
        </Card.CardContent>
      </Card.Card>
    </div>
  );
}
