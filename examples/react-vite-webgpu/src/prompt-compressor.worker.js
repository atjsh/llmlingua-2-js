import { InterruptableStoppingCriteria } from "@huggingface/transformers";
import { PromptCompressorLLMLingua2 } from '@atjsh/llmlingua-2';


const modelName = "atjsh/llmlingua-2-js-xlm-roberta-large-meetingbank";
const stopping_criteria = new InterruptableStoppingCriteria();

/**
 * @type {PromptCompressorLLMLingua2}
 */
let promptCompressor = null;


async function check() {
    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error("WebGPU is not supported (no adapter found)");
        }
    } catch (e) {
        self.postMessage({
            status: "error",
            data: e.toString(),
        });
    }
}

async function load(loadConfig) {
    self.postMessage({
        status: "loading",
        data: "Loading model...",
    });

    const dtype = loadConfig.dtype || "int8";

    promptCompressor = new PromptCompressorLLMLingua2(modelName, { dtype, device: "webgpu" });
    await promptCompressor.init();

    self.postMessage({ status: "ready" });
}

async function generate(messages) {
    self.postMessage({ status: "start" });

    console.log({ messages });

    const input = messages[messages.length - 1]
    const compressionRate = (input.compressionRate / 100).toFixed(5)
    const inputText = input.content;

    const result = await promptCompressor.compress_prompt(inputText, { rate: compressionRate })

    console.log({ result });
    self.postMessage({
        status: "update",
        output: result,
    })


    // Send the output back to the main thread
    self.postMessage({
        status: "complete",
        output: result,
    });
}


// Listen for messages from the main thread
self.addEventListener("message", async (e) => {
    const { type, data } = e.data;

    switch (type) {
        case "check":
            check();
            break;

        case "load":
            load(data);
            break;

        case "generate":
            stopping_criteria.reset();
            generate(data);
            break;

        case "interrupt":
            stopping_criteria.interrupt();
            break;

        case "reset":
            // past_key_values_cache = null;
            stopping_criteria.reset();
            break;
    }
});
