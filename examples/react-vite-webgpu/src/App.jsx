import { useEffect, useRef, useState } from "react";

import Chat from "./components/Chat";
import ArrowRightIcon from "./components/icons/ArrowRightIcon";
import StopIcon from "./components/icons/StopIcon";
import Progress from "./components/Progress";
import { EXAMPLES } from "./long-texts";
import RetryIcon from "./components/icons/RetryIcon";

const STICKY_SCROLL_THRESHOLD = 120;

const AVAILABLE_MODELS = {
  tinybert: {
    name: "atjsh/llmlingua-2-js-tinybert-meetingbank",
    description: "60MB (Experimental, English Only Recommended)",
    dtypes: ["fp32"],
  },
  mobilebert: {
    name: "atjsh/llmlingua-2-js-mobilebert-meetingbank",
    description: "100MB (Experimental, English Only Recommended)",
    dtypes: ["fp32"],
  },
  bert: {
    name: "Arcoldd/llmlingua4j-bert-base-onnx",
    description: "700MB",
    dtypes: ["fp32"],
  },
  roberta: {
    name: "atjsh/llmlingua-2-js-xlm-roberta-large-meetingbank",
    description: "500MB ~ 1GB",
    dtypes: ["int8", "q8", "uint8", "q4", "bnb4"],
  },
};

function App() {
  // Create a reference to the worker object.
  const worker = useRef(null);

  const textareaRef = useRef(null);
  const chatContainerRef = useRef(null);

  // Model loading and progress
  const [status, setStatus] = useState(null);
  const [deviceMode, setDeviceMode] = useState(null);
  const [error, setError] = useState(null);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [progressItems, setProgressItems] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [modelKey, setModelKey] = useState("bert");
  const [dtype, setDtype] = useState(AVAILABLE_MODELS[modelKey].dtypes[0]);

  // Inputs and outputs
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [tps, setTps] = useState(null);
  const [numTokens] = useState(null);
  const [compressionRate, setCompressionRate] = useState(95);

  function onEnter(message) {
    setMessages((prev) => [
      ...prev,
      { role: "user", content: message, compressionRate },
    ]);
    setTps(null);
    setIsRunning(true);
    setInput("");
  }

  function onInterrupt() {
    // NOTE: We do not set isRunning to false here because the worker
    // will send a 'complete' message when it is done.
    worker.current.postMessage({ type: "interrupt" });
  }

  useEffect(() => {
    resizeInput();
  }, [input]);

  function resizeInput() {
    if (!textareaRef.current) return;

    const target = textareaRef.current;
    target.style.height = "auto";
    const newHeight = Math.min(Math.max(target.scrollHeight, 24), 200);
    target.style.height = `${newHeight}px`;
  }

  // We use the `useEffect` hook to setup the worker as soon as the `App` component is mounted.
  useEffect(() => {
    // Create the worker if it does not yet exist.
    if (!worker.current) {
      worker.current = new Worker(
        new URL("./prompt-compressor.worker.js", import.meta.url),
        {
          type: "module",
        }
      );
      worker.current.postMessage({ type: "check" }); // Do a feature check
    }

    // Create a callback function for messages from the worker thread.
    const onMessageReceived = (e) => {
      switch (e.data.status) {
        case "loading":
          // Model file start load: add a new progress item to the list.
          setStatus("loading");
          setLoadingMessage(e.data.data);
          break;

        case "initiate":
          setProgressItems((prev) => [...prev, e.data]);
          break;

        case "progress":
          // Model file progress: update one of the progress items.
          setProgressItems((prev) =>
            prev.map((item) => {
              if (item.file === e.data.file) {
                return { ...item, ...e.data };
              }
              return item;
            })
          );
          break;

        case "done":
          // Model file loaded: remove the progress item from the list.
          setProgressItems((prev) =>
            prev.filter((item) => item.file !== e.data.file)
          );
          break;

        case "ready":
          // Pipeline ready: the worker is ready to accept messages.
          setStatus("ready");
          setDeviceMode(e.data.device);
          break;

        case "start":
          {
            // Start generation
            setMessages((prev) => [
              ...prev,
              { role: "assistant", content: "" },
            ]);
          }
          break;

        case "update":
          {
            // Generation update: update the output text.
            // Parse messages
            const { output } = e.data;
            setMessages((prev) => {
              const cloned = [...prev];
              const last = cloned.at(-1);
              cloned[cloned.length - 1] = {
                ...last,
                content: last.content + output.result,
                inputLength: output.inputLength,
                compressedLength: output.compressedLength,
                time: output.time,
              };
              return cloned;
            });
          }
          break;

        case "complete":
          // Generation complete: re-enable the "Generate" button
          setIsRunning(false);
          break;

        case "error":
          setError(e.data.data);
          setLoadingMessage(e.data.data);
          break;
      }
    };

    const onErrorReceived = (e) => {
      console.error("Worker error:", e);
    };

    // Attach the callback function as an event listener.
    worker.current.addEventListener("message", onMessageReceived);
    worker.current.addEventListener("error", onErrorReceived);

    // Define a cleanup function for when the component is unmounted.
    return () => {
      worker.current.removeEventListener("message", onMessageReceived);
      worker.current.removeEventListener("error", onErrorReceived);
    };
  }, []);

  // Send the messages to the worker thread whenever the `messages` state changes.
  useEffect(() => {
    if (messages.filter((x) => x.role === "user").length === 0) {
      // No user messages yet: do nothing.
      return;
    }
    if (messages.at(-1).role === "assistant") {
      // Do not update if the last message is from the assistant
      return;
    }
    setTps(null);
    worker.current.postMessage({ type: "generate", data: messages });
  }, [messages, isRunning]);

  useEffect(() => {
    if (!chatContainerRef.current || !isRunning) return;
    const element = chatContainerRef.current;
    if (
      element.scrollHeight - element.scrollTop - element.clientHeight <
      STICKY_SCROLL_THRESHOLD
    ) {
      element.scrollTop = element.scrollHeight;
    }
  }, [messages, isRunning]);

  return (
    <div className="flex flex-col h-screen mx-auto items justify-end text-gray-800 dark:text-gray-200 bg-white dark:bg-gray-900">
      {status === null && messages.length === 0 && (
        <div className="h-full overflow-auto scrollbar-thin flex justify-center items-center flex-col relative">
          <div className="flex flex-col items-center mb-1 max-w-[500px] text-center">
            <h1 className="text-4xl font-bold mb-1">
              🤖🗜️ <br /> LLM Context Compressor!
            </h1>
          </div>

          <div className="flex flex-col items-center px-4">
            <p className="max-w-[514px] text-center">
              Long txt, Markdown, or Source codes... <br />
              Compress them before sending to LLMs, <br /> to save costs and
              improve performance.
            </p>
            <br />
            <p className="max-w-[514px] mb-4 text-center">
              Check out the source code for this demo on{" "}
              <a
                href="https://github.com/atjsh/llmlingua-2-js"
                target="_blank"
                rel="noreferrer"
                className="underline"
              >
                GitHub
              </a>
              . <br />
              built with{" "}
              <a
                href="https://www.npmjs.com/package/@atjsh/llmlingua-2"
                target="_blank"
                rel="noreferrer"
                className="underline"
              >
                llmlingua-2-js
              </a>
              .
            </p>

            {error && (
              <div className="text-red-500 text-center mb-2">
                <p className="mb-1">
                  Unable to load model due to the following error:
                </p>
                <p className="text-sm">{error}</p>
              </div>
            )}

            <div className="flex flex-col gap-3 items-center mb-4">
              <div className="flex flex-col gap-2">
                <div className="flex flex-col gap-2">
                  <label htmlFor="model">Model</label>
                  <select
                    className="border dark:bg-gray-700 px-4 py-2 rounded-lg bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-200"
                    value={modelKey}
                    onChange={(e) => {
                      setModelKey(e.target.value);
                      setDtype(AVAILABLE_MODELS[e.target.value].dtypes[0]);
                    }}
                  >
                    {Object.keys(AVAILABLE_MODELS).map((key) => (
                      <optgroup
                        label={AVAILABLE_MODELS[key].description}
                        key={key}
                      >
                        <option key={key} value={key}>
                          {AVAILABLE_MODELS[key].name}
                        </option>
                      </optgroup>
                    ))}
                  </select>
                </div>
                <div className="flex flex-col gap-2">
                  <label htmlFor="dtype">Dtype</label>
                  <select
                    className="border dark:bg-gray-700 px-4 py-2 rounded-lg bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-200"
                    value={dtype}
                    onChange={(e) => setDtype(e.target.value)}
                  >
                    {AVAILABLE_MODELS[modelKey].dtypes.map((dtype) => (
                      <option key={dtype} value={dtype}>
                        {dtype}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <button
                className="border px-4 py-2 rounded-lg bg-blue-700 text-white hover:bg-blue-900 disabled:bg-blue-100 disabled:cursor-not-allowed select-none"
                onClick={() => {
                  worker.current.postMessage({
                    type: "load",
                    data: {
                      dtype,
                      modelKind: modelKey,
                      modelName: AVAILABLE_MODELS[modelKey].name,
                    },
                  });
                  setStatus("loading");
                }}
                disabled={status !== null || error !== null}
              >
                Click to Start
              </button>
            </div>
          </div>
        </div>
      )}
      {status === "loading" && (
        <>
          <div className="w-full max-w-[500px] text-left mx-auto p-4 bottom-0 mt-auto">
            <p className="text-center mb-1">{loadingMessage}</p>
            {progressItems.map(({ file, progress, total }, i) => (
              <Progress
                key={i}
                text={file}
                percentage={progress}
                total={total}
              />
            ))}
          </div>
        </>
      )}

      {status === "ready" && (
        <div
          ref={chatContainerRef}
          className="overflow-y-auto scrollbar-thin w-full flex flex-col items-center h-full"
        >
          <Chat messages={messages} />
          {messages.length === 0 && (
            <div>
              {EXAMPLES.map((msg, i) => (
                <div
                  key={i}
                  className="m-1 border dark:border-gray-600 rounded-md p-2 bg-gray-100 dark:bg-gray-700 cursor-pointer"
                  onClick={() => onEnter(msg)}
                >
                  <div className="text-xs text-gray-500 dark:text-gray-300">
                    Long Context Example {i + 1}:
                  </div>
                  <span className="font-bold">
                    {msg.length.toLocaleString()}
                  </span>{" "}
                  characters
                  <div className="text-xs opacity-60">
                    {msg.slice(0, 50) + "..."}
                  </div>
                </div>
              ))}
            </div>
          )}
          <p className="text-center text-sm min-h-6 text-gray-500 dark:text-gray-300">
            {tps && messages.length > 0 && (
              <>
                {!isRunning && (
                  <span>
                    Generated {numTokens} tokens in{" "}
                    {(numTokens / tps).toFixed(2)} seconds&nbsp;&#40;
                  </span>
                )}
                {
                  <>
                    <span className="font-medium text-center mr-1 text-black dark:text-white">
                      {tps.toFixed(2)}
                    </span>
                    <span className="text-gray-500 dark:text-gray-300">
                      tokens/second
                    </span>
                  </>
                }
                {!isRunning && (
                  <>
                    <span className="mr-1">&#41;.</span>
                    <span
                      className="underline cursor-pointer"
                      onClick={() => {
                        worker.current.postMessage({ type: "reset" });
                        setMessages([]);
                      }}
                    >
                      Reset
                    </span>
                  </>
                )}
              </>
            )}
          </p>
        </div>
      )}

      <div className="flex flex-col items-center mb-4">
        <div className="mt-2 dark:bg-gray-700 rounded-lg w-[600px] max-w-[80%] max-h-[200px] mx-auto relative mb-3 flex">
          {status === "ready" && (
            <textarea
              ref={textareaRef}
              className="scrollbar-thin w-[550px] dark:bg-gray-700 px-3 py-1 rounded-lg bg-transparent border-none outline-none text-gray-800 disabled:text-gray-400 dark:text-gray-200 placeholder-gray-500 dark:placeholder-gray-400 disabled:placeholder-gray-200 resize-none disabled:cursor-not-allowed"
              placeholder={
                isRunning
                  ? "running..."
                  : messages.length > 0
                  ? "change compression rate, or try new prompt"
                  : "enter the prompt"
              }
              type="text"
              rows={1}
              value={input}
              disabled={status !== "ready"}
              onKeyDown={(e) => {
                if (
                  input.length > 0 &&
                  !isRunning &&
                  e.key === "Enter" &&
                  !e.shiftKey
                ) {
                  e.preventDefault(); // Prevent default behavior of Enter key
                  onEnter(input);
                }
              }}
              onInput={(e) => setInput(e.target.value)}
            />
          )}
          {status !== "ready" ? (
            <></>
          ) : isRunning ? (
            <div className="cursor-pointer" onClick={onInterrupt}>
              <StopIcon className="h-5 w-5 p-1 rounded-md text-gray-800 dark:text-gray-100 absolute right-2 bottom-2" />
            </div>
          ) : input.length > 0 ? (
            <div className="cursor-pointer" onClick={() => onEnter(input)}>
              <ArrowRightIcon
                className={`h-5 w-5 p-1 bg-gray-800 dark:bg-gray-100 text-white dark:text-black rounded-md absolute right-2 bottom-2`}
              />
            </div>
          ) : messages.length > 0 ? (
            <div
              className="cursor-pointer"
              onClick={() => onEnter(messages[messages.length - 2].content)}
            >
              <RetryIcon
                className={`h-5 w-5 p-1 bg-gray-800 dark:bg-gray-100 text-white dark:text-black rounded-md absolute right-2 bottom-2`}
              />
            </div>
          ) : (
            <div>
              <ArrowRightIcon
                className={`h-5 w-5 p-1 bg-gray-200 dark:bg-gray-600 text-gray-50 dark:text-gray-800 rounded-md absolute right-2 bottom-2`}
              />
            </div>
          )}
        </div>
        <div className="flex flex-col items-center">
          <div className="flex items-center gap-2">
            <label htmlFor="compressionRate">Compression Rate</label>
            <input
              type="range"
              name="compressionRate"
              id="compressionRate"
              min="0"
              max="100"
              value={compressionRate}
              onChange={(e) => setCompressionRate(e.target.value)}
              disabled={status !== "ready"}
            />
            <span className="text-sm text-gray-500 dark:text-gray-300">
              {compressionRate}%
            </span>
          </div>
          <p className="text-xs text-gray-400 text-center mb-3">
            Warning. This is a <b>Experimental Technology Demo</b>. <br />
            Open the <b>browser console</b> for live logs.{" "}
            {deviceMode && (
              <>
                Engine: <span>{deviceMode}</span>
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;
