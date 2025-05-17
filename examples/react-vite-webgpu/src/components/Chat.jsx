import BotIcon from "./icons/BotIcon";
import UserIcon from "./icons/UserIcon";

import "./Chat.css";

export default function Chat({ messages }) {
  const empty = messages.length === 0;

  return (
    <div
      className={`flex-1 p-6 max-w-[960px] w-full ${
        empty ? "flex flex-col items-center justify-end" : "space-y-4"
      }`}
    >
      {empty ? (
        <>
          <div className="text-xl">Ready!</div>
          <p>Pick a example text from the list, or enter your own text.</p>
        </>
      ) : (
        messages.map((msg, i) => (
          <div key={`message-${i}`} className="flex items-start space-x-4">
            {msg.role === "assistant" ? (
              <>
                <BotIcon className="h-6 w-6 min-h-6 min-w-6 my-3 text-gray-500 dark:text-gray-300" />
                <div className="bg-gray-200 dark:bg-gray-700 rounded-lg p-4">
                  <p className="min-h-6 text-gray-800 dark:text-gray-200 overflow-wrap-anywhere">
                    {msg.content.length > 0 ? (
                      <div>
                        <div className="markdown">{msg.content}</div>
                        <br />
                        <hr />
                        <br />
                        <div>
                          <p>
                            before: {msg.inputLength.toLocaleString()}, after:{" "}
                            {msg.compressedLength.toLocaleString()}
                          </p>
                          <p>
                            🗑️{" "}
                            {(
                              100 -
                              (msg.compressedLength / msg.inputLength) * 100
                            ).toFixed(2)}
                            % characters removed
                          </p>
                        </div>
                      </div>
                    ) : (
                      <span className="h-6 flex items-center gap-1">
                        <span className="w-2.5 h-2.5 bg-gray-600 dark:bg-gray-300 rounded-full animate-pulse"></span>
                        <span className="w-2.5 h-2.5 bg-gray-600 dark:bg-gray-300 rounded-full animate-pulse animation-delay-200"></span>
                        <span className="w-2.5 h-2.5 bg-gray-600 dark:bg-gray-300 rounded-full animate-pulse animation-delay-400"></span>
                      </span>
                    )}
                  </p>
                </div>
              </>
            ) : (
              <>
                <UserIcon className="h-6 w-6 min-h-6 min-w-6 my-3 text-gray-500 dark:text-gray-300" />
                <div className="bg-blue-500 text-white rounded-lg p-4">
                  <p className="min-h-6 overflow-wrap-anywhere">
                    {msg.content}
                  </p>
                </div>
              </>
            )}
          </div>
        ))
      )}
    </div>
  );
}
