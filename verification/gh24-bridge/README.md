# GH-24 bridge verification

Temporary, history-only checks for the Transformers.js v3/v4 bridge. Remove
this entire directory in the cleanup commit. The fast checks also freeze the
legacy terminal-token drop and XLM active-ID-0 probability shift so those
3.0-only corrections cannot leak into 2.0.5. A pinned real XLM-R tokenizer
plus mock logits also freezes its leading-space and UTF-8 output bytes without
downloading the multi-gigabyte model.

Run the fast package/import/type/mock checks against every supported runtime:

```sh
node verification/gh24-bridge/run.mjs
```

Also run the pinned TinyBERT CPU/fp32 golden under every runtime:

```sh
LLMLINGUA_RUN_MODEL_E2E=1 node verification/gh24-bridge/run.mjs
```

Pass one or more exact Transformers.js versions to narrow a diagnostic run:

```sh
node verification/gh24-bridge/run.mjs 3.5.2 4.2.0
```

The runner builds and packs the repository, installs the tarball into isolated
OS temporary directories, and deletes those directories when finished. It does
not modify the root manifest or lockfile.
