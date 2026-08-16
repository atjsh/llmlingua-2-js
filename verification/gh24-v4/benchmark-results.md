# GH-24 v4 benchmark results

Non-gating measurements captured on 2026-08-16. Lower is better. Each value is
the median result from five fresh processes.

| Case | Implementation | Wall median / p95 (ms) | CPU median / p95 (ms) | Post-GC RSS / process max (MiB) |
| --- | --- | ---: | ---: | ---: |
| Mock `[50,512,2]` | published 2.0.5 | 17.018 / 18.891 | 17.105 / 19.239 | 159.734 / 171.063 |
| Mock `[50,512,2]` | 3.0.0 candidate | 16.845 / 22.492 | 16.814 / 36.448 | 167.578 / 182.313 |
| TinyBERT CPU/fp32 | published 2.0.5 | 3.583 / 5.456 | 18.519 / 33.792 | 281.656 / 281.656 |
| TinyBERT CPU/fp32 | 3.0.0 candidate | 3.523 / 5.698 | 18.278 / 32.742 | 269.531 / 269.531 |

The candidate's median wall/CPU changes were -1.0%/-1.7% for saturated mock
post-processing and -1.7%/-1.3% for warm TinyBERT compression: effectively no
regression. Mock post-GC RSS/peak RSS were 4.9%/6.6% higher, while TinyBERT
process RSS was 4.3% lower. The mock candidate's p95 was noisy and higher, so
this evidence is directional rather than a release gate.

## Method and provenance

- Host: Apple M4 Mac mini, 24 GiB RAM, macOS 26.5.2 (25F84), arm64.
- Runtime: Node 22.16.0 and Transformers.js 4.2.0 for both implementations.
- Baseline: published `@atjsh/llmlingua-2@2.0.5`.
- Candidate base: Git `a2a086066997e4aad3bf435ac837b50e547f4a61`; candidate
  `prompt-compressor.ts` SHA-256
  `a1ad164aff569d0278dc76657a27d4189a9975dc7392140a65807f47e369222d`.
- Mock: five fresh processes, 20 warmups and 50 samples each, exact
  50-by-512-by-2 logits, with model work limited to deterministic tensor
  construction.
- Model: pinned `atjsh/llmlingua-2-js-tinybert-meetingbank` revision
  `a9af82841d3f815c9c492b13791c6517154791d3`, sorted
  `src/e2e/long-texts.ts` `EXAMPLES[1]`, rate 0.5, five fresh processes with
  5 warmups and 20 samples each. Model loading and downloads are excluded.
- Every measured output must match the committed byte length and SHA-256 in
  `benchmark-golden.json`. Output differences between 2.0.5 and 3.0.0 are the
  intentional corrected selection semantics covered by the E2E suite.
- RSS is process-level and separate-process comparison is noisy. No GPU result
  was recorded by this CPU benchmark; Chromium WASM/WebGPU qualification is
  handled by the browser verification harness.
