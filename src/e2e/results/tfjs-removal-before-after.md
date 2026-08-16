# TFJS removal: E2E before and after

This comparison uses [`EXAMPLES[5]`](../long-texts.ts) with:

- Model: `atjsh/llmlingua-2-js-tinybert-meetingbank`
- Revision: `a9af82841d3f815c9c492b13791c6517154791d3`
- Runtime: Node CPU, fp32
- Compression rate: `0.5`
- Token counter: `o200k_base`

| Implementation | Length | SHA-256 |
| --- | ---: | --- |
| Before: `@tensorflow/tfjs@4.22.0`, commit `82764ae` | 1431 | `b70a2a334efd2b2887db255c37865705c941d6028704335a2d5a44e325a56008` |
| After: Transformers.js `softmax` | 1431 | `b70a2a334efd2b2887db255c37865705c941d6028704335a2d5a44e325a56008` |

The outputs are byte-for-byte identical.

## Before

```text
madam court, read docket 1239? docket 1239. committee on government operations, december 1st, 2021, docket 1239 order authorizing sheltered market program conformity requirements general laws. chapter 30 b section 18. authorization contracts for goods professional services support services. authorization no six contracts, awarded june 30th, 2022. sheltered market program disadvantaged, minority women vendors, disparity city ' s 2020 disparities. study submits report recommending pass., clerk. recognizes edwards, committee. edwards. floor.,, sponsored. cannes. conformance recommendations disparity study opt pilot program mass general laws 30 section 18., ' s recommendations issue, disparity minority contractors women contractors contracts city boston. shepherd move six contracts groups disadvantage. ' s. promise. city government, financial benefits, accessible city boston. recommend pass vote.. edward acceptance committee report passage docket 1239 court, call roll? docket 1239 arroyo. arroyo. baker baker barker braden braden campbell campbell. edwards. sabby george sabby george councilor flaherty councilor flaherty councilor flynn councilor flynn yes councilor jane yes councilor janey councilor me councilor me councilor murphy
councilor murphy. yes. o ' malley.. madam president, 1239 unanimous vote. dockett 1239 passed move action. clerk, read docket 0863. docket 0863 discuss pest control illegal city boston
```

## After

```text
madam court, read docket 1239? docket 1239. committee on government operations, december 1st, 2021, docket 1239 order authorizing sheltered market program conformity requirements general laws. chapter 30 b section 18. authorization contracts for goods professional services support services. authorization no six contracts, awarded june 30th, 2022. sheltered market program disadvantaged, minority women vendors, disparity city ' s 2020 disparities. study submits report recommending pass., clerk. recognizes edwards, committee. edwards. floor.,, sponsored. cannes. conformance recommendations disparity study opt pilot program mass general laws 30 section 18., ' s recommendations issue, disparity minority contractors women contractors contracts city boston. shepherd move six contracts groups disadvantage. ' s. promise. city government, financial benefits, accessible city boston. recommend pass vote.. edward acceptance committee report passage docket 1239 court, call roll? docket 1239 arroyo. arroyo. baker baker barker braden braden campbell campbell. edwards. sabby george sabby george councilor flaherty councilor flaherty councilor flynn councilor flynn yes councilor jane yes councilor janey councilor me councilor me councilor murphy
councilor murphy. yes. o ' malley.. madam president, 1239 unanimous vote. dockett 1239 passed move action. clerk, read docket 0863. docket 0863 discuss pest control illegal city boston
```

## Performance

Measurements were taken on an Apple M4 Mac mini (10 cores, 24 GB RAM),
Node.js 22.16.0, and Chromium 151. The postprocessing benchmark uses
deterministic logits shaped `[50, 512, 2]` and measures only the code changed
by this refactor; model loading and inference are excluded.

### CPU postprocessing

Five fresh Node processes ran 20 warmups and 31 measured calls each (155
samples per implementation).

| Metric | Before: TFJS CPU | After: HF softmax | Difference |
| --- | ---: | ---: | ---: |
| Wall time, median | 14.250 ms | 4.397 ms | 3.24x faster |
| Wall time, p95 | 17.791 ms | 5.276 ms | 3.37x faster |
| Process CPU time, median | 14.364 ms | 4.280 ms | 3.36x less CPU |

### Browser GPU postprocessing

Headed Chromium used WebGL2 through ANGLE Metal on the Apple M4. Both pages
loaded Transformers.js; the before page additionally loaded TFJS. Seven
warmups and 31 measured calls were used.

| Metric | Before: TFJS WebGL2 | After: HF softmax | Difference |
| --- | ---: | ---: | ---: |
| Synchronized wall time, median | 25.795 ms | 3.970 ms | 6.50x faster |
| Synchronized wall time, p95 | 32.525 ms | 4.620 ms | 7.04x faster |
| Browser task time per call | 26.668 ms | 4.079 ms | 6.54x lower |

The after path is browser CPU JavaScript, not a replacement GPU kernel. It
avoids uploading host logits to TFJS WebGL and synchronously downloading 50
slices. Transformers.js model-device selection remains unchanged. Default
headless Chromium used software SwiftShader rather than the physical GPU; in
that environment the median was 8.110 ms before and 4.055 ms after (2.00x).

### RAM and retained GPU memory

| Measurement | Before: TFJS | After: HF softmax |
| --- | ---: | ---: |
| Node ready-to-score RSS | 133.766 MiB | 88.703 MiB |
| Additional Node RSS from importing TFJS | 19.641 MiB | 0 MiB |
| Browser JS heap after library load and GC | 5.103 MiB | 3.353 MiB |
| Retained per Node call | 52 tensors / 500 KiB | None after GC |
| Retained per WebGL call | 52 tensors / 500 KiB logical / 600 KiB GPU | None |

The old production path did not dispose its TFJS tensors. After 100 additional
Node calls it retained 5,200 tensors and 48.828 MiB of TFJS-tracked memory.
After 38 browser calls it retained 1,976 tensors, 18.55 MiB logical tensor
memory, and 22.27 MiB of tracked GPU memory. The new path creates no TFJS
registry objects or GPU allocations and its live heap returned to baseline
after garbage collection.

### Whole E2E compression

With the model and tokenizer already loaded, two warmups and seven measured
compressions of `EXAMPLES[5]` produced:

| Implementation | Median | p95 |
| --- | ---: | ---: |
| Before: TFJS | 75.799 ms | 86.604 ms |
| After: HF softmax | 75.669 ms | 77.214 ms |

The median changed by -0.130 ms (-0.17%), which is effectively unchanged.
Model inference consumed roughly 91–92% of the end-to-end time. With only
seven samples, the reported p95 is the maximum and should be treated as noisy.
