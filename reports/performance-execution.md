# Bounded execution and exact noise caching

## Result

The output-only browser path is faster with unchanged final float32 output on the tested RTX 5080 / Edge 153 setup. Six warm samples per configuration were interleaved with alternating order after warming each renderer once.

| Source resolution | Conservative median | UI-path median | Reduction |
| --- | ---: | ---: | ---: |
| 1280 x 720 | 604.85 ms | 558.65 ms | 7.6% |
| 1920 x 1080 | 1393.15 ms | 1322.30 ms | 5.1% |

720p conservative samples: 606.9, 660.4, 604.4, 603.6, 603.0, 605.3 ms. UI-path samples: 556.8, 560.0, 557.4, 559.9, 556.5, 561.8 ms.

1080p conservative samples: 1389.4, 1397.0, 1393.1, 1393.2, 1395.5, 1389.6 ms. UI-path samples: 1323.6, 1320.4, 1320.7, 1321.0, 1323.6, 1326.4 ms.

Both configurations use the same current build and model. Conservative options are `maxInFlightBatches: 1`, `cacheNoise: false`, and `readHead: true`. The UI path uses four batches, cached noise, and `readHead: false`. This measures the combined change; it does not attribute savings to individual optimizations. Composition now precedes optional head readback in both configurations, so the control is not an exact historical checkout.

Measurements cover `engine.run()` with profiling disabled and a deterministic gradient. They exclude model import, pipeline creation, input capture, display conversion, PNG creation, and hash verification. Two renderer instances remain resident during each paired test. Results are specific to this device/browser and do not establish original-driver image parity.

## Implementation

- The CUDA `nr_noise` kernel preserves the original noise arithmetic and stores three already-published half values in two packed words per padded pixel. `nr_preprocess_cached` reads those values and constructs all remaining feature lanes normally. The original preprocessing kernel remains available. The one-entry GPU cache is keyed by padded dimensions and seed and cleared with workspace reset. It adds 7.875 MiB at 720p or 16.875 MiB at 1080p, outside the existing plan and pool budgets.
- Prepared execution submits up to four batches before awaiting completion. Each batch still contains at most 32 graph operations by default. GPU queue ordering preserves dependencies and uniform uploads. Stable prepared-plan buffers remain alive throughout execution. Streamed execution and capture boundaries retain their completion waits. Cancellation stops future work and waits for already submitted work before reuse.
- `readHead: false` avoids the head download and CPU expansion; `head` is null. API calls default to returning the head for compatibility. Final composition reads the GPU head directly and runs before any requested head download.
- Warm 720p readback falls from 23,003,136 to 14,745,600 bytes; 1080p falls from 50,872,320 to 33,177,600 bytes. The final float32 image is still downloaded.
- Engine timing reports host phases, including explicit waits and completion/readback. The UI separately measures input preparation, engine execution, presentation, and PNG creation. These are wall-clock timings, not per-kernel GPU measurements. The existing diagnostic GPU profiler remains available.

No WebCuda compiler/runtime change was needed. The implementation uses its existing CUDA compiler, ordered queue submission, uniform snapshots, and buffer APIs. No vendored WebCuda divergence was introduced by this work.

## Validation

- 33 host tests passed, including new option validation.
- 197 GPU checks passed across 49 generated pipelines. Cached preprocessing matches all feature bits for seeds 0, 219, and 4294967295 with reflected padding, masked history, and varied controls.
- Real-model prepared/cached and streamed/original-preprocessing paths match head and output bit for bit across image, seed, conditioning, history, motion, and feature-override changes. The output-only path matches too. Cancellation/reuse, workspace clearing, and resizing are covered.
- All 75 intermediate checkpoint hashes match the saved reference at 128 x 96 on initial and reused execution. Head and output hashes match as well.
- Full-resolution final output hashes match the paired control and prior canonical values at 720p and 1080p.
- Real image/GLB/glTF UI inference, cancellation, export, auxiliary-input validation, and the new frame-timing data passed.

Run `npm run benchmark:execution` with `NR_DLL` and `NR_BROWSER` set to local files. Detailed measurements go to ignored `reports/execution-comparison.json`. The regular benchmark also accepts `NR_INFLIGHT`, `NR_NOISE_CACHE`, and `NR_READ_HEAD` for controlled experiments. No model files or assets are distributed.
