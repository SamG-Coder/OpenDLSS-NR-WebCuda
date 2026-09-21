# Scheduling and model-decoding performance

Measured 2026-09-22 on the local RTX 5080 / Microsoft Edge WebGPU adapter with the locally supplied NR 310.8.0.0 model. Baseline: commit `0b17cac`. The input is a deterministic RGB gradient with alpha 1, seed 0, and the renderer's default conditioning. Both versions process the same full source resolution.

| 1280 × 720, three runs | Baseline | Optimized |
| --- | ---: | ---: |
| Renderer elapsed time, median | 5485.3 ms | 2283.9 ms |
| Elapsed range | 5470.7–5497.4 ms | 2275.2–2318.6 ms |
| CPU weight decoding, median | 2757.1 ms | 678.0 ms |
| Queue-completion waits per run | 654 | 178 |
| Compute submissions per run | 655 | 178 |

This is a **2.40× speedup / 58.4% reduction in elapsed time**. The benchmark measures `NeuralRenderer.run`, including allocations, uploads, preprocessing, all 653 graph dispatches, composition, readback, and cleanup. DLL parsing, shader initialization, output hashing, PNG conversion, and UI work are excluded. CPU decoding is measured around model matrix/vector/prior methods. Queue-wait durations are host-observed waits, not GPU timestamp measurements.

At **1920 × 1080**, optimized runs took 3574.8, 3556.4, and 3549.0 ms (median **3556.4 ms**). There is no paired 1080p baseline in this benchmark session. All three full-resolution outputs were finite and had identical hashes.

## Changes

- Decode FP8 and half values through small lookup tables instead of repeating exponentiation for every weight. Packed weight layouts are unchanged. The tables occupy about 257 KiB.
- Record up to eight graph operations per command batch, flushing earlier when retired resources reach 64 MiB. One operation can exceed that threshold; this is a scheduling threshold, not a total-memory cap.
- Reuse the existing 256 MiB activation pool in GPU command order. Keep other retired resources alive until their submitted work finishes, then destroy them. Decoded weights are still released; no permanent GPU copy of the expanded model is added.
- Flush at capture boundaries. Discard pending commands on cancellation or failure and release all inference-owned resources. Progress retains its per-operation callback; browser updates and cancellation can occur between batches.

## Correctness

All three optimized 720p runs matched all three baseline runs byte for byte, including signed floating-point representations:

| Full float32 array | SHA-256 |
| --- | --- |
| Padded network head | `04acbfd71bf25ee2340da48d61bbccaca7e40bd2081fc0e9682f3921dd77375a` |
| Composited RGBA output | `92c0bcf0e7f54837200fd59d2d5d3a1bebf5fee321fc40705afc4f23ef216698` |

Verification passed: 24 host tests; 44 GPU checks, including native numerical fixtures, all 75 synthetic capture boundaries, batched output, resource cleanup, and cancellation/restart; real-DLL image and GLB/glTF UI checks. These checks preserve the previous implementation's behavior, but do not establish original-driver image parity.

Reproduce with `npm run benchmark` and `NR_DLL`, optionally setting `NR_BROWSER`, `NR_WIDTH`, `NR_HEIGHT`, `NR_RUNS`, `NR_REPORT`, and `NR_COMPARE`. Raw measurement reports stay local and ignored. No DLLs, model weights, or input/output image assets are included in this report.

The remaining work is predominantly GPU execution, transfers, and allocation/cleanup, plus about 0.68 s of CPU decoding per run. The CUDA kernels still use scalar arithmetic to preserve native rounding behavior. Kernel-level timing should guide subsequent tiling or fusion changes; this result is still a seconds-per-frame renderer.
