# Packed activations and prepared execution

Measured 22 September 2026 with Edge 153.0.4234.48, RTX 5080, and a locally supplied compatible model. Three runs per resolution: one cold render followed by two warm renders. Timing covers `renderer.run`, including preprocessing, network execution, head readback/decoding, composition, and output readback. It excludes DLL import, pipeline creation, UI conversion, and PNG export. GPU profiling was disabled. These are local measurements, not guarantees for other browsers or GPUs.

| Resolution | Previous warm mean | New warm runs | New warm mean | Reduction in time | New first render |
|---|---:|---:|---:|---:|---:|
| 1280 × 720 | 1009.4 ms | 741.9 / 736.7 ms | 739.3 ms | 26.8% | 1234.0 ms |
| 1920 × 1080 | 2343.4 ms | 1620.8 / 1605.7 ms | 1613.3 ms | 31.2% | 2146.5 ms |

The previous measurements are documented in [the attention report](performance-attention.md). Rendering remains a still-frame workflow, not realtime.

## Changes

- Graph publications already rounded to FP8 or half now remain packed between kernels. The arithmetic, accumulation order, and rounding points are unchanged. Input features and public results remain float32.
- CUDA storage variants are generated from the authored CUDA bodies and compiled normally by WebCuda. Atomic packed stores preserve neighboring byte/half lanes. There is no WGSL source patching.
- GEMM and merge raw outputs that have no consumers skip the store and use a four-byte placeholder instead of allocating a full tensor.
- A resolution-specific execution plan assigns buffers using non-overlapping inclusive tensor lifetimes. Inputs and outputs in the same dispatch cannot alias. The plan retains graph bind groups and invalidates them when its resources change.
- Readback expands packed values through lookup tables. Packing alone was approximately break-even for speed; avoiding repeated allocations/bindings and inefficient CPU decoding made the combined path faster.

| Memory / repeated work | 720p | 1080p |
|---|---:|---:|
| Previous largest individual activation buffer | 504 MiB | 1080 MiB |
| New largest individual activation buffer | 189 MiB | 405 MiB |
| Retained prepared activation slots, including zero buffer | 445.95 MiB | 956.17 MiB |
| Retained frame buffer pool in this benchmark | 28.13 MiB | 63.28 MiB |
| Separate model cache | 144.51 MiB | 144.51 MiB |
| Working buffer allocations, warm render | 0 | 0 |
| Bind groups created, warm render | 2 | 2 |
| Data uploads, warm render | 14,745,600 bytes | 33,177,600 bytes |

Warm uploads contain only the changing RGBA proxy in this benchmark. Both paths still execute 653 graph dispatches. The new plan retains all 653 graph bindings; the two fresh bindings are preprocessing and composition. Batches remain bounded to eight graph operations for cancellation responsiveness.

The prepared-plan budget defaults to 1 GiB. Above the budget, execution falls back to streaming with the same packed storage. The separate frame/streaming pool defaults to 256 MiB. These are distinct budgets, not a total VRAM limit; model buffers, readback staging, and browser resources are additional. `clearWorkspace()` releases both plan and pool. Clearing weights invalidates bindings as well.

## Correctness and validation

- All 75 intermediate checkpoints matched the previous path bit for bit at 128 × 96, on both initial and reused plans.
- Full float32 network-head and composed-output hashes matched the previous version at 720p and 1080p on every measured run.
- Local real-model tests compared the prepared path with expanded float storage while changing proxy pixels, seed, conditioning, history, motion vectors, and precomputed features. All head/output words matched, and distinct inputs produced distinct outputs.
- 30 host tests passed, including inclusive lifetimes, same-dispatch alias exclusion, dead-output detection, allocation-failure cleanup, budgets, and updated resolution checks.
- 92 GPU checks passed, including native CUDA fixtures, exhaustive packed publication checks across half values, signed zeros, NaNs, dirty packed words, odd tails, plan reuse, resizing, cancellation, profiling, cache release, and budget fallback.
- Browser UI tests passed with a local DLL, real image and GLB/glTF inference, exact selected output dimensions, cancellation, export, and no model uploads.

The compact storage path is compared against this project's reconstructed numerical implementation. These tests do not establish original-driver capture parity.

## Reproduce locally

Build with `npm run build`. Set `NR_DLL` to your own compatible local DLL, and `NR_BROWSER` if Playwright needs an installed Edge/Chrome executable. Run `npm run benchmark`; set `NR_WIDTH=1920` and `NR_HEIGHT=1080` for 1080p. Use `NR_ACTIVATIONS=float` and `NR_EXECUTION=streamed` for the expanded reference path. `NR_REPORT` selects an ignored local JSON report, `NR_COMPARE` checks hashes against a saved report, and `NR_CAPTURE=1` includes intermediate hashes. Run `npm run test:prepared` with the same local DLL for changing-input comparisons.

No DLLs, model tensors, or private captures are distributed with this report.
