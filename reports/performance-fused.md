# Fused local attention and prepared batches

Measured 22 September 2026 using Edge 153.0.4234.48 and RTX 5080 with a locally supplied compatible model. Each normal benchmark has one first render and two repeated renders. Timing includes preprocessing, network execution, composition, and result readback, but excludes DLL import, pipeline setup, UI conversion, and PNG export. GPU profiling is disabled for frame timings.

| Resolution | Previous warm mean | New warm runs | New warm mean | Reduction in time | New first render |
|---|---:|---:|---:|---:|---:|
| 1280 × 720 | 739.3 ms | 607.0 / 604.1 ms | 605.6 ms | 18.1% | 1254.6 ms |
| 1920 × 1080 | 1613.3 ms | 1382.1 / 1492.4 ms | 1437.3 ms | 10.9% | 2074.5 ms |

The baseline is the [packed activation and prepared execution version](performance-prepared.md). Cold rendering did not consistently improve: the previous first renders measured 1234.0 / 2146.5 ms. These small local samples show repeated-render improvements, not guarantees for other hardware or browsers. The 1080p frame timings varied more than 720p; all recorded runs are included above.

## Implementation

The new `nr_local_attention` CUDA kernel combines local scores, softmax, and weighted output. Eight queries share each window tile in a 128-thread workgroup. Scores and quantized probabilities stay in shared memory. Key storage is reused for values, and shared rows are padded. Total shared storage is 11,552 bytes, below WebGPU's 16 KiB baseline limit.

The softmax's eight partial sums are parallelized across threads, followed by the original even/odd half reduction. Every half/FP8 rounding point and F13 accumulation group remains explicit. This is exact fusion of the reconstructed arithmetic, not a replacement softmax algorithm.

A lane owns four consecutive output channels and assembles one packed FP8 word. That path uses one atomic exchange per word, avoiding competing per-byte compare/exchange loops. The other kernels keep their existing packed stores. CUDA storage variants are compiled through WebCuda without WGSL patching.

The graph replaces 62 local three-pass attention operations with 62 fused operations. Eight global attention operations retain the previous tiled score/output kernels and separate softmax. This removes 186 local temporary resources and reduces graph dispatches from 653 to 529. Retained prepared activation slots fall from 445.95 to 431.46 MiB at 720p, and from 956.17 to 925.00 MiB at 1080p. The largest individual buffer remains 189 / 405 MiB; this change does not further reduce that limit.

Prepared execution batches up to 32 graph operations instead of eight. Normal benchmark submissions fall from 84 to 19. Cancellation remains checked between operations and after submitted batches; pending batches can be discarded, but in-flight GPU work must complete. The streamed fallback retains eight-operation and memory-pressure bounds. `graphBatchSize` can be set from 1 to 64.

## Profiling and selection

Separate GPU timestamp runs at 720p measured score/softmax/output attention at **126.69 ms before**, and fused-local plus unchanged global attention at **60.91 ms after**, about 52% less GPU time. The latter comprises 53.07 ms fused local attention and 7.84 ms global attention. Normalization is separate and excluded from both totals. GEMM remains the largest GPU cost, around 451 ms in the final diagnostic run.

The initial four-query fused kernel was slower than the previous path. Parallel softmax and an eight-query tile were necessary to make fusion worthwhile. Larger prepared batches provide an additional end-to-end gain. Timestamped runs submit individual diagnostic passes and must not be compared directly with normal wall-clock timings.

## Validation

- 31 host tests passed, including fused graph structure, retained global operations, unchanged checkpoint identities, lifetime planning, and batch-option validation.
- 140 GPU checks passed. New cases compare fused output against independent scalar score/softmax/output passes using nonzero inputs, all four window shifts, 1 × 1 and odd dimensions, surplus 2D dispatch groups, packed inputs, float/half/FP8 outputs, and sentinel tail guards. Existing native CUDA fixtures also passed.
- All 75 intermediate checkpoint hashes matched the previous reference at 128 × 96 on initial and reused plans.
- Network-head and composed-output float32 hashes matched the preceding version at 720p and 1080p on every measured run.
- Local real-model changing-input tests matched the unfused float-storage reference bit for bit when changing proxy pixels, seed, conditioning, history, motion vectors, and feature overrides.
- Browser tests passed with a local DLL, image and GLB/glTF rendering, selected output sizes, cancellation, exports, and no model uploads.

These checks validate this project's reconstructed implementation. Original-driver capture parity remains unverified.

## Reproduce

Build with `npm run build`. Set `NR_DLL` to your own local compatible DLL and `NR_BROWSER` to an installed supported browser if needed. Run `npm run benchmark`; use `NR_WIDTH=1920` and `NR_HEIGHT=1080` for 1080p. Defaults are now `NR_ATTENTION=fused` and `NR_BATCH=32`. Set `NR_ATTENTION=tiled` and `NR_BATCH=8` for the preceding path; compare modes with the same batch size to isolate fusion. `NR_PROFILE=1` collects GPU timestamps, `NR_CAPTURE=1` collects checkpoint hashes, and `NR_COMPARE` compares against an ignored local baseline JSON report. `npm run test:prepared` exercises changing inputs against unfused float storage.

No DLLs, model weights, private captures, or upstream project snapshots are distributed.
