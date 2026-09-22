# Tiled attention and matrix-shape selection

Measured 2026-09-22 on the local RTX 5080 / Edge with the locally supplied NR 310.8.0.0 model and the same deterministic gradient and controls as the [previous tiled-GEMM report](performance-tiled.md). Baseline commit: `add00e9`. No model data or images are distributed.

## Normal renderer timing

Profiling and intermediate capture are disabled for these measurements. Each resolution has one initial render and two subsequent renders on the same model session. Times include frame uploads, allocations, compute, readback, and cleanup; DLL parsing, shader initialization, hashes, and UI conversion are excluded.

| Resolution | Previous repeat runs | New initial run | New repeat runs |
| --- | ---: | ---: | ---: |
| 1280 × 720 | 1215.1, 1206.4 ms | 1493.0 ms | 1007.4, 1011.4 ms |
| 1920 × 1080 | 2701.5, 2710.5 ms | 2785.4 ms | 2345.4, 2341.3 ms |

Average repeat-frame time decreases by **16.6% at 720p** and **13.4% at 1080p** relative to the preceding report. These are local measurements, not universal device guarantees. Resolution, conditioning, packed weight values, and arithmetic publication points are unchanged.

## GPU timestamps and tile exploration

Diagnostic passes change submission overhead, so their wall times are not used above. On the second 720p diagnostic run:

| Operation | Previous implementation | Final implementation |
| --- | ---: | ---: |
| Attention scores | 143.21 ms | 68.46 ms |
| Attention weighted output | 96.62 ms | 30.36 ms |
| All GEMM operations | 476.79 ms | 435.45 ms |

Attention GPU time drops by about **59%**. GEMM GPU time improves a further **9%** over the previous fixed tile. Normalization, softmax, preprocessing, and composition are unchanged.

Before choosing the default, all three matrix tile variants were run against the real model and checked against the same output hashes. The second 720p profiled runs measured 475.41 ms for fixed 4 × 16, 436.32 ms for 8 × 8, and 438.00 ms for 8 × 16 across FP8 GEMMs (excluding the small scalar FP16 projections). Shape-specific differences were larger: for K=128/N=384 the totals were 25.39, 23.30, and 18.14 ms respectively; for K=64/N=192 they were 14.52, 13.12, and 16.89 ms.

The default is a compact shape heuristic derived from these measurements: 8 × 16 for most contraction shapes and the K=128/N=384 projection, 8 × 8 for the other large shapes, and 4 × 16 for fewer than eight rows. The 512-column contraction exception retains 8 × 8. Explicit fixed-tile and scalar modes remain available. This is not startup autotuning and should be remeasured before making claims about other GPUs.

## Implementation

- `nr_scores_tiled` shares four 32-channel queries and sixteen 32-channel keys per workgroup. It computes a 4 × 16 score tile with the original two F13 groups, prior values, and exponential publication.
- `nr_attend_tiled` shares a 4 × 16 attention-weight tile and sixteen 16-channel value vectors. It preserves key iteration order, grouped accumulation, global inverse scaling, and FP8 publication.
- Local tiles are indexed within shifted 8 × 8 windows. Global tiles use padded keys. Invalid query/key positions and surplus 2D dispatch groups retain the reference behavior; every workgroup lane reaches required barriers.
- GEMM variants are generated from one authored CUDA template, `kernels/gemm-tiled.cu`. They share the packed weight decoder in `kernels/packed.cuh`. No hand-written WGSL or reduced-precision approximation is introduced.
- Cache budgets and model uploads are unchanged: 144.51 MiB resident model, a maximum 256 MiB working-buffer cache, and zero model-weight uploads on repeat renders.

## Verification and reproduction

All full-resolution benchmark runs match the previous float32 head and final-output hashes. A separate real-model run at 128 × 96 compares the old fixed GEMM/scalar-attention path with the new defaults: **all 75 intermediate capture-point hashes match**, as do both final arrays.

Verification passed **27 host tests, 83 GPU checks, 24 native CUDA fixture cases**, and the real-model image/GLB/glTF UI checks. GPU coverage includes every GEMM tile variant, odd matrix tails, half mode, partitioned accumulation, 2D dispatch padding, all four local attention phases, odd image dimensions, padded global keys, native attention fixtures, and output-tail sentinels. Existing cache, cancellation, resizing, and cleanup tests remain enabled.

`npm run benchmark` accepts:

- `NR_PROFILE=1` for GPU timestamps; leave it unset for normal timing.
- `NR_GEMM=auto|tiled|tile8x8|tile8x16|scalar` (default auto).
- `NR_ATTENTION=tiled|scalar` (default tiled).
- `NR_CAPTURE=1` to hash every intermediate capture point. This adds readbacks and hashing, so it is a correctness mode, not normal frame timing.
- `NR_COMPARE=<report.json>` to require the previous output hashes; if the baseline contains intermediate hashes, those must match as well.

Existing DLL, browser, dimensions, run-count, cache-budget, and output-path options still apply. Raw JSON reports remain ignored and local. API equivalents are the `gemmMode` and `attentionMode` renderer creation options; the existing `capture` callback supplies intermediate arrays.
