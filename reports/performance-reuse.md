# Vector exponent reductions and attention operand reuse

This pass follows `51e74ea`. Its saved generated artifacts are the baseline, with the same renderer configuration, model and input on both sides.

Measured 2026-09-22 on RTX 5080 / Edge 153.0.4234.48, median of six alternating warm image renders:

| Resolution | Previous path | Final path | Less render time |
| --- | ---: | ---: | ---: |
| 1280 x 720 | 228.20 ms | 182.95 ms | 19.8% |
| 1920 x 1080 | 501.40 ms | 406.40 ms | 18.9% |

Every timed output SHA-256 matched. These image-render timings include preprocessing, network, composition and output readback, excluding setup and hashing. They are separate from the common-feature network-only upstream comparison.

## Fresh upstream comparison

Identical precomputed feature upload, complete network and padded float32 head readback. Six alternating warm samples, followed by three separate GPU profiles. No preprocessing, composition or UI in this scope.

| Resolution | Ours | Upstream | Ours / upstream |
| --- | ---: | ---: | ---: |
| 512x512 | 71.30 ms | 50.20 ms | 1.42x |
| 1280x720 | 201.50 ms | 148.75 ms | 1.35x |
| 1920x1080 | 446.35 ms | 359.90 ms | 1.24x |

All cold, warm and profiled heads matched exactly at all three resolutions.

| Targeted GPU family | Previous 720p profile | Final 720p profile | Previous 1080p profile | Final 1080p profile |
| --- | ---: | ---: | ---: | ---: |
| FP8 GEMM | 149.11 ms | 115.77 ms | 324.37 ms | 253.99 ms |
| Local normalization + attention | 55.01 ms | 42.77 ms | 116.81 ms | 92.58 ms |

Family timings are medians of three per-family GPU sums. Previous values are from the immediately preceding audit; the final ours/upstream wall measurements were paired in this run. These profiles verify that both targeted families improved.

## Implementation

- FP8 GEMM reduces the eight output exponents in two float vectors. Shared half exponents are exact small integers, so per-term float-to-integer conversions are unnecessary. Integer conversion happens after the reduction when constructing power-of-two scales.
- Local attention keeps its score accumulators small, using float exponents and exact float F13 sums to eliminate per-term integer conversions. Power-of-two scales are constructed once per group.
- The value stage computes two adjacent output channels per lane, keeping all 512 lanes active. Each probability and exponent serves both channels, with sums and exponent reductions carried in float2 vectors. The final owner still packs four channels; redundant quantization before this packing is removed.
- `vector-f13.cuh` shares the exact F13 accumulation and exponent reduction helpers. Truncated products and partial sums remain exactly representable float32 integers. Sixteen-term group order, normalization, softmax reduction and half rounding are preserved.
- WebCuda lowers component-wise CUDA `fmaxf`/`fminf` helpers to WGSL vector extrema. CUDA remains the authored numerical source; generated WGSL is not patched.

The workgroup size and 31,680-byte local-attention shared allocation remain unchanged, as do the execution-plan memory budget, model caches and smaller-device fallback. Global attention and FP16/F24 endpoints are unchanged.

## Verification

All 326 GPU checks and 39 NR host tests passed. All 75 captured intermediate boundaries at 128 x 96 matched the saved reference, together with the head and final output hashes. The WebCuda suite passed 778 tests, including vector extrema lane ordering. The compiler change is upstream as `db1e3d2` and synchronized into this repository.

## Reproduction

Save the previous generated artifacts to an ignored `reports/*-generated/` directory before rebuilding. Set local `NR_DLL`, `NR_BROWSER`, `NR_ENHANCED=1`, `NR_PREVIOUS_GENERATED=reports/pre-reuse-generated` and run `npm run benchmark:execution`. Six alternating warm samples per resolution include preprocessing, network, composition and output readback, excluding setup and hashing.
