# Shared exponent reuse and explicit CUDA loop unrolling

This pass builds on `b8db727`, which already includes eight-output GEMM, cached metadata/SiLU, guarded half publication and fused local normalization. The comparison uses saved artifacts from that revision with all of those options enabled on both sides.

Measured on 2026-09-22, RTX 5080 / Edge 153.0.4234.48:

| Resolution | Previously shipped path | Complete new path | Less render time |
| --- | ---: | ---: | ---: |
| 1280 x 720 | 306.40 ms | 228.25 ms | 25.5% |
| 1920 x 1080 | 672.70 ms | 506.85 ms | 24.7% |

These are medians of six alternating warm samples per resolution. Every timed output hash matched exactly. The intermediate version with word loading, shared exponents and unrolling but integer GEMM sums improved its paired baseline by only 6.8% / 7.0%; the final table includes the four-lane float sums as well. The execution-plan allocation and model caches are unchanged from the preceding release.

## Changes

- GEMM stages aligned four-byte FP8 input words and paired weight words once, then extracts their components into shared half pairs. Previously each byte extraction issued its own scalar word read. The specialized graph shapes have four-element-aligned strides and channel dimensions.
- WebCuda now honors bounded `#pragma unroll` loops in emitted WGSL. GEMM's eight-pair exponent and product reductions and its four-column loading loop are expanded in iteration order. Local attention's sixteen-term exponent and product reductions are also expanded. This is compiler lowering from CUDA, not an edit to generated WGSL.
- GEMM retains F13 sums in two four-lane float vectors. Each truncated term and ordered partial sum is an exactly representable integer, well below the float32 integer limit. Conversion to integer happens once per output at publication instead of on every product. WebCuda combines side-effect-free component-wise CUDA vector helpers into vector WGSL without changing the arithmetic tree.
- Local attention caches Q, K, probability and V exponents when staging the values. The exponent fields are exact small integers stored as half values. The same exponent is reused across dot products instead of repeatedly extracting float bits. Zero operands use a sentinel below the minimum accumulator exponent.
- All 512 workgroup lanes cooperate on the 96 Q/K norms. The original half-rounded square pairs and reduction tree are preserved. Temporary shared storage is reused for K/V after normalization.

The attention tile now uses 31,680 shared bytes, within a 32 KiB device limit. Smaller devices retain the separate-normalization attention path. Global attention and FP16/F24 endpoints retain their previous implementation. No tensor-core or approximate arithmetic path is introduced.

## Validation and reproduction

All 326 final GPU checks passed, including all specialized GEMM shapes, local normalization/attention shifts and zero norms. All 75 intermediate boundaries at 128 x 96 matched the saved reference, along with the exact head and final output hashes. The NR host suite passed 39 tests. WebCuda's host suite passed 777 tests, including preservation of loops with counter aliases, mutations or control-flow exits and component-wise vector arithmetic semantics. Compiler changes are upstream in WebCuda `f82d847` and synchronized into this repository.

For the paired warm image benchmark, save the previous revision's `generated` directory under an ignored `reports/*-generated/` directory before rebuilding. Set `NR_ENHANCED=1`, `NR_PREVIOUS_GENERATED=reports/pre-fragments-generated`, local `NR_DLL` and `NR_BROWSER`, then run `npm run benchmark:execution`. Six alternating samples at each resolution time preprocessing, network execution, composition and output readback; model setup and hashing are excluded. Both sides use the same model, input and renderer options. These timings must not be mixed with the earlier network-only upstream comparison.

The compiler implementation follows the role of [`#pragma unroll` in the CUDA language reference](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html), with a bounded subset documented in WebCuda. Dynamic or unsupported loop forms retain their original control flow.
