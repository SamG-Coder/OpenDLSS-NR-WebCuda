# Wider CUDA execution and cached numerics

This report describes revision `b8db727`. See [the subsequent exponent/vector optimization](performance-fragments.md) for the current kernels and shared-memory requirement.

Measured 2026-09-22 on RTX 5080 / Edge 153.0.4234.48. Both configurations use the same local model, native-half support, CUDA __clz, packed activations, noise cache, and four batches in flight. The baseline disables wideGemm and normalizeAttention; the candidate enables both. Six alternating warm samples per resolution; image preprocessing, network, composition and output readback are timed. Setup and hashing are outside timing.

| Resolution | Previous path | Complete new path | Less time |
| --- | ---: | ---: | ---: |
| 1280 x 720 | 372.20 ms | 300.50 ms | 19.3% |
| 1920 x 1080 | 877.05 ms | 666.85 ms | 24.0% |

Every timed output SHA-256 matched. These image-render timings are distinct from the common-feature network-only upstream comparison; do not combine the two scopes into a new upstream speed ratio.

Execution-plan memory falls from 431.46 to 369.49 MiB at 720p and from 925.00 to 791.68 MiB at 1080p. These figures exclude model weights and the additional 258 KiB numerical lookup cache.

Final validation passed 326 GPU checks across all 174 compiled pipelines, with all 75 captured intermediate boundaries matching the saved reference. The NR host suite passed 39 tests and the WebCuda host suite passed 771 tests. Every output in the alternating timed comparison matched exactly.

## Final implementation

- `gemm-wide.cu`: 32x32 output tiles, eight outputs per invocation (two rows by four columns), 128 threads. A K slab of 32 operands serves two ordered 16-term F13 groups. Paired half operands and exponents are shared; weights use a padded/transposed shared layout. The shared allocation is 8,448 bytes.
- Each invocation owns every packed output word it writes, including raw half pairs. Ordinary u32 stores replace compare/exchange retries. The emitted wide kernel has non-atomic output bindings and workgroup-only barriers.
- `lookup.cu`: one GPU pass builds 256 value/exponent metadata entries and the exact 65,536-entry half-input SiLU table. The tables are model-independent and retained across renders, avoiding repeated weight/activation decoding and repeated SiLU arithmetic. They occupy 258 KiB, separately from model weights and the execution-plan budget.
- `fast-half.cuh`: native half conversion publishes exactly representable F13 float sums only when the result lies in the finite normal-half range. Zero, subnormal and overflow cases retain the existing software conversion. FP16/F24 endpoint GEMMs keep their original arithmetic.
- `attention-normalized.cu`: 32 query rows on 512 threads, exact normalization from raw half QKV, and owned packed output stores. It removes 62 normalization passes and tensors; the graph has 467 operations instead of 529. Shared storage is 21,248 bytes. Global attention is unchanged.

All numerical implementation is authored as CUDA and compiled by WebCuda. There is no production WGSL patching or incorporated upstream browser shader source.

## Runtime selection and caching

`wideGemm: true` and `normalizeAttention: true` are the defaults. Wide GEMM additionally requires shader-f16, a matching specialized packed-FP8 shape, N divisible by four, packed output formats, and the existing one-time matrix weight bound check. Other configurations use the previous GEMM path.

Fused normalization requires packed activations, fused local attention mode, shader-f16, at least 21,248 shared bytes, and a 512-thread X workgroup. Unsupported devices and modes retain separate normalization and the previous attention path. WebCuda commit `7cc771c` adds the explicit `useAdapterWorkgroupLimits` option; its default remains unchanged for other callers. The renderer opts in when fused normalization is requested. Supplied devices are checked without assuming their limits can be raised.

The numerical lookup buffers are initialized lazily before the first wide GEMM. They are reused across ordinary image and seed changes. Explicit weight-cache/workspace clearing, workspace resets, and disposal release them; a later wide GEMM rebuilds them. Their first build is excluded from warm timing. Existing plan invalidation includes both new mode flags.

## Reproduction

With local NR_DLL and NR_BROWSER set, build with `npm run build`. Use `NR_ENHANCED=1` with `npm run benchmark:execution` to compare the previous path with both optimizations. NR_WIDE=1 isolates wide GEMM against the previous GEMM; NR_NORMALIZE=1 compares fused normalization on top of wide GEMM. `npm run benchmark` uses current defaults; set NR_WIDE=0 and/or NR_NORMALIZE=0 to disable the corresponding feature.

The earlier four-output and eight-output-only timings were development experiments. The table above measures the complete final kernel, cached metadata/SiLU and guarded half publication together. These improvements do not establish a twofold speedup or eliminate the remaining gap to upstream.
