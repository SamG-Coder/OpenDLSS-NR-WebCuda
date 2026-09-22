# Exact GEMM specialization

## Decision and measurement

Enable `specializeGemm: true` by default for supported packed tiled GEMMs. Specialization improved full-frame performance at both target resolutions without changing the checked outputs. Tile geometry, group accumulation, residual initialization, partition boundaries, activation, and publication arithmetic remain unchanged.

Six warm samples per configuration were alternated after one warmup per renderer on RTX 5080 / Edge 153. Both paths retain the preceding noise cache and four-batch execution and omit head download. Only specialization differs.

| Source resolution | Generic median | Specialized median | Less engine time |
| --- | ---: | ---: | ---: |
| 1280 x 720 | 557.70 ms | 436.80 ms | 21.7% |
| 1920 x 1080 | 1323.20 ms | 1014.85 ms | 23.3% |

720p generic samples: 558.6, 559.5, 556.8, 556.3, 558.3, 557.1 ms. Specialized: 436.5, 438.3, 439.3, 437.1, 436.2, 435.0 ms.

1080p generic samples: 1321.8, 1430.2, 1322.8, 1323.6, 1322.4, 1324.7 ms. Specialized: 1010.9, 1014.5, 1015.8, 1013.2, 1015.2, 1019.3 ms.

This is engine timing with profiling disabled and a deterministic gradient. It excludes DLL import, pipeline setup, source capture, display conversion, PNG export, and output hashing. Two renderer instances remain resident during the paired measurements. These figures do not establish performance on other GPUs or original-driver image parity.

## Implementation

The build derives 41 distinct configurations from the real packed graph. The existing 8 x 8 and 8 x 16 compact CUDA kernels are specialized before WebCuda compilation: K/N, batches, strides, partition, half mode, residual and activation flags, storage formats, and raw-output enable become literals. Rows remain a runtime parameter so different image sizes reuse these kernels.

The renderer matches the complete specialization key, removes fixed parameters from the runtime scalar binding, and retains the original dispatch geometry. It falls back to the original kernel when no matching specialization exists. FP16 projections, float activation storage, scalar kernels, and experimental multi-output kernels retain their existing paths. The graph cache includes the specialization option, preserving prepared-binding correctness.

This makes constants visible to the downstream shader compiler. Reduced dynamic branching, indexing, and conversion-path overhead are plausible explanations for the gain; no hardware instruction-counter measurement was made, so the experiment does not isolate those contributions.

All sources remain CUDA. No generated WGSL is patched, and no WebCuda compiler/runtime modification or upstream sync was necessary.

## Costs and fallback

The feature adds 41 generated artifacts and GPU pipelines when enabled with a compatible mode. Setup cost depends heavily on browser/driver compilation caches. In the paired session, initial generic/specialized setup took approximately 5.91/5.10 seconds; a subsequent pair took 154/214 ms. Their order and shared driver caches make these unsuitable as independent cold-start comparisons. This change claims warm inference improvement, not faster startup.

Set `specializeGemm: false` to omit specialized pipeline loading and use the previous kernels. Incompatible storage and GEMM modes also skip their loading. No new weight cache or activation allocation is introduced.

## Diagnostics

The synthetic GEMM benchmark now includes specialization alongside the established tiles. It passed exact output and live-raw checks and measured a lower median than the shape-selected generic tile in all 36 representative cases at the default 4096-row cap. Those cases are not weighted by their full-model execution frequencies.

`NR_ABLATE=1` additionally compiles an ordinary-FMA diagnostic from CUDA in the benchmark only. It replaces grouped exponent scanning/truncation with FMA and retains half publication at group boundaries. This is numerically different, is explicitly excluded from parity checks, and has no renderer mode or shipped inference artifact. It was faster in aggregate in the synthetic test, supporting further investigation of arithmetic overhead without establishing a usable quality-preserving replacement. Timings cannot be interpreted as additive percentages of shader cost: changing arithmetic may also change compiler scheduling and resource use.

## Validation and reproduction

- 35 host tests cover specialization identity, dynamic rows, CUDA transformation boundaries, fallback eligibility, option validation, and pipeline-loading opt-out.
- 239 GPU checks pass across 90 generated pipelines. All 41 specializations match scalar packed GEMM, including partial row tiles, surplus dispatch groups, residual formats, raw-output enable, partitions, and publication modes.
- Real-model reference versus specialized execution matches bit for bit across changed image/seed/controls/history/motion/features, output-only execution, cancellation/reuse, and resize.
- All 75 intermediate checkpoint hashes match the saved reference at 128 x 96 on initial and reused execution, together with head/output hashes.
- Real image/3D UI inference, cancellation, export, and frame timing passed.
- Full-frame output hashes match the prior canonical values at 720p and 1080p.

Set `NR_DLL` and `NR_BROWSER` to local files. Use `NR_SPECIALIZE=1` with `npm run benchmark:execution` to compare identical execution settings with specialization off/on. Set `NR_REPORT` to an ignored local output path. The regular `npm run benchmark` defaults to specialization enabled; `NR_SPECIALIZE=0` disables it. `npm run benchmark:gemm` checks and measures the isolated kernels without a DLL; `NR_ABLATE=1` adds the non-equivalent diagnostic.

No DLL, model weights, native captures, or example media are distributed.
