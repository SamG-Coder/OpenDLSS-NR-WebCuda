# Prepared FP8 model and integer GEMM

The model can now be prepared for the generated browser shaders while retaining its original FP8 codes. This adds two opt-in backends, `prepared-half` and `prepared-integer`. Both passed hardware GPU numerical validation, including all 75 captured model boundaries and exact network-head and image outputs at 128 x 96. Warm 720p/1080p comparisons found **both prepared backends slower than the standard backend**: prepared half took about 13% longer and prepared integer took about 3.8 times as long. The existing `half` backend remains the default and the recommended mode on the tested RTX 5080.

## Representation and cache

`prepareModel()` reconstructs the ordinary matrix codes from the native model and packs each FP8 matrix as `[batch][K32 slab][N32 tile][K4 group][column]`. Each uint32 contains four consecutive K codes for one output column. Both new kernels use this representation, without separate copies for each backend or output resolution.

The same buffer has a metadata tail: one byte per `[batch][K16 group][column]`. The low four bits encode the maximum clamped nonzero exponent plus six. Bits 4, 5, and 6 indicate all-zero weights, constant exponent among nonzero weights, and no zero weights. Bit 7 is reserved. There are still eight storage-buffer bindings per GEMM.

Preparation of the local model verified these sizes:

| Data | Bytes | MiB |
|---|---:|---:|
| Original FP8 codes, repacked | 143,831,040 | 137.17 |
| Exponent metadata | 8,989,440 | 8.57 |
| Combined prepared matrices | 152,820,480 | 145.74 |

All 358 FP8 matrices are eligible. The two half-precision endpoints retain their original representation. Metadata adds 6.25% to the FP8 payload; this pass does not compress the weight codes below one byte each. CPU prepared data is retained for GPU-cache clearing and re-upload. It is released when the renderer is disposed. GPU uploads remain cached across renders.

IndexedDB stores prepared matrices locally. Its key includes a hash derived from the actual tensor bytes and names, the format version, and the matrix specification. Resolution is deliberately absent. Records validate their identity, dimensions, layout version, descriptor fields, and data digest before reuse. Missing storage, quota failures, stale entries and corrupt entries fall back to preparation. No model data is uploaded or included in the repository/site. A selected DLL or model folder is still required to identify and supply the model each session.

## Arithmetic

Let `EA` and `EB` be the maximum nonzero activation and weight exponents for one 16-term group. Their sum is only an upper bound: the two maxima can occur at different K positions. The shader skips the paired exponent scan only when the accumulator exponent already dominates that bound, the weight group is all zero, or the weights have a constant exponent and no zeros. Other groups keep the paired scan.

For finite nonzero FP8 operands, `v = sign * m * 2^(e-3)`, where `m` is the integer significand and `e` is clamped to at least -6. Given the actual group exponent `E`, the integer path computes the magnitude of each F13 term from `mA*mB`, shifted left by `7-d` or right by `d-7`, with `d=E-eA-eB`. It applies the sign after shifting, preserving truncation toward zero. Large shifts explicitly produce zero. This replaces product arithmetic without changing the intended residual seed, 16-term half publication, partition order, SiLU lookup, or packed output publication.

| Backend | Tile | Threads | Shared bytes | Requires shader-f16 |
|---|---|---:|---:|---|
| Prepared half | 32 x 32 | 128 | 8,704 | Yes |
| Prepared integer | 32 x 32 | 128 | 2,432 | No |

The integer path stages packed codes and uses software fixed-point-to-half publication. Its smaller shared-memory footprint is a structural difference, **not evidence of a speedup**. Extra shifts, extraction and branching may offset the memory savings. Both prepared variants currently use a fixed 32 x 32 tile. `gemmTile` continues to control the original wide fallback kernels.

Half preparation uses the existing bounded-weight eligibility check. Unsupported devices, shapes, missing specializations or nonfinite weight codes retain the existing layout and kernels. The renderer reports prepared versus fallback GEMM dispatch counts, and the comparison harness rejects an experiment that silently runs no prepared kernels.

## Completed verification

- All 62 host tests passed, including exhaustive checks of 1,047,464 finite FP8 pair/exponent combinations, additional large-shift cases, exponent-bound counterexamples, residual seeds, cancellation and partition publication.
- The build produced 338 CUDA-generated kernels, including 41 specializations for each new backend. A CPU artifact audit verified their eight bindings, rows-only scalar arguments, shared-memory requirements and feature requirements.
- A CPU graph-dispatch test verified that 358 FP8 operations select prepared buffers, two endpoints fall back, warm renders reuse weight uploads/bindings, and clearing GPU weights allows re-upload from retained preparation. This test does not execute shaders.
- An actual Edge IndexedDB test with GPU access disabled verified all 358 matrices across cold preparation, page reload, deliberately corrupted/stale records, and cache clearing. Complete prepared-data digests matched. It observed zero GPU requests or uploads.
- In one actual-DLL browser-storage test, cold preparation and writes took 1.816 seconds; reuse after reload took 0.425 seconds. These are individual CPU setup observations, not inference measurements or an isolated performance study.
- Hardware GPU validation passed **492 checks with zero failures**, and all **338 CUDA-generated kernels** created GPU pipelines. The prepared variants were exercised across all 41 specialization shapes, adverse exponent-metadata patterns, full-range finite integer weights, packed output/raw tail guards, and surplus dispatches, including 2D dispatches.
- With the actual local model at **128 x 96**, each prepared backend matched **all 75 captured boundaries**, the network head, and the final image against the existing baseline. Each run dispatched **358 prepared FP8 GEMMs and two original half-precision endpoint GEMMs**. Both produced head SHA-256 `55953855fb590273902c7bf9dd3b4b269688ace4852e90c29ce4ddc2cb969aaf` and output SHA-256 `7534fc82a40716832e620535e85b38b96916130a11958463078df7daf5728482`.

The hardware checks used an **NVIDIA GeForce RTX 5080**, **driver 616.64**, and **Edge 153.0.4234.48**. Local, ignored raw reports are `reports/prepared-gpu.json`, `reports/prepared-half-boundaries.json`, and `reports/prepared-integer-boundaries.json`. The integer artifacts do not require `shader-f16`, but these hardware runs used a device with that feature; execution on a device without it has not been tested. Boundary capture adds readbacks and synchronization, so its elapsed times are correctness-run observations, not performance results.

## Running comparisons

In the browser, open **Model → Model preparation → Compute backend**. Prepared modes remain marked experimental. The same panel controls local persistence and clears saved preparation.

For a matched-tile comparison:

```powershell
$env:NR_DLL='C:\path\to\nvngx_dlssnr.dll'
$env:NR_BROWSER='C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
$env:NR_TILE='32x32'
$env:NR_RUNS='20'
$env:NR_BASELINE_GEMM_BACKEND='half'
$env:NR_GEMM_BACKEND='prepared-half'
$env:NR_REPORT='reports/prepared-half-comparison.json'
npm run benchmark:execution
$env:NR_BASELINE_GEMM_BACKEND='prepared-half'
$env:NR_GEMM_BACKEND='prepared-integer'
$env:NR_REPORT='reports/prepared-integer-comparison.json'
npm run benchmark:execution
```

Both backends then use the same 32 x 32 tile. A separate comparison with `NR_TILE=auto` measures against the current production tile policy. The execution harness checks final output hashes, alternates the order, and records setup costs separately. `npm run benchmark` accepts the same `NR_GEMM_BACKEND` selection with `NR_CAPTURE=1` and `NR_COMPARE` for intermediate-boundary checks. `npm run test:model-cache` exercises storage with a synthetic model by default and never requests a GPU; supplying `NR_DLL` exercises the local model instead.

No WebCuda compiler/runtime changes were required for this pass.

## Performance comparison

### Prepared half versus the standard backend

The completed comparison used 20 measured warm renders per backend and resolution, alternated their order each round, and excluded one warmup per backend plus renderer setup. Both used the same generated shaders, renderer version, attention settings, and final-image readback. `NR_TILE=auto` preserves the standard backend's current tile selection; prepared half uses fixed 32 x 32 tiles. **These measurements compare complete backends and include the tile-policy difference; they do not isolate the effect of packing or exponent metadata.** The local raw report is `reports/prepared-half-comparison.json`.

| Resolution | Backend | Median ms | Q1–Q3 ms | IQR ms | Min–max ms |
|---|---|---:|---:|---:|---:|
| 1280 x 720 | Standard half, auto tiles | 190.10 | 189.22–191.52 | 2.30 | 185.90–194.20 |
| 1280 x 720 | Prepared half, 32 x 32 | 214.90 | 213.35–215.95 | 2.60 | 211.60–217.30 |
| 1920 x 1080 | Standard half, auto tiles | 392.00 | 390.25–395.75 | 5.50 | 387.60–410.30 |
| 1920 x 1080 | Prepared half, 32 x 32 | 444.10 | 441.67–447.10 | 5.43 | 439.20–469.90 |

Prepared half increased median render time by **13.05% at 720p** and **13.29% at 1080p**. Every paired round was slower, and the complete sample ranges did not overlap at either resolution. The comparison therefore provides clear evidence of a regression for the prepared-half backend on this system. Keep the standard backend as the default.

For each alternating round, the paired percentage difference is `100 * (prepared / standard - 1)`. Positive values mean slower:

| Resolution | Median paired difference | Q1–Q3 | Min–max |
|---|---:|---:|---:|
| 1280 x 720 | +12.62% | +11.99% to +13.83% | +9.46% to +16.68% |
| 1920 x 1080 | +13.33% | +12.48% to +13.70% | +9.75% to +20.56% |

Quartiles use linear interpolation between sorted samples; IQR is Q3 minus Q1. Median paired differences need not equal the percentage difference between the two backend medians.

Renderer creation took 27.238 seconds for standard half versus 46.224 seconds for prepared half in the 720p case, and 27.305 versus 44.285 seconds in the 1080p case. These are single setup observations that include pipeline creation and, for prepared half, model preparation/cache loading. They are not preparation-only measurements. The first candidate setup prepared and saved 358 matrices; the second reused all 358 from IndexedDB.

Every measured candidate render dispatched **358 prepared GEMMs and two endpoint fallbacks**; every baseline render used the original 360 GEMMs. Warmup and all measured outputs matched across backends. The final-image SHA-256 hashes were `92c0bcf0e7f54837200fd59d2d5d3a1bebf5fee321fc40705afc4f23ef216698` at 720p and `43d9ba3263578ed4a5f4b5f2e15d7c8b4e32966e90ce863cc18d6c91f6df2532` at 1080p.

### Prepared integer versus the standard backend

The integer comparison used the same procedure: 20 measured warm renders per backend and resolution, alternating order, one excluded warmup per backend, matching attention and readback settings, and a fresh standard-half baseline. It also uses `NR_TILE=auto` for the baseline and fixed 32 x 32 prepared tiles, so it compares complete backend implementations rather than isolating integer arithmetic. The local raw report is `reports/prepared-integer-comparison.json`.

| Resolution | Backend | Median ms | Q1–Q3 ms | IQR ms | Min–max ms |
|---|---|---:|---:|---:|---:|
| 1280 x 720 | Standard half, auto tiles | 190.30 | 188.17–191.08 | 2.90 | 186.80–194.10 |
| 1280 x 720 | Prepared integer, 32 x 32 | 734.20 | 733.12–735.10 | 1.98 | 731.40–739.70 |
| 1920 x 1080 | Standard half, auto tiles | 393.75 | 392.20–395.20 | 3.00 | 389.10–398.40 |
| 1920 x 1080 | Prepared integer, 32 x 32 | 1491.80 | 1490.32–1495.35 | 5.03 | 1485.40–1501.20 |

Prepared integer took **3.86 times as long at 720p** and **3.79 times as long at 1080p**, increasing median render time by 285.81% and 278.87%, respectively. Every paired round was slower, with no overlap between the complete sample ranges. The smaller shared-memory allocation did not produce a performance advantage in this implementation on the tested GPU.

| Resolution | Median paired difference | Q1–Q3 | Min–max |
|---|---:|---:|---:|
| 1280 x 720 | +287.36% | +283.83% to +290.28% | +277.59% to +294.43% |
| 1920 x 1080 | +279.38% | +278.00% to +280.66% | +275.73% to +282.83% |

Renderer creation took 27.046 seconds for standard half versus 90.360 seconds for prepared integer in the 720p case, and 27.146 versus 89.352 seconds in the 1080p case. These individual observations include pipeline creation and model preparation/cache loading. The first candidate setup prepared and saved all 358 matrices; the second reused all 358 from IndexedDB. Cache reuse therefore worked, but the complete renderer setup remained substantially slower. These measurements do not separately attribute setup time to compiler work and model preparation.

All **40 measured integer renders** used **358 prepared GEMMs and two endpoint fallbacks**. Every baseline used the original 360 GEMMs. Warmup and measured outputs matched their paired baselines and produced the same 720p and 1080p output hashes listed above. Integer numerical correctness also passed the specialization suite and all 75 model boundaries. Neither prepared backend should replace the current default based on these results; determining which source changes caused the slowdown requires a separate controlled comparison.
