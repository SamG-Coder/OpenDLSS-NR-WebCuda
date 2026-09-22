# Prepared FP8 model and integer GEMM

The model can now be prepared for the generated browser shaders while retaining its original FP8 codes. This adds two opt-in backends, `prepared-half` and `prepared-integer`. The existing `half` backend remains the default. **Hardware GPU numerical validation and speed comparisons for the new backends are pending:** a separate GPU campaign was running during this work, so only CPU/build/browser-storage checks were performed.

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

The GPU suite has been extended with prepared variants across all specializations, adverse metadata patterns, full-range finite integer weights, packed tail guards and surplus 2D dispatches. **These GPU checks have not yet been run for this pass.** Once the other campaign is finished, run them before judging either backend's speed. Then compare all 75 model boundaries and final output against the existing reference, followed by alternating warm 720p/1080p timing samples.

## Running comparisons

In the browser, open **Model → Model preparation → Compute backend**. Prepared modes remain marked experimental. The same panel controls local persistence and clears saved preparation.

For a matched-tile comparison after GPU validation:

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
