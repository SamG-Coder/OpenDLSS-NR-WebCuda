# Shape-specific CUDA tiles and compact dispatch grids

This pass follows `6aacccb`. The paired baseline uses that revision's renderer and saved generated shaders, preserving its old dispatch grid as well as its kernels. Measurements use RTX 5080 / Edge 153.0.4234.48 on 2026-09-22.

## Real-model image renders

Twenty alternating warm samples per engine and resolution, including preprocessing, network, composition and final float32 output readback. Setup, source creation, hashing and UI conversion are excluded. Both sides skip the optional head readback. Every cold and timed output hash matched exactly.

| Resolution | Previous renderer median | New renderer median | Less render time |
| --- | ---: | ---: | ---: |
| 1280 x 720 | 236.70 ms | 226.80 ms | 4.2% |
| 1920 x 1080 | 479.05 ms | 452.50 ms | 5.5% |

There was background GPU activity and substantial timing variation during this session. At 720p the previous/new sample ranges were 193.70–297.00 / 193.70–285.50 ms; at 1080p they were 417.70–580.10 / 388.10–539.60 ms. The candidate was faster in 11 of 20 corresponding 720p rounds and 15 of 20 1080p rounds. These are modest observed gains, with weaker evidence at 720p; absolute times should not be compared with the quieter session in the preceding report.

The initial six-sample run was noisy enough to show a 1080p median regression (431.50 to 461.20 ms), despite exact outputs. The larger run above retained all twenty samples rather than filtering slow results. No additional timing runs were used to select a favorable result. Shader setup remained approximately 28–29 seconds for both engines. The upstream renderer was not rebenchmarked in this pass.

## Changes

Large dispatches previously used `x = min(groups, limit)` and `y = ceil(groups / limit)`. At a 65,535-workgroup dimension limit, a request for 69,120 groups launched 131,070 groups. Out-of-range tiles preserved output correctness but still executed their arithmetic. The renderer now uses `y = ceil(groups / limit)` and `x = ceil(groups / y)`, launching exactly 69,120 groups in this example. The general formula adds fewer than `y` surplus groups, with both dimensions within the device limit. All affected kernels already flatten `blockIdx.xy` through `gridDim.x`.

The same CUDA GEMM template now generates three tile shapes. Each thread still owns eight outputs; paired products, ordered 16-term F13 groups, partition accumulation, half rounding and packed output ownership are unchanged.

| Tile, rows x columns | Threads | Shared bytes |
| --- | ---: | ---: |
| 32 x 32 | 128 | 8,448 |
| 64 x 32 | 256 | 12,544 |
| 32 x 64 | 256 | 12,800 |

The default selection uses 64 x 32 for K32/N32 and K64/N128; 32 x 64 for K32/N64, K32/N128, K64/N64, K64/N192, K64/N256, K128/N384 and K256/N768; and the established 32 x 32 tile elsewhere. This changes 98 of the graph's 358 FP8 GEMM dispatches. Deep matrices keep the smaller workgroup because the larger tiles measured slower there.

Only one wide variant per specialization is loaded. Device checks cover shared storage, invocation count and X dimension. Missing or unsupported variants fall back to 32 x 32, then the existing smaller half/scalar paths. Native-half support and the bounded-weight check remain required. Weight and activation caches do not grow, and no WebCuda compiler change is required for this pass.

## Isolated tile measurements

The DLL-free shape benchmark covers 45 specialization/row combinations per resolution, accounting for all 358 FP8 GEMM calls. It uses deterministic synthetic values, exact packed-output and raw-output comparisons, two warmup passes, and seven rotating samples of three dispatches. GPU timestamps exclude compilation, upload and comparison readbacks.

The following weighted sums use the **corrected dispatch grid on both sides**. They isolate tile selection and exclude the separate benefit of the grid fix. They are estimates from isolated kernels, not whole-network or frame timings.

| Resolution | All 32 x 32 | Selected tiles | Less estimated FP8 time |
| --- | ---: | ---: | ---: |
| 1280 x 720 | 108.42 ms | 106.87 ms | 1.4% |
| 1920 x 1080 | 231.24 ms | 225.71 ms | 2.4% |

Initial shape measurements exposed the grid-padding problem. Tile choices were made from a second complete run after fixing it; padding-driven speedups are not attributed to tile size.

## Verification

- All 408 GPU checks passed, including pipeline creation for 256 generated kernels and exact output/raw-output comparisons for every new tile specialization.
- Partial row and column tiles include a 65-row, 96-column case, surplus workgroups and packed-write sentinels.
- All 44 host tests passed, including compact-grid coverage, device-limit fallback, artifact fallback and single-variant loading.
- All 75 intermediate boundaries at 128 x 96 matched the saved reference, together with the full head and final-output hashes.
- Every full-image output in both paired real-model runs matched at 720p and 1080p.
- The static Pages build and repository-subpath smoke test passed, covering module loading, image and glTF inputs, decoder assets and all generated kernels.

## Reproduction

Run `npm run build`, `npm test`, `npm run test:gpu`, and `npm run benchmark:tiles`. Set `NR_BROWSER` to the browser executable and `NR_REPORT` for the ignored JSON output. `NR_RESOLUTIONS` selects dimensions; `NR_TILE_FILTER` limits shapes. No DLL is needed for the shape benchmark.

The real-model comparison uses `npm run benchmark:execution` with local `NR_DLL`, `NR_BROWSER`, `NR_ENHANCED=1`, `NR_RUNS=20`, `NR_PREVIOUS_GENERATED=reports/pre-tiles-generated`, and `NR_PREVIOUS_ENGINE=reports/pre-tiles-generated/engine.js`. Save the prior artifacts and engine before editing; the saved engine must use root-relative imports and `/generated/` URLs. `NR_TILE` forces a specific tile for a candidate comparison. The default remains `auto`.

Saved engines, generated snapshots, JSON measurements, DLLs and model data remain local and ignored. The public repository contains the authored CUDA, runtime selection, benchmark scripts and this aggregate report.
