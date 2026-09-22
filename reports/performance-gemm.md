# Multi-output GEMM experiment

## Decision

The GEMM benchmark and six exact CUDA variants are implemented. The default `gemmMode: 'auto'` selection remains unchanged: this experiment did not establish a dependable frame-time improvement across both 720p and 1080p on the reference RTX 5080 / Edge setup. Experimental pipelines load only when explicitly selected, so default model setup does not pay for them.

## Implemented

Each thread computes four adjacent output columns and reuses each loaded activation across four independent accumulators. Variants cover 4 × 32, 8 × 32, 16 × 16, 16 × 32, 32 × 32, and 16 × 64 output tiles. All are built from `kernels/gemm-multi.cu`, with normal float storage and CUDA-generated compact storage variants.

Residual initialization, F13/F24 product groups, partition boundaries, half publication, and SiLU retain the reference behavior. Full aligned output groups use one packed FP8 word store or two FP16 word stores. Partial columns and unaligned row starts use the existing masked atomic stores to preserve neighboring values. The two small half-weight projection operations keep the established scalar kernel in full-model execution.

`gemmMode: 'multi-auto'` is an experimental selector: it uses the 32 × 32 variant for selected K=32/64 configurations with at least 16384 rows, and the established 8 × 16 tile elsewhere. It is exposed for continued measurement, not presented as a verified universal optimization. The original `auto`, `tiled`, `tile8x8`, `tile8x16`, and `scalar` modes remain available.

No expanded weight cache, additional model quantization, or approximate arithmetic is introduced.

## Benchmark

`npm run benchmark:gemm` requires WebGPU timestamp queries but no DLL or model assets. It derives 36 FP8 shape/configuration cases from the 720p graph, including matrix dimensions, batches, strides, partitions, residual types, and output publication types. It uses deterministic synthetic values and caps rows at 4096 by default. The cap is recorded alongside each original row count; timings must not be treated as full-resolution frame measurements.

Every candidate is warmed and compared with the scalar packed reference before timing. Three timestamped samples per candidate run in rotating order, and the median is recorded. Both outputs and live raw outputs are checked. The eight candidates are the existing 8 × 8 / 8 × 16 tiles and six new variants. Results are written to ignored local JSON files.

Small shape tests frequently favored the existing kernels. Full-model timestamp profiles showed a more nuanced result: some high-row-count, narrow matrices benefited, while wider matrices regressed. For example, a measured 720p profile summed K=32/N=128 expansion operations to about 38.9 ms using the 32 × 32 variant, versus 43.6 ms using the existing 8 × 16 tile. Conversely, the K=N=512 residual operations grew from about 17.9 ms to 37.5 ms. These figures are from separate diagnostic runs and are evidence for selective experimentation, not a frame-speed claim.

In normal 720p runs, forcing the 32 × 32 variant measured 593.8 / 598.9 ms, and forcing the existing 8 × 16 tile measured 588.5 / 586.9 ms. The preceding default report measured 607.0 / 604.1 ms. These small differences did not justify changing the default, especially with inconsistent 1080p results from candidate selectors. A cooperative packed-load variant and an exact integer-product prototype were also evaluated; neither was retained. The current source keeps the validated float-accumulator variants and existing default selection.

## Validation

- 33 host checks passed, including unchanged default selection and skipping experimental pipelines during default setup.
- 194 GPU checks passed across 47 generated pipelines. New checks cover every multi-output tile, F13 and F24, partitions, broadcast inputs, residuals, SiLU, FP8/FP16 output packing, odd column tails, extra 2D dispatch groups, and sentinel guards.
- All 75 intermediate hashes matched the reference with the multi-output path at 128 × 96 on initial and reused execution plans.
- Full network-head and composed-output hashes matched the preceding implementation during 720p full-model experiments and candidate 1080p runs.
- The synthetic shape benchmark checked every candidate against scalar packed GEMM for every benchmark case.

These checks establish parity with the reconstructed numerical implementation, not original-driver capture parity.

## Use

Run `npm run build` first. Set `NR_BROWSER` if Playwright needs your installed Edge/Chrome executable, then run `npm run benchmark:gemm`. `NR_GEMM_ROWS` accepts 1–65536; `NR_REPORT` selects the local JSON report path.

For full-model comparisons, supply your compatible local DLL via `NR_DLL` and use the existing `npm run benchmark` command with `NR_GEMM=multi32x32` or another explicit mode. Use `NR_GEMM=auto` for the established default. `NR_PROFILE=1` collects diagnostic timestamps, and `NR_COMPARE` verifies hashes against a saved local report. Compare ordinary frame times with profiling disabled.

No DLLs, weights, private captures, or upstream project snapshots are distributed.
