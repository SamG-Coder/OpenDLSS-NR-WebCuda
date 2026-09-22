# Precomputed model operands

This experiment moves weight-side arithmetic out of every render and into model preparation. It is exposed as `gemmBackend: 'precomputed-half'`. It preserves the input and output resolution, the standard automatic tile policy, and the existing vector F13 accumulation. Its objective is less repeated computation, with additional storage accepted as the cost.

## What is precomputed

For each original FP8 weight, `precomputeMatrix()` calculates the exact binary16 representation of `weight * 4` and the clamped weight exponent used by the existing GEMM. It packs adjacent K operands into native half2 pairs. Zero weights retain their signed zero value and use the existing exponent sentinel of -128.

The buffer contains two planes with ordering `[batch][K32 slab][N32 tile][K pair][column]`: scaled half2 values followed by half2 exponents. The generated shader loads these directly through a CUDA `const __half2*`. Weight byte extraction, numerical-table lookup, scaling and float-to-half pair construction no longer happen during rendering. Activation preparation remains dynamic because it depends on the image and previous layers.

The exponent combination with the current activations and accumulator, products, ordered 16-term F13 reduction, half publication, partition addition, SiLU and output publication retain the standard implementation. Those values depend on runtime inputs; this representation does not precompute complete layer outputs.

The local model has 358 eligible FP8 matrices containing 143,831,040 weights. Their value plane and exponent plane each occupy 287,662,080 bytes, totaling **575,324,160 bytes (548.67 MiB)**. This is four bytes per original FP8 weight. Both original half-precision endpoint matrices keep their existing path. GPU buffers are uploaded once and reused across frames.

## Runtime and persistent cache

The backend uses the same 32 x 32, 64 x 32, and 32 x 64 tile policy as Standard, with the same workgroup/shared-memory requirements. CUDA kernels remain the source of generated WGSL. WebCuda supports the native half2 storage-buffer interface; no hand-written replacement WGSL is used.

IndexedDB uses the distinct `half-operands-v1` layout identity, the hash of actual model tensor contents, and the full matrix specification. Records authenticate both planes before reuse. Existing compact prepared records remain valid for their original backends and cannot be interpreted as these operands. Preparation is independent of output resolution. Missing browser storage falls back to preparation from the selected local model.

Devices without `shader-f16`, insufficient workgroup limits, unavailable specializations, unsupported matrix shapes, and out-of-bound half products retain their original kernels and buffers. Standard remains the default while this experiment is evaluated.

## Validation and comparison

CPU validation checked all 254 finite FP8 codes, signed zero, multi-batch/slab/tile ordering, cache integrity and format separation. Exhaustive inspection of the actual local model matched **all 143,831,040 scaled weights and all corresponding exponent operands exactly**.

The NR host suite passed all **72 tests**. Hardware GPU validation passed **616 checks with zero failures** and created all **461 kernel pipelines**, including 123 precomputed variants. Every precomputed tile matched the scalar reference's packed output and raw output, including partial tiles, output guards and surplus two-dimensional dispatches.

The companion WebCuda suite passed **782 host tests** and compiled all ten examples. Its separate hardware test passed 41,656 exact paired half products plus native half/half2 storage reads and writes. CPU-oracle storage checks cover all 65,536 binary16 encodings and pointer offsets. By-value half kernel arguments remain unsupported; the new interface uses typed storage pointers.

With the actual local model, both cold and warm correctness runs matched **all 75 captured boundaries, the network head and final image** at 128 x 96. Each dispatched 358 precomputed GEMMs and two original half endpoints. The warm run performed zero matrix decoding and uploaded only its 196,608-byte input image; model operands remained resident. Boundary captures force readbacks and are not frame-speed measurements.

Head SHA-256: `55953855fb590273902c7bf9dd3b4b269688ace4852e90c29ce4ddc2cb969aaf`.
Image SHA-256: `7534fc82a40716832e620535e85b38b96916130a11958463078df7daf5728482`.

The checks used RTX 5080, driver 616.64 and Edge 153.0.4234.48. Local raw reports remain ignored: `reports/precomputed-gpu.json`, `reports/precomputed-half-boundaries.json` and `reports/precomputed-half-comparison.json`. WebCuda's committed change is [`f0f3699`](https://github.com/SamG-Coder/cuda-webshader/commit/f0f3699b498cfe6fe5419e072a4f4e2faa63b781), synced into this project.

## Full-resolution performance

Both modes used identical automatic tile selection, attention settings and output readback. Twenty warm renders per backend and resolution alternated execution order; setup and one warmup per backend were excluded. This comparison changes the weight representation and staging, without the tile-policy mismatch present in the older prepared experiments.

| Resolution | Standard median | Precomputed median | Render-time change |
|---|---:|---:|---:|
| 1280 x 720 | 187.10 ms | 190.65 ms | +1.90% |
| 1920 x 1080 | 386.95 ms | 393.30 ms | +1.64% |

This run provides **no evidence of a speedup**. The precomputed backend was slower in 19 of 20 paired rounds at 720p and all 20 at 1080p. Standard remains the recommended default; precomputed operands remain available as an experiment. Weight decoding and conversion were removed, but input-dependent products, exponent combinations and F13 reductions still execute. The larger operand loads may offset the removed arithmetic; this run does not isolate that cause.

| Resolution / backend | Q1–Q3 ms | Min–max ms |
|---|---:|---:|
| 720p Standard | 185.07–188.82 | 183.90–202.90 |
| 720p Precomputed | 190.05–192.68 | 186.20–205.80 |
| 1080p Standard | 385.10–388.12 | 382.80–390.10 |
| 1080p Precomputed | 392.17–394.40 | 390.20–396.20 |

Quartiles use linear interpolation. Median paired render-time increases were 1.83% at 720p and 1.55% at 1080p. No extra benchmark repetitions were used to pursue this small difference.

All 40 measured candidate renders used **358 precomputed GEMMs and two endpoint fallbacks**. Every measured and warmup image matched its paired baseline. Output SHA-256 hashes: `92c0bcf0e7f54837200fd59d2d5d3a1bebf5fee321fc40705afc4f23ef216698` at 720p and `43d9ba3263578ed4a5f4b5f2e15d7c8b4e32966e90ce863cc18d6c91f6df2532` at 1080p.

Renderer creation took 27.525 seconds for Standard and 45.856 seconds for precomputed at 720p; 27.547 and 43.484 seconds at 1080p. These individual setup observations include pipeline compilation and model preparation/cache loading. The first precomputed setup wrote 358 matrices; the second loaded all 358 from IndexedDB with no preparation misses. Persistent caching works, but this implementation also loads fallback pipelines and does not improve total setup time.

To compare against Standard at identical automatic tiles and original 720p/1080p resolutions:

```powershell
$env:NR_DLL='C:\path\to\nvngx_dlssnr.dll'
$env:NR_BROWSER='C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
$env:NR_TILE='auto'
$env:NR_RUNS='20'
$env:NR_BASELINE_GEMM_BACKEND='half'
$env:NR_GEMM_BACKEND='precomputed-half'
$env:NR_REPORT='reports/precomputed-half-comparison.json'
npm run benchmark:execution
```

The harness alternates execution order, excludes setup and one warmup per backend, checks every output hash, and records effective dispatch counts and cache statistics. Earlier compact FP8 and integer backend experiments are documented separately in [model-preparation.md](model-preparation.md).
