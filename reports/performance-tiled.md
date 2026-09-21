# Tiled CUDA GEMM and persistent working buffers

Measured 2026-09-22 on the local RTX 5080 / Edge 153.0.4234.48 with the same locally supplied NR model, gradient input, seed, and controls as the [packed-cache report](performance-packed.md). The preceding implementation is commit `74772ac`. No proprietary binaries, tensors, or image assets are distributed.

## Normal rendering

These runs have GPU profiling **disabled**. Each resolution was run three times in one renderer: the first run prepares the model cache, then two runs reuse it. Wall times cover the entire renderer call, including allocations, uploads, compute, readback, and cleanup, but exclude DLL parsing, shader setup, hashing, and UI conversion.

| Resolution | Previous packed scalar repeat runs | New first render | New repeat runs |
| --- | ---: | ---: | ---: |
| 1280 × 720 | 1878.1, 1881.9 ms | 1668.7 ms | 1215.1, 1206.4 ms |
| 1920 × 1080 | 3696.1, 3698.5 ms | 3301.7 ms | 2701.5, 2710.5 ms |

The average of the two repeat runs improves by **35.6% at 720p** and **26.8% at 1080p**, relative to the previous report on this system. Hardware load and browser scheduling affect wall time; these are local measurements, not cross-device guarantees. Output dimensions remain source-sized.

## GPU measurements

Diagnostic mode records actual timestamp queries around each dispatch and returns the scalar parameters, duration, per-kernel totals, and summed GPU time. It uses individual timestamped passes, changing submission overhead; its wall time must not substitute for the normal-render timings above. Unsupported devices explicitly report that timestamp queries are unavailable. Query and readback resources are released after profiling, including cancellation.

Two diagnostic runs were taken for each mode using the same final buffer-cache policy. On the second 720p run:

| GPU work | Scalar reference | Tiled mode |
| --- | ---: | ---: |
| Matrix multiplication, all kernels | 890.52 ms | 476.79 ms |
| Attention scores | 144.01 ms | 143.21 ms |
| Attention weighted output | 96.49 ms | 96.62 ms |
| Normalization | 31.06 ms | 31.90 ms |

The tiled change reduces measured matrix-multiplication GPU time by about **46.5%**. No changes to attention arithmetic are included.

## Implementation

`nr_gemm_tiled` is authored in CUDA. A 64-thread group computes four rows by sixteen columns, with 64 shared input floats and 256 shared weight floats (1280 bytes). Input values are reused across columns; unpacked FP8 weights are reused across four rows and both exponent/accumulation loops. Both barriers are reached by every thread, including partial row/column tiles and surplus groups in a 2D dispatch. Accumulation order, F13/F24 grouping, partitions, residual seeding, SiLU, and half/FP8 publication are preserved.

FP8 matrix operations use the tile by default. The two small FP16 projection matrices retain the scalar packed kernel; packed model storage remains 144.51 MiB. `gemmMode: 'scalar'` retains the packed scalar execution path for comparison. The tiled kernel also has half-mode regression coverage.

The working-buffer pool now persists between same-resolution frames. It evicts the oldest free size bucket when needed instead of letting unused cached buffers prevent useful reuse. Evicted buffers are destroyed only after their queued users finish; host writes do not reuse buffers referenced by unsubmitted commands. The default cache cap is 256 MiB, independent of the model cache. Resolution changes release the old pool. Oversized buffers still require fresh allocations, and the cap is not a bound on total active GPU memory.

At the end of these runs, the retained working buffers occupied **185.63 MiB at 720p** and **206.72 MiB at 1080p**. Repeat 720p renders created 202 working buffers and reused buffers 891 times; repeat 1080p renders created 321 and reused 772 times. These reuse counts include reuse within the same frame as well as across frames. This does not claim allocation-free rendering.

## Reproduction and validation

Use the existing `npm run benchmark` with `NR_DLL`, optionally `NR_BROWSER`, dimensions, and output report path. Additional options:

- `NR_PROFILE=1`: diagnostic per-dispatch GPU timestamps.
- `NR_GEMM=scalar` or `tiled`: select the packed GEMM implementation (default tiled).
- `NR_WORKSPACE_MIB=0`: disable the working-buffer cache; default is 256 MiB.
- `NR_COMPARE=<previous-report.json>`: require matching network-head and final-output hashes.

API equivalents are `NeuralRenderer.create(model, {gemmMode, workspaceCacheBytes})` and `renderer.run({...inputs, profile: true})`. Call `clearWorkspace()` while idle to release working buffers, independently of `clearWeightCache()`. Disposal releases both caches.

All normal and diagnostic real-model runs matched the preceding full float32 head/output hashes at both resolutions. Verification passed **27 host tests and 57 GPU checks**, including tiled/scalar comparison for non-multiple tile dimensions, surplus 2D groups, half mode, residuals, SiLU, quantization, and partitioned accumulation; explicit unsupported profiling; buffer-budget reuse, resolution changes, cancellation/restart, and release. Existing native numerical fixtures, synthetic capture boundaries, and real-model image/GLB/glTF UI tests also passed.
