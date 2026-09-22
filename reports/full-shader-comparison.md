# Full generated-shader and execution comparison

Measured 2026-09-22 on RTX 5080 / Edge 153.0.4234.48. This compares the entire executed network, not just a representative GEMM shader. WebCuda includes native half support and the new production CUDA `__clz` lowering from `b8e8cd0`. The renderer was measured with the __clz working-tree change atop `62782e4`; raw reports include the measured kernel/compiler SHA-256 fingerprints. Upstream is the unmodified `9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1` checkout.

## End-to-end network timing

| Source resolution | Ours, median | Upstream, median | Ours / upstream |
| --- | ---: | ---: | ---: |
| 512 x 512 | 136.20 ms | 50.45 ms | 2.70x |
| 1280 x 720 | 391.75 ms | 146.85 ms | 2.67x |
| 1920 x 1080 | 921.30 ms | 358.80 ms | 2.57x |

Both received the exact same precomputed float32 features and local model. One cold run preceded six alternating warm samples each. Timed scope: feature upload, complete graph execution, and padded float32 head readback. Preprocessing, display composition, UI, loading and compilation are excluded. No downscaling. Network heads were finite and bit-identical on every paired iteration and on all three additional profiled iterations per renderer/resolution.

## GPU time by complete operation family

Three separate timestamped runs follow the normal comparison. Tables report the median sum per family, with equivalent fused and unfused work grouped together. Profiling splits each dispatch into its own pass, so these timings locate GPU work; they are not normal frame wall time. Do not subtract them from the separate wall timings to infer CPU overhead.

| Family | 720p ours | 720p upstream | 1080p ours | 1080p upstream |
| --- | ---: | ---: | ---: | ---: |
| Conversion / pool / merge | 2.60 ms | 0.90 ms | 5.85 ms | 2.20 ms |
| FP16 GEMM | 4.68 ms | 1.80 ms | 10.37 ms | 3.84 ms |
| FP8 GEMM | 264.31 ms | 81.72 ms | 641.00 ms | 184.54 ms |
| Local normalization + attention | 82.53 ms | 27.49 ms | 177.42 ms | 58.88 ms |
| Global normalization + attention | 8.83 ms | 10.46 ms | 32.36 ms | 58.66 ms |

At 720p the median total GPU time was 362.94 ms vs 122.35 ms, a 240.59 ms gap. FP8 GEMMs explain about 182.59 ms (76%) and local normalization/attention another 55.04 ms (23%). The 62 separate local normalization passes alone cost about 32.62 ms; our local attention costs 50.26 ms, versus their combined 27.49 ms. Global attention is already competitive and is faster in our implementation at 1080p. Family medians need not sum exactly to the median total.

## Actual shader and pipeline inventory

The harness intercepts the upstream device during setup and records actual shader-module strings and asynchronous pipeline descriptors, including override constants. It captured 11 shader modules and 384 async pipeline descriptors per resolution: four FP8 GEMM module variants, window attention, FP16 GEMM, global ViT, generic ops, preprocessing and two setup-only table builders. Preprocessing and table builders are not timed network passes. Our GPU profiles reference 50 active generated entry points: 41 native-half GEMM specializations plus the other operation families. All 358 FP8 GEMM calls selected the native-half path; the speed gap is not an accidental fallback.

| Graph operation | Ours dispatches | Upstream dispatches |
| --- | ---: | ---: |
| FP8 GEMM | 358 | 358 |
| FP16 adapter/head GEMM | 2 | 2 |
| Local normalization | 62 | 0 (fused) |
| Local attention | 62 | 62 |
| Global normalization + attention | 32 | 16 |
| Conversion, downsampling and merges | 13 | 13 |
| Total | 529 | 451 |

The 78-dispatch difference is exactly 62 fused local normalizations plus 16 eliminated global-attention passes. Most time is nevertheless in FP8 GEMMs, whose dispatch count is identical.

## FP8 matrix multiplication: the dominant difference

| Property | Our emitted WGSL | Their captured production WGSL |
| --- | --- | --- |
| Outputs per invocation | 1 | 8 (two rows by four columns) |
| Output tile | 8x8 or 8x16 | 32x32 |
| K step between shared synchronization | 16 | 32, retaining two ordered 16-term arithmetic groups |
| Shared operand representation | vec2<f16>, paired K values | vec4<f16>, vector loads and more output reuse |
| Shared weight/exponent layout | K-major tile | Transposed/padded vector columns |
| Exact half products | Yes, bounded weights and x4 operands | Same bound/scaling idea |
| Accumulation/publication | Integer F13; software exact integer-to-half conversion with native bit scan | Exactly bounded float integer sums and native half publication in vector groups |
| Packed FP8 outputs | Per-lane compare/exchange retries | One owner writes the whole word |
| Weight decoding | Arithmetic byte decode and exponent extraction | 256-entry metadata lookup plus word-oriented loads |
| SiLU publication | Software arithmetic and half rounding | Precomputed half-input activation/code lookup |
| Shape specialization | Fixed matrix parameters; rows stay runtime | Pipeline overrides include rows, dimensions, strides and layouts |
| Small reduction loops | Retained in WGSL | Explicitly unrolled |

At 720p, ours dispatches 26,387,328 FP8 workgroups versus 2,012,448 upstream (13.1x). That is more groups for the same mathematical outputs, not 13.1x more matrix arithmetic. Smaller tiles expose more indexing, loading, barriers and group scheduling per result. More outputs per thread also permits non-atomic word publication. Wider tiles may increase register pressure; the earlier multi-output CUDA trial without the current half/exponent layout did not give reliable wins.

The representative 8x16 half tile uses 1,536 bytes of shared memory; their tile128 GEMM uses 8,704 bytes. The upstream device limit of 32 KiB is not the actual GEMM allocation. Requesting more shared memory alone will not change our GEMM behavior.

## Local and global attention

Our local shader uses 128 threads, eight query rows per workgroup, f32 shared arrays (11,552 bytes), and a separate normalization pass that writes a packed tensor. Their captured local shader uses 512 threads, 32 queries, 32 KiB of shared storage, fused normalization from raw QKV, vector/shared exponent fragments, bounded native-half value products, unrolled reductions and single-owner packed stores. The normalized tensor is not written to global memory. This combination addresses both the 32.62 ms normalization cost and repeated K/V loading; fusion alone is not proven to save the entire normalization duration because the fused shader must still perform its arithmetic.

Global ViT attention differs: our tiled score, softmax and value passes plus normalization use 32 dispatches total; theirs combines attention into eight workgroup-shared dispatches plus eight normalization passes. Fewer dispatches are not automatically faster: at 1080p our grouped GPU time is 32.36 ms versus 58.66 ms. Preserve this advantage while optimizing the other families.

## FP16 endpoints and remaining ops

Their two FP16 GEMMs compute four output columns per invocation, reuse eight loaded activation values and publish complete packed words. Ours uses scalar-output packed GEMM and software F24 arithmetic. Both retain exact fixed-point-to-half behavior. The 720p difference is about 2.88 ms, much smaller than FP8 GEMMs.

Their conversion, downsampling, upsampling and post-blend shaders likewise operate on four outputs and store packed words. Our scalar outputs generally go through the generic atomic storage helpers. The whole remaining family differs by roughly 1.7 ms at 720p. It is a valid cleanup target, but not the cause of the large gap.

## Compiler, scheduling and memory

WebCuda translates the CUDA loops and packing logic faithfully. It does not currently turn scalar-output CUDA into multi-output tiled kernels, fuse neighboring graph operations, generate activation lookup tables, or fully specialize called helper bodies. Those are source/graph/runtime responsibilities. Generated code still contains dead constant branches, repeated helper parameters and fixed loops; downstream driver optimization may eliminate them, so WGSL length is not an instruction count.

Compiler-specific issues include missing intrinsic lowering (now fixed for __clz), conservative storage barriers based on whole-kernel read/write usage, and leaving constant helper arguments/loops to the driver. CUDA strong atomicCAS correctly needs retry behavior over WGSL weak compare/exchange. Removing that retry without changing output ownership is not valid.

Our prepared graph reuses bindings and buffers but submits bounded batches (32 operations, up to four in flight). Their recorder replays the graph into one command buffer and normally one compute pass. Our scheduling supports bounded cancellation and pooled graph memory. The measured GPU-family gap is already large, so JavaScript scheduling alone cannot explain the slowdown. No driver ISA/register/occupancy capture was taken, and no precise CPU-overhead number is inferred from these profiles.

| Allocation counter | 720p ours | 720p upstream | 1080p ours | 1080p upstream |
| --- | ---: | ---: | ---: |
| Reusable plan / allocated activations | 431.5 MiB | 1905.0 MiB | 925.0 MiB | 4078.6 MiB |

These counters cover different strategies and are not total VRAM usage. Weight caches are about 144.5 MiB each. Their eager weight layout and async pipeline compilation move more work into setup; our first inference uploads/caches matrices lazily. Cold times are recorded separately and are not included in warm medians.

## Measured generated-code experiments

| Experiment | 720p baseline -> candidate | 1080p baseline -> candidate | Status |
| --- | --- | --- | --- |
| Production __clz vs restoring old loops | 407.65 -> 380.40 ms | 959.65 -> 900.45 ms | Shipped CUDA + compiler change; 6.7% / 6.2% lower wall time |
| Remove duplicate FP8 encode/decode before packed store | 376.50 -> 369.85 ms | 882.95 -> 873.30 ms | Diagnostic only; 1.8% / 1.1%, samples overlap |
| Remove extra storage barriers (previous audit) | 408.05 -> 408.10 ms | 953.70 -> 950.35 ms | No meaningful measured gain |

These experiments use the complete output-image renderer, unlike the common-feature network benchmark above. Compare each paired row internally. Each used six alternating warm samples and checked every output hash. Publication removes only the redundant quantize/decode step for packed-FP8 half-GEMM destinations; it does not remove output atomics. Both diagnostic edits remain outside production. The bit-scan benchmark now restores the old loop across generated kernels in the baseline, so it tests the production change rather than editing only half GEMMs.

## Implementation priorities established by this comparison

1. Rebuild the FP8 CUDA tile around multiple output columns per thread, larger shared reuse, vector fragments and one owner per packed output word. Evaluate it with the current native-half/exponent path, exact partition order and per-boundary parity. This targets the measured 76% share of the 720p gap.
2. Fuse local normalization into local attention and test a wider query tile. Add an explicit adapter-workgroup-limit option to WebCuda if needed, while retaining the existing path for devices below 32 KiB.
3. Isolate native half publication, metadata/SiLU tables, and reduction unrolling in separate exactness/performance tests. A broad f16 substitution is not justified solely by exact half products; subnormals, overflow and signed zero require validation.
4. Keep global attention unchanged initially. Direct packed-code publication is a small measured opportunity, not a substitute for the two dominant changes.

## Validation and reproduction

WebCuda: 770 host tests, kernel compilation, and 4,096 GPU patterns checked as both signed and unsigned __clz operands. Renderer: 37 host tests, 281 GPU checks, all 75 captured network boundaries plus head/output matching the saved reference for two runs, and cross-port exact heads at 512, 720p and 1080p including profiling.

Set local NR_UPSTREAM, NR_MODEL and NR_BROWSER, then NR_PROFILE=1 and run `npm run benchmark:upstream`. Setup captures raw shader strings and pipeline descriptors only into ignored JSON reports. Normal six-sample timings run before profiling. For output-image experiments set NR_DLL and NR_AUDIT=bitscan, publication or barriers, then run `npm run audit:generated`. Upstream code, model files, DLLs and raw captures are not redistributed.

The comparison covers every executed network family. Preprocessing, final display composition, image loading and UI export are excluded deliberately; their inputs and application flows are not identical. Cross-port equality on the shared generated feature inputs is not proof of temporal quality or all possible inputs. GPU timestamps locate work, but they do not prove individual register, cache or bank-conflict costs.
