# Architecture and optimization research review

Reviewed 2026-09-22 against the current CUDA sources, graph, engine, WebCuda compiler/runtime, browser UI, and existing local benchmark reports. This is a research review, not a new speedup measurement. No inference behavior changed.

## End-to-end execution

1. `src/dll-model.js` parses a user-selected DLL as a PE resource container. The DLL is not executed. The embedded model is validated and exposed through `src/model.js`; 153 tensors supply the reconstructed graph. Packed matrices, scales, and priors are uploaded lazily and retained in a roughly 144.5 MiB GPU cache.
2. `web/app.js` converts an image canvas to float32 RGBA. `web/scene.js` uses Three.js WebGL to render a GLB/glTF object, copies that canvas to a 2D canvas, and takes the same CPU pixel path. The network receives a rendered image, not mesh vertices. Consequently NR cost follows pixel dimensions regardless of whether the source was an image or a 3D object.
3. `kernels/frame.cu` optionally reprojects history, reflects/pads the input, generates noise, and assembles 16 feature lanes containing noise, source/history colour, and controls. Its noise functions deliberately use emulated binary64 arithmetic. A 1280 x 720 source has a 1344 x 768 padded field, with a 63 MiB float32 feature tensor.
4. `src/graph.js` builds the encoder, global bottleneck, decoder with skip connections, and output head. At 720p the fused graph contains 360 GEMMs, 70 normalizations, 62 fused local-attention operations, eight each of global scores/softmax/attend, six pools, six merges, and one input publication: 529 operations. Preprocessing, reprojection, and composition are outside that count. The GEMMs represent 207,888,580,608 logical multiply-accumulates per frame, excluding attention and all emulation overhead.
5. CUDA sources compile through WebCuda to WGSL. Packed FP8/FP16 storage is decoded into shader arithmetic. The exact GEMMs scan group exponents, truncate products to a shared fixed-point scale, accumulate integers, and publish rounded half values at specified boundaries. This is much more work than ordinary matrix multiply-add. Small FP16 projections use the scalar packed path; most matrices use the established tiled kernels.
6. `src/execution-plan.js` assigns disjoint tensor lifetimes to reusable GPU allocations. Prepared execution retains buffers and bindings. `src/engine.js` submits batches of 32 graph operations and then awaits `runtime.idle()`, which calls `onSubmittedWorkDone()`.
7. The engine reads and expands the network head on the CPU before dispatching composition, although composition itself reads the existing GPU head. It then reads the final float32 RGBA result. The UI converts it to RGBA8 and draws through a 2D canvas. These CPU conversions are outside the existing engine-only performance measurements.

## What the measurements establish

The second, warmed measurement in local `profile-fused-final.json` reports:

| GPU work | Time | Share of timestamp total |
| --- | ---: | ---: |
| GEMMs, combined | 450.83 ms | 79.8% |
| Fused local attention | 53.07 ms | 9.4% |
| Normalization | 33.71 ms | 6.0% |
| Preprocessing | 17.13 ms | 3.0% |
| Global scores/softmax/attend | 7.84 ms | 1.4% |
| Remaining kernels | 2.61 ms | 0.5% |
| Total | 565.20 ms | 100% |

Rounding affects percentage sums. Diagnostic profiling changes submission behavior, so this total must not be subtracted from a separate normal-frame timing to infer CPU overhead. Previous ordinary warm measurements were approximately 0.61 seconds at 720p and 1.44 seconds at 1080p. They use a synthetic gradient, not a representative visual-quality dataset.

GEMM is the clear first target for a large speedup. These measurements do not identify whether its dominant constraint is instruction throughput, register pressure, memory traffic, barriers, or atomic contention. The six multi-output tile experiments did not establish a consistent full-frame gain. Changing more tile sizes without separating those costs is weak evidence-driven optimization.

## Browser capability check

A fresh headless Edge 153 probe on the local NVIDIA Blackwell adapter exposed `shader-f16`, `subgroups`, `subgroup-size-control`, and `timestamp-query`. It did not expose `chromium-experimental-subgroup-matrix`; subgroup matrix configurations were unavailable. This describes the tested browser process, not every browser/backend or the user's existing tab.

The vendored runtime requests subgroups but not `shader-f16`. The reviewed compiler has subgroup shuffle/vote support and f32 FMA lowering; the NR matrix kernels currently use workgroup shared memory rather than subgroup matrix operations. An FP8 buffer does not imply native FP8 matrix instructions.

## Research and official documentation

### Hardware-aware diagnosis before another kernel rewrite

[Berkeley Lab's Roofline methodology](https://crd.lbl.gov/assets/Uploads/SciDAC19-Poster-Roofline-SWWilliams.pdf) relates attained throughput to data movement and hardware ceilings, including integer work. For this project, logical GEMM FLOPs alone omit substantial exponent, conversion, truncation, and synchronization costs. Use controlled kernel ablations and, where accessible, hardware counters; WebGPU timestamps alone cannot prove a bandwidth bottleneck.

[TVM](https://arxiv.org/abs/1802.04799) combines graph optimizations with hardware-specific scheduling and automated search. The applicable lesson is to specialize stable graph parameters and measure schedules on the actual device. We need not replace our CUDA pipeline with TVM to apply that lesson.

Candidate: specialize activation formats, half mode, residual presence, publication mode, and common K/N/partition values in CUDA before compilation. This can expose constant branches and loop bounds to the compiler while retaining operation order. The driver may already remove some overhead; extra pipeline count and startup cost must be measured.

### Attention fusion has limits here

[FlashAttention](https://arxiv.org/abs/2205.14135) reduces attention memory traffic using tiling and on-chip intermediates. Our existing local fusion already applies the relevant data-reuse principle. It is not a literal FlashAttention implementation. The native reconstruction explicitly rounds scores and probabilities and uses a prescribed reduction order; substituting online softmax does not automatically preserve those results. Global attention is only about 7.84 ms in this profile, so further global fusion has limited immediate leverage.

### Native half arithmetic and matrix units are separate capabilities

[Chrome's f16 documentation](https://developer.chrome.com/blog/new-in-webgpu-120) explains explicit feature negotiation and WGSL `enable f16`. Our adapter supports this feature, but enabling it does not transform existing f32 kernels or provide Tensor Core access. Native f16 storage/conversion merits an isolated experiment. It cannot replace grouped F13/F24 arithmetic without validation.

[Dawn's subgroup-matrix proposal](https://dawn.googlesource.com/dawn/+/refs/heads/main/docs/dawn/features/subgroup_matrix.md) describes matrix primitives, supported configurations, and experimental feature negotiation. The document lists Vulkan/Metal paths and no D3D support, and does not define ULP bounds for matrix arithmetic. The actual local feature probe is the stronger evidence for this deployment: the feature was not exposed. Treat matrix hardware as a capability-gated research backend, not a portable Pages dependency or an exact numerical replacement.

[Numerical Behavior of NVIDIA Tensor Cores](https://eprints.maths.manchester.ac.uk/2784/) studies architecture-dependent arithmetic in Volta, Turing, and Ampere. It supports caution about assuming conventional accumulation semantics, but does not independently validate this project's Ada-style F13/F24 reconstruction or Blackwell behavior. Existing parity checks establish agreement with the reconstruction; original-driver capture parity remains unverified.

## Recommended sequence

1. **Instrument the real end-to-end frame.** Separate input capture/CPU conversion, uploads, preprocessing, graph encoding/submission, GPU execution, head/output readbacks, and presentation. Add timestamps without forcing one submission per graph operation. Retain the old profiler for detailed diagnostics. Measure cold startup separately from warm frames.
2. **Remove unnecessary CPU waits and head downloads.** Add an output-only path that composes before optional head readback. Keep the existing API/reference path available. Prepared plans can explore several bounded in-flight batches, using queue ordering for dependencies. Audit uniform-buffer writes, resource retirement, cancellation latency, and error cleanup before removing waits; streamed execution has different lifetime requirements.
3. **Cache the exact noise field.** Noise depends on padded x/y and seed, not the image, camera, history, or conditioning. Split noise generation from feature assembly, keep a bounded GPU cache keyed by padded geometry and seed, and preserve all current arithmetic. Three noise lanes can use half storage because the existing code already publishes them to half. Changing the seed invalidates the cache. The 17.13 ms preprocessing cost is an upper bound on potential savings, not the expected improvement: feature assembly and cache reads remain.
4. **Diagnose and specialize the dominant GEMMs.** Use real shapes/row counts and separate compute, decoding, shared-memory, and publication costs. Study subgroup-assisted whole-word output packing in the established one-output-per-thread tiles independently from four-output compute tiling. That isolates contention without simultaneously changing the accumulator/register layout. Require exact checkpoint parity and full-frame wins at both target resolutions.
5. **Build a GPU-resident frame path for interactive 3D.** A shared WebGPU device/render target can avoid the current WebGL canvas-to-CPU-to-WebGPU route. Retain output/history as GPU resources and download only for export or inspection. This requires renderer integration and colour/alpha validation; camera movement still requires valid motion/history handling. It improves frame overhead but cannot eliminate the network's roughly 451 ms GEMM cost.
6. **Evaluate approximate arithmetic only with a quality harness.** Ordinary f32 accumulation is a useful diagnostic for the cost of exact emulation and a possible opt-in mode. Evaluate native f16 and any future matrix primitives separately. Compare portraits, game scenes, fine textures, controls, and valid temporal sequences using float error, clipping/non-finite checks, visual differences, and temporal stability. Better numerical precision does not by itself guarantee equivalent network output. Exact remains the baseline.

## Acceptance criteria

Exact optimizations must preserve all 75 checkpoint comparisons on representative inputs, final head/output comparisons, temporal-input changes, odd dimensions, cancellation/reuse, and both prepared/streamed paths where affected. Existing original-driver parity limitations remain explicit.

Performance experiments should alternate baseline/candidate, use more warm samples than the prior two-frame reports, record browser/adapter/features, and report distributions at 720p and 1080p. Use profiling-disabled frame runs for speed claims. Distinguish engine timing from the complete user-visible frame. A quality-changing backend also needs measured image and temporal evidence before it can be recommended.

The immediate implementation recommendation is measurement plus an output-only, bounded asynchronous execution path, followed by exact noise caching. For a substantially larger gain, investigate GEMM instruction cost and compiler specialization; keep approximate arithmetic as a measured alternative rather than assuming it is the answer.
