# OpenDLSS-NR for WebCuda

CUDA reconstruction of [OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR)'s native implementation, compiled into WebGPU compute shaders by [cuda-webshader](https://github.com/SamG-Coder/cuda-webshader).

The source repository has no `.cu` files at the pinned revision. This project reconstructs CUDA from its native GLSL kernels, PTX generators, C++ graph, weight loader, numerical reference, and specifications. **No code from `ports/browser-webgpu` is used.**

All inference and frame-processing kernels are authored in `kernels/*.cu` with shared numerical code in `kernels/numeric.cuh`. JavaScript handles model decoding, graph scheduling, GPU resources, and the browser interface. WGSL is generated, never hand-maintained.

## Status

Implemented: the full 71-block graph (including block 39's transition-only operation), dense and expert FFNs, 512-channel branch FFNs, global ViT attention, shifted windows, encoder/decoder transitions, preprocessing, motion-based history reprojection, temporal composition, model loading with SHA-256 verification, and a native-fixture parity runner.

This is a **correctness-focused backend**, with 529 graph dispatches per frame and fused local attention and shape-selected GEMM for packed FP8 matrices. It emulates Ada's grouped F13/F24 accumulation, preserves half and FP8 publication points, and stores published activations as packed FP8/FP16. Prepared execution plans, packed model caching, and working-buffer reuse reduce overhead; rendering is still not realtime. Large resolutions can exceed the device's buffer limit and are rejected before inference.

**Real-model browser inference has passed** with a locally supplied `nvngx_dlssnr.dll` version 310.8.0.0: all 153 tensors loaded, and all 529 dispatches completed at source resolution on an RTX 5080 in Edge, including 720p and 1080p. Original-capture parity remains unverified; successful inference does not establish image equivalence to NVIDIA's implementation. Model weights and original captures are not distributed.

The browser workspace accepts images or local GLB/glTF objects. Objects are rendered with Three.js into an sRGB display proxy, then processed by the same CUDA-derived NR network. This is a still-frame workflow with an interactive camera; it does not reproduce the original Filament renderer, its HDR display pipeline, or its custom postprocessing styles. Temporal users supply motion vectors; this project does not generate scene motion vectors.

## Live app and attribution

[Open the browser app](https://samg-coder.github.io/OpenDLSS-NR-WebCuda/) · [Source repository](https://github.com/SamG-Coder/OpenDLSS-NR-WebCuda)

This project is based on the native implementation and specifications of **[maanHimself/OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR)**. Credit for the original reverse-engineered network and numerical specification belongs to that project. This repository contains our CUDA reconstruction and browser workspace, not a copy of the original project or its browser port. [WebCuda](https://github.com/SamG-Coder/cuda-webshader) compiles the CUDA kernels for WebGPU. Upstream license notices are retained.

**No NVIDIA DLLs, drivers, SDK libraries, model weights, extracted tensors, or native captures are included in this repository or the Pages site.** Supply your own compatible DLL through the local file picker. The hosted app reads it in your browser and does not upload it. There is no server-side inference or automatic model download.

## Run

Requires Node.js 20+ and a WebGPU-capable browser:

```powershell
npm install
npm run build
npm start
```

Open **http://127.0.0.1:8090**. Under **Model → Choose NR DLL**, choose your `nvngx_dlssnr.dll` (tested with version 310.8.0.0). The browser reads its embedded `WEIGHTS_HT` resource, validates all 153 tensor layouts, and uses the model with the CUDA-derived WebGPU kernels. No extraction tool or native DLL execution is needed. An incompatible DLL produces an error and preserves any previously loaded model.

Alternatively, expand **Extracted model folder** and select a directory containing:

```text
manifest.json
model/
  <stage files named by manifest.stages[].file>
```

Choose **Image** or **3D object**, then click **Render with NR**. Images retain their original dimensions by default. Custom image sizing preserves the aspect ratio; 3D captures use the specified width and height. The NR output and exported PNG exactly match that selected resolution. The internal padded field is at least 320 pixels per axis but is cropped back to the selected dimensions. There is no silent downscaling: unsupported dimensions and GPU buffer limits produce an explicit error. Each axis must be at least 33 pixels. This backend can be slow at large sizes; interactive cancellation takes effect between bounded GPU batches.

Preview and NR memory checks are separate. Before model setup, no guessed WebGPU storage-buffer limit is imposed. After setup, the actual NR device limit determines whether inference is available; exceeding it leaves the source preview usable. A 512 × 512 frame requires a 54 MiB intermediate NR buffer and has been verified through preview and real inference on the test GPU. The preview has its own 16-megapixel allocation guard and, for 3D, checks the WebGL texture limit. These limits are distinct from model geometry size and total GPU VRAM.

NR now explicitly requests the adapter's supported storage-buffer and allocation limits instead of WebCuda's conservative 256 MiB default. On the tested RTX 5080 / Edge adapter, this permits buffers up to almost 2 GiB. Real-model benchmarks with the packed cache complete on repeat renders at **1280 × 720** (about 0.44 seconds on the output-only path) and **1920 × 1080** (about 1.01 seconds on the output-only path), with matching source/output dimensions and no rescaling. A prepared graph retains reusable activation slots within a 1 GiB budget; larger plans fall back to streamed execution. A separate 256 MiB pool handles frame buffers and the streamed fallback. Both reset when resolution changes. Actual hardware limits still apply.

For GLB, select the file. For glTF, select the complete folder or the glTF plus its buffers and textures together. The viewer includes orbit/zoom/pan, object framing, three lighting presets, exposure, background colour, field of view, and animation clip/time selection for a still pose. Draco, Meshopt, and KTX2 decoder support is configured locally through Three.js. Browser tests cover ordinary GLB and external-buffer glTF files; compressed-asset decoding has not been separately fixture-tested. All app libraries are served locally after `npm install`; asset loading never fetches a model's missing files from a remote server.

The inspector exposes tone, structure, automatic masking, skin structure (follow or custom), style conditioning, and noise seed. Optional temporal inputs accept a previous output image at the exact selected resolution and little-endian float32 RGBA motion vectors (XY=current-to-previous UV displacement, Z=validity). Without motion, history must already be aligned. Image history is an 8-bit convenience path; use the JS API for full float precision. A precomputed feature file can override preprocessing: little-endian float32, 16 lanes per padded pixel, with the required byte length shown in the UI. Invalid sizes and non-finite float data are rejected. Skin overrides require automatic masking. Style is network conditioning, not an extra colour-grading pass.

Compare input/output side by side or with a wipe slider. Export the output as PNG and the selected configuration as JSON. Configuration export records settings and filenames; it does not bundle assets or implement session import. Files remain on the device and are not uploaded.

Tone and structure default to **1**, with automatic structure masking enabled, matching the native demo's `NrControls`. Setting both to zero produces a nearly unchanged image in the tested examples. The initial browser implementation incorrectly defaulted both controls to zero; the earlier portrait and GTA V comparisons used those settings. Style remains zero (no style preset).

## Validation

```powershell
npm test
npm run build
# Optional: use a separate upstream checkout for native reference comparisons.
# No upstream checkout or native reference source is bundled here.
$env:NR_NATIVE_SOURCE = 'C:/path/to/OpenDLSS-NR/src'
# Generate native GPU fixtures and compare GEMMs to that reference:
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/test-native.ps1
# Pass -VcVars if your Visual Studio installation differs from the default.
$env:NR_BROWSER = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
npm run test:gpu
# Optionally exercise the real DLL file picker and full inference:
$env:NR_DLL = 'C:/path/to/your/nvngx_dlssnr.dll'
npm run test:ui
npm run test:real
# Optional full-resolution regression (requires the same local DLL):
$env:NR_WIDTH = '1280'
$env:NR_HEIGHT = '720'
npm run test:real
```

Alternatively install Playwright's browser with `npx playwright install chromium` and omit `NR_BROWSER`. Without native fixtures, the browser suite reports that native comparison was not run; it still runs numerical and synthetic-graph checks.

Validation covers:

- All 65,536 half bit patterns for half publication and E4M3 conversion, including signed zero, ties, saturation, and NaN publication.
- FP8 and f16 matrix multiplication, batches, residual seeds, and split-K; native CUDA is checked against the original independent CPU arithmetic reference.
- Native CUDA versus WebGPU for normalization, window/global attention, softmax, pooling, merging, preprocessing, reprojection, and composition.
- Full graph scheduling and all 75 comparable boundaries, using zero weights and deliberately tiny **test-only** geometry.
- Native geometry examples, weight packing, phase continuity, model validation, and fixture rejection behavior.
- DLL resource parsing, corrupt-file rejection, tensor metadata validation, and safe replacement of a loaded model. The optional real-model browser test checks successful completion, PNG availability, and nonconstant output differing from its input; it is not an original-capture parity test.

NaN payloads are excluded from cross-backend component bit comparisons. Real fixture head comparison is strict, including NaN bit patterns and signed zero.

For real captures, select a fixture directory in the browser's validation section. The runner validates lengths and declared checks, requires all 75 boundaries to be captured or explicitly omitted, compares repeated production runs, and compares the instrumented run to production. It supports `boundaries`, `head`, and `output` checks, with the source's one-code tolerance only for RGBA8 captures. Proxy fixtures currently require proxy dimensions equal to source dimensions.

## API

```javascript
import {modelFromDll} from './src/dll-model.js';
import {NeuralRenderer} from './src/engine.js';

const model = await modelFromDll(dllInput.files[0], {onProgress: console.log});
const renderer = await NeuralRenderer.create(model);
const {head, output, geometry} = await renderer.run({
  width, height,
  proxy, // Float32Array, interleaved RGBA, display-code values in [0,1]
  seed: 0,
  conditioning: {style: 0, localTone: 1, localStructure: 1,
                 skinStructure: -1, autoMask: 1},
  onProgress: ({index, total, label}) => console.log(index, total, label),
});
renderer.dispose();
```

The extracted-folder API remains available as `Model.load(readFile)` from `src/model.js`, where `readFile(relativePath)` returns an `ArrayBuffer`. For offline export, `node scripts/extract-model.mjs <dll-path> <new-output-directory>` writes a manifest and hashed stage files using the same DLL parser as the browser. Keep DLLs and model exports local; `models/` is ignored by Git.

`inputFeatures` may replace preprocessing: it must be a `Float32Array` of padded width × padded height × 16. The head is padded RGBA f32; `output` is source-sized RGBA f32 and is returned when a proxy is supplied.

For temporal rendering, pass `history` (previous RGBA output) and `motion` (RGBA floats, xy = current-to-previous UV displacement, z = validity). Without motion, history is assumed already reprojected and its alpha is used as the validity mask. Invalid history falls back to the current proxy. Output is truncated to the half grid like the native compositor.

The deterministic Box–Muller path uses software binary64 for trigonometry and log evaluation through WebCuda. This avoids GPU-dependent WGSL approximation errors seen in testing. Original-driver transcendental parity is still a capture-level question; use recorded `inputFeatures` to isolate network arithmetic from preprocessing.

## Performance

The current output-only UI path measured **436.80 ms at 720p** and **1014.85 ms at 1080p** (medians of six alternating warm samples per configuration). Exact GEMM specialization reduced engine time by **21.7% / 23.3%** against identical settings with specialization disabled (557.70 / 1323.20 ms). Final float32 output hashes match at both resolutions. These timings exclude source capture, display conversion, and PNG export; the UI measures those separately. See [specialization measurements](reports/performance-specialization.md) and the preceding [execution and noise-cache measurements](reports/performance-execution.md).

The renderer now uploads **packed FP8/FP16 matrices once per loaded model**, then keeps them in WebCuda GPU buffers. Matrix values are unpacked by `nr_gemm_packed` in CUDA; no additional quantization is applied. Small scale and attention-prior buffers remain float32. The complete GPU model cache is **144.5 MiB**, compared with 556.0 MiB for all expanded weights. The previous renderer streamed those expanded weights rather than keeping all 556 MiB resident at once; caching adds persistent memory alongside the activation buffers.

The first render populates the cache lazily. Repeat renders upload only changing frame inputs and perform no matrix, scale, or prior decoding. The cache survives image, camera, controls, and resolution changes. Loading a replacement model disposes the previous renderer and its cache. API users can call `renderer.clearWeightCache()` while idle to reclaim model buffers; the next render repopulates them. Device loss requires creating a new renderer, and a page reload requires selecting the DLL again. This is an in-memory cache, not browser disk storage.

The preceding fused-attention implementation rendered the test gradient at **0.61 s for repeat 720p frames** and **1.44 s for repeat 1080p frames**, versus 0.74 s / 1.61 s in the preceding version (about 18% / 11% less time). The two measured 1080p warm runs were 1.38 s and 1.49 s. Full float32 head and final-output hashes match. First renders populate the model cache and measured about 1.25 s / 2.07 s; cold-start speed did not consistently improve. Measurements exclude model import, shader setup, and UI conversion; see the [fused attention report](https://github.com/SamG-Coder/OpenDLSS-NR-WebCuda/blob/main/reports/performance-fused.md).

The default `attentionMode: 'fused'` combines local score calculation, softmax, and weighted output in one CUDA kernel. Eight queries share a window tile using 11,552 bytes of workgroup memory, preserving the native reduction order, shifted-window mapping, and half/FP8 publication points. Packed local outputs use one word store per four channels rather than contended per-byte compare/exchange loops. Global attention retains its tiled kernels. `attentionMode: 'tiled'` selects the previous three-pass path; `'scalar'` selects the scalar reference. GEMM variants share one CUDA template, compiled as 4 × 16, 8 × 8, and 8 × 16 tiles. The default `gemmMode: 'auto'` selects by matrix shape using measurements on the reference RTX 5080; it is a fixed heuristic, not runtime hardware autotuning. Explicit modes `tiled` (4 × 16), `tile8x8`, `tile8x16`, and `scalar` remain available for comparisons. The small FP16 projection operations keep the scalar packed kernel.

The default `specializeGemm: true` uses 41 CUDA-generated specializations of the established packed 8 � 8 / 8 � 16 kernels. Fixed matrix dimensions, strides, operation flags, and storage formats become compile-time constants; row counts remain dynamic. Arithmetic order and publication points are unchanged. A complete configuration key selects the artifact, with the original kernel as fallback. FP16 projections and incompatible modes retain their existing path. `specializeGemm: false` disables selection and loading of the extra pipelines. Specialization adds shader setup work; the measured improvement is warm inference, not startup. No additional weight or activation cache is allocated.

An experimental four-output-per-thread CUDA GEMM is available through `gemmMode: 'multi4x32'`, `'multi8x32'`, `'multi16x16'`, `'multi16x32'`, `'multi32x32'`, and `'multi16x64'`. The optional `'multi-auto'` mode limits it to selected large, narrow matrices. **These modes are not defaults:** correctness passed, but profiling and frame measurements did not establish a consistent improvement across 720p and 1080p. Default setup does not load their pipelines. See the [GEMM experiment report](https://github.com/SamG-Coder/OpenDLSS-NR-WebCuda/blob/main/reports/performance-gemm.md).

`npm run benchmark:gemm` runs a DLL-free GPU timestamp benchmark with deterministic synthetic values and 36 shape/configuration cases derived from the real graph. It compares the existing tiles with all six new variants and the applicable specialization, verifies outputs against the scalar packed kernel, rotates sample order, and reports median GPU times. Set `NR_GEMM_ROWS` to change the default 4096-row cap (1�65536), and `NR_REPORT` to choose an ignored local JSON result file. Set `NR_ABLATE=1` to add a benchmark-only, numerically different FMA diagnostic; it is not parity checked or available for inference. This is a bounded kernel benchmark, not a substitute for real-model frame timing. GPU timestamp support is required.

Intermediate tensors now retain their existing published precision: FP8 outputs use one byte per element, and half outputs use two. Computation still uses the same CUDA arithmetic and rounding points. Unread raw outputs are not written. The largest individual buffer at 720p falls from 504 MiB to 189 MiB; 1080p falls from 1080 MiB to 405 MiB. Input features, frame inputs, and public result arrays remain float32.

A prepared execution plan assigns tensors with disjoint lifetimes to reusable GPU buffers and retains graph bindings. On repeat renders at the same resolution, the tested path creates no working buffers and only two frame-level bind groups, instead of recreating every binding. The plan retains about 431 MiB at 720p or 925 MiB at 1080p, in addition to model weights and frame buffers. A configurable `planCacheBytes` budget defaults to 1 GiB; plans exceeding it use streamed execution. It is a cache budget, not a promise that every device has that much free VRAM.

Prepared execution batches up to 32 graph operations. By default it queues up to four graph batches before awaiting GPU completion (`maxInFlightBatches: 4`, valid range 1�8); set 1 to wait after each batch. `graphBatchSize` accepts 1�64. Smaller bounds allow more frequent cancellation checks from browser events. Streamed execution retains its eight-operation bound and memory-pressure waits. Cancellation cannot interrupt already submitted GPU work.

Preprocessing caches the exact three half-precision noise lanes for the current padded dimensions and seed. The cache uses two packed words per padded pixel (7.875 MiB at 720p, 16.875 MiB at 1080p), retains one field, and resets on seed/resolution changes or `clearWorkspace()`. Image, camera, controls, and history can change without regenerating that field. `cacheNoise: false` uses the original preprocessing kernel. This cache is separate from the plan and workspace budgets.

The UI requests `run({..., readHead: false})`, which returns `head: null` and downloads only the final output. API calls keep `readHead: true` by default. Composition runs before any optional head download. `result.timings` reports host wall-clock phases: setup, graph encoding/host work excluding explicit waits, explicit graph waits, composition/readback completion, and total including cleanup. These phases are not GPU timestamp measurements; capture callbacks are included in graph host work. The UI measures input preparation, engine execution, display conversion, and PNG export, with a breakdown in the result tooltip and `output` canvas's `data-timings` attribute. Its displayed total includes all these phases.

`npm run benchmark:execution` requires the local DLL/browser environment variables and alternates six warm samples per configuration at 720p and 1080p. It compares conservative execution (one batch in flight, original preprocessing, head returned) with the UI path (four batches, noise cache, no head download), checks output hashes, and writes ignored `reports/execution-comparison.json`. Both configurations use the current build; this is not a historical checkout comparison. Set `NR_SPECIALIZE=1` to instead compare specialization off/on with identical cached output-only execution, and `NR_REPORT` to select the local report path.

`activationStorage: 'float'` selects the expanded reference storage. `executionMode: 'streamed'` disables the prepared plan. Frame buffers and streamed tensors use a separate 256 MiB pool with eviction. `workspaceCacheBytes: 0` disables that pool; combine it with `executionMode: 'streamed'` and `cacheNoise: false` to disable all working-buffer retention. `renderer.clearWorkspace()` releases both the plan and pool while idle. Resolution changes reset both; clearing weights also invalidates graph bindings. Cancellation and changing temporal inputs preserve correct reuse.

Use `renderer.run({...inputs, profile: true})` to collect per-dispatch GPU timestamps and totals by kernel in `result.profile`. Unsupported devices return an explicit reason instead of estimated GPU timings. Profiling submits individual timestamped passes, so its wall time is not comparable to normal frame timing. Normal runs do not allocate timestamp queries. The result also reports working-buffer allocation/reuse counts, whether a prepared plan was used, and its retained byte count.

Run your own local benchmark with a compatible DLL:

```powershell
$env:NR_DLL = 'C:\path\to\nvngx_dlssnr.dll'
$env:NR_BROWSER = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
$env:NR_WIDTH = '1280'
$env:NR_HEIGHT = '720'
$env:NR_RUNS = '3'
npm run build
npm run benchmark
```

Results are written to ignored `reports/performance.json`. Set `NR_REPORT` to save another report and `NR_COMPARE` to a baseline report to require matching network-head and final-output SHA-256 hashes. Set `NR_SPECIALIZE=0` to disable GEMM specialization. Set `NR_INFLIGHT=1` for one graph batch in flight, `NR_NOISE_CACHE=0` to disable the noise cache, or `NR_READ_HEAD=0` to omit head download. Set `NR_PROFILE=1` for GPU timing, `NR_GEMM=scalar|tiled|tile8x8|tile8x16|auto` to select matrix kernels, `NR_ATTENTION=scalar|tiled|fused` to select attention kernels, `NR_BATCH=8` to restore smaller prepared batches, `NR_ACTIVATIONS=float|packed` to select activation storage, `NR_EXECUTION=streamed|prepared` to select scheduling, or `NR_WORKSPACE_MIB=0` to disable the buffer pool. `NR_CAPTURE=1` records hashes for all 75 intermediate capture points; when the baseline contains those hashes, `NR_COMPARE` checks them too. Compare on the same hardware and browser, and use profiling-disabled runs for frame-speed comparisons. The benchmark uses a generated gradient and never distributes the DLL or its tensors.

Native half GEMMs are enabled by default (`nativeHalf: true`) when `shader-f16` is available. Each cached FP8 matrix is checked once for decoded magnitudes no greater than 9. Eligible packed-input GEMMs use paired half products and shared operand exponents, retaining the existing F13 accumulation and output publication. Unsupported devices, matrices outside the bound, and other GEMM configurations use the original path. Use `nativeHalf: false` or `NR_HALF=0` with `npm run benchmark` to disable it. Set `NR_HALF=1` with `npm run benchmark:execution` for an isolated off/on comparison. See [measured results and exactness checks](reports/performance-half.md).

## WebCuda change

WebCuda is vendored under `vendor/webcuda` so a checkout is self-contained. The vendored compiler includes this change developed against WebCuda:

`25673c6` — **Support CUDA scalar float/integer bit reinterpretation intrinsics**

It adds `__float_as_uint`, `__uint_as_float`, `__float_as_int`, and `__int_as_float` to the compiler and CPU oracle, with argument validation and regression tests. The patch is in `patches/`. Its full existing test suite passed: **762 tests**. The original development commit is identified here for provenance; its patch is included.

`0b581fb` — **Allow callers to request adapter-sized storage buffers** adds the explicit `useAdapterBufferLimits` runtime option. NR enables it; other WebCuda callers retain their previous defaults. The change is included in the vendored runtime and as patch 0002. The updated WebCuda suite passed **765 tests**.

Native local/shared `__half` and `__half2` support is synced from WebCuda commit `5aa80e3`, pushed to its own repository. That upstream history also contains the two earlier patches as `390f6c1` and `166056c`. Validation: 768 host tests, existing kernel compilation, and 41,656 exact GPU half-product pairs. Kernel half storage-buffer parameters are outside the supported subset.

## Provenance and license

Original work in this project is **Copyright (c) 2026 SamG-Coder**, licensed under the [MIT License](LICENSE). Credit and retained notices for upstream-derived portions and dependencies are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); the source mapping is in [PROVENANCE.md](PROVENANCE.md). Native reference files are not distributed; optional native tests read a separate checkout selected by `NR_NATIVE_SOURCE`. No model weights are distributed. This project is not affiliated with NVIDIA.

## GitHub Actions and Pages

The workflow runs the public-file audit, host tests, CUDA compilation, and a browser smoke test against the static site at a repository subpath. Pushes to `main` deploy the tested artifact to GitHub Pages; pull requests build and test without deployment. CI does not receive DLLs or weights. Hosted runners do not establish hardware GPU or native-model parity.

`npm run build:site` creates an allowlisted `site/` containing only the browser app, generated kernels, required libraries, and license/attribution files. `npm run test:site` verifies this artifact (install Playwright Chromium first or set `NR_BROWSER`).

The [generated WGSL audit](reports/generated-output-audit.md) separates measured compiler-output experiments from CUDA algorithm/layout differences. `npm run audit:generated` is a local diagnostic requiring your DLL; its shader edits do not modify production artifacts.
