# OpenDLSS-NR for WebCuda

CUDA reconstruction of [OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR)'s native implementation, compiled into WebGPU compute shaders by [cuda-webshader](https://github.com/SamG-Coder/cuda-webshader).

The source repository has no `.cu` files at the pinned revision. This project reconstructs CUDA from its native GLSL kernels, PTX generators, C++ graph, weight loader, numerical reference, and specifications. **No code from `ports/browser-webgpu` is used.**

All inference and frame-processing kernels are authored in `kernels/*.cu` with shared numerical code in `kernels/numeric.cuh`. JavaScript handles model decoding, graph scheduling, GPU resources, and the browser interface. WGSL is generated, never hand-maintained.

## Status

Implemented: the full 71-block graph (including block 39's transition-only operation), dense and expert FFNs, 512-channel branch FFNs, global ViT attention, shifted windows, encoder/decoder transitions, preprocessing, motion-based history reprojection, temporal composition, model loading with SHA-256 verification, and a native-fixture parity runner.

This is a **scalar correctness backend**, with 653 graph dispatches per frame. It emulates Ada's grouped F13/F24 accumulation, preserves half and FP8 publication points, and keeps decoded activations in f32 buffers. Scheduling and model decoding are optimized with bounded command batches and lookup tables; the scalar kernels are not realtime. Large resolutions can exceed the device's buffer limit and are rejected before inference.

**Real-model browser inference has passed** with a locally supplied `nvngx_dlssnr.dll` version 310.8.0.0: all 153 tensors loaded, and all 653 dispatches completed at source resolution on an RTX 5080 in Edge, including 720p and 1080p. Original-capture parity remains unverified; successful inference does not establish image equivalence to NVIDIA's implementation. Model weights and original captures are not distributed.

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

Choose **Image** or **3D object**, then click **Render with NR**. Images retain their original dimensions by default. Custom image sizing preserves the aspect ratio; 3D captures use the specified width and height. The NR output and exported PNG exactly match that selected resolution. The internal padded field is at least 320 pixels per axis but is cropped back to the selected dimensions. There is no silent downscaling: unsupported dimensions and GPU buffer limits produce an explicit error. Each axis must be at least 33 pixels. This scalar backend can be slow at large sizes; cancellation takes effect between GPU dispatches.

Preview and NR memory checks are separate. Before model setup, no guessed WebGPU storage-buffer limit is imposed. After setup, the actual NR device limit determines whether inference is available; exceeding it leaves the source preview usable. A 512 × 512 frame requires a 144 MiB intermediate NR buffer and has been verified through preview and real inference on the test GPU. The preview has its own 16-megapixel allocation guard and, for 3D, checks the WebGL texture limit. These limits are distinct from model geometry size and total GPU VRAM.

NR now explicitly requests the adapter's supported storage-buffer and allocation limits instead of WebCuda's conservative 256 MiB default. On the tested RTX 5080 / Edge adapter, this permits buffers up to almost 2 GiB. Real-model benchmarks with the packed cache complete on repeat renders at **1280 × 720** (about 1.88 seconds) and **1920 × 1080** (about 3.70 seconds), with matching source/output dimensions and no rescaling. The reusable temporary-buffer pool is bounded to 256 MiB; larger retired buffers are destroyed after GPU completion. Actual hardware limits still apply.

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

The renderer now uploads **packed FP8/FP16 matrices once per loaded model**, then keeps them in WebCuda GPU buffers. Matrix values are unpacked by `nr_gemm_packed` in CUDA; no additional quantization is applied. Small scale and attention-prior buffers remain float32. The complete GPU model cache is **144.5 MiB**, compared with 556.0 MiB for all expanded weights. The previous renderer streamed those expanded weights rather than keeping all 556 MiB resident at once; caching adds persistent memory alongside the activation buffers.

The first render populates the cache lazily. Repeat renders upload only changing frame inputs and perform no matrix, scale, or prior decoding. The cache survives image, camera, controls, and resolution changes. Loading a replacement model disposes the previous renderer and its cache. API users can call `renderer.clearWeightCache()` while idle to reclaim model buffers; the next render repopulates them. Device loss requires creating a new renderer, and a page reload requires selecting the DLL again. This is an in-memory cache, not browser disk storage.

On the tested RTX 5080 / Edge system, the final 720p benchmark took **2.31 s on the first render and 1.88 s on repeat renders**, versus 2.28 s before packed caching. At 1080p it took **4.11 s initially and 3.70 s on repeat renders**, versus 3.56 s before: scalar GPU unpacking is a tradeoff, not a universal speedup. Both full float32 outputs match the preceding implementation byte for byte at both resolutions. Measurements use a synthetic gradient and exclude model import, shader setup, and UI image conversion. See the [packed-cache report](https://github.com/SamG-Coder/OpenDLSS-NR-WebCuda/blob/main/reports/performance-packed.md) and [earlier scheduling optimization](https://github.com/SamG-Coder/OpenDLSS-NR-WebCuda/blob/main/reports/performance.md).

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

Results are written to ignored `reports/performance.json`. Set `NR_REPORT` to save another report and `NR_COMPARE` to a baseline report to require matching network-head and final-output SHA-256 hashes. Compare on the same hardware and browser. The benchmark uses a generated gradient and never distributes the DLL or its tensors.

## WebCuda change

WebCuda is vendored under `vendor/webcuda` so a checkout is self-contained. The vendored compiler includes this change developed against WebCuda:

`25673c6` — **Support CUDA scalar float/integer bit reinterpretation intrinsics**

It adds `__float_as_uint`, `__uint_as_float`, `__float_as_int`, and `__int_as_float` to the compiler and CPU oracle, with argument validation and regression tests. The patch is in `patches/`. Its full existing test suite passed: **762 tests**. The original development commit is identified here for provenance; its patch is included.

`0b581fb` — **Allow callers to request adapter-sized storage buffers** adds the explicit `useAdapterBufferLimits` runtime option. NR enables it; other WebCuda callers retain their previous defaults. The change is included in the vendored runtime and as patch 0002. The updated WebCuda suite passed **765 tests**.

## Provenance and license

Original work in this project is **Copyright (c) 2026 SamG-Coder**, licensed under the [MIT License](LICENSE). Credit and retained notices for upstream-derived portions and dependencies are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); the source mapping is in [PROVENANCE.md](PROVENANCE.md). Native reference files are not distributed; optional native tests read a separate checkout selected by `NR_NATIVE_SOURCE`. No model weights are distributed. This project is not affiliated with NVIDIA.

## GitHub Actions and Pages

The workflow runs the public-file audit, host tests, CUDA compilation, and a browser smoke test against the static site at a repository subpath. Pushes to `main` deploy the tested artifact to GitHub Pages; pull requests build and test without deployment. CI does not receive DLLs or weights. Hosted runners do not establish hardware GPU or native-model parity.

`npm run build:site` creates an allowlisted `site/` containing only the browser app, generated kernels, required libraries, and license/attribution files. `npm run test:site` verifies this artifact (install Playwright Chromium first or set `NR_BROWSER`).
