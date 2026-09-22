# Source provenance

Original work in this repository is copyright (c) 2026 SamG-Coder. See [LICENSE](LICENSE) for the project license and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for retained upstream notices.

- Native specification/source: `https://github.com/maanHimself/OpenDLSS-NR`, revision `9d08f41` (full revision recorded in `reports/validation-summary.json`). The reconstruction was built from native sources and documentation. The browser port is inspected separately for performance comparisons; its shader implementations are not incorporated or redistributed.
- Compiler/runtime: `https://github.com/SamG-Coder/cuda-webshader`, revision `db1e3d2`. The bit-reinterpretation and adapter-buffer-limit patches are now upstream (`390f6c1`, `166056c`), alongside native local/shared half support (`5aa80e3`) and 32-bit CUDA `__clz` support (`b8e8cd0`) and opt-in adapter workgroup storage limits (`7cc771c`) and bounded loop unrolling/component-wise vector lowering (`f82d847`) and vector extrema reductions (`db1e3d2`). Historical patches remain in `patches/`.
- Optional 3D input preview: `three` 0.186.0 (MIT), installed from npm and served locally. Uses the official GLTFLoader, OrbitControls, Draco/KTX2 loaders and Meshopt decoder. Three.js rasterizes an input frame; NR inference remains in the reconstructed CUDA kernels.

| Reconstructed code | Native source |
| --- | --- |
| `kernels/numeric.cuh`, `kernels/gemm.cu` | `src/numeric.h`, `src/reference.cpp`, `shaders/common.glsl`, `shaders/gemm_fp8.comp`, `shaders/gemm_f16.comp` |
| `kernels/attention.cu` | `shaders/window_normalize.comp`, `shaders/window_attend.comp`, `shaders/global_normalize.comp`, `shaders/global_attend.comp` |
| `kernels/ops.cu` | `shaders/ops.comp`, `demo/shaders/nr_composite.comp`, `src/main.cpp` output comparison |
| `kernels/frame.cu` | `shaders/preprocess.comp`, `demo/shaders/nr_preprocess.comp`, `demo/shaders/nr_common.glsl`, `docs/frame.md` |
| `src/geometry.js`, `src/graph.js` | `src/nr_graph.cpp`, `docs/network.md` |
| `src/model.js` | `src/nr_model.cpp`, `docs/weights.md` |
| `src/dll-model.js`, `scripts/extract-model.mjs` | Static inspection of the locally supplied `nvngx_dlssnr.dll` 310.8.0.0 PE resource and serialized tensor records, validated against the native graph layouts |
| `src/parity.js` | fixture contract in `src/main.cpp`, `docs/numerics.md` |

The optional native test generator reads `numeric.h` and `reference.cpp` from a separate user-supplied upstream checkout (`NR_NATIVE_SOURCE`). Neither those reference files nor the upstream project are distributed in this repository or the Pages site.

The portable kernel decomposition, software preprocessing functions, JavaScript scheduler, UI, and tests were written for this reconstruction. No performance or whole-model bit-exactness claims from upstream are inherited by this project.

The tested DLL has SHA-256 `e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e`. Its `WEIGHTS_HT` resource contains 153 tensors across 71 blocks, with 147,683,778 payload bytes. Neither the DLL nor its weights are committed or redistributed.
