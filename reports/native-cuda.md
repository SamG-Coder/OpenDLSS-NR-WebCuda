# Production CUDA sources compiled with NVCC

This benchmark executes **this project's CUDA kernels**, not the upstream PTX implementation. It traces the production `NeuralRenderer` to retain kernel selection, automatic tile selection, scalar arguments, packed model uploads, and activation-buffer reuse. The generated CUDA sources are compiled into separate translation units without changing their arithmetic. No upstream numerical kernels are included.

The harness and all model/input/output bytes are generated under ignored `build/native-benchmark/`. Nothing is added to the public site. The production renderer is unchanged.

## Reproduction on Windows

1. Run `npm run build` to generate CUDA variants.
2. Run `node scripts/prepare-native-benchmark.mjs`. `NR_MODEL` can specify the local model directory (default `models/nr`, containing `manifest.json` and `model/`).
3. In an MSVC x64 developer shell, run `node scripts/compile-native-benchmark.mjs`. `NVCC` overrides the CUDA compiler path and `NR_CUDA_ARCH` overrides the default `native` target. Four compilation processes run concurrently. The compiler uses `-O2 --fmad=false`, without fast math; the unsupported-host-compiler override accommodates the locally installed VS 18 toolchain.
4. From the repository root, run `build/native-benchmark/runner.exe 10`. The argument is the measured frame count; three untimed executions precede measurement at each resolution.
5. Run `node scripts/check-native-benchmark.mjs` to compare native heads with the browser using identical inputs and collect five browser wall-clock samples per resolution. `NR_BROWSER` overrides the Edge executable.

## Measurement scope

The two cases are 1280 x 720 and 1920 x 1080, with the production graph's padding. Both use real model weights and deterministic sine-wave input features. The runner compiles 51 kernel variants and executes 467 network dispatches per frame. The numerical lookup initializer is executed once outside timing.

Native timings use CUDA events around kernel launches, excluding allocation, weight upload, input reset, and head readback. The graph reuses the feature buffer, so its original contents are restored before **every** execution. Browser timings include input upload and JavaScript scheduling but disable head readback. These two timing scopes are not identical; their difference must not be attributed entirely to the shader compiler.

Output validation compares every padded head element bit-for-bit after expanding its packed format and rejects nonfinite values. This is a full-head comparison for two synthetic inputs, not a claim of validation across arbitrary scenes.

## Local measurements, 2026-09-22

RTX 5080, driver 616.64, CUDA toolkit 13.3, native target sm_120. Ten native samples after three untimed executions:

| Resolution | Native reported median* | Native minimum |
|---|---:|---:|
| 1280 x 720 | 157.806 ms | 157.190 ms |
| 1920 x 1080 | 334.130 ms | 331.411 ms |

*The executable reports the upper middle sample for an even sample count.

The matched browser runs had median wall times of 195.0 ms at 720p and 416.9 ms at 1080p. The native heads matched the browser bit-for-bit: 4,128,768 values at 720p and 8,847,360 at 1080p, with zero mismatches and zero nonfinite values. Browser wall time was approximately 24-25% higher than native GPU time; the different timing scopes prevent interpreting this as a pure compiler overhead measurement.

Earlier measurements of **upstream's separate native implementation** on this GPU were 3.096 ms and 6.025 ms, respectively. Those results use its PTX tensor-core route and its own synthetic inputs. They are not measurements of this project's kernels.

Compiling the current software arithmetic implementation to native CUDA does not reproduce upstream's tensor-core performance. This isolates an important limitation of the kernel approach itself; it does not establish that all remaining browser overhead is unavoidable.
