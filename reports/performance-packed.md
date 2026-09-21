# Persistent packed model cache

Measured 2026-09-22 using the same local RTX 5080 / Edge setup, NR 310.8.0.0 model, gradient, seed, and controls as the [previous report](performance.md). The preceding implementation is commit `24d6126`. No proprietary binaries or tensors are distributed.

## Representation and lifetime

The native matrix codes are rearranged once into row-major packed words, without converting FP8 to FP16 or requantizing anything. Each `uint32` holds four FP8 values or two FP16 values. `nr_gemm_packed` extracts and decodes the original values in CUDA while preserving the existing accumulation and publication order. The float-input `nr_gemm` remains available as a reference and for synthetic model providers.

Matrices, float32 scales, and float32 attention priors are cached in renderer-owned WebCuda buffers. The total is **151,526,056 bytes / 144.51 MiB**, about **74% smaller** than the complete 556.01 MiB expanded representation. This is not a claim of a 74% reduction in total live GPU memory: the previous backend streamed weights and freed them during each render, whereas the new cache stays resident alongside activation buffers.

The first render populates 639 model buffers lazily. Subsequent renders reuse them at any source resolution. Cancellation retains valid cached weights and discards pending commands; inference-owned working buffers are released. `clearWeightCache()` releases the model cache while idle; disposal or replacing the renderer releases it too. There is no disk cache, network upload, or persistence across reloads.

## Timing and uploads

Three final runs per resolution; run 1 populates the cache, runs 2–3 reuse it. These are whole renderer API times, including inference and cleanup, excluding DLL parsing, shader setup, output hashing, and UI conversion.

| Resolution | Previous median | First packed render | Repeat packed renders |
| --- | ---: | ---: | ---: |
| 1280 × 720 | 2283.9 ms | 2310.5 ms | 1878.1, 1881.9 ms |
| 1920 × 1080 | 3556.4 ms | 4114.5 ms | 3696.1, 3698.5 ms |

Repeat 720p renders are about 18% faster than the preceding implementation; repeat 1080p renders are about 4% slower. GPU unpacking adds arithmetic to the scalar GEMM, so storage and transfer savings do not guarantee faster execution at every resolution. Both FP8 and FP16 matrices remain packed in the final implementation.

First-render CPU model preparation measured about 401 ms. Repeat renders measured zero matrix/vector/prior preparation time at the benchmark timer's resolution (the single compositor blend scalar is still read on the host).

| Resolution | First-render data upload | Repeat-render data upload |
| --- | ---: | ---: |
| 1280 × 720 | 166,271,660 bytes | 14,745,604 bytes |
| 1920 × 1080 | 184,703,660 bytes | 33,177,604 bytes |

Repeat uploads are exactly the source RGBA float32 image plus the four-byte zero placeholder. Model uploads are **zero**; uniform uploads are tracked separately. The benchmark now asserts this invariant and records cache bytes and data uploads in its ignored JSON report.

## Verification

All final runs matched the previous full float32 network-head and composited-output hashes at both resolutions. Output dimensions remain source-sized. The 720p hashes are in the preceding report; 1080p hashes are:

- Head: `ffb9462478b9af9da50c79b84bc841305b1c9035e1cb5c50dcf43faa5341951c`
- Output: `43d9ba3263578ed4a5f4b5f2e15d7c8b4e32966e90ce863cc18d6c91f6df2532`

Host tests cover raw code preservation, tensor offsets, batches, padded half columns, and bounds rejection. GPU tests decode all 256 FP8 and 65,536 half patterns, including signed zero and non-finite encodings, and verify repeated-cache use, zero repeat weight uploads, explicit release, and recovery from a partially populated cache after cancellation. Existing native arithmetic fixtures and capture tests remain enabled. Real-model UI tests cover images, GLB/glTF, controls, cancellation, and export.
