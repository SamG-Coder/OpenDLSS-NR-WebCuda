# Separate upstream WebGPU comparison

Measured 2026-09-22 on RTX 5080 / Edge 153. The upstream browser implementation is WebGPU, not CPU inference. Its JavaScript CPU numerics are an arithmetic reference, not a complete browser CPU renderer.

## Result

Their WebGPU port is about three times faster in this controlled network benchmark. Network-head values match bit for bit at every tested resolution and on every paired iteration.

| Valid source resolution | Our median | Upstream median | Upstream speed ratio |
| --- | ---: | ---: | ---: |
| 512 x 512 | 156.65 ms | 51.80 ms | 3.02x |
| 1280 x 720 | 453.35 ms | 143.30 ms | 3.16x |
| 1920 x 1080 | 1038.65 ms | 344.85 ms | 3.01x |

The requested 720p upstream samples were **143.9, 142.2, 142.7, 144.6, 141.2, and 145.3 ms**. Our corresponding samples were **457.7, 458.4, 450.6, 454.0, 449.0, and 452.7 ms**. Both used a 1344 x 768 padded field for a valid 1280 x 720 source. No downscaling occurred.

## Comparison contract

- Upstream revision: `9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1`, confirmed to be upstream HEAD when tested. The checkout was unmodified.
- Our renderer revision: `54681f81efec4c66412ea064d681f3085abdc76a`, with exact specialization enabled.
- Both used the same locally extracted model stage files. Our loader verified stage hashes.
- A deterministic gradient and our preprocessing kernel generated one common float32 feature array per resolution, outside timing. Both received exactly those bytes. This deliberately removes differences in noise implementations from the comparison.
- Timing starts before uploading features and ends after float32 network-head readback. It includes graph scheduling and GPU completion. It excludes preprocessing, final image composition, presentation, PNG encoding, model loading, and shader setup.
- One untimed warmup per renderer, followed by six interleaved samples per renderer with alternating execution order. Hashing and element comparison are outside timing. Both renderers remained resident during each resolution's comparison.
- Both ran on the NVIDIA Blackwell adapter in the same Edge process. Each used its own normal device configuration; this compares the implementations as supplied, not kernels under equalized features or limits.
- Every output element was finite. There were zero bit mismatches and zero maximum absolute head error at all three resolutions on all seven paired iterations. This is cross-port agreement for the tested features, not an original-driver capture validation or a temporal-quality test.

The 720p head hash was `04acbfd71bf25ee2340da48d61bbccaca7e40bd2081fc0e9682f3921dd77375a` for both. At 1080p it was `ffb9462478b9af9da50c79b84bc841305b1c9035e1cb5c50dcf43faa5341951c` for both.

These numbers should not be directly compared with the earlier output-only frame benchmarks: those upload source pixels, preprocess, compose, and download the final image; this test uploads the larger feature tensor and downloads the padded head.

## Observed implementation differences

Upstream requested `shader-f16` and 32768 bytes of workgroup storage; our runtime used 16384 bytes and did not request `shader-f16`. Upstream recorded 451 graph dispatches; ours recorded 529.

The upstream [matrix variant composition](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1/ports/browser-webgpu/src/matmul/variants.js) includes vector loads/reductions, shared exponent data, unrolled groups, word-oriented weight loads, weight/activation lookup tables, and native half products. Its [bounded-half transform](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1/ports/browser-webgpu/src/matmul/bounded-half.js) relies on validated model-weight bounds and scales operands to keep products exactly representable as normal half values. This preserves the later emulated accumulation rather than replacing it with ordinary half accumulation.

These differences identify promising experiments. The benchmark does not isolate how much time each saves; attributing the entire gap to f16, shared memory, or dispatch count would be unsupported.

Memory also differs. Upstream reported approximately 1905 MiB of allocated activation tensors at 720p, versus approximately 431 MiB in our reusable graph plan. At 1080p those figures were approximately 4079 MiB versus 925 MiB. These counters cover different allocation strategies and exclude some resources; they are not total VRAM measurements. Both weight caches were approximately 144.5 MiB.

Observed setup was about 29 seconds for upstream and 8.4 seconds for our renderer in this session. These are not equivalent startup boundaries: upstream creation loads weights and records/compiles the graph, while our model load was measured separately (about 259 ms), and our first run includes lazy weight upload and graph preparation. Browser/driver cache state also affects both. The headline result concerns warm inference only.

## Next optimization direction

Investigate native half/vector support in WebCuda and an independently authored CUDA GEMM using provably exact bounded products, retaining the existing accumulation and publication rules. Validate weight bounds and every checkpoint before enabling it. Compare larger shared-memory tiles separately. Do not substitute approximate accumulation merely because the upstream path is faster.

No upstream browser implementation code was copied into our renderer, and neither renderer nor WebCuda was changed for this comparison.

## Reproduce

Set `NR_UPSTREAM` to an unmodified upstream repository, `NR_MODEL` to the local extracted model directory, and optionally `NR_BROWSER` to Edge/Chrome. Run `npm run benchmark:upstream`. The harness mounts those paths only on a temporary localhost server, imports the external implementation separately, and writes ignored `reports/upstream-comparison.json`. `NR_REPORT` overrides the report path. The upstream checkout, model files, and raw outputs are not published.
