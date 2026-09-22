# Exact native half GEMMs

Measured locally on RTX 5080, Edge 153, with the same local model and generated gradient as the earlier performance tests. Six warm samples per configuration alternate order; times are host wall time for the output-only renderer, including composition/readback. Setup is excluded. Both paths retain specialization, packed activations, noise caching, and four batches in flight.

| Resolution | Original median | Half median | Reduction |
| --- | ---: | ---: | ---: |
| 1280 x 720 | 447.40 ms | 407.45 ms | 8.9% |
| 1920 x 1080 | 1035.60 ms | 964.75 ms | 6.8% |

720p original samples: 464.2, 449.5, 440.3, 451.4, 445.3, 439.7 ms. Half: 407.9, 409.5, 407.0, 402.1, 400.4, 410.3 ms.
1080p original: 1013.3, 1024.3, 1058.9, 1043.0, 1057.0, 1028.2 ms. Half: 956.9, 958.9, 970.6, 973.7, 979.6, 948.4 ms.

An earlier noisier paired run measured 506.8 -> 470.45 ms at 720p and 1079.2 -> 1050.3 ms at 1080p. A naive half-product implementation without shared operand exponents regressed; it was replaced. These are hardware-specific measurements, not a universal speed guarantee, and do not close the gap to the separate upstream WebGPU implementation.

## Exactness and implementation

Original CUDA source is in `kernels/gemm-half.cu`. Two adjacent K operands share a half2. Multiplying both operands by four keeps every nonzero finite E4 activation product with |weight| <= 9 in the normal binary16 range [2^-14, 64512]. At most eight significant bits are needed. Integer F13 accumulation, exponent selection, residual handling, partition boundaries, SiLU, and quantization remain unchanged. Operand exponents are computed once during shared-tile loading and summed as exact half2 integers. This uses shader-f16, not tensor cores.

Each persistent FP8 matrix receives a one-time bound check. Other matrices and devices without shader-f16 fall back. `nativeHalf: false` restores the previous path.

Validation: 37 renderer host tests; 281 GPU checks (131 generated pipelines), including every specialization with tail guards; all 41,656 signed bounded E4 product pairs; changing-image/history/motion/control/feature prepared-execution comparisons; all 75 captured network boundaries, head, and output matching the saved reference for two 128 x 96 runs. All six samples at each of 720p and 1080p produced identical final output hashes between configurations.

Reproduce with local `NR_DLL` and `NR_BROWSER`, `npm run build`, then `NR_HALF=1` and `npm run benchmark:execution`. No DLL or model weights are distributed. Raw reports stay local.
