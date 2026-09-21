#include "numeric.cuh"
// One scalar output per thread. Reproduces Ada F13/F24 dot grouping, not a
// mathematically equivalent f32 GEMM. Weights are relaid to [batch][K][N].
__global__ void nr_gemm(const float* input, const float* weights, const float* residual, const float* scales, float* raw, float* output, unsigned rows, unsigned K, unsigned N, unsigned batches, unsigned inputStride, unsigned inputBatchStride, unsigned partition, int halfMode, int silu, int hasResidual, int quantize) {
  unsigned i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i >= rows * batches * N) return;
  unsigned col = i % N;
  unsigned batch = i / N % batches;
  unsigned row = i / N / batches;
  unsigned base = row * inputStride + batch * inputBatchStride;
  unsigned group = halfMode != 0 ? 8u : 16u;
  int frac = halfMode != 0 ? 24 : 13;
  int minExp = halfMode != 0 ? -14 : -6;
  float initial = hasResidual != 0 ? nr_half(residual[i] * scales[batch * N + col]) : 0.0f;
  float acc = initial;
  float total = 0.0f;
  for (unsigned kb = 0u; kb < K; kb += group) {
    int e = acc != 0.0f ? nr_exp(acc, -14) : -21;
    for (unsigned j = 0u; j < group; j += 1u) {
      float a = input[base + kb + j];
      float b = weights[(batch * K + kb + j) * N + col];
      if (a != 0.0f && b != 0.0f) e = max(e, nr_exp(a, minExp) + nr_exp(b, minExp));
    }
    if (isfinite(acc)) {
      int sum = (int)truncf(acc * nr_pow2(frac - e));
      for (unsigned j = 0u; j < group; j += 1u) {
        float a = input[base + kb + j];
        float b = weights[(batch * K + kb + j) * N + col];
        sum += (int)truncf((a * b) * nr_pow2(frac - e));
      }
      acc = nr_fixed_half(sum, e - frac);
    }
    if (partition != 0u && (kb + group) % partition == 0u) {
      total = kb < partition ? acc : nr_half(total + acc);
      acc = 0.0f;
    }
  }
  if (partition != 0u) acc = total;
  if (silu != 0) acc = nr_silu(acc);
  raw[i] = acc;
  output[i] = quantize != 0 ? nr_quant(acc) : acc;
}

// Packed, row-major model values. The accumulation order matches nr_gemm exactly.
__device__ float nr_packed_weight(unsigned word, unsigned index, int halfMode) {
  if (halfMode != 0) return nr_from_half((word >> ((index & 1u) * 16u)) & 65535u);
  unsigned code = (word >> ((index & 3u) * 8u)) & 255u;
  // The native model loader maps both E4 NaN encodings to positive zero.
  if ((code & 127u) == 127u) return 0.0f;
  return nr_e4_decode(code);
}
__global__ void nr_gemm_packed(const float* input, const unsigned* weights, const float* residual, const float* scales, float* raw, float* output, unsigned rows, unsigned K, unsigned N, unsigned batches, unsigned inputStride, unsigned inputBatchStride, unsigned partition, int halfMode, int silu, int hasResidual, int quantize) {
  unsigned i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i >= rows * batches * N) return;
  unsigned col = i % N;
  unsigned batch = i / N % batches;
  unsigned row = i / N / batches;
  unsigned base = row * inputStride + batch * inputBatchStride;
  unsigned group = halfMode != 0 ? 8u : 16u;
  int frac = halfMode != 0 ? 24 : 13;
  int minExp = halfMode != 0 ? -14 : -6;
  float initial = hasResidual != 0 ? nr_half(residual[i] * scales[batch * N + col]) : 0.0f;
  float acc = initial;
  float total = 0.0f;
  for (unsigned kb = 0u; kb < K; kb += group) {
    int e = acc != 0.0f ? nr_exp(acc, -14) : -21;
    for (unsigned j = 0u; j < group; j += 1u) {
      float a = input[base + kb + j];
      float b = nr_packed_weight(weights[((batch * K + kb + j) * N + col) >> (halfMode != 0 ? 1u : 2u)], (batch * K + kb + j) * N + col, halfMode);
      if (a != 0.0f && b != 0.0f) e = max(e, nr_exp(a, minExp) + nr_exp(b, minExp));
    }
    if (isfinite(acc)) {
      int sum = (int)truncf(acc * nr_pow2(frac - e));
      for (unsigned j = 0u; j < group; j += 1u) {
        float a = input[base + kb + j];
        float b = nr_packed_weight(weights[((batch * K + kb + j) * N + col) >> (halfMode != 0 ? 1u : 2u)], (batch * K + kb + j) * N + col, halfMode);
        sum += (int)truncf((a * b) * nr_pow2(frac - e));
      }
      acc = nr_fixed_half(sum, e - frac);
    }
    if (partition != 0u && (kb + group) % partition == 0u) {
      total = kb < partition ? acc : nr_half(total + acc);
      acc = 0.0f;
    }
  }
  if (partition != 0u) acc = total;
  if (silu != 0) acc = nr_silu(acc);
  raw[i] = acc;
  output[i] = quantize != 0 ? nr_quant(acc) : acc;
}
