#include <cuda_fp16.h>
#include "numeric.cuh"
#include "packed.cuh"
#define NR_TILE_ROWS 4
#define NR_TILE_COLS 16
#define NR_TILE_ENTRY nr_gemm_tiled
// Exact only for finite E4 activations and decoded |weights| <= 9.
// Scaling each operand by four puts every nonzero product in the normal half range.
// Half products are exact (at most eight significant bits); F13 accumulation is unchanged.
__global__ void NR_TILE_ENTRY(const float* input, const unsigned* weights, const float* residual, const float* scales, float* raw, float* output, unsigned rows, unsigned K, unsigned N, unsigned batches, unsigned inputStride, unsigned inputBatchStride, unsigned partition, int halfMode, int silu, int hasResidual, int quantize) {
  __shared__ __half2 tileA[NR_TILE_ROWS * 8];
  __shared__ __half2 tileB[8 * NR_TILE_COLS];
  __shared__ __half2 expA[NR_TILE_ROWS * 8];
  __shared__ __half2 expB[8 * NR_TILE_COLS];
  unsigned tid = threadIdx.x;
  unsigned tile = blockIdx.x + blockIdx.y * gridDim.x;
  unsigned columns = (N + NR_TILE_COLS - 1u) / NR_TILE_COLS;
  unsigned colBase = tile % columns * NR_TILE_COLS;
  unsigned batch = tile / columns % batches;
  unsigned rowBase = tile / columns / batches * NR_TILE_ROWS;
  unsigned row = rowBase + tid / NR_TILE_COLS;
  unsigned col = colBase + tid % NR_TILE_COLS;
  unsigned i = (row * batches + batch) * N + col;
  unsigned group = 16u;
  int frac = 13;
  int minExp = -6;
  float acc = row < rows && col < N && hasResidual != 0 ? nr_half(residual[i] * scales[batch * N + col]) : 0.0f;
  float total = 0.0f;
  for (unsigned kb = 0u; kb < K; kb += group) {
    for (unsigned t = tid; t < NR_TILE_ROWS * 8u; t += NR_TILE_ROWS * NR_TILE_COLS) {
      unsigned inputRow = rowBase + t / 8u;
      unsigned k = kb + (t % 8u) * 2u;
      float a0 = inputRow < rows && k < K ? input[inputRow * inputStride + batch * inputBatchStride + k] : 0.0f;
      float a1 = inputRow < rows && k + 1u < K ? input[inputRow * inputStride + batch * inputBatchStride + k + 1u] : 0.0f;
      tileA[t] = __floats2half2_rn(a0 * 4.0f, a1 * 4.0f);
      expA[t] = __floats2half2_rn(a0 != 0.0f ? (float)nr_exp(a0, -6) : -128.0f, a1 != 0.0f ? (float)nr_exp(a1, -6) : -128.0f);
    }
    for (unsigned t = tid; t < 8u * NR_TILE_COLS; t += NR_TILE_ROWS * NR_TILE_COLS) {
      unsigned k = kb + (t / NR_TILE_COLS) * 2u;
      unsigned wc = colBase + t % NR_TILE_COLS;
      unsigned wi = (batch * K + k) * N + wc;
      float b0 = k < K && wc < N ? nr_packed_weight(weights[wi >> 2u], wi, 0) : 0.0f;
      float b1 = k + 1u < K && wc < N ? nr_packed_weight(weights[(wi + N) >> 2u], wi + N, 0) : 0.0f;
      tileB[t] = __floats2half2_rn(b0 * 4.0f, b1 * 4.0f);
      expB[t] = __floats2half2_rn(b0 != 0.0f ? (float)nr_exp(b0, -6) : -128.0f, b1 != 0.0f ? (float)nr_exp(b1, -6) : -128.0f);
    }
    __syncthreads();
    int e = acc != 0.0f ? nr_exp(acc, -14) : -21;
    for (unsigned j = 0u; j < 8u; j += 1u) {
      float2 exponents = __half22float2(__hadd2(expA[tid / NR_TILE_COLS * 8u + j], expB[j * NR_TILE_COLS + tid % NR_TILE_COLS]));
      e = max(e, max((int)exponents.x, (int)exponents.y));
    }
    if (isfinite(acc)) {
      int sum = (int)truncf(acc * nr_pow2(frac - e));
      for (unsigned j = 0u; j < 8u; j += 1u) {
        __half2 a = tileA[tid / NR_TILE_COLS * 8u + j];
        __half2 b = tileB[j * NR_TILE_COLS + tid % NR_TILE_COLS];
        float2 products = __half22float2(__hmul2(a, b));
        sum += (int)truncf(products.x * nr_pow2(frac - e - 4));
        sum += (int)truncf(products.y * nr_pow2(frac - e - 4));
      }
      acc = nr_fixed_half(sum, e - frac);
    }
    if (partition != 0u && (kb + group) % partition == 0u) {
      total = kb < partition ? acc : nr_half(total + acc);
      acc = 0.0f;
    }
    __syncthreads();
  }
  if (partition != 0u) acc = total;
  if (silu != 0) acc = nr_silu(acc);
  if (row < rows && col < N) {
    raw[i] = acc;
    output[i] = quantize != 0 ? nr_quant(acc) : acc;
  }
}
