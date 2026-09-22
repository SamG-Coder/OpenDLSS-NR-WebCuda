#include "numeric.cuh"
#include "packed.cuh"
#define NR_MULTI_ROWS 8
#define NR_MULTI_COLS 32
#define NR_MULTI_ENTRY nr_gemm_multi8x32
// Four adjacent outputs share each input load. Accumulators retain independent
// native F13/F24 groups, residual seeding, and partition publication.
// Every lane reaches both barriers, including row and column tail lanes.
__global__ void NR_MULTI_ENTRY(const float* input, const unsigned* weights, const float* residual, const float* scales, float* raw, float* output, unsigned rows, unsigned K, unsigned N, unsigned batches, unsigned inputStride, unsigned inputBatchStride, unsigned partition, int halfMode, int silu, int hasResidual, int quantize) {
  __shared__ float tileA[NR_MULTI_ROWS * 16];
  __shared__ float tileB[16 * NR_MULTI_COLS];
  unsigned tid = threadIdx.x;
  unsigned tile = blockIdx.x + blockIdx.y * gridDim.x;
  unsigned columns = (N + NR_MULTI_COLS - 1u) / NR_MULTI_COLS;
  unsigned colBase = tile % columns * NR_MULTI_COLS;
  unsigned batch = tile / columns % batches;
  unsigned rowBase = tile / columns / batches * NR_MULTI_ROWS;
  unsigned row = rowBase + tid / (NR_MULTI_COLS / 4u);
  unsigned col = colBase + tid % (NR_MULTI_COLS / 4u) * 4u;
  unsigned i = (row * batches + batch) * N + col;
  unsigned group = halfMode != 0 ? 8u : 16u;
  int frac = halfMode != 0 ? 24 : 13;
  int minExp = halfMode != 0 ? -14 : -6;
  float acc0 = row < rows && col + 0u < N && hasResidual != 0 ? nr_half(residual[i + 0u] * scales[batch * N + col + 0u]) : 0.0f;
  float total0 = 0.0f;
  float acc1 = row < rows && col + 1u < N && hasResidual != 0 ? nr_half(residual[i + 1u] * scales[batch * N + col + 1u]) : 0.0f;
  float total1 = 0.0f;
  float acc2 = row < rows && col + 2u < N && hasResidual != 0 ? nr_half(residual[i + 2u] * scales[batch * N + col + 2u]) : 0.0f;
  float total2 = 0.0f;
  float acc3 = row < rows && col + 3u < N && hasResidual != 0 ? nr_half(residual[i + 3u] * scales[batch * N + col + 3u]) : 0.0f;
  float total3 = 0.0f;
  for (unsigned kb = 0u; kb < K; kb += group) {
    for (unsigned t = tid; t < NR_MULTI_ROWS * 16u; t += NR_MULTI_ROWS * NR_MULTI_COLS / 4u) {
      unsigned inputRow = rowBase + t / 16u;
      unsigned inputK = t % 16u;
      tileA[t] = inputRow < rows && inputK < group && kb + inputK < K ? input[inputRow * inputStride + batch * inputBatchStride + kb + inputK] : 0.0f;
    }
    for (unsigned t = tid; t < 16u * NR_MULTI_COLS; t += NR_MULTI_ROWS * NR_MULTI_COLS / 4u) {
      unsigned wk = t / NR_MULTI_COLS;
      unsigned wc = colBase + t % NR_MULTI_COLS;
      unsigned wi = (batch * K + kb + wk) * N + wc;
      tileB[t] = wk < group && kb + wk < K && wc < N ? nr_packed_weight(weights[wi >> (halfMode != 0 ? 1u : 2u)], wi, halfMode) : 0.0f;
    }
    __syncthreads();
    int e0 = acc0 != 0.0f ? nr_exp(acc0, -14) : -21;
    int e1 = acc1 != 0.0f ? nr_exp(acc1, -14) : -21;
    int e2 = acc2 != 0.0f ? nr_exp(acc2, -14) : -21;
    int e3 = acc3 != 0.0f ? nr_exp(acc3, -14) : -21;
    for (unsigned j = 0u; j < group; j += 1u) {
      float a = tileA[tid / (NR_MULTI_COLS / 4u) * 16u + j];
      float b0 = tileB[j * NR_MULTI_COLS + tid % (NR_MULTI_COLS / 4u) * 4u + 0u];
      if (a != 0.0f && b0 != 0.0f) e0 = max(e0, nr_exp(a, minExp) + nr_exp(b0, minExp));
      float b1 = tileB[j * NR_MULTI_COLS + tid % (NR_MULTI_COLS / 4u) * 4u + 1u];
      if (a != 0.0f && b1 != 0.0f) e1 = max(e1, nr_exp(a, minExp) + nr_exp(b1, minExp));
      float b2 = tileB[j * NR_MULTI_COLS + tid % (NR_MULTI_COLS / 4u) * 4u + 2u];
      if (a != 0.0f && b2 != 0.0f) e2 = max(e2, nr_exp(a, minExp) + nr_exp(b2, minExp));
      float b3 = tileB[j * NR_MULTI_COLS + tid % (NR_MULTI_COLS / 4u) * 4u + 3u];
      if (a != 0.0f && b3 != 0.0f) e3 = max(e3, nr_exp(a, minExp) + nr_exp(b3, minExp));
    }
    int sum0 = isfinite(acc0) ? (int)truncf(acc0 * nr_pow2(frac - e0)) : 0;
    int sum1 = isfinite(acc1) ? (int)truncf(acc1 * nr_pow2(frac - e1)) : 0;
    int sum2 = isfinite(acc2) ? (int)truncf(acc2 * nr_pow2(frac - e2)) : 0;
    int sum3 = isfinite(acc3) ? (int)truncf(acc3 * nr_pow2(frac - e3)) : 0;
    for (unsigned j = 0u; j < group; j += 1u) {
      float a = tileA[tid / (NR_MULTI_COLS / 4u) * 16u + j];
      float b0 = tileB[j * NR_MULTI_COLS + tid % (NR_MULTI_COLS / 4u) * 4u + 0u];
      if (isfinite(acc0)) sum0 += (int)truncf((a * b0) * nr_pow2(frac - e0));
      float b1 = tileB[j * NR_MULTI_COLS + tid % (NR_MULTI_COLS / 4u) * 4u + 1u];
      if (isfinite(acc1)) sum1 += (int)truncf((a * b1) * nr_pow2(frac - e1));
      float b2 = tileB[j * NR_MULTI_COLS + tid % (NR_MULTI_COLS / 4u) * 4u + 2u];
      if (isfinite(acc2)) sum2 += (int)truncf((a * b2) * nr_pow2(frac - e2));
      float b3 = tileB[j * NR_MULTI_COLS + tid % (NR_MULTI_COLS / 4u) * 4u + 3u];
      if (isfinite(acc3)) sum3 += (int)truncf((a * b3) * nr_pow2(frac - e3));
    }
    if (isfinite(acc0)) acc0 = nr_fixed_half(sum0, e0 - frac);
    if (partition != 0u && (kb + group) % partition == 0u) {
      total0 = kb < partition ? acc0 : nr_half(total0 + acc0);
      acc0 = 0.0f;
    }
    if (isfinite(acc1)) acc1 = nr_fixed_half(sum1, e1 - frac);
    if (partition != 0u && (kb + group) % partition == 0u) {
      total1 = kb < partition ? acc1 : nr_half(total1 + acc1);
      acc1 = 0.0f;
    }
    if (isfinite(acc2)) acc2 = nr_fixed_half(sum2, e2 - frac);
    if (partition != 0u && (kb + group) % partition == 0u) {
      total2 = kb < partition ? acc2 : nr_half(total2 + acc2);
      acc2 = 0.0f;
    }
    if (isfinite(acc3)) acc3 = nr_fixed_half(sum3, e3 - frac);
    if (partition != 0u && (kb + group) % partition == 0u) {
      total3 = kb < partition ? acc3 : nr_half(total3 + acc3);
      acc3 = 0.0f;
    }
    __syncthreads();
  }
  if (partition != 0u) acc0 = total0;
  if (silu != 0) acc0 = nr_silu(acc0);
  if (row < rows && col + 0u < N) {
    raw[i + 0u] = acc0;
    output[i + 0u] = quantize != 0 ? nr_quant(acc0) : acc0;
  }
  if (partition != 0u) acc1 = total1;
  if (silu != 0) acc1 = nr_silu(acc1);
  if (row < rows && col + 1u < N) {
    raw[i + 1u] = acc1;
    output[i + 1u] = quantize != 0 ? nr_quant(acc1) : acc1;
  }
  if (partition != 0u) acc2 = total2;
  if (silu != 0) acc2 = nr_silu(acc2);
  if (row < rows && col + 2u < N) {
    raw[i + 2u] = acc2;
    output[i + 2u] = quantize != 0 ? nr_quant(acc2) : acc2;
  }
  if (partition != 0u) acc3 = total3;
  if (silu != 0) acc3 = nr_silu(acc3);
  if (row < rows && col + 3u < N) {
    raw[i + 3u] = acc3;
    output[i + 3u] = quantize != 0 ? nr_quant(acc3) : acc3;
  }
}
