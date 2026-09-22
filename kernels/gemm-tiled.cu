#include "numeric.cuh"
#include "packed.cuh"
#define NR_TILE_ROWS 4
#define NR_TILE_COLS 16
#define NR_TILE_ENTRY nr_gemm_tiled
// Compile tile variants from this CUDA source. Every lane reaches both barriers.
__global__ void NR_TILE_ENTRY(const float* input, const unsigned* weights, const float* residual, const float* scales, float* raw, float* output, unsigned rows, unsigned K, unsigned N, unsigned batches, unsigned inputStride, unsigned inputBatchStride, unsigned partition, int halfMode, int silu, int hasResidual, int quantize) {
  __shared__ float tileA[NR_TILE_ROWS * 16];
  __shared__ float tileB[16 * NR_TILE_COLS];
  unsigned tid = threadIdx.x;
  unsigned tile = blockIdx.x + blockIdx.y * gridDim.x;
  unsigned columns = (N + NR_TILE_COLS - 1u) / NR_TILE_COLS;
  unsigned colBase = tile % columns * NR_TILE_COLS;
  unsigned batch = tile / columns % batches;
  unsigned rowBase = tile / columns / batches * NR_TILE_ROWS;
  unsigned row = rowBase + tid / NR_TILE_COLS;
  unsigned col = colBase + tid % NR_TILE_COLS;
  unsigned i = (row * batches + batch) * N + col;
  unsigned group = halfMode != 0 ? 8u : 16u;
  int frac = halfMode != 0 ? 24 : 13;
  int minExp = halfMode != 0 ? -14 : -6;
  float acc = row < rows && col < N && hasResidual != 0 ? nr_half(residual[i] * scales[batch * N + col]) : 0.0f;
  float total = 0.0f;
  for (unsigned kb = 0u; kb < K; kb += group) {
    for (unsigned t = tid; t < NR_TILE_ROWS * 16u; t += NR_TILE_ROWS * NR_TILE_COLS) {
      unsigned inputRow = rowBase + t / 16u;
      unsigned inputK = t % 16u;
      tileA[t] = inputRow < rows && inputK < group && kb + inputK < K ? input[inputRow * inputStride + batch * inputBatchStride + kb + inputK] : 0.0f;
    }
    for (unsigned t = tid; t < 16u * NR_TILE_COLS; t += NR_TILE_ROWS * NR_TILE_COLS) {
      unsigned wk = t / NR_TILE_COLS;
      unsigned wc = colBase + t % NR_TILE_COLS;
      unsigned wi = (batch * K + kb + wk) * N + wc;
      tileB[t] = wk < group && kb + wk < K && wc < N ? nr_packed_weight(weights[wi >> (halfMode != 0 ? 1u : 2u)], wi, halfMode) : 0.0f;
    }
    __syncthreads();
    int e = acc != 0.0f ? nr_exp(acc, -14) : -21;
    for (unsigned j = 0u; j < group; j += 1u) {
      float a = tileA[tid / NR_TILE_COLS * 16u + j];
      float b = tileB[j * NR_TILE_COLS + tid % NR_TILE_COLS];
      if (a != 0.0f && b != 0.0f) e = max(e, nr_exp(a, minExp) + nr_exp(b, minExp));
    }
    if (isfinite(acc)) {
      int sum = (int)truncf(acc * nr_pow2(frac - e));
      for (unsigned j = 0u; j < group; j += 1u) {
        float a = tileA[tid / NR_TILE_COLS * 16u + j];
        float b = tileB[j * NR_TILE_COLS + tid % NR_TILE_COLS];
        sum += (int)truncf((a * b) * nr_pow2(frac - e));
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
