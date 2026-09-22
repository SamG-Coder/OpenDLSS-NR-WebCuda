#include "numeric.cuh"
__global__ void nr_normalize(const float* qkv, const float* scales, float* output, unsigned rows, unsigned heads, int globalMode) {
  unsigned i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i >= rows * heads) return;
  unsigned base = i * 96u;
  for (unsigned part = 0u; part < 3u; part += 1u) {
    float norm = 1.0f;
    if (part < 2u) {
      float r[16];
      for (unsigned c = 0u; c < 16u; c += 1u) {
        float a = qkv[base + part * 32u + c];
        float b = qkv[base + part * 32u + c + 16u];
        float high = nr_half(b * b);
        r[c] = globalMode != 0 ? nr_half(a * a + high) : nr_half(fmaf(a, a, high));
      }
      for (unsigned stride = 8u; stride > 0u; stride >>= 1u) {
        for (unsigned c = 0u; c < stride; c += 1u) r[c] = nr_half(r[c] + r[c + stride]);
      }
      // A zero row publishes zeros, including the 0*infinity norm case.
      norm = r[0] == 0.0f ? 0.0f : nr_half(rsqrtf(r[0]));
    }
    for (unsigned c = 0u; c < 32u; c += 1u) {
      float v = qkv[base + part * 32u + c];
      if (part < 2u) v = nr_half(v * norm);
      if (part == 0u) {
        if (globalMode != 0) v = nr_half(v * nr_half(5.656854249492381f));
        v = nr_half(v * nr_half(scales[i % heads]));
      }
      output[base + part * 32u + c] = nr_quant(v);
    }
  }
}
__device__ unsigned nr_physical_to_natural(unsigned k) {
  unsigned tile = k >> 4u;
  unsigned inner = k & 15u;
  return ((tile >> 1u) * 4u + (inner >> 2u)) * 8u + (tile & 1u) * 4u + (inner & 3u);
}
// q/k rows use natural pixels; keys use the native physical reduction order.
__global__ void nr_scores(const float* qkv, const float* prior, float* scores, unsigned width, unsigned height, unsigned heads, unsigned shiftX, unsigned shiftY, unsigned padded, int globalMode) {
  unsigned i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  unsigned keys = globalMode != 0 ? padded : 64u;
  unsigned rows = width * height;
  if (i >= rows * heads * keys) return;
  unsigned key = i % keys;
  unsigned head = i / keys % heads;
  unsigned row = i / keys / heads;
  int target = (int)key;
  unsigned localQ = 0u;
  if (globalMode == 0) {
    unsigned x = row % width + shiftX;
    unsigned y = row / width + shiftY;
    localQ = (y % 8u) * 8u + x % 8u;
    unsigned natural = nr_physical_to_natural(key);
    int tx = (int)(x / 8u * 8u + natural % 8u) - (int)shiftX;
    int ty = (int)(y / 8u * 8u + natural / 8u) - (int)shiftY;
    target = tx < 0 || ty < 0 || tx >= (int)width || ty >= (int)height ? -1 : ty * (int)width + tx;
  }
  float acc = globalMode == 0 ? prior[(head * 64u + localQ) * 64u + key] : 0.0f;
  for (unsigned kb = 0u; kb < 32u; kb += 16u) {
    int e = acc != 0.0f ? nr_exp(acc, -14) : -21;
    for (unsigned c = 0u; c < 16u; c += 1u) {
      float a = qkv[(row * heads + head) * 96u + kb + c];
      float b = target < 0 || target >= (int)rows ? 0.0f : qkv[((unsigned)target * heads + head) * 96u + 32u + kb + c];
      if (a != 0.0f && b != 0.0f) e = max(e, nr_exp(a, -6) + nr_exp(b, -6));
    }
    int sum = (int)truncf(acc * nr_pow2(13 - e));
    for (unsigned c = 0u; c < 16u; c += 1u) {
      float a = qkv[(row * heads + head) * 96u + kb + c];
      float b = target < 0 || target >= (int)rows ? 0.0f : qkv[((unsigned)target * heads + head) * 96u + 32u + kb + c];
      sum += (int)truncf(a * b * nr_pow2(13 - e));
    }
    acc = nr_fixed_half(sum, e - 13);
  }
  scores[i] = nr_exp_weight(acc, globalMode);
}
__global__ void nr_softmax(const float* scores, float* weights, float* inverse, unsigned rows, unsigned heads, unsigned keys, int globalMode) {
  unsigned i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i >= rows * heads) return;
  float total = 0.0f;
  for (unsigned base = 0u; base < keys; base += 64u) {
    float t[8];
    for (unsigned g = 0u; g < 8u; g += 1u) {
      float sum = 0.0f;
      for (unsigned j = 0u; j < 4u; j += 1u) {
        unsigned k = i * keys + base + g + j * 16u;
        float pair = nr_half(scores[k] + scores[k + 8u]);
        sum = j == 0u ? pair : nr_half(sum + pair);
      }
      t[g] = sum;
    }
    float even = nr_half(nr_half(nr_half(t[0] + t[2]) + t[4]) + t[6]);
    float odd = nr_half(nr_half(nr_half(t[1] + t[3]) + t[5]) + t[7]);
    total = nr_half(total + nr_half(even + odd));
  }
  if (globalMode != 0) total = nr_half(total - nr_half(nr_exp_weight(0.0f, 1) * (float)(keys - rows)));
  float inv = nr_half(1.0f / total);
  inverse[i] = inv;
  for (unsigned k = 0u; k < keys; k += 1u) weights[i * keys + k] = nr_quant(globalMode != 0 ? scores[i * keys + k] : nr_half(scores[i * keys + k] * inv));
}
__global__ void nr_attend(const float* qkv, const float* weights, const float* inverse, float* output, unsigned width, unsigned height, unsigned heads, unsigned shiftX, unsigned shiftY, unsigned keys, int globalMode) {
  unsigned i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  unsigned rows = width * height;
  if (i >= rows * heads * 32u) return;
  unsigned c = i % 32u;
  unsigned head = i / 32u % heads;
  unsigned row = i / 32u / heads;
  float acc = 0.0f;
  for (unsigned kb = 0u; kb < keys; kb += 16u) {
    float a[16];
    float b[16];
    int e = acc != 0.0f ? nr_exp(acc, -14) : -21;
    for (unsigned j = 0u; j < 16u; j += 1u) {
      unsigned key = kb + j;
      int target = (int)key;
      if (globalMode == 0) {
        unsigned x = row % width + shiftX;
        unsigned y = row / width + shiftY;
        unsigned natural = nr_physical_to_natural(key);
        int tx = (int)(x / 8u * 8u + natural % 8u) - (int)shiftX;
        int ty = (int)(y / 8u * 8u + natural / 8u) - (int)shiftY;
        target = tx < 0 || ty < 0 || tx >= (int)width || ty >= (int)height ? -1 : ty * (int)width + tx;
      }
      a[j] = weights[(row * heads + head) * keys + key];
      b[j] = target < 0 || target >= (int)rows ? 0.0f : qkv[((unsigned)target * heads + head) * 96u + 64u + c];
      if (a[j] != 0.0f && b[j] != 0.0f) e = max(e, nr_exp(a[j], -6) + nr_exp(b[j], -6));
    }
    if (isfinite(acc)) {
      int sum = (int)truncf(acc * nr_pow2(13 - e));
      for (unsigned j = 0u; j < 16u; j += 1u) sum += (int)truncf(a[j] * b[j] * nr_pow2(13 - e));
      acc = nr_fixed_half(sum, e - 13);
    }
  }
  if (globalMode != 0) acc = nr_half(acc * inverse[row * heads + head]);
  output[i] = nr_quant(acc);
}

// Window-relative coordinates are shared by every query/key in a tile.
__device__ int nr_window_row(unsigned natural, unsigned window, unsigned width, unsigned height, unsigned shiftX, unsigned shiftY) {
  unsigned windowsX = (width + shiftX + 7u) / 8u;
  int x = (int)((window % windowsX) * 8u + natural % 8u) - (int)shiftX;
  int y = (int)((window / windowsX) * 8u + natural / 8u) - (int)shiftY;
  return x < 0 || y < 0 || x >= (int)width || y >= (int)height ? -1 : y * (int)width + x;
}
__global__ void nr_scores_tiled(const float* qkv, const float* prior, float* scores, unsigned width, unsigned height, unsigned heads, unsigned shiftX, unsigned shiftY, unsigned padded, int globalMode) {
  __shared__ float queries[128];
  __shared__ float keyValues[512];
  unsigned tid = threadIdx.x;
  unsigned tile = blockIdx.x + blockIdx.y * gridDim.x;
  unsigned rows = width * height;
  unsigned keys = globalMode != 0 ? padded : 64u;
  unsigned keyTiles = keys / 16u;
  unsigned queryTiles = globalMode != 0 ? (rows + 3u) / 4u : 16u;
  unsigned keyBase = tile % keyTiles * 16u;
  unsigned queryBase = tile / keyTiles % queryTiles * 4u;
  unsigned head = tile / keyTiles / queryTiles % heads;
  unsigned window = tile / keyTiles / queryTiles / heads;
  for (unsigned t = tid; t < 128u; t += 64u) {
    unsigned natural = queryBase + t / 32u;
    int row = globalMode != 0 ? (int)natural : nr_window_row(natural, window, width, height, shiftX, shiftY);
    queries[t] = row < 0 || row >= (int)rows ? 0.0f : qkv[((unsigned)row * heads + head) * 96u + t % 32u];
  }
  for (unsigned t = tid; t < 512u; t += 64u) {
    unsigned key = keyBase + t / 32u;
    int row = globalMode != 0 ? (int)key : nr_window_row(nr_physical_to_natural(key), window, width, height, shiftX, shiftY);
    keyValues[t] = row < 0 || row >= (int)rows ? 0.0f : qkv[((unsigned)row * heads + head) * 96u + 32u + t % 32u];
  }
  __syncthreads();
  unsigned localQ = queryBase + tid / 16u;
  unsigned key = keyBase + tid % 16u;
  int row = globalMode != 0 ? (int)localQ : nr_window_row(localQ, window, width, height, shiftX, shiftY);
  float acc = globalMode == 0 ? prior[(head * 64u + localQ) * 64u + key] : 0.0f;
  for (unsigned kb = 0u; kb < 32u; kb += 16u) {
    int e = acc != 0.0f ? nr_exp(acc, -14) : -21;
    for (unsigned c = 0u; c < 16u; c += 1u) {
      float a = queries[tid / 16u * 32u + kb + c];
      float b = keyValues[tid % 16u * 32u + kb + c];
      if (a != 0.0f && b != 0.0f) e = max(e, nr_exp(a, -6) + nr_exp(b, -6));
    }
    int sum = (int)truncf(acc * nr_pow2(13 - e));
    for (unsigned c = 0u; c < 16u; c += 1u) {
      float a = queries[tid / 16u * 32u + kb + c];
      float b = keyValues[tid % 16u * 32u + kb + c];
      sum += (int)truncf(a * b * nr_pow2(13 - e));
    }
    acc = nr_fixed_half(sum, e - 13);
  }
  if (row >= 0 && row < (int)rows && (globalMode == 0 || window == 0u)) scores[((unsigned)row * heads + head) * keys + key] = nr_exp_weight(acc, globalMode);
}
__global__ void nr_attend_tiled(const float* qkv, const float* weights, const float* inverse, float* output, unsigned width, unsigned height, unsigned heads, unsigned shiftX, unsigned shiftY, unsigned keys, int globalMode) {
  __shared__ float tileWeights[64];
  __shared__ float tileValues[256];
  unsigned tid = threadIdx.x;
  unsigned tile = blockIdx.x + blockIdx.y * gridDim.x;
  unsigned rows = width * height;
  unsigned queryTiles = globalMode != 0 ? (rows + 3u) / 4u : 16u;
  unsigned channelBase = tile % 2u * 16u;
  unsigned queryBase = tile / 2u % queryTiles * 4u;
  unsigned head = tile / 2u / queryTiles % heads;
  unsigned window = tile / 2u / queryTiles / heads;
  unsigned localQ = queryBase + tid / 16u;
  int row = globalMode != 0 ? (int)localQ : nr_window_row(localQ, window, width, height, shiftX, shiftY);
  unsigned channel = channelBase + tid % 16u;
  float acc = 0.0f;
  for (unsigned kb = 0u; kb < keys; kb += 16u) {
    tileWeights[tid] = row < 0 || row >= (int)rows ? 0.0f : weights[((unsigned)row * heads + head) * keys + kb + tid % 16u];
    for (unsigned t = tid; t < 256u; t += 64u) {
      unsigned key = kb + t / 16u;
      int target = globalMode != 0 ? (int)key : nr_window_row(nr_physical_to_natural(key), window, width, height, shiftX, shiftY);
      tileValues[t] = target < 0 || target >= (int)rows ? 0.0f : qkv[((unsigned)target * heads + head) * 96u + 64u + channelBase + t % 16u];
    }
    __syncthreads();
    int e = acc != 0.0f ? nr_exp(acc, -14) : -21;
    for (unsigned j = 0u; j < 16u; j += 1u) {
      float a = tileWeights[tid / 16u * 16u + j];
      float b = tileValues[j * 16u + tid % 16u];
      if (a != 0.0f && b != 0.0f) e = max(e, nr_exp(a, -6) + nr_exp(b, -6));
    }
    if (isfinite(acc)) {
      int sum = (int)truncf(acc * nr_pow2(13 - e));
      for (unsigned j = 0u; j < 16u; j += 1u) sum += (int)truncf(tileWeights[tid / 16u * 16u + j] * tileValues[j * 16u + tid % 16u] * nr_pow2(13 - e));
      acc = nr_fixed_half(sum, e - 13);
    }
    __syncthreads();
  }
  if (row >= 0 && row < (int)rows && (globalMode == 0 || window == 0u)) {
    if (globalMode != 0) acc = nr_half(acc * inverse[(unsigned)row * heads + head]);
    output[((unsigned)row * heads + head) * 32u + channel] = nr_quant(acc);
  }
}
