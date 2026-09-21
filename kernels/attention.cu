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
