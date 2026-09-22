#include "numeric.cuh"
// WGSL sin/cos accuracy varies by backend. Evaluate the Box-Muller elementary
// functions in binary64 (WebCuda lowers this to integer limbs), then publish f32.
// Inputs here are bounded: angles in [0,2*pi], uniforms in (0,1].
__device__ float nr_noise_sin(float value) {
  double x = (double)value;
  if (x > 3.141592653589793) x = x - 6.283185307179586;
  double term = x;
  double sum = x;
  for (int n = 1; n < 14; n += 1) {
    term = term * (-x * x) / (double)((2 * n) * (2 * n + 1));
    sum = sum + term;
  }
  return (float)sum;
}
__device__ float nr_noise_cos(float value) {
  double x = (double)value;
  if (x > 3.141592653589793) x = x - 6.283185307179586;
  double term = 1.0;
  double sum = 1.0;
  for (int n = 1; n < 14; n += 1) {
    term = term * (-x * x) / (double)((2 * n - 1) * (2 * n));
    sum = sum + term;
  }
  return (float)sum;
}
__device__ float nr_noise_log2(float value) {
  unsigned bits = __float_as_uint(value);
  int exponent = (int)((bits >> 23u) & 255u) - 127;
  double m = (double)__uint_as_float((bits & 8388607u) | 1065353216u);
  double z = (m - 1.0) / (m + 1.0);
  double term = z;
  double sum = z;
  for (int n = 1; n < 18; n += 1) {
    term = term * z * z;
    sum = sum + term / (double)(2 * n + 1);
  }
  return (float)((double)exponent + sum * 2.8853900817779268);
}
__device__ float nr_uniform(unsigned x) {
  x = (x >> ((x >> 28u) + 4u)) ^ x;
  x *= 0x108ef2d9u;
  return (float)(((x >> 30u) ^ (x >> 8u)) + 1u) * __uint_as_float(0x33800000u);
}
__device__ unsigned nr_mirror(unsigned x, unsigned size) {
  // Same reflected-then-clamped sampling as demo/shaders/nr_preprocess.comp.
  if (x < size) return x;
  return (unsigned)max(2 * (int)size - (int)x - 2, 0);
}
__global__ void nr_preprocess(const float* proxy, const float* history, float* features, unsigned width, unsigned height, unsigned fullWidth, unsigned fullHeight, unsigned seed, float autoMask, float localTone, float localStructure, float skinStructure, float style, int useHistory) {
  unsigned i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i >= fullWidth * fullHeight) return;
  unsigned x = i % fullWidth;
  unsigned y = i / fullWidth;
  unsigned p = (nr_mirror(y, height) * width + nr_mirror(x, width)) * 4u;
  unsigned hash = (x * 0x8da6b343u) ^ (seed * 0x9e3779b9u) ^ (y * 0xd8163841u) ^ 0x243f6a88u;
  hash = (hash >> ((hash >> 28u) + 4u)) ^ hash;
  hash *= 0x108ef2d9u;
  hash = (hash >> 22u) ^ hash;
  float u0 = nr_uniform(hash * 0x2c9277b5u + 0xac564b05u);
  float u1 = nr_uniform(hash * 0xfa6dc5f9u + 0x4712a88eu);
  float u2 = nr_uniform(hash * 0xcaa5b80du + 0x21dd796bu);
  float u3 = nr_uniform(hash * 0x83232c31u + 0x3463e0acu);
  float r0 = (float)sqrt((double)(nr_noise_log2(u0) * __uint_as_float(0x3f317218u) * -2.0f));
  float r1 = (float)sqrt((double)(nr_noise_log2(u2) * __uint_as_float(0x3f317218u) * -2.0f));
  float a0 = u1 * __uint_as_float(0x40c90fdbu);
  float a1 = u3 * __uint_as_float(0x40c90fdbu);
  unsigned b = i * 16u;
  features[b] = nr_half(r0 * nr_noise_cos(a0));
  features[b + 1u] = nr_half(r0 * nr_noise_sin(a0));
  features[b + 2u] = nr_half(r1 * nr_noise_cos(a1));
  features[b + 3u] = 1.0f;
  for (unsigned c = 0u; c < 3u; c += 1u) {
    features[b + 4u + c] = nr_half(nr_half(nr_half(proxy[p + c]) - 0.5f) * 0.125f);
    float h = useHistory != 0 && history[p + 3u] > 0.0f ? history[p + c] : proxy[p + c];
    features[b + 7u + c] = nr_half(nr_half(nr_half(h) - 0.5f) * 0.125f);
  }
  features[b + 10u] = nr_half(style / 128.0f);
  features[b + 11u] = nr_half(localTone);
  features[b + 12u] = nr_half(autoMask > 0.0f ? 1.0f : localStructure);
  features[b + 13u] = nr_half(autoMask > 0.0f ? (skinStructure < 0.0f ? localStructure : skinStructure) : -1.0f);
  features[b + 14u] = nr_half(autoMask > 0.0f ? localStructure : -1.0f);
  features[b + 15u] = 0.0f;
}
// Cache exactly the three published half noise lanes, padded to two words per pixel.
__global__ void nr_noise(unsigned* noise, unsigned fullWidth, unsigned fullHeight, unsigned seed) {
  unsigned i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i >= fullWidth * fullHeight) return;
  unsigned x = i % fullWidth;
  unsigned y = i / fullWidth;
  unsigned hash = (x * 0x8da6b343u) ^ (seed * 0x9e3779b9u) ^ (y * 0xd8163841u) ^ 0x243f6a88u;
  hash = (hash >> ((hash >> 28u) + 4u)) ^ hash;
  hash *= 0x108ef2d9u;
  hash = (hash >> 22u) ^ hash;
  float u0 = nr_uniform(hash * 0x2c9277b5u + 0xac564b05u);
  float u1 = nr_uniform(hash * 0xfa6dc5f9u + 0x4712a88eu);
  float u2 = nr_uniform(hash * 0xcaa5b80du + 0x21dd796bu);
  float u3 = nr_uniform(hash * 0x83232c31u + 0x3463e0acu);
  float r0 = (float)sqrt((double)(nr_noise_log2(u0) * __uint_as_float(0x3f317218u) * -2.0f));
  float r1 = (float)sqrt((double)(nr_noise_log2(u2) * __uint_as_float(0x3f317218u) * -2.0f));
  float a0 = u1 * __uint_as_float(0x40c90fdbu);
  float a1 = u3 * __uint_as_float(0x40c90fdbu);
  noise[i * 2u] = nr_half_bits(r0 * nr_noise_cos(a0)) | (nr_half_bits(r0 * nr_noise_sin(a0)) << 16u);
  noise[i * 2u + 1u] = nr_half_bits(r1 * nr_noise_cos(a1));
}
__global__ void nr_preprocess_cached(const unsigned* noise, const float* proxy, const float* history, float* features, unsigned width, unsigned height, unsigned fullWidth, unsigned fullHeight, unsigned seed, float autoMask, float localTone, float localStructure, float skinStructure, float style, int useHistory) {
  unsigned i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i >= fullWidth * fullHeight) return;
  unsigned x = i % fullWidth;
  unsigned y = i / fullWidth;
  unsigned p = (nr_mirror(y, height) * width + nr_mirror(x, width)) * 4u;
  unsigned b = i * 16u;
  features[b] = nr_from_half(noise[i * 2u] & 65535u);
  features[b + 1u] = nr_from_half(noise[i * 2u] >> 16u);
  features[b + 2u] = nr_from_half(noise[i * 2u + 1u] & 65535u);
  features[b + 3u] = 1.0f;
  for (unsigned c = 0u; c < 3u; c += 1u) {
    features[b + 4u + c] = nr_half(nr_half(nr_half(proxy[p + c]) - 0.5f) * 0.125f);
    float h = useHistory != 0 && history[p + 3u] > 0.0f ? history[p + c] : proxy[p + c];
    features[b + 7u + c] = nr_half(nr_half(nr_half(h) - 0.5f) * 0.125f);
  }
  features[b + 10u] = nr_half(style / 128.0f);
  features[b + 11u] = nr_half(localTone);
  features[b + 12u] = nr_half(autoMask > 0.0f ? 1.0f : localStructure);
  features[b + 13u] = nr_half(autoMask > 0.0f ? (skinStructure < 0.0f ? localStructure : skinStructure) : -1.0f);
  features[b + 14u] = nr_half(autoMask > 0.0f ? localStructure : -1.0f);
  features[b + 15u] = 0.0f;
}
// Motion is RGBA: current-to-previous UV displacement in xy, validity in z.
__global__ void nr_reproject(const float* history, const float* motion, const float* proxy, float* output, unsigned width, unsigned height) {
  unsigned i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i >= width * height) return;
  float x = (float)(i % width) + motion[i * 4u] * (float)width;
  float y = (float)(i / width) + motion[i * 4u + 1u] * (float)height;
  int valid = motion[i * 4u + 2u] > 0.0f && x >= -0.5f && y >= -0.5f && x <= (float)width - 0.5f && y <= (float)height - 0.5f ? 1 : 0;
  if (valid == 0) {
    for (unsigned c = 0u; c < 3u; c += 1u) output[i * 4u + c] = proxy[i * 4u + c];
    output[i * 4u + 3u] = 0.0f;
    return;
  }
  int ix = (int)floorf(x);
  int iy = (int)floorf(y);
  float fx = x - (float)ix;
  float fy = y - (float)iy;
  unsigned x0 = (unsigned)min(max(ix, 0), (int)width - 1);
  unsigned x1 = (unsigned)min(max(ix + 1, 0), (int)width - 1);
  unsigned y0 = (unsigned)min(max(iy, 0), (int)height - 1);
  unsigned y1 = (unsigned)min(max(iy + 1, 0), (int)height - 1);
  for (unsigned c = 0u; c < 3u; c += 1u) {
    float a = history[(y0 * width + x0) * 4u + c];
    float b = history[(y0 * width + x1) * 4u + c];
    float d = history[(y1 * width + x0) * 4u + c];
    float e = history[(y1 * width + x1) * 4u + c];
    float top = fmaf(fx, b - a, a);
    float bottom = fmaf(fx, e - d, d);
    output[i * 4u + c] = fmaf(fy, bottom - top, top);
  }
  output[i * 4u + 3u] = 1.0f;
}
