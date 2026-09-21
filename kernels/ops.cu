#include "numeric.cuh"
__global__ void nr_numeric(const float* input, float* output, unsigned* codes, unsigned count) {
  unsigned i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  output[i * 3u] = nr_half(input[i]);
  output[i * 3u + 1u] = nr_quant(input[i]);
  output[i * 3u + 2u] = nr_silu(nr_half(input[i]));
  codes[i] = nr_e4_code(input[i]);
}
__global__ void nr_publish(const float* input, float* output, unsigned count, int quantize) {
  unsigned i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i < count) output[i] = quantize != 0 ? nr_quant(input[i]) : nr_half(input[i]);
}
__global__ void nr_pool(const float* input, float* output, unsigned iw, unsigned ih, unsigned ow, unsigned oh, unsigned channels) {
  unsigned i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i >= ow * oh * channels) return;
  unsigned c = i % channels;
  unsigned x = (i / channels % ow) * 2u;
  unsigned y = (i / channels / ow) * 2u;
  float v = 0.0f;
  if (x + 1u < iw && y + 1u < ih) {
    unsigned p = (y * iw + x) * channels + c;
    float a = nr_half(input[p] + input[p + channels]);
    float b = nr_half(input[p + iw * channels] + input[p + (iw + 1u) * channels]);
    v = nr_half(nr_half(a + b) * 0.25f);
  }
  output[i] = nr_quant(v);
}
__global__ void nr_merge(const float* low, const float* skip, const float* scaleA, const float* scaleB, float* raw, float* output, unsigned iw, unsigned ow, unsigned oh, unsigned channels, int post) {
  unsigned i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i >= ow * oh * channels) return;
  unsigned c = i % channels;
  unsigned x = i / channels % ow;
  unsigned y = i / channels / ow;
  float v = low[((y / 2u) * iw + x / 2u) * channels + c];
  if (post != 0) v = nr_half(v * scaleA[c]);
  v = nr_half(fmaf(skip[i], scaleB[c], v));
  raw[i] = v;
  output[i] = nr_quant(v);
}
__global__ void nr_compose(const float* proxy, const float* head, const float* history, float* output, unsigned width, unsigned height, unsigned fullWidth, float blendScale, int useHistory) {
  unsigned i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i >= width * height) return;
  unsigned h = ((i / width) * fullWidth + i % width) * 4u;
  float blend = useHistory != 0 && history[i * 4u + 3u] > 0.0f ? fminf(fmaxf(fminf(fmaxf(blendScale, 0.0f), 1.0f) / (1.0f + exp2f(head[h + 3u] * __uint_as_float(0xbfb8aa3bu))), 0.0f), 1.0f) : 0.0f;
  for (unsigned c = 0u; c < 3u; c += 1u) {
    float centered = fmaf(proxy[i * 4u + c], 0.125f, -0.0625f);
    float v = fminf(fmaxf(fmaf(head[h + c], 0.03125f, centered) * 8.0f + 0.5f, 0.0f), 1.0f);
    output[i * 4u + c] = nr_truncate_half(fmaf(blend, history[i * 4u + c] - v, v));
  }
  output[i * 4u + 3u] = 1.0f;
}
