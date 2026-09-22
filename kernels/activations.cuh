#pragma once
// Storage only: values are already published to FP8 or half by the graph.
// Formats: 0=float32, 1=E4M3, 2=half. Atomics preserve neighboring packed lanes.
__device__ float nr_activation_load(const unsigned* data, unsigned index, int format) {
  if (format == 0) return __uint_as_float(data[index]);
  if (format == 2) return nr_from_half((data[index >> 1u] >> ((index & 1u) * 16u)) & 65535u);
  return nr_e4_decode((data[index >> 2u] >> ((index & 3u) * 8u)) & 255u);
}
__device__ void nr_activation_store(unsigned* data, unsigned index, float value, int format) {
  if (format == 0) { atomicExch(&data[index], __float_as_uint(value)); return; }
  unsigned word = format == 2 ? index >> 1u : index >> 2u;
  unsigned shift = format == 2 ? (index & 1u) * 16u : (index & 3u) * 8u;
  unsigned mask = (format == 2 ? 65535u : 255u) << shift;
  unsigned bits = (format == 2 ? nr_half_bits(value) : nr_e4_code(value)) << shift;
  unsigned old = data[word];
  while (true) {
    unsigned observed = atomicCAS(&data[word], old, (old & ~mask) | bits);
    if (observed == old) return;
    old = observed;
  }
}

// Four aligned lanes have a single owner; no compare/exchange retries are needed.
__device__ void nr_activation_store4(unsigned* data, unsigned index, float a, float b, float c, float d, int format) {
  if (format == 1) {
    unsigned word = nr_e4_code(a) | (nr_e4_code(b) << 8u) | (nr_e4_code(c) << 16u) | (nr_e4_code(d) << 24u);
    atomicExch(&data[index >> 2u], word);
  } else if (format == 2) {
    atomicExch(&data[index >> 1u], nr_half_bits(a) | (nr_half_bits(b) << 16u));
    atomicExch(&data[(index >> 1u) + 1u], nr_half_bits(c) | (nr_half_bits(d) << 16u));
  } else {
    nr_activation_store(data, index, a, format);
    nr_activation_store(data, index + 1u, b, format);
    nr_activation_store(data, index + 2u, c, format);
    nr_activation_store(data, index + 3u, d, format);
  }
}
