#pragma once
__device__ float nr_packed_weight(unsigned word, unsigned index, int halfMode) {
  if (halfMode != 0) return nr_from_half((word >> ((index & 1u) * 16u)) & 65535u);
  unsigned code = (word >> ((index & 3u) * 8u)) & 255u;
  // The native model loader maps both E4 NaN encodings to positive zero.
  if ((code & 127u) == 127u) return 0.0f;
  return nr_e4_decode(code);
}
