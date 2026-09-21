#pragma once
// Reconstructed from OpenDLSS-NR src/numeric.h and src/reference.cpp (MIT).
// Float storage is used deliberately; every half publication is explicit.
__device__ unsigned nr_rshift(unsigned v, unsigned s) {
  if (s == 0u) return v;
  if (s > 31u) return 0u;
  unsigned q = v >> s;
  unsigned r = v & ((1u << s) - 1u);
  unsigned h = 1u << (s - 1u);
  return q + ((r > h || (r == h && (q & 1u) != 0u)) ? 1u : 0u);
}
__device__ unsigned nr_half_bits(float v) {
  unsigned b = __float_as_uint(v);
  unsigned sign = (b >> 16u) & 32768u;
  unsigned e = (b >> 23u) & 255u;
  unsigned m = b & 8388607u;
  if (e == 255u) return sign | (m != 0u ? 32256u : 31744u);
  int he = (int)e - 112;
  if (he >= 31) return sign | 31744u;
  if (he <= 0) {
    if (he < -10) return sign;
    return sign | nr_rshift(m | 8388608u, (unsigned)(14 - he));
  }
  unsigned r = nr_rshift(m, 13u);
  if (r == 1024u) { r = 0u; he += 1; }
  if (he >= 31) return sign | 31744u;
  return sign | ((unsigned)he << 10u) | r;
}
__device__ float nr_from_half(unsigned b) {
  unsigned sign = (b & 32768u) << 16u;
  unsigned e = (b >> 10u) & 31u;
  unsigned m = b & 1023u;
  if (e == 0u) {
    if (m == 0u) return __uint_as_float(sign);
    int shift = 0;
    while ((m & 1024u) == 0u) { m <<= 1u; shift += 1; }
    return __uint_as_float(sign | ((unsigned)(113 - shift) << 23u) | ((m & 1023u) << 13u));
  }
  if (e == 31u) return __uint_as_float(sign | 2139095040u | (m << 13u));
  return __uint_as_float(sign | ((e + 112u) << 23u) | (m << 13u));
}
__device__ float nr_half(float v) { return nr_from_half(nr_half_bits(v)); }
__device__ float nr_truncate_half(float v) {
  unsigned b = __float_as_uint(v);
  unsigned sign = (b >> 16u) & 32768u;
  unsigned e = (b >> 23u) & 255u;
  unsigned m = b & 8388607u;
  if (e == 255u) return nr_from_half(sign | (m != 0u ? 32256u : 31744u));
  int he = (int)e - 112;
  if (he >= 31) return nr_from_half(sign | 31744u);
  if (he <= 0) return nr_from_half(he < -10 ? sign : sign | ((m | 8388608u) >> (unsigned)(14 - he)));
  return nr_from_half(sign | ((unsigned)he << 10u) | (m >> 13u));
}
__device__ unsigned nr_e4_code(float v) {
  unsigned h = nr_half_bits(v);
  if ((h & 31744u) == 31744u && (h & 1023u) != 0u) return 0u;
  unsigned sign = (h >> 8u) & 128u;
  unsigned e = (h >> 10u) & 31u;
  unsigned m = h & 1023u;
  unsigned code = 0u;
  if (e == 31u) code = 126u;
  else if (e <= 8u) code = min(nr_rshift(e == 0u ? m : m + 1024u, e == 0u ? 15u : 16u - e), 8u);
  else {
    unsigned ee = e - 8u;
    unsigned mm = nr_rshift(m, 7u);
    if (mm == 8u) { mm = 0u; ee += 1u; }
    code = min((ee << 3u) | mm, 126u);
  }
  return sign | code;
}
__device__ float nr_e4_decode(unsigned code) {
  unsigned e = (code >> 3u) & 15u;
  unsigned m = code & 7u;
  float v = e == 0u ? (float)m * 0.001953125f : (1.0f + (float)m * 0.125f) * __uint_as_float((e + 120u) << 23u);
  return __uint_as_float(__float_as_uint(v) | ((code & 128u) << 24u));
}
__device__ float nr_quant(float v) { return nr_e4_decode(nr_e4_code(v)); }
__device__ float nr_silu(float v) {
  float x = nr_half(fminf(fmaxf(v, -4.0f), 4.0f));
  float p = nr_half(fmaf(-0.055908203125f, fabsf(x), 0.447265625f));
  p = nr_half(fmaf(x, p, 0.89453125f));
  return nr_half(v * p);
}
__device__ int nr_exp(float v, int floorExp) {
  return max((int)((__float_as_uint(v) >> 23u) & 255u) - 127, floorExp);
}
__device__ float nr_pow2(int e) { return __uint_as_float((unsigned)(e + 127) << 23u); }
// Exact fixed-point integer -> half, avoiding f32 double-rounding for F24.
__device__ float nr_fixed_half(int sum, int exponent) {
  if (sum == 0) return 0.0f;
  unsigned sign = sum < 0 ? 32768u : 0u;
  unsigned mag = (unsigned)(sum < 0 ? -sum : sum);
  unsigned msb = 0u;
  unsigned t = mag;
  while (t > 1u) { t >>= 1u; msb += 1u; }
  int e = (int)msb + exponent;
  unsigned h = sign;
  if (e >= -14) {
    unsigned sig = msb > 10u ? nr_rshift(mag, msb - 10u) : mag << (10u - msb);
    if (sig >= 2048u) { sig = 1024u; e += 1; }
    if (e >= 16) h |= 31744u;
    else h |= ((unsigned)(e + 15) << 10u) | (sig - 1024u);
  } else {
    int shift = exponent + 24;
    unsigned m = shift >= 0 ? mag << (unsigned)shift : nr_rshift(mag, (unsigned)(-shift));
    h |= min(m, 1024u);
  }
  return nr_from_half(h);
}
__device__ float nr_exp_weight(float score, int globalMode) {
  float x = globalMode != 0 ? nr_half(fmaf(score, nr_half(0.08953947f), nr_half(1.70936143f))) : nr_half(fmaf(score, 0.044921875f, 1.30078125f));
  x = globalMode != 0 ? fminf(fmaxf(x, 1.439453125f), 1.9775390625f) : fminf(fmaxf(x, 1.03125f), 1.5693359375f);
  unsigned b = nr_half_bits(x);
  return nr_from_half(globalMode != 0 ? ((b << 4u) + 16384u) & 65535u : ((b << 5u) ^ 32768u) & 65535u);
}
