#pragma once
// F13 sums are exactly representable as f32. Native conversion is used only for
// normal, finite binary16 results; zero/subnormal/overflow cases retain software rounding.
__device__ float nr_fast_fixed_half(int sum,int exponent){
  float value=(float)sum*nr_pow2(exponent);
  float magnitude=fabsf(value);
  if(sum>=-16777216 && sum<=16777216 && magnitude>=0.00006103515625f && magnitude<65520.0f)
    return __half2float(__float2half_rn(value));
  return nr_fixed_half(sum,exponent);
}
