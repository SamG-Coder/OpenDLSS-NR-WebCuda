#pragma once
// F13 terms and partial sums are exact f32 integers. Component-wise helpers
// remain valid CUDA; WebCuda lowers these expression trees to vector WGSL.
__device__ float4 nr_accumulate4(float4 sum,float4 product,float4 scale){
  return make_float4(sum.x+truncf(product.x*scale.x),sum.y+truncf(product.y*scale.y),sum.z+truncf(product.z*scale.z),sum.w+truncf(product.w*scale.w));
}
__device__ float4 nr_max4(float4 a,float4 b){
  return make_float4(fmaxf(a.x,b.x),fmaxf(a.y,b.y),fmaxf(a.z,b.z),fmaxf(a.w,b.w));
}
__device__ float4 nr_exponent4(float4 a){
  return make_float4(a.x!=0.0f?(float)nr_exp(a.x,-14):-21.0f,a.y!=0.0f?(float)nr_exp(a.y,-14):-21.0f,a.z!=0.0f?(float)nr_exp(a.z,-14):-21.0f,a.w!=0.0f?(float)nr_exp(a.w,-14):-21.0f);
}
__device__ float2 nr_max2(float2 a,float2 b){
  return make_float2(fmaxf(a.x,b.x),fmaxf(a.y,b.y));
}
__device__ float2 nr_accumulate2(float2 sum,float2 product,float2 scale){
  return make_float2(sum.x+truncf(product.x*scale.x),sum.y+truncf(product.y*scale.y));
}
