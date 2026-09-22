#include <cuda_fp16.h>
#include "numeric.cuh"
#include "fast-half.cuh"
#include "packed.cuh"
#include "activations.cuh"
// Every F13 term and partial sum is an integer below the exact f32 limit.
// Keep four ordered accumulators together without a per-product float/int cast.
__device__ float4 nr_accumulate4(float4 sum,float4 product,float4 scale){
  return make_float4(sum.x+truncf(product.x*scale.x),sum.y+truncf(product.y*scale.y),sum.z+truncf(product.z*scale.z),sum.w+truncf(product.w*scale.w));
}
// Requires packed FP8 input, bounded weights, four-element-aligned dimensions
// and input strides, and packed outputs. The build enforces this specialization.
// Each invocation owns eight outputs across two rows; each row has four consecutive columns and therefore every output word it writes.
__device__ void nr_wide_store4(unsigned* data, unsigned index, float a, float b, float c, float d, int format) {
  if(format == 1) data[index >> 2u] = nr_e4_code(a) | (nr_e4_code(b) << 8u) | (nr_e4_code(c) << 16u) | (nr_e4_code(d) << 24u);
  else {
    data[index >> 1u] = nr_half_bits(a) | (nr_half_bits(b) << 16u);
    data[(index >> 1u) + 1u] = nr_half_bits(c) | (nr_half_bits(d) << 16u);
  }
}
__global__ void nr_gemm_wide(const float* metadata, const float* siluTable, const unsigned* input, const unsigned* weights, const unsigned* residual, const float* scales, unsigned* raw, unsigned* output, unsigned rows, unsigned K, unsigned N, unsigned batches, unsigned inputStride, unsigned inputBatchStride, unsigned partition, int halfMode, int silu, int hasResidual, int quantize, int inputFormat, int residualFormat, int rawFormat, int outputFormat, int rawEnabled) {
  __shared__ __half2 tileA[512];
  __shared__ __half2 tileB[544];
  __shared__ __half2 expA[512];
  __shared__ __half2 expB[544];
  unsigned tid=threadIdx.x;
  unsigned tile=blockIdx.x+blockIdx.y*gridDim.x;
  unsigned columns=(N+31u)/32u;
  unsigned colBase=tile%columns*32u;
  unsigned batch=tile/columns%batches;
  unsigned rowBase=tile/columns/batches*32u;
  unsigned localRow=tid/8u*2u;
  unsigned row=rowBase+localRow;
  unsigned col=colBase+tid%8u*4u;
  unsigned i=(row*batches+batch)*N+col;
  unsigned next=i+batches*N;
  float acc0=row+0u<rows && col+0u<N && hasResidual!=0 ? nr_half(nr_activation_load(residual,i+0u,residualFormat)*scales[batch*N+col+0u]):0.0f;
  float total0=0.0f;
  float acc1=row+0u<rows && col+1u<N && hasResidual!=0 ? nr_half(nr_activation_load(residual,i+1u,residualFormat)*scales[batch*N+col+1u]):0.0f;
  float total1=0.0f;
  float acc2=row+0u<rows && col+2u<N && hasResidual!=0 ? nr_half(nr_activation_load(residual,i+2u,residualFormat)*scales[batch*N+col+2u]):0.0f;
  float total2=0.0f;
  float acc3=row+0u<rows && col+3u<N && hasResidual!=0 ? nr_half(nr_activation_load(residual,i+3u,residualFormat)*scales[batch*N+col+3u]):0.0f;
  float total3=0.0f;
  float acc4=row+1u<rows && col+0u<N && hasResidual!=0 ? nr_half(nr_activation_load(residual,next+0u,residualFormat)*scales[batch*N+col+0u]):0.0f;
  float total4=0.0f;
  float acc5=row+1u<rows && col+1u<N && hasResidual!=0 ? nr_half(nr_activation_load(residual,next+1u,residualFormat)*scales[batch*N+col+1u]):0.0f;
  float total5=0.0f;
  float acc6=row+1u<rows && col+2u<N && hasResidual!=0 ? nr_half(nr_activation_load(residual,next+2u,residualFormat)*scales[batch*N+col+2u]):0.0f;
  float total6=0.0f;
  float acc7=row+1u<rows && col+3u<N && hasResidual!=0 ? nr_half(nr_activation_load(residual,next+3u,residualFormat)*scales[batch*N+col+3u]):0.0f;
  float total7=0.0f;
  for(unsigned kb=0u;kb<K;kb+=32u){
    // Load aligned FP8 words once, then stage paired K values.
    for(unsigned t=tid;t<256u;t+=128u){
      unsigned ar=rowBase+t/8u,ak=kb+(t%8u)*4u;
      unsigned ai=ar*inputStride+batch*inputBatchStride+ak;
      unsigned word=ar<rows && ak<K?input[ai>>2u]:0u;
      unsigned c0=word&255u,c1=(word>>8u)&255u,c2=(word>>16u)&255u,c3=word>>24u;
      unsigned a=t*2u;
      tileA[a]=__floats2half2_rn(metadata[c0*2u],metadata[c1*2u]);
      tileA[a+1u]=__floats2half2_rn(metadata[c2*2u],metadata[c3*2u]);
      expA[a]=__floats2half2_rn(metadata[c0*2u+1u],metadata[c1*2u+1u]);
      expA[a+1u]=__floats2half2_rn(metadata[c2*2u+1u],metadata[c3*2u+1u]);
    }
    unsigned bk=kb+(tid/8u)*2u,bc=colBase+(tid%8u)*4u;
    unsigned wi=(batch*K+bk)*N+bc;
    unsigned w0=bk<K && bc<N?weights[wi>>2u]:0u;
    unsigned w1=bk+1u<K && bc<N?weights[(wi+N)>>2u]:0u;
    #pragma unroll
    for(unsigned lane=0u;lane<4u;lane+=1u){
      unsigned c0=(w0>>(lane*8u))&255u,c1=(w1>>(lane*8u))&255u;
      unsigned b=(tid%8u*4u+lane)*17u+tid/8u;
      tileB[b]=__floats2half2_rn(metadata[c0*2u],metadata[c1*2u]);
      expB[b]=__floats2half2_rn(metadata[c0*2u+1u],metadata[c1*2u+1u]);
    }
    __syncthreads();
    // Two ordered native groups share the same staged K slab.
    for(unsigned part=0u;part<2u;part+=1u){
      unsigned kg=kb+part*16u;
      if(kg<K){
        int e0=acc0!=0.0f?nr_exp(acc0,-14):-21;
        int e1=acc1!=0.0f?nr_exp(acc1,-14):-21;
        int e2=acc2!=0.0f?nr_exp(acc2,-14):-21;
        int e3=acc3!=0.0f?nr_exp(acc3,-14):-21;
        int e4=acc4!=0.0f?nr_exp(acc4,-14):-21;
        int e5=acc5!=0.0f?nr_exp(acc5,-14):-21;
        int e6=acc6!=0.0f?nr_exp(acc6,-14):-21;
        int e7=acc7!=0.0f?nr_exp(acc7,-14):-21;
        #pragma unroll
        for(unsigned j=0u;j<8u;j+=1u){
          unsigned k=part*8u+j;
          __half2 a0=expA[localRow*16u+k];
          __half2 a1=expA[(localRow+1u)*16u+k];
          __half2 b0=expB[(tid%8u*4u+0u)*17u+k];
          float2 ex0=__half22float2(__hadd2(a0,b0));
          e0=max(e0,max((int)ex0.x,(int)ex0.y));
          float2 ex4=__half22float2(__hadd2(a1,b0));
          e4=max(e4,max((int)ex4.x,(int)ex4.y));
          __half2 b1=expB[(tid%8u*4u+1u)*17u+k];
          float2 ex1=__half22float2(__hadd2(a0,b1));
          e1=max(e1,max((int)ex1.x,(int)ex1.y));
          float2 ex5=__half22float2(__hadd2(a1,b1));
          e5=max(e5,max((int)ex5.x,(int)ex5.y));
          __half2 b2=expB[(tid%8u*4u+2u)*17u+k];
          float2 ex2=__half22float2(__hadd2(a0,b2));
          e2=max(e2,max((int)ex2.x,(int)ex2.y));
          float2 ex6=__half22float2(__hadd2(a1,b2));
          e6=max(e6,max((int)ex6.x,(int)ex6.y));
          __half2 b3=expB[(tid%8u*4u+3u)*17u+k];
          float2 ex3=__half22float2(__hadd2(a0,b3));
          e3=max(e3,max((int)ex3.x,(int)ex3.y));
          float2 ex7=__half22float2(__hadd2(a1,b3));
          e7=max(e7,max((int)ex7.x,(int)ex7.y));
        }
        float4 sums0=make_float4(isfinite(acc0)?truncf(acc0*nr_pow2(13-e0)):0.0f,isfinite(acc1)?truncf(acc1*nr_pow2(13-e1)):0.0f,isfinite(acc2)?truncf(acc2*nr_pow2(13-e2)):0.0f,isfinite(acc3)?truncf(acc3*nr_pow2(13-e3)):0.0f);
        float4 scales0=make_float4(nr_pow2(9-e0),nr_pow2(9-e1),nr_pow2(9-e2),nr_pow2(9-e3));
        float4 sums1=make_float4(isfinite(acc4)?truncf(acc4*nr_pow2(13-e4)):0.0f,isfinite(acc5)?truncf(acc5*nr_pow2(13-e5)):0.0f,isfinite(acc6)?truncf(acc6*nr_pow2(13-e6)):0.0f,isfinite(acc7)?truncf(acc7*nr_pow2(13-e7)):0.0f);
        float4 scales1=make_float4(nr_pow2(9-e4),nr_pow2(9-e5),nr_pow2(9-e6),nr_pow2(9-e7));
        #pragma unroll
        for(unsigned j=0u;j<8u;j+=1u){
          unsigned k=part*8u+j;
          __half2 a0=tileA[localRow*16u+k];
          __half2 a1=tileA[(localRow+1u)*16u+k];
          __half2 b0=tileB[(tid%8u*4u+0u)*17u+k];
          float2 p0=__half22float2(__hmul2(a0,b0));
          float2 p4=__half22float2(__hmul2(a1,b0));
          __half2 b1=tileB[(tid%8u*4u+1u)*17u+k];
          float2 p1=__half22float2(__hmul2(a0,b1));
          float2 p5=__half22float2(__hmul2(a1,b1));
          __half2 b2=tileB[(tid%8u*4u+2u)*17u+k];
          float2 p2=__half22float2(__hmul2(a0,b2));
          float2 p6=__half22float2(__hmul2(a1,b2));
          __half2 b3=tileB[(tid%8u*4u+3u)*17u+k];
          float2 p3=__half22float2(__hmul2(a0,b3));
          float2 p7=__half22float2(__hmul2(a1,b3));
          sums0=nr_accumulate4(sums0,make_float4(p0.x,p1.x,p2.x,p3.x),scales0);
          sums0=nr_accumulate4(sums0,make_float4(p0.y,p1.y,p2.y,p3.y),scales0);
          sums1=nr_accumulate4(sums1,make_float4(p4.x,p5.x,p6.x,p7.x),scales1);
          sums1=nr_accumulate4(sums1,make_float4(p4.y,p5.y,p6.y,p7.y),scales1);
        }
        if(isfinite(acc0))acc0=nr_fast_fixed_half((int)sums0.x,e0-13);
        if(partition!=0u && (kg+16u)%partition==0u){total0=kg<partition?acc0:nr_half(total0+acc0);acc0=0.0f;}
        if(isfinite(acc1))acc1=nr_fast_fixed_half((int)sums0.y,e1-13);
        if(partition!=0u && (kg+16u)%partition==0u){total1=kg<partition?acc1:nr_half(total1+acc1);acc1=0.0f;}
        if(isfinite(acc2))acc2=nr_fast_fixed_half((int)sums0.z,e2-13);
        if(partition!=0u && (kg+16u)%partition==0u){total2=kg<partition?acc2:nr_half(total2+acc2);acc2=0.0f;}
        if(isfinite(acc3))acc3=nr_fast_fixed_half((int)sums0.w,e3-13);
        if(partition!=0u && (kg+16u)%partition==0u){total3=kg<partition?acc3:nr_half(total3+acc3);acc3=0.0f;}
        if(isfinite(acc4))acc4=nr_fast_fixed_half((int)sums1.x,e4-13);
        if(partition!=0u && (kg+16u)%partition==0u){total4=kg<partition?acc4:nr_half(total4+acc4);acc4=0.0f;}
        if(isfinite(acc5))acc5=nr_fast_fixed_half((int)sums1.y,e5-13);
        if(partition!=0u && (kg+16u)%partition==0u){total5=kg<partition?acc5:nr_half(total5+acc5);acc5=0.0f;}
        if(isfinite(acc6))acc6=nr_fast_fixed_half((int)sums1.z,e6-13);
        if(partition!=0u && (kg+16u)%partition==0u){total6=kg<partition?acc6:nr_half(total6+acc6);acc6=0.0f;}
        if(isfinite(acc7))acc7=nr_fast_fixed_half((int)sums1.w,e7-13);
        if(partition!=0u && (kg+16u)%partition==0u){total7=kg<partition?acc7:nr_half(total7+acc7);acc7=0.0f;}
      }
    }
    __syncthreads();
  }
  if(partition!=0u)acc0=total0;
  if(silu!=0)acc0=siluTable[nr_half_bits(acc0)];
  if(partition!=0u)acc1=total1;
  if(silu!=0)acc1=siluTable[nr_half_bits(acc1)];
  if(partition!=0u)acc2=total2;
  if(silu!=0)acc2=siluTable[nr_half_bits(acc2)];
  if(partition!=0u)acc3=total3;
  if(silu!=0)acc3=siluTable[nr_half_bits(acc3)];
  if(partition!=0u)acc4=total4;
  if(silu!=0)acc4=siluTable[nr_half_bits(acc4)];
  if(partition!=0u)acc5=total5;
  if(silu!=0)acc5=siluTable[nr_half_bits(acc5)];
  if(partition!=0u)acc6=total6;
  if(silu!=0)acc6=siluTable[nr_half_bits(acc6)];
  if(partition!=0u)acc7=total7;
  if(silu!=0)acc7=siluTable[nr_half_bits(acc7)];
  if(row+0u<rows && col+3u<N){
    if(rawEnabled!=0)nr_wide_store4(raw,i,acc0,acc1,acc2,acc3,rawFormat);
    nr_wide_store4(output,i,acc0,acc1,acc2,acc3,outputFormat);
  }
  if(row+1u<rows && col+3u<N){
    if(rawEnabled!=0)nr_wide_store4(raw,next,acc4,acc5,acc6,acc7,rawFormat);
    nr_wide_store4(output,next,acc4,acc5,acc6,acc7,outputFormat);
  }
}
