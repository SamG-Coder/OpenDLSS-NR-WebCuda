#define NR_PREPARED_INTEGER 0
#include <cuda_fp16.h>
#include "numeric.cuh"
#if !NR_PREPARED_INTEGER
#include "fast-half.cuh"
#endif
#include "packed.cuh"
#include "activations.cuh"
#include "vector-f13.cuh"
#if NR_PREPARED_INTEGER
#define NR_PREPARED_FIXED_HALF nr_fixed_half
#else
#define NR_PREPARED_FIXED_HALF nr_fast_fixed_half
#endif
#define NR_PREPARED_ROWS 32
#define NR_PREPARED_COLS 32
#define NR_PREPARED_COLUMN_GROUPS 8
#define NR_PREPARED_A_PAIRS 512
#define NR_PREPARED_B_PAIRS 544
// Experimental exact FP8 backends. Model preparation stores four consecutive
// K codes per word in [batch][K32][N32][K4][column] order, followed by one
// exponent/flag byte per [batch][K16][column]. K and N must be multiples of 32.
// Both backends publish each ordered 16-term group exactly as the reference.
__device__ int nr_prepared_code_exp(unsigned code) {
  return (code&127u)==0u?-128:max((int)((code>>3u)&15u),1)-7;
}
__device__ int nr_prepared_initial_exp(int accExp,int activationExp,unsigned info) {
  if((info&16u)!=0u)return accExp;
  int bound=activationExp+(int)(info&15u)-6;
  return (info&96u)==96u?max(accExp,bound):accExp;
}
__device__ int nr_prepared_scan(int accExp,int activationExp,unsigned info) {
  int bound=activationExp+(int)(info&15u)-6;
  return (info&16u)==0u && (info&96u)!=96u && accExp<bound?1:0;
}
#if NR_PREPARED_INTEGER
// The unsigned magnitude is shifted before applying sign. A signed right
// shift would round negative terms down instead of truncating toward zero.
__device__ int nr_prepared_term(unsigned a,unsigned b,int exponent) {
  unsigned aa=a&127u,bb=b&127u;
  if(aa==0u || bb==0u || aa==127u || bb==127u)return 0;
  unsigned ae=(a>>3u)&15u,be=(b>>3u)&15u;
  unsigned am=(ae!=0u?8u:0u)+(a&7u),bm=(be!=0u?8u:0u)+(b&7u);
  int distance=exponent-(max((int)ae,1)-7)-(max((int)be,1)-7);
  unsigned magnitude=am*bm;
  if(distance<0)return 0;
  if(distance<=7)magnitude<<=(unsigned)(7-distance);
  else if(distance<15)magnitude>>=(unsigned)(distance-7);
  else magnitude=0u;
  int term=(int)magnitude;
  return ((a^b)&128u)!=0u?-term:term;
}
#endif
__device__ void nr_prepared_store4(unsigned* data, unsigned index, float a, float b, float c, float d, int format) {
  if(format == 1) data[index >> 2u] = nr_e4_code(a) | (nr_e4_code(b) << 8u) | (nr_e4_code(c) << 16u) | (nr_e4_code(d) << 24u);
  else {
    data[index >> 1u] = nr_half_bits(a) | (nr_half_bits(b) << 16u);
    data[(index >> 1u) + 1u] = nr_half_bits(c) | (nr_half_bits(d) << 16u);
  }
}
__global__ void nr_gemm_prepared(const float* metadata, const float* siluTable, const unsigned* input, const unsigned* weights, const unsigned* residual, const float* scales, unsigned* raw, unsigned* output, unsigned rows, unsigned K, unsigned N, unsigned batches, unsigned inputStride, unsigned inputBatchStride, unsigned partition, int halfMode, int silu, int hasResidual, int quantize, int inputFormat, int residualFormat, int rawFormat, int outputFormat, int rawEnabled) {
#if NR_PREPARED_INTEGER
  __shared__ unsigned tileA[256];
  __shared__ unsigned tileB[288];
#else
  __shared__ __half2 tileA[NR_PREPARED_A_PAIRS];
  __shared__ __half2 tileB[NR_PREPARED_B_PAIRS];
  __shared__ __half2 expA[NR_PREPARED_A_PAIRS];
  __shared__ __half2 expB[NR_PREPARED_B_PAIRS];
#endif
  __shared__ int activationMax[64];
  unsigned tid=threadIdx.x;
  unsigned tile=blockIdx.x+blockIdx.y*gridDim.x;
  unsigned columns=(N+NR_PREPARED_COLS-1u)/NR_PREPARED_COLS;
  unsigned colBase=tile%columns*NR_PREPARED_COLS;
  unsigned batch=tile/columns%batches;
  unsigned rowBase=tile/columns/batches*NR_PREPARED_ROWS;
  unsigned localRow=tid/NR_PREPARED_COLUMN_GROUPS*2u;
  unsigned row=rowBase+localRow;
  unsigned col=colBase+tid%NR_PREPARED_COLUMN_GROUPS*4u;
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
    // Shared row maxima are conservative bounds, never substitutes for the
    // paired-product maximum unless the weight group has a constant exponent
    // and no zero values. Read aligned words to avoid another shared barrier.
    if(tid<64u){
      unsigned ar=rowBase+tid/2u,ak=kb+(tid%2u)*16u;
      int maximum=-128;
      #pragma unroll
      for(unsigned j=0u;j<4u;j+=1u){
        unsigned ai=ar*inputStride+batch*inputBatchStride+ak+j*4u;
        unsigned word=ar<rows?input[ai>>2u]:0u;
        maximum=max(maximum,nr_prepared_code_exp(word&255u));
        maximum=max(maximum,nr_prepared_code_exp((word>>8u)&255u));
        maximum=max(maximum,nr_prepared_code_exp((word>>16u)&255u));
        maximum=max(maximum,nr_prepared_code_exp(word>>24u));
      }
      activationMax[tid]=maximum;
    }
    for(unsigned t=tid;t<256u;t+=128u){
      unsigned ar=rowBase+t/8u,ak=kb+(t%8u)*4u;
      unsigned ai=ar*inputStride+batch*inputBatchStride+ak;
      unsigned word=ar<rows?input[ai>>2u]:0u;
#if NR_PREPARED_INTEGER
      tileA[t]=word;
#else
      unsigned c0=word&255u,c1=(word>>8u)&255u,c2=(word>>16u)&255u,c3=word>>24u;
      unsigned a=t*2u;
      tileA[a]=__floats2half2_rn(metadata[c0*2u],metadata[c1*2u]);
      tileA[a+1u]=__floats2half2_rn(metadata[c2*2u],metadata[c3*2u]);
      expA[a]=__floats2half2_rn(metadata[c0*2u+1u],metadata[c1*2u+1u]);
      expA[a+1u]=__floats2half2_rn(metadata[c2*2u+1u],metadata[c3*2u+1u]);
#endif
      unsigned column=t%32u,k4=t/32u;
      unsigned wi=(((batch*(K/32u)+kb/32u)*(N/32u)+colBase/32u)*8u+k4)*32u+column;
      unsigned packedWeight=weights[wi];
#if NR_PREPARED_INTEGER
      tileB[column*9u+k4]=packedWeight;
#else
      unsigned w0=packedWeight&255u,w1=(packedWeight>>8u)&255u,w2=(packedWeight>>16u)&255u,w3=packedWeight>>24u;
      unsigned b=column*17u+k4*2u;
      tileB[b]=__floats2half2_rn(metadata[w0*2u],metadata[w1*2u]);
      tileB[b+1u]=__floats2half2_rn(metadata[w2*2u],metadata[w3*2u]);
      expB[b]=__floats2half2_rn(metadata[w0*2u+1u],metadata[w1*2u+1u]);
      expB[b+1u]=__floats2half2_rn(metadata[w2*2u+1u],metadata[w3*2u+1u]);
#endif
    }
    __syncthreads();
    // Two ordered native groups share the same staged K slab.
    for(unsigned part=0u;part<2u;part+=1u){
      unsigned kg=kb+part*16u;
      if(kg<K){
        unsigned metadataOffset=batches*K*N/4u;
        unsigned metadataIndex=(batch*(K/16u)+kg/16u)*N+col;
        unsigned infoWord=weights[metadataOffset+(metadataIndex>>2u)];
        unsigned info0=(infoWord>>0u)&255u;
        unsigned info1=(infoWord>>8u)&255u;
        unsigned info2=(infoWord>>16u)&255u;
        unsigned info3=(infoWord>>24u)&255u;
        int activationExp0=activationMax[localRow*2u+part];
        int activationExp1=activationMax[(localRow+1u)*2u+part];
        int seed0=acc0!=0.0f?nr_exp(acc0,-14):-21;
        int e0=nr_prepared_initial_exp(seed0,activationExp0,info0);
        int scan0=nr_prepared_scan(seed0,activationExp0,info0);
        int seed1=acc1!=0.0f?nr_exp(acc1,-14):-21;
        int e1=nr_prepared_initial_exp(seed1,activationExp0,info1);
        int scan1=nr_prepared_scan(seed1,activationExp0,info1);
        int seed2=acc2!=0.0f?nr_exp(acc2,-14):-21;
        int e2=nr_prepared_initial_exp(seed2,activationExp0,info2);
        int scan2=nr_prepared_scan(seed2,activationExp0,info2);
        int seed3=acc3!=0.0f?nr_exp(acc3,-14):-21;
        int e3=nr_prepared_initial_exp(seed3,activationExp0,info3);
        int scan3=nr_prepared_scan(seed3,activationExp0,info3);
        int seed4=acc4!=0.0f?nr_exp(acc4,-14):-21;
        int e4=nr_prepared_initial_exp(seed4,activationExp1,info0);
        int scan4=nr_prepared_scan(seed4,activationExp1,info0);
        int seed5=acc5!=0.0f?nr_exp(acc5,-14):-21;
        int e5=nr_prepared_initial_exp(seed5,activationExp1,info1);
        int scan5=nr_prepared_scan(seed5,activationExp1,info1);
        int seed6=acc6!=0.0f?nr_exp(acc6,-14):-21;
        int e6=nr_prepared_initial_exp(seed6,activationExp1,info2);
        int scan6=nr_prepared_scan(seed6,activationExp1,info2);
        int seed7=acc7!=0.0f?nr_exp(acc7,-14):-21;
        int e7=nr_prepared_initial_exp(seed7,activationExp1,info3);
        int scan7=nr_prepared_scan(seed7,activationExp1,info3);
        if(scan0!=0 || scan1!=0 || scan2!=0 || scan3!=0 || scan4!=0 || scan5!=0 || scan6!=0 || scan7!=0){
#if NR_PREPARED_INTEGER
          #pragma unroll
          for(unsigned j=0u;j<16u;j+=1u){
            unsigned k=part*4u+j/4u,shift=(j%4u)*8u;
            int a0=nr_prepared_code_exp((tileA[localRow*8u+k]>>shift)&255u);
            int a1=nr_prepared_code_exp((tileA[(localRow+1u)*8u+k]>>shift)&255u);
            int b0=nr_prepared_code_exp((tileB[(tid%8u*4u+0u)*9u+k]>>shift)&255u);
            int b1=nr_prepared_code_exp((tileB[(tid%8u*4u+1u)*9u+k]>>shift)&255u);
            int b2=nr_prepared_code_exp((tileB[(tid%8u*4u+2u)*9u+k]>>shift)&255u);
            int b3=nr_prepared_code_exp((tileB[(tid%8u*4u+3u)*9u+k]>>shift)&255u);
            if(scan0!=0)e0=max(e0,a0+b0);
            if(scan1!=0)e1=max(e1,a0+b1);
            if(scan2!=0)e2=max(e2,a0+b2);
            if(scan3!=0)e3=max(e3,a0+b3);
            if(scan4!=0)e4=max(e4,a1+b0);
            if(scan5!=0)e5=max(e5,a1+b1);
            if(scan6!=0)e6=max(e6,a1+b2);
            if(scan7!=0)e7=max(e7,a1+b3);
          }
#else
          #pragma unroll
          for(unsigned j=0u;j<8u;j+=1u){
            unsigned k=part*8u+j;
            __half2 a0=expA[localRow*16u+k];
            __half2 a1=expA[(localRow+1u)*16u+k];
            __half2 b0=expB[(tid%8u*4u+0u)*17u+k];
            __half2 b1=expB[(tid%8u*4u+1u)*17u+k];
            __half2 b2=expB[(tid%8u*4u+2u)*17u+k];
            __half2 b3=expB[(tid%8u*4u+3u)*17u+k];
            if(scan0!=0){float2 pair=__half22float2(__hadd2(a0,b0));e0=max(e0,(int)fmaxf(pair.x,pair.y));}
            if(scan1!=0){float2 pair=__half22float2(__hadd2(a0,b1));e1=max(e1,(int)fmaxf(pair.x,pair.y));}
            if(scan2!=0){float2 pair=__half22float2(__hadd2(a0,b2));e2=max(e2,(int)fmaxf(pair.x,pair.y));}
            if(scan3!=0){float2 pair=__half22float2(__hadd2(a0,b3));e3=max(e3,(int)fmaxf(pair.x,pair.y));}
            if(scan4!=0){float2 pair=__half22float2(__hadd2(a1,b0));e4=max(e4,(int)fmaxf(pair.x,pair.y));}
            if(scan5!=0){float2 pair=__half22float2(__hadd2(a1,b1));e5=max(e5,(int)fmaxf(pair.x,pair.y));}
            if(scan6!=0){float2 pair=__half22float2(__hadd2(a1,b2));e6=max(e6,(int)fmaxf(pair.x,pair.y));}
            if(scan7!=0){float2 pair=__half22float2(__hadd2(a1,b3));e7=max(e7,(int)fmaxf(pair.x,pair.y));}
          }
#endif
        }
        float4 sums0=make_float4(isfinite(acc0)?truncf(acc0*nr_pow2(13-e0)):0.0f,isfinite(acc1)?truncf(acc1*nr_pow2(13-e1)):0.0f,isfinite(acc2)?truncf(acc2*nr_pow2(13-e2)):0.0f,isfinite(acc3)?truncf(acc3*nr_pow2(13-e3)):0.0f);
        float4 sums1=make_float4(isfinite(acc4)?truncf(acc4*nr_pow2(13-e4)):0.0f,isfinite(acc5)?truncf(acc5*nr_pow2(13-e5)):0.0f,isfinite(acc6)?truncf(acc6*nr_pow2(13-e6)):0.0f,isfinite(acc7)?truncf(acc7*nr_pow2(13-e7)):0.0f);
#if NR_PREPARED_INTEGER
        #pragma unroll
        for(unsigned j=0u;j<16u;j+=1u){
          unsigned k=part*4u+j/4u,shift=(j%4u)*8u;
          unsigned a0=(tileA[localRow*8u+k]>>shift)&255u;
          unsigned a1=(tileA[(localRow+1u)*8u+k]>>shift)&255u;
          unsigned b0=(tileB[(tid%8u*4u+0u)*9u+k]>>shift)&255u;
          unsigned b1=(tileB[(tid%8u*4u+1u)*9u+k]>>shift)&255u;
          unsigned b2=(tileB[(tid%8u*4u+2u)*9u+k]>>shift)&255u;
          unsigned b3=(tileB[(tid%8u*4u+3u)*9u+k]>>shift)&255u;
          sums0=make_float4(sums0.x+(float)nr_prepared_term(a0,b0,e0),sums0.y+(float)nr_prepared_term(a0,b1,e1),sums0.z+(float)nr_prepared_term(a0,b2,e2),sums0.w+(float)nr_prepared_term(a0,b3,e3));
          sums1=make_float4(sums1.x+(float)nr_prepared_term(a1,b0,e4),sums1.y+(float)nr_prepared_term(a1,b1,e5),sums1.z+(float)nr_prepared_term(a1,b2,e6),sums1.w+(float)nr_prepared_term(a1,b3,e7));
        }
#else
        float4 scales0=make_float4(nr_pow2(9-e0),nr_pow2(9-e1),nr_pow2(9-e2),nr_pow2(9-e3));
        float4 scales1=make_float4(nr_pow2(9-e4),nr_pow2(9-e5),nr_pow2(9-e6),nr_pow2(9-e7));
        #pragma unroll
        for(unsigned j=0u;j<8u;j+=1u){
          unsigned k=part*8u+j;
          __half2 a0=tileA[localRow*16u+k];
          __half2 a1=tileA[(localRow+1u)*16u+k];
          __half2 b0=tileB[(tid%NR_PREPARED_COLUMN_GROUPS*4u+0u)*17u+k];
          float2 p0=__half22float2(__hmul2(a0,b0));
          float2 p4=__half22float2(__hmul2(a1,b0));
          __half2 b1=tileB[(tid%NR_PREPARED_COLUMN_GROUPS*4u+1u)*17u+k];
          float2 p1=__half22float2(__hmul2(a0,b1));
          float2 p5=__half22float2(__hmul2(a1,b1));
          __half2 b2=tileB[(tid%NR_PREPARED_COLUMN_GROUPS*4u+2u)*17u+k];
          float2 p2=__half22float2(__hmul2(a0,b2));
          float2 p6=__half22float2(__hmul2(a1,b2));
          __half2 b3=tileB[(tid%NR_PREPARED_COLUMN_GROUPS*4u+3u)*17u+k];
          float2 p3=__half22float2(__hmul2(a0,b3));
          float2 p7=__half22float2(__hmul2(a1,b3));
          sums0=nr_accumulate4(sums0,make_float4(p0.x,p1.x,p2.x,p3.x),scales0);
          sums0=nr_accumulate4(sums0,make_float4(p0.y,p1.y,p2.y,p3.y),scales0);
          sums1=nr_accumulate4(sums1,make_float4(p4.x,p5.x,p6.x,p7.x),scales1);
          sums1=nr_accumulate4(sums1,make_float4(p4.y,p5.y,p6.y,p7.y),scales1);
        }
#endif
        if(isfinite(acc0))acc0=NR_PREPARED_FIXED_HALF((int)sums0.x,e0-13);
        if(partition!=0u && (kg+16u)%partition==0u){total0=kg<partition?acc0:nr_half(total0+acc0);acc0=0.0f;}
        if(isfinite(acc1))acc1=NR_PREPARED_FIXED_HALF((int)sums0.y,e1-13);
        if(partition!=0u && (kg+16u)%partition==0u){total1=kg<partition?acc1:nr_half(total1+acc1);acc1=0.0f;}
        if(isfinite(acc2))acc2=NR_PREPARED_FIXED_HALF((int)sums0.z,e2-13);
        if(partition!=0u && (kg+16u)%partition==0u){total2=kg<partition?acc2:nr_half(total2+acc2);acc2=0.0f;}
        if(isfinite(acc3))acc3=NR_PREPARED_FIXED_HALF((int)sums0.w,e3-13);
        if(partition!=0u && (kg+16u)%partition==0u){total3=kg<partition?acc3:nr_half(total3+acc3);acc3=0.0f;}
        if(isfinite(acc4))acc4=NR_PREPARED_FIXED_HALF((int)sums1.x,e4-13);
        if(partition!=0u && (kg+16u)%partition==0u){total4=kg<partition?acc4:nr_half(total4+acc4);acc4=0.0f;}
        if(isfinite(acc5))acc5=NR_PREPARED_FIXED_HALF((int)sums1.y,e5-13);
        if(partition!=0u && (kg+16u)%partition==0u){total5=kg<partition?acc5:nr_half(total5+acc5);acc5=0.0f;}
        if(isfinite(acc6))acc6=NR_PREPARED_FIXED_HALF((int)sums1.z,e6-13);
        if(partition!=0u && (kg+16u)%partition==0u){total6=kg<partition?acc6:nr_half(total6+acc6);acc6=0.0f;}
        if(isfinite(acc7))acc7=NR_PREPARED_FIXED_HALF((int)sums1.w,e7-13);
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
    if(rawEnabled!=0)nr_prepared_store4(raw,i,acc0,acc1,acc2,acc3,rawFormat);
    nr_prepared_store4(output,i,acc0,acc1,acc2,acc3,outputFormat);
  }
  if(row+1u<rows && col+3u<N){
    if(rawEnabled!=0)nr_prepared_store4(raw,next,acc4,acc5,acc6,acc7,rawFormat);
    nr_prepared_store4(output,next,acc4,acc5,acc6,acc7,outputFormat);
  }
}
