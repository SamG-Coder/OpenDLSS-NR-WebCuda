#include <cuda_fp16.h>
#include "attention.cu"
#include "fast-half.cuh"
#include "activations.cuh"
__global__ void nr_local_attention_normalized(const unsigned* qkv, const float* prior, const float* scales, unsigned* output, unsigned width, unsigned height, unsigned heads, unsigned shiftX, unsigned shiftY) {
  __shared__ float queries[1024];
  __shared__ float tileData[2112];
  __shared__ float probabilities[2080];
  __shared__ float norms[96];
  // Exponents are small exact integers; half storage keeps the tile below 32 KiB.
  __shared__ __half queryExp[1024];
  __shared__ __half tileExp[2112];
  __shared__ __half probabilityExp[2080];
  unsigned tid = threadIdx.x;
  unsigned tile = blockIdx.x + blockIdx.y * gridDim.x;
  unsigned queryBase = tile % 2u * 32u;
  unsigned head = tile / 2u % heads;
  unsigned window = tile / 2u / heads;
  unsigned windows = ((width + shiftX + 7u) / 8u) * ((height + shiftY + 7u) / 8u);
  if (window >= windows) return;
  // All 512 lanes cooperate on the 96 Q/K norms. Preserve the original
  // half-rounded pair and tree reduction order in temporary shared storage.
  for(unsigned t=tid;t<1536u;t+=512u){
    unsigned n=t/16u,c=t%16u;
    int r=nr_window_row(n<64u?nr_physical_to_natural(n):queryBase+n-64u,window,width,height,shiftX,shiftY);
    unsigned base=((unsigned)max(r,0)*heads+head)*96u+(n<64u?32u:0u);
    float a=r<0?0.0f:nr_activation_load(qkv,base+c,2);
    float b=r<0?0.0f:nr_activation_load(qkv,base+c+16u,2);
    tileData[t]=nr_half(fmaf(a,a,nr_half(b*b)));
  }
  __syncthreads();
  for(unsigned stride=8u;stride>0u;stride>>=1u){
    for(unsigned t=tid;t<96u*stride;t+=512u){
      unsigned slot=t/stride*16u+t%stride;
      tileData[slot]=nr_half(tileData[slot]+tileData[slot+stride]);
    }
    __syncthreads();
  }
  if(tid<96u){float sum=tileData[tid*16u];norms[tid]=sum==0.0f?0.0f:nr_half(rsqrtf(sum));}
  __syncthreads();
  for (unsigned t = tid; t < 1024u; t += 512u) {
    int row = nr_window_row(queryBase + t / 32u, window, width, height, shiftX, shiftY);
    queries[t] = row < 0 ? 0.0f : nr_quant(nr_half(nr_half(nr_activation_load(qkv,((unsigned)row * heads + head) * 96u + t % 32u,2)*norms[64u+t/32u])*nr_half(scales[head])));
    queryExp[t]=__float2half_rn(queries[t]==0.0f?-128.0f:(float)nr_exp(queries[t],-6));
  }
  for (unsigned t = tid; t < 2048u; t += 512u) {
    int row = nr_window_row(nr_physical_to_natural(t / 32u), window, width, height, shiftX, shiftY);
    tileData[(t / 32u) * 33u + t % 32u] = row < 0 ? 0.0f : nr_quant(nr_half(nr_activation_load(qkv,((unsigned)row * heads + head) * 96u + 32u + t % 32u,2)*norms[t/32u]));
    unsigned slot=(t/32u)*33u+t%32u;
    tileExp[slot]=__float2half_rn(tileData[slot]==0.0f?-128.0f:(float)nr_exp(tileData[slot],-6));
  }
  __syncthreads();
  unsigned query = tid / 16u;
  for (unsigned keyBase = 0u; keyBase < 64u; keyBase += 16u) {
    unsigned key = keyBase + tid % 16u;
    float acc = prior[(head * 64u + queryBase + query) * 64u + key];
    for (unsigned kb = 0u; kb < 32u; kb += 16u) {
      int e = acc != 0.0f ? nr_exp(acc, -14) : -21;
      #pragma unroll
      for (unsigned c = 0u; c < 16u; c += 1u) {
        e = max(e, (int)__half2float(queryExp[query*32u+kb+c]) + (int)__half2float(tileExp[key*33u+kb+c]));
      }
      int sum = (int)truncf(acc * nr_pow2(13 - e));
      #pragma unroll
      for (unsigned c = 0u; c < 16u; c += 1u) sum += (int)truncf(queries[query * 32u + kb + c] * tileData[key * 33u + kb + c] * nr_pow2(13 - e));
      acc = nr_fast_fixed_half(sum, e - 13);
    }
    probabilities[query * 65u + key] = nr_exp_weight(acc, 0);
  }
  __syncthreads();
  // Parallelize each query's eight partial sums without changing reduction order.
  if (tid < 256u) {
    unsigned q = tid / 8u;
    unsigned g = tid % 8u;
    float sum = 0.0f;
    for (unsigned j = 0u; j < 4u; j += 1u) {
      unsigned k = q * 65u + g + j * 16u;
      float pair = nr_half(probabilities[k] + probabilities[k + 8u]);
      sum = j == 0u ? pair : nr_half(sum + pair);
    }
    queries[tid] = sum;
  }
  __syncthreads();
  if (tid < 32u) {
    unsigned t = tid * 8u;
    float even = nr_half(nr_half(nr_half(queries[t] + queries[t + 2u]) + queries[t + 4u]) + queries[t + 6u]);
    float odd = nr_half(nr_half(nr_half(queries[t + 1u] + queries[t + 3u]) + queries[t + 5u]) + queries[t + 7u]);
    float total = nr_half(0.0f + nr_half(even + odd));
    queries[512u + tid] = nr_half(1.0f / total);
  }
  __syncthreads();
  for (unsigned k = tid % 16u; k < 64u; k += 16u) {
    unsigned slot=query*65u+k;
    probabilities[slot] = nr_quant(nr_half(probabilities[slot] * queries[512u + query]));
    probabilityExp[slot]=__float2half_rn(probabilities[slot]==0.0f?-128.0f:(float)nr_exp(probabilities[slot],-6));
  }
  for (unsigned t = tid; t < 2048u; t += 512u) {
    int row = nr_window_row(nr_physical_to_natural(t / 32u), window, width, height, shiftX, shiftY);
    tileData[(t / 32u) * 33u + t % 32u] = row < 0 ? 0.0f : nr_quant(nr_activation_load(qkv,((unsigned)row * heads + head) * 96u + 64u + t % 32u,2));
    unsigned slot=(t/32u)*33u+t%32u;
    tileExp[slot]=__float2half_rn(tileData[slot]==0.0f?-128.0f:(float)nr_exp(tileData[slot],-6));
  }
  __syncthreads();
  int row = nr_window_row(queryBase + query, window, width, height, shiftX, shiftY);
  for (unsigned channelBase = 0u; channelBase < 32u; channelBase += 16u) {
    unsigned channel = channelBase + tid % 16u;
    float acc = 0.0f;
    for (unsigned kb = 0u; kb < 64u; kb += 16u) {
      int e = acc != 0.0f ? nr_exp(acc, -14) : -21;
      #pragma unroll
      for (unsigned j = 0u; j < 16u; j += 1u) {
        e = max(e, (int)__half2float(probabilityExp[query*65u+kb+j]) + (int)__half2float(tileExp[(kb+j)*33u+channel]));
      }
      if (isfinite(acc)) {
        int sum = (int)truncf(acc * nr_pow2(13 - e));
        #pragma unroll
        for (unsigned j = 0u; j < 16u; j += 1u) sum += (int)truncf(probabilities[query * 65u + kb + j] * tileData[(kb + j) * 33u + channel] * nr_pow2(13 - e));
        acc = nr_fast_fixed_half(sum, e - 13);
      }
    }
    queries[query * 32u + channel] = nr_quant(acc);
  }
  __syncthreads();
  // A lane owns four consecutive channels, allowing one packed-word store.
  if (tid < 256u) {
    unsigned channel = tid % 8u * 4u;
    unsigned q = tid / 8u;
    int target = nr_window_row(queryBase + q, window, width, height, shiftX, shiftY);
    if (target >= 0) {
      unsigned base = ((unsigned)target * heads + head) * 32u + channel;
      unsigned source = q * 32u + channel;
      output[base>>2u]=nr_e4_code(queries[source]) | (nr_e4_code(queries[source+1u])<<8u) | (nr_e4_code(queries[source+2u])<<16u) | (nr_e4_code(queries[source+3u])<<24u);
    }
  }
}
