#include <cuda_fp16.h>
#include "attention.cu"
#include "fast-half.cuh"
#include "activations.cuh"
// Exact local cosine normalization; input is the graph's raw half QKV tensor.
__device__ float nr_fused_norm(const unsigned* qkv,unsigned base){
  float r[16];
  for(unsigned c=0u;c<16u;c+=1u){
    float a=nr_activation_load(qkv,base+c,2);
    float b=nr_activation_load(qkv,base+c+16u,2);
    r[c]=nr_half(fmaf(a,a,nr_half(b*b)));
  }
  for(unsigned stride=8u;stride>0u;stride>>=1u)for(unsigned c=0u;c<stride;c+=1u)r[c]=nr_half(r[c]+r[c+stride]);
  return r[0]==0.0f?0.0f:nr_half(rsqrtf(r[0]));
}
__global__ void nr_local_attention_normalized(const unsigned* qkv, const float* prior, const float* scales, unsigned* output, unsigned width, unsigned height, unsigned heads, unsigned shiftX, unsigned shiftY) {
  __shared__ float queries[1024];
  __shared__ float tileData[2112];
  __shared__ float probabilities[2080];
  __shared__ float norms[96];
  unsigned tid = threadIdx.x;
  unsigned tile = blockIdx.x + blockIdx.y * gridDim.x;
  unsigned queryBase = tile % 2u * 32u;
  unsigned head = tile / 2u % heads;
  unsigned window = tile / 2u / heads;
  unsigned windows = ((width + shiftX + 7u) / 8u) * ((height + shiftY + 7u) / 8u);
  if (window >= windows) return;
  if(tid < 64u){
    int kr=nr_window_row(nr_physical_to_natural(tid),window,width,height,shiftX,shiftY);
    norms[tid]=kr<0?0.0f:nr_fused_norm(qkv,((unsigned)kr*heads+head)*96u+32u);
  }
  if(tid < 32u){
    int qr=nr_window_row(queryBase+tid,window,width,height,shiftX,shiftY);
    norms[64u+tid]=qr<0?0.0f:nr_fused_norm(qkv,((unsigned)qr*heads+head)*96u);
  }
  __syncthreads();
  for (unsigned t = tid; t < 1024u; t += 512u) {
    int row = nr_window_row(queryBase + t / 32u, window, width, height, shiftX, shiftY);
    queries[t] = row < 0 ? 0.0f : nr_quant(nr_half(nr_half(nr_activation_load(qkv,((unsigned)row * heads + head) * 96u + t % 32u,2)*norms[64u+t/32u])*nr_half(scales[head])));
  }
  for (unsigned t = tid; t < 2048u; t += 512u) {
    int row = nr_window_row(nr_physical_to_natural(t / 32u), window, width, height, shiftX, shiftY);
    tileData[(t / 32u) * 33u + t % 32u] = row < 0 ? 0.0f : nr_quant(nr_half(nr_activation_load(qkv,((unsigned)row * heads + head) * 96u + 32u + t % 32u,2)*norms[t/32u]));
  }
  __syncthreads();
  unsigned query = tid / 16u;
  for (unsigned keyBase = 0u; keyBase < 64u; keyBase += 16u) {
    unsigned key = keyBase + tid % 16u;
    float acc = prior[(head * 64u + queryBase + query) * 64u + key];
    for (unsigned kb = 0u; kb < 32u; kb += 16u) {
      int e = acc != 0.0f ? nr_exp(acc, -14) : -21;
      for (unsigned c = 0u; c < 16u; c += 1u) {
        float a = queries[query * 32u + kb + c];
        float b = tileData[key * 33u + kb + c];
        if (a != 0.0f && b != 0.0f) e = max(e, nr_exp(a, -6) + nr_exp(b, -6));
      }
      int sum = (int)truncf(acc * nr_pow2(13 - e));
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
  for (unsigned k = tid % 16u; k < 64u; k += 16u) probabilities[query * 65u + k] = nr_quant(nr_half(probabilities[query * 65u + k] * queries[512u + query]));
  for (unsigned t = tid; t < 2048u; t += 512u) {
    int row = nr_window_row(nr_physical_to_natural(t / 32u), window, width, height, shiftX, shiftY);
    tileData[(t / 32u) * 33u + t % 32u] = row < 0 ? 0.0f : nr_quant(nr_activation_load(qkv,((unsigned)row * heads + head) * 96u + 64u + t % 32u,2));
  }
  __syncthreads();
  int row = nr_window_row(queryBase + query, window, width, height, shiftX, shiftY);
  for (unsigned channelBase = 0u; channelBase < 32u; channelBase += 16u) {
    unsigned channel = channelBase + tid % 16u;
    float acc = 0.0f;
    for (unsigned kb = 0u; kb < 64u; kb += 16u) {
      int e = acc != 0.0f ? nr_exp(acc, -14) : -21;
      for (unsigned j = 0u; j < 16u; j += 1u) {
        float a = probabilities[query * 65u + kb + j];
        float b = tileData[(kb + j) * 33u + channel];
        if (a != 0.0f && b != 0.0f) e = max(e, nr_exp(a, -6) + nr_exp(b, -6));
      }
      if (isfinite(acc)) {
        int sum = (int)truncf(acc * nr_pow2(13 - e));
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
