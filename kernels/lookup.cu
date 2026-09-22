#include "numeric.cuh"
// Model-independent numerical tables, built once on the same GPU as inference.
__global__ void nr_lookup_tables(float* metadata,float* silu){
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<256u){float v=nr_e4_decode(i);metadata[2u*i]=v*4.0f;metadata[2u*i+1u]=v==0.0f?-128.0f:(float)nr_exp(v,-6);}
  if(i<65536u)silu[i]=nr_silu(nr_from_half(i));
}
