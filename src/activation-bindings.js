export const activationBindings={
  nr_local_attention:['qkv','output'],
  nr_publish:['input','output'],nr_pool:['input','output'],nr_merge:['low','skip','raw','output'],nr_compose:['head'],
  nr_gemm:['input','residual','raw','output'],nr_gemm_packed:['input','residual','raw','output'],
  nr_gemm_tiled:['input','residual','raw','output'],nr_gemm_tile8x8:['input','residual','raw','output'],nr_gemm_tile8x16:['input','residual','raw','output'],
  nr_normalize:['qkv','output'],nr_scores:['qkv','scores'],nr_scores_tiled:['qkv','scores'],
  nr_softmax:['scores','weights','inverse'],nr_attend:['qkv','weights','inverse','output'],nr_attend_tiled:['qkv','weights','inverse','output'],
};
