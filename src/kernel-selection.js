// Shape choices measured on the reference RTX 5080. Explicit modes remain
// available for profiling other devices; this is not a hardware autotuner.
export function gemmKernel({rows,K,N,halfMode},mode='auto') {
  if(halfMode||mode==='scalar')return 'nr_gemm_packed';
  if(mode==='tile8x8')return 'nr_gemm_tile8x8';
  if(mode==='tile8x16')return 'nr_gemm_tile8x16';
  if(mode==='tiled'||rows<8)return 'nr_gemm_tiled';
  return (N<=K&&N!==512)||(K===128&&N===384)?'nr_gemm_tile8x16':'nr_gemm_tile8x8';
}

export function dispatchGroups(entry,s,count) {
  const tile={nr_gemm_tiled:[4,16],nr_gemm_tile8x8:[8,8],nr_gemm_tile8x16:[8,16]}[entry];
  if(tile)return Math.ceil(s.rows/tile[0])*s.batches*Math.ceil(s.N/tile[1]);
  if(entry==='nr_scores_tiled'||entry==='nr_attend_tiled'){
    const score=entry==='nr_scores_tiled';
    return s.heads*(s.globalMode?Math.ceil(s.width*s.height/4)*(score?s.padded/16:2):Math.ceil((s.width+s.shiftX)/8)*Math.ceil((s.height+s.shiftY)/8)*(score?64:32));
  }
  return Math.ceil(count/64);
}
