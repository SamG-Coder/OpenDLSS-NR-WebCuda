// Only graph-invariant scalar parameters are specialized. Row counts stay dynamic.
export const gemmConstants=['K','N','batches','inputStride','inputBatchStride','partition','halfMode','silu','hasResidual','quantize','inputFormat','residualFormat','rawFormat','outputFormat','rawEnabled'];
export function specializedGemm(entry,scalars) {
  if(!/^nr_gemm_tile8x(8|16)_compact$/.test(entry)||scalars.halfMode)return null;
  if(!gemmConstants.every(key=>Number.isSafeInteger(scalars[key])&&scalars[key]>=0))return null;
  return entry+'_s'+gemmConstants.map(key=>scalars[key]).join('_');
}
export function dynamicGemmScalars(scalars) {
  return Object.fromEntries(Object.entries(scalars).filter(([key])=>!gemmConstants.includes(key)));
}

// Check original FP8 codes once when a matrix is cached. Reject NaN codes conservatively.
export function boundedHalfWeights(words){
  if(!(words instanceof Uint32Array))return false;
  for(const word of words)for(let shift=0;shift<32;shift+=8)if(((word>>>shift)&127)>81)return false;
  return true;
}
