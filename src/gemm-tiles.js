// All variants retain eight outputs per thread and the same ordered F13 arithmetic.
export const wideTiles=Object.freeze(Object.fromEntries([[32,32],[64,32],[32,64]].map(([rows,cols])=>{
  const name=`${rows}x${cols}`;
  return [name,Object.freeze({name,rows,cols,threads:rows*cols/8,sharedBytes:(rows*16+cols*17)*8,suffix:name==='32x32'?'_wide_half':`_wide${name}_half`})];
})));
export function wideTileForEntry(entry){
  return Object.values(wideTiles).find(tile=>entry.endsWith(tile.suffix));
}
export function supportsWideTile(tile,limits){
  return tile.sharedBytes<=limits?.maxComputeWorkgroupStorageSize&&tile.threads<=limits?.maxComputeInvocationsPerWorkgroup&&tile.threads<=limits?.maxComputeWorkgroupSizeX;
}
export function preferredWideTile(base){
  // Provisional policy from contended 720p/1080p RTX 5080 measurements.
  // See reports/performance-tiles.md; confirm performance in a quiet GPU run.
  // Rows remain dynamic; this chooses one pipeline per fixed matrix signature.
  const match=/_compact_s(\d+)_(\d+)_/.exec(base);
  if(!match)return '32x32';
  const [K,N]=match.slice(1).map(Number);
  if(K===32&&N===32)return '64x32';
  if(K===32&&(N===64||N===128))return '32x64';
  if(K===64&&N===128)return '64x32';
  if(K===64&&[64,192,256].includes(N))return '32x64';
  if((K===128&&N===384)||(K===256&&N===768))return '32x64';
  return '32x32';
}
export function selectWideGemm(base,available,limits,mode='auto'){
  const preferred=mode==='auto'?preferredWideTile(base):mode;
  for(const name of new Set([preferred,'32x32'])){
    const tile=wideTiles[name],entry=base+tile.suffix;
    if(available[entry]&&supportsWideTile(tile,limits))return {entry,tile};
  }
  return null;
}
export function wideDispatchGroups(tile,{rows,N,batches}){
  return Math.ceil(rows/tile.rows)*batches*Math.ceil(N/tile.cols);
}
