import {wideTiles,preferredWideTile,supportsWideTile} from './gemm-tiles.js';

// Keep the arithmetic and tile policy identical to the standard wide kernels.
// Only the model operands change: values and exponents are already half2 pairs.
export const precomputedTiles=Object.freeze(Object.fromEntries(Object.entries(wideTiles).map(([name,tile])=>[
  name,Object.freeze({...tile,suffix:name==='32x32'?'_precomputed_half':`_precomputed${name}_half`})
])));
export function precomputedTileForEntry(entry){
  return Object.values(precomputedTiles).find(tile=>entry.endsWith(tile.suffix));
}
export function selectPrecomputedGemm(base,available,limits,mode='auto'){
  if(!base)return null;
  const preferred=mode==='auto'?preferredWideTile(base):mode;
  for(const name of new Set([preferred,'32x32'])){
    const tile=precomputedTiles[name];
    if(!tile)continue;
    const entry=base+tile.suffix;
    if(available[entry]&&supportsWideTile(tile,limits))return {entry,tile};
  }
  return null;
}
