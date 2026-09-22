import {createGraph} from './graph.js';
import {createPreparedModelCache,createPreparedRecord,validatePreparedRecord,hashModelTensors} from './prepared-model-cache.js';
export {clearPreparedModelCache} from './prepared-model-cache.js';

export const PREPARED_FORMAT_VERSION=1;
export const preparedMatrixKey=spec=>JSON.stringify(spec);

function dimensions(spec) {
  const {K,N,batches=1,halfMode=false}=spec||{};
  if(halfMode||![K,N,batches].every(v=>Number.isSafeInteger(v)&&v>0)||K%32||N%32||!Number.isSafeInteger(K*N*batches))return null;
  return {K,N,batches};
}

// Payload: [batch][K32 slab][N32 tile][K4 group][column], four K codes per word.
// Tail: one byte per [batch][K16 group][column]. Its low nibble is exponent+6;
// bits 4/5/6 mean all-zero, one nonzero exponent, and zero-free respectively.
export function prepareMatrix(rowMajorWords,spec) {
  const shape=dimensions(spec);
  if(!shape)throw Error('Prepared matrices require FP8 weights and K/N multiples of 32.');
  const {K,N,batches}=shape,values=K*N*batches,payloadWordLength=values/4,metadataByteLength=values/16;
  if(!(rowMajorWords instanceof Uint32Array)||rowMajorWords.length!==payloadWordLength)throw Error('Prepared matrix source length mismatch.');
  const words=new Uint32Array(payloadWordLength+metadataByteLength/4);
  let boundedHalf=true;
  // Reading numerical words explicitly preserves byte order even for sliced arrays.
  const codeAt=i=>(rowMajorWords[i>>>2]>>>((i&3)*8))&255;
  for(let batch=0;batch<batches;batch++)for(let slab=0;slab<K/32;slab++)for(let tile=0;tile<N/32;tile++) {
    const tileBase=((batch*(K/32)+slab)*(N/32)+tile)*256;
    for(let group=0;group<8;group++)for(let column=0;column<32;column++) {
      const source=(batch*K+slab*32+group*4)*N+tile*32+column;
      let packed=0;
      for(let lane=0;lane<4;lane++) {
        const code=codeAt(source+lane*N),magnitude=code&127;
        if(magnitude===127)throw Error('Nonfinite FP8 code in prepared matrix.');
        if(magnitude>81)boundedHalf=false;
        packed|=code<<(lane*8);
      }
      words[tileBase+group*32+column]=packed;
    }
  }
  for(let batch=0;batch<batches;batch++)for(let group=0;group<K/16;group++)for(let column=0;column<N;column++) {
    let maximum=-6,first=-128,constant=true,zeroFree=true,nonzero=false;
    for(let lane=0;lane<16;lane++) {
      const code=codeAt((batch*K+group*16+lane)*N+column);
      if((code&127)===0){zeroFree=false;continue;}
      const exponent=Math.max((code>>>3)&15,1)-7;
      if(first===-128)first=exponent;
      else if(exponent!==first)constant=false;
      maximum=Math.max(maximum,exponent);nonzero=true;
    }
    const metadata=(maximum+6)|(!nonzero?16:0)|(constant?32:0)|(zeroFree?64:0);
    const index=(batch*(K/16)+group)*N+column;
    words[payloadWordLength+(index>>>2)]|=metadata<<((index&3)*8);
  }
  return {words,boundedHalf,K,N,batches,payloadWordLength,metadataWordOffset:payloadWordLength,metadataByteLength};
}

export async function prepareModel(model,{cache=true,onProgress=()=>{}}={}) {
  const graph=createGraph(1280,720,{activationStorage:'packed',fuseLocalAttention:true});
  const specs=new Map();
  for(const op of graph.ops)for(const spec of Object.values(op.bindings))if(spec?.kind==='matrix')specs.set(preparedMatrixKey(spec),spec);
  const matrices=new Map(),stats={matrixCount:specs.size,preparedCount:0,skippedCount:0,cacheHits:0,cacheMisses:0,cacheWrites:0,cacheRejected:0,
    payloadBytes:0,metadataBytes:0,preparedBytes:0,cacheAvailable:false,skipped:[]};
  // A secure browser context has SHA-256. Preparation still works without local storage.
  const modelHash=globalThis.crypto?.subtle?await hashModelTensors(model,onProgress):null;
  const persistent=cache&&modelHash?await createPreparedModelCache():null;
  stats.cacheAvailable=!!persistent;
  let completed=0,lastYield=performance.now();
  try {
    for(const [matrixKey,spec] of specs) {
      completed++;
      const shape=dimensions(spec);
      if(!shape){
        stats.skippedCount++;stats.skipped.push({matrixKey,reason:'unsupported-shape'});
        onProgress({phase:'prepare',completed,total:specs.size,matrixKey,skipped:true,cacheHit:false});
        continue;
      }
      const cacheKey=`${PREPARED_FORMAT_VERSION}/${modelHash}/${matrixKey}`;
      const identity={modelHash,matrixKey,formatVersion:PREPARED_FORMAT_VERSION,...shape};
      let descriptor=null;
      if(persistent) {
        const record=await persistent.get(cacheKey);
        if(record) {
          descriptor=await validatePreparedRecord(record,identity);
          if(!descriptor){stats.cacheRejected++;await persistent.delete(cacheKey);}
        }
      }
      const cacheHit=!!descriptor;
      if(descriptor)stats.cacheHits++;
      else {
        stats.cacheMisses++;
        try {
          descriptor=prepareMatrix(model.packedMatrix(spec.name,spec.offset,spec.K,spec.N,spec),spec);
        } catch(error) {
          // NaN encodings cannot use the prepared arithmetic; retain the existing kernel.
          if(!/Nonfinite FP8 code/.test(error.message))throw error;
          stats.skippedCount++;stats.skipped.push({matrixKey,reason:'nonfinite-fp8'});
        }
        if(descriptor&&persistent?.writable) {
          const record=await createPreparedRecord(descriptor,identity);
          if(await persistent.set(cacheKey,record))stats.cacheWrites++;
        }
      }
      if(descriptor) {
        matrices.set(matrixKey,descriptor);stats.preparedCount++;
        stats.payloadBytes+=descriptor.payloadWordLength*4;stats.metadataBytes+=descriptor.metadataByteLength;stats.preparedBytes+=descriptor.words.byteLength;
      }
      onProgress({phase:'prepare',completed,total:specs.size,matrixKey,cacheHit,skipped:!descriptor,...shape});
      if(performance.now()-lastYield>32){await new Promise(resolve=>setTimeout(resolve,0));lastYield=performance.now();}
    }
  } finally {persistent?.close();}
  return {matrices,stats,modelHash,formatVersion:PREPARED_FORMAT_VERSION};
}
