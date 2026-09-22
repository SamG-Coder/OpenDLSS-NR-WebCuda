import {createGraph} from './graph.js';
import {e4} from './model.js';
import {preparedMatrixKey} from './model-preparation.js';
import {createPreparedModelCache,createPreparedRecord,validatePreparedRecord,hashModelTensors} from './prepared-model-cache.js';

export const PRECOMPUTED_FORMAT_VERSION='half-operands-v1';

function dimensions(spec) {
  const {K,N,batches=1,halfMode=false}=spec||{};
  if(halfMode||![K,N,batches].every(v=>Number.isSafeInteger(v)&&v>0)||K%32||N%32||!Number.isSafeInteger(K*N*batches))return null;
  return {K,N,batches};
}

// Every finite FP8 value times four, and every exponent below, is exactly
// representable as a normal binary16 number (or signed zero). No rounding occurs.
const floatValue=new Float32Array(1),floatBits=new Uint32Array(floatValue.buffer);
function exactHalf(value) {
  floatValue[0]=value;
  const bits=floatBits[0],sign=(bits>>>16)&0x8000;
  return (bits&0x7fffffff)===0?sign:sign|((((bits>>>23)&255)-112)<<10)|((bits&0x7fffff)>>>13);
}
const weightHalf=new Uint16Array(256),exponentHalf=new Uint16Array(256);
for(let code=0;code<256;code++) {
  if((code&127)===127)continue;
  weightHalf[code]=exactHalf(e4(code)*4);
  exponentHalf[code]=exactHalf((code&127)===0?-128:Math.max((code>>>3)&15,1)-7);
}

// The shader consumes both planes without per-weight FP8 decoding, scaling,
// exponent extraction or half2 construction. Each word contains consecutive K
// operands at [batch][K32 slab][N32 tile][K pair][column], low K first.
export function precomputeMatrix(rowMajorWords,spec) {
  const shape=dimensions(spec);
  if(!shape)throw Error('Precomputed matrices require FP8 weights and K/N multiples of 32.');
  const {K,N,batches}=shape,values=K*N*batches,payloadWordLength=values/2,metadataByteLength=values*2;
  if(!(rowMajorWords instanceof Uint32Array)||rowMajorWords.length!==values/4)throw Error('Precomputed matrix source length mismatch.');
  const words=new Uint32Array(values);
  let boundedHalf=true;
  const codeAt=i=>(rowMajorWords[i>>>2]>>>((i&3)*8))&255;
  for(let batch=0;batch<batches;batch++)for(let slab=0;slab<K/32;slab++)for(let tile=0;tile<N/32;tile++) {
    const tileBase=((batch*(K/32)+slab)*(N/32)+tile)*512;
    for(let pair=0;pair<16;pair++)for(let column=0;column<32;column++) {
      const source=(batch*K+slab*32+pair*2)*N+tile*32+column;
      const a=codeAt(source),b=codeAt(source+N),magnitudeA=a&127,magnitudeB=b&127;
      if(magnitudeA===127||magnitudeB===127)throw Error('Nonfinite FP8 code in precomputed matrix.');
      if(magnitudeA>81||magnitudeB>81)boundedHalf=false;
      const destination=tileBase+pair*32+column;
      words[destination]=weightHalf[a]|(weightHalf[b]<<16);
      words[payloadWordLength+destination]=exponentHalf[a]|(exponentHalf[b]<<16);
    }
  }
  return {words,boundedHalf,K,N,batches,payloadWordLength,metadataWordOffset:payloadWordLength,metadataByteLength};
}

export async function precomputeModel(model,{cache=true,onProgress=()=>{}}={}) {
  const graph=createGraph(1280,720,{activationStorage:'packed',fuseLocalAttention:true});
  const specs=new Map();
  for(const op of graph.ops)for(const spec of Object.values(op.bindings))if(spec?.kind==='matrix')specs.set(preparedMatrixKey(spec),spec);
  const matrices=new Map(),stats={matrixCount:specs.size,preparedCount:0,skippedCount:0,cacheHits:0,cacheMisses:0,cacheWrites:0,cacheRejected:0,
    payloadBytes:0,metadataBytes:0,valueBytes:0,exponentBytes:0,preparedBytes:0,cacheAvailable:false,skipped:[]};
  const modelHash=globalThis.crypto?.subtle?await hashModelTensors(model,onProgress):null;
  const persistent=cache&&modelHash?await createPreparedModelCache():null;
  stats.cacheAvailable=!!persistent;
  let completed=0,lastYield=performance.now();
  try {
    for(const [matrixKey,spec] of specs) {
      completed++;
      const shape=dimensions(spec);
      if(!shape) {
        stats.skippedCount++;stats.skipped.push({matrixKey,reason:'unsupported-shape'});
        onProgress({phase:'prepare',completed,total:specs.size,matrixKey,skipped:true,cacheHit:false});
        continue;
      }
      const cacheKey=`${PRECOMPUTED_FORMAT_VERSION}/${modelHash}/${matrixKey}`;
      const identity={modelHash,matrixKey,formatVersion:PRECOMPUTED_FORMAT_VERSION,layout:PRECOMPUTED_FORMAT_VERSION,...shape};
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
          descriptor=precomputeMatrix(model.packedMatrix(spec.name,spec.offset,spec.K,spec.N,spec),spec);
        } catch(error) {
          if(!/Nonfinite FP8 code/.test(error.message))throw error;
          stats.skippedCount++;stats.skipped.push({matrixKey,reason:'nonfinite-fp8'});
        }
        if(descriptor&&persistent?.writable) {
          if(await persistent.set(cacheKey,await createPreparedRecord(descriptor,identity)))stats.cacheWrites++;
        }
      }
      if(descriptor) {
        matrices.set(matrixKey,descriptor);stats.preparedCount++;
        stats.valueBytes+=descriptor.payloadWordLength*4;stats.exponentBytes+=descriptor.metadataByteLength;
        stats.payloadBytes+=descriptor.payloadWordLength*4;stats.metadataBytes+=descriptor.metadataByteLength;
        stats.preparedBytes+=descriptor.words.byteLength;
      }
      onProgress({phase:'prepare',completed,total:specs.size,matrixKey,cacheHit,skipped:!descriptor,...shape});
      if(performance.now()-lastYield>32){await new Promise(resolve=>setTimeout(resolve,0));lastYield=performance.now();}
    }
  } finally {persistent?.close();}
  return {matrices,stats,modelHash,formatVersion:PRECOMPUTED_FORMAT_VERSION};
}
