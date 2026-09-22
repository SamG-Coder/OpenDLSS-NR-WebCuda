import test from 'node:test';
import assert from 'node:assert/strict';
import {e4,half} from '../src/model.js';
import {precomputeMatrix,PRECOMPUTED_FORMAT_VERSION} from '../src/precomputed-model.js';
import {prepareMatrix} from '../src/model-preparation.js';
import {createPreparedRecord,validatePreparedRecord} from '../src/prepared-model-cache.js';

function pack(codes) {
  const words=new Uint32Array(codes.length/4);
  codes.forEach((code,i)=>words[i>>>2]|=code<<((i&3)*8));return words;
}
function operands(prepared,k,n,batch=0) {
  const {K,N,words}=prepared;
  const index=(((batch*(K/32)+Math.floor(k/32))*(N/32)+Math.floor(n/32))*16+Math.floor((k%32)/2))*32+(n%32);
  const shift=(k&1)*16;
  return {value:half((words[index]>>>shift)&65535),exponent:half((words[prepared.metadataWordOffset+index]>>>shift)&65535)};
}

test('Precomputed operands exactly reproduce all finite FP8 values, scaling and clamped exponents',()=>{
  const codes=Uint8Array.from({length:32*32},(_,i)=>i%256===127||i%256===255?0:i%256);
  const prepared=precomputeMatrix(pack(codes),{K:32,N:32}),seen=new Set();
  for(let k=0;k<32;k++)for(let n=0;n<32;n++) {
    const code=codes[k*32+n],operand=operands(prepared,k,n),value=e4(code);
    assert.equal(operand.value,value*4,`scaled value code ${code}`);
    assert.equal(operand.exponent,value===0?-128:Math.max(Math.floor(Math.log2(Math.abs(value))),-6),`exponent code ${code}`);
    seen.add(code);
  }
  assert.equal(seen.size,254);
  assert(Object.is(operands(prepared,4,0).value,-0));
  assert.equal(prepared.words.byteLength,codes.length*4);
  assert.equal(prepared.payloadWordLength,codes.length/2);
  assert.equal(prepared.metadataWordOffset,codes.length/2);
  assert.equal(prepared.metadataByteLength,codes.length*2);
});

test('Precomputed operand ordering spans batches, K slabs and N tiles without changing the source',()=>{
  const spec={K:96,N:64,batches:3},codes=Uint8Array.from({length:spec.K*spec.N*spec.batches},(_,i)=>{
    const code=(i*37+(i>>>5))&255;return(code&127)===127?128:code;
  });
  // Exercise a sliced source view as well as the explicit numerical byte order.
  const backing=new Uint32Array(codes.length/4+9),source=backing.subarray(3,3+codes.length/4);
  source.set(pack(codes));const copy=source.slice(),prepared=precomputeMatrix(source,spec);
  for(let b=0;b<spec.batches;b++)for(let k=0;k<spec.K;k++)for(let n=0;n<spec.N;n++) {
    const code=codes[(b*spec.K+k)*spec.N+n],operand=operands(prepared,k,n,b);
    assert.equal(operand.value,e4(code)*4);
    assert.equal(operand.exponent,(code&127)===0?-128:Math.max((code>>>3)&15,1)-7);
  }
  assert.deepEqual(source,copy);assert.equal(prepared.boundedHalf,false);
});

test('Precomputed matrices validate shape, length, finite codes and the native half product bound',()=>{
  for(const spec of [{K:16,N:32},{K:32,N:4},{K:32,N:32,halfMode:true},{K:32,N:32,batches:0}])assert.throws(()=>precomputeMatrix(new Uint32Array(256),spec),/multiples of 32/);
  assert.throws(()=>precomputeMatrix(new Uint32Array(255),{K:32,N:32}),/length mismatch/);
  for(const code of [127,255]){const words=new Uint32Array(256);words[42]=code<<16;assert.throws(()=>precomputeMatrix(words,{K:32,N:32}),/Nonfinite/);}
  const words=new Uint32Array(256).fill(0xd151d151);
  assert.equal(precomputeMatrix(words,{K:32,N:32}).boundedHalf,true);
  words[0]=0x52;assert.equal(precomputeMatrix(words,{K:32,N:32}).boundedHalf,false);
});

test('Precomputed cache authenticates both operand planes and cannot reinterpret compact FP8 records',async()=>{
  const spec={K:32,N:64,batches:2},source=new Uint32Array(spec.K*spec.N*spec.batches/4).fill(0x81380001);
  const descriptor=precomputeMatrix(source,spec);
  const identity={...spec,modelHash:'a'.repeat(64),matrixKey:'test',formatVersion:PRECOMPUTED_FORMAT_VERSION,layout:PRECOMPUTED_FORMAT_VERSION};
  const record=await createPreparedRecord(descriptor,identity);
  assert.deepEqual(await validatePreparedRecord(record,identity),descriptor);
  for(const index of [0,descriptor.payloadWordLength,descriptor.words.length-1]) {
    const changed=record.words.slice();changed[index]^=0x100;
    assert.equal(await validatePreparedRecord({...record,words:changed},identity),null);
  }
  for(const changes of [{layout:undefined},{layout:'other'},{boundedHalf:false},{metadataWordOffset:0},{metadataByteLength:1},{formatVersion:1}])assert.equal(await validatePreparedRecord({...record,...changes},identity),null);
  assert.equal(await validatePreparedRecord(record,{...identity,layout:undefined}),null);
  assert.equal(await validatePreparedRecord(record,{...identity,layout:'other'}),null);
  const compact=prepareMatrix(source,spec),compactIdentity={...spec,modelHash:identity.modelHash,matrixKey:identity.matrixKey,formatVersion:1};
  const compactRecord=await createPreparedRecord(compact,compactIdentity);
  assert.deepEqual(await validatePreparedRecord(compactRecord,compactIdentity),compact);
  assert.equal(await validatePreparedRecord(compactRecord,identity),null);
  assert.equal(await validatePreparedRecord({...record,layout:undefined,formatVersion:1},compactIdentity),null);
});
