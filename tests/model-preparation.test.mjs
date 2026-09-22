import test from 'node:test';
import assert from 'node:assert/strict';
import {prepareMatrix,preparedMatrixKey,PREPARED_FORMAT_VERSION} from '../src/model-preparation.js';
import {createPreparedRecord,validatePreparedRecord,hashModelTensors,createPreparedModelCache,clearPreparedModelCache} from '../src/prepared-model-cache.js';

function pack(codes) {
  const words=new Uint32Array(codes.length/4);
  codes.forEach((code,i)=>words[i>>>2]|=code<<((i&3)*8));return words;
}
function unpackPrepared(prepared,k,n,batch=0) {
  const {K,N,words}=prepared;
  const index=(((batch*(K/32)+Math.floor(k/32))*(N/32)+Math.floor(n/32))*8+Math.floor((k%32)/4))*32+(n%32);
  return(words[index]>>>((k%4)*8))&255;
}
function metadata(prepared,group,column,batch=0) {
  const index=(batch*(prepared.K/16)+group)*prepared.N+column;
  return(prepared.words[prepared.metadataWordOffset+(index>>>2)]>>>((index&3)*8))&255;
}

test('Prepared packing roundtrips every code across K slabs, N tiles and batches',()=>{
  const spec={K:96,N:64,batches:3},codes=Uint8Array.from({length:spec.K*spec.N*spec.batches},(_,i)=>{
    const code=(i*37+(i>>>5))&255;return(code&127)===127?128:code;
  });
  const source=pack(codes),copy=source.slice(),prepared=prepareMatrix(source,spec);
  for(let b=0;b<spec.batches;b++)for(let k=0;k<spec.K;k++)for(let n=0;n<spec.N;n++)assert.equal(unpackPrepared(prepared,k,n,b),codes[(b*spec.K+k)*spec.N+n]);
  assert.deepEqual(source,copy);assert.equal(prepared.boundedHalf,false);
  assert.equal(prepared.words.byteLength,codes.length*17/16);
  assert.equal(prepared.metadataWordOffset,codes.length/4);
  assert.equal(prepared.metadataByteLength,codes.length/16);
});

test('Prepared exponent metadata preserves zero, subnormal and constant-exponent distinctions',()=>{
  const K=32,N=32,codes=new Uint8Array(K*N);
  for(let k=0;k<K;k++) {
    codes[k*N]=k%2?128:0; // Both signs of zero remain byte-identical.
    codes[k*N+1]=1+(k%7); // All subnormals share the clamped exponent -6.
    codes[k*N+2]=k?0:0x38; // Constant nonzero exponent, but not zero-free.
    codes[k*N+3]=k%2?0x40:0x30;
    codes[k*N+4]=0x38;
    codes[k*N+5]=k===7?0x7e:1;
  }
  const prepared=prepareMatrix(pack(codes),{K,N});
  assert.equal(metadata(prepared,0,0),0x30);
  assert.equal(metadata(prepared,0,1),0x60);
  assert.equal(metadata(prepared,0,2),0x26);
  assert.equal(metadata(prepared,1,2),0x30);
  assert.equal(metadata(prepared,0,3),0x47);
  assert.equal(metadata(prepared,0,4),0x66);
  assert.equal(metadata(prepared,0,5),0x4e);
  assert.equal(unpackPrepared(prepared,1,0),128);
  for(let g=0;g<2;g++)for(let n=0;n<N;n++)assert.equal(metadata(prepared,g,n)&128,0);
});

test('Independent operand exponent maxima are only an upper bound for pairwise products',()=>{
  const codes=new Uint8Array(32*32).fill(1);codes[0]=0x58;
  const prepared=prepareMatrix(pack(codes),{K:32,N:32});
  const weightMaximum=(metadata(prepared,0,0)&15)-6;
  const activationExponents=Array(16).fill(-6);activationExponents[1]=4;
  const weightExponents=Array(16).fill(-6);weightExponents[0]=4;
  assert.equal(weightMaximum,4);
  assert.equal(Math.max(...activationExponents)+weightMaximum,8);
  assert.equal(Math.max(...activationExponents.map((a,i)=>a+weightExponents[i])),-2);
});

test('Prepared matrices reject unsupported shapes, mismatched lengths and nonfinite FP8',()=>{
  for(const spec of [{K:16,N:32},{K:32,N:4},{K:32,N:32,halfMode:true},{K:32,N:32,batches:0}])assert.throws(()=>prepareMatrix(new Uint32Array(256),spec),/multiples of 32/);
  assert.throws(()=>prepareMatrix(new Uint32Array(255),{K:32,N:32}),/length mismatch/);
  for(const code of [127,255]){const words=new Uint32Array(256);words[42]=code<<16;assert.throws(()=>prepareMatrix(words,{K:32,N:32}),/Nonfinite/);}
  const words=new Uint32Array(256).fill(0xd151d151);
  assert.equal(prepareMatrix(words,{K:32,N:32}).boundedHalf,true);
  words[0]=0x52;assert.equal(prepareMatrix(words,{K:32,N:32}).boundedHalf,false);
});

test('Prepared cache validates contents, identity, version and descriptor metadata',async()=>{
  const spec={kind:'matrix',name:'test',offset:0,K:32,N:64,batches:2,halfMode:false};
  assert.equal(preparedMatrixKey(spec),JSON.stringify(spec));
  const prepared=prepareMatrix(new Uint32Array(spec.K*spec.N*spec.batches/4).fill(0x38383838),spec);
  const identity={modelHash:'a'.repeat(64),matrixKey:preparedMatrixKey(spec),formatVersion:PREPARED_FORMAT_VERSION,...spec};
  const record=await createPreparedRecord(prepared,identity);
  assert.deepEqual(await validatePreparedRecord(record,identity),prepared);
  for(const changes of [{formatVersion:99},{modelHash:'b'.repeat(64)},{matrixKey:'different'},{K:64},{boundedHalf:false},{metadataByteLength:1},{schema:99},{words:new Uint32Array(1)}]){
    assert.equal(await validatePreparedRecord({...record,...changes},identity),null);
  }
  const changed=record.words.slice();changed[changed.length-1]^=0x100;
  assert.equal(await validatePreparedRecord({...record,words:changed},identity),null);
  assert.equal(await validatePreparedRecord({...record,integritySha256:'0'.repeat(64)},identity),null);
});

test('Model identity hashes actual tensor content and names, independent of manifest claims and insertion order',async()=>{
  const model={tensors:new Map([['z',new Uint8Array([1,2])],['a',new Uint8Array([3])]]),manifest:{source:{resourceSha256:'unverified'}}};
  const progress=[],first=await hashModelTensors(model,p=>progress.push(p));
  assert.match(first,/^[0-9a-f]{64}$/);assert.equal(progress.length,2);
  assert.equal(await hashModelTensors({tensors:new Map(Array.from(model.tensors).reverse())}),first);
  model.manifest.source.resourceSha256='different';assert.equal(await hashModelTensors(model),first);
  model.tensors.get('z')[0]=9;assert.notEqual(await hashModelTensors(model),first);
  assert.notEqual(await hashModelTensors({tensors:new Map([['other',new Uint8Array([1,2,3])]])}),first);
});

test('Browser cache is optional and handles unavailable or rejected IndexedDB',async()=>{
  assert.equal(await createPreparedModelCache({indexedDB:null}),null);
  assert.equal(await createPreparedModelCache({indexedDB:{open(){throw Error('disabled');}}}),null);
  assert.equal(await clearPreparedModelCache(),false);
});

test('Browser cache reuses records, clears locally, and disables writes after quota failure',async()=>{
  // A small IndexedDB contract double exercises transaction completion and aborts.
  const records=new Map();let failWrites=false,closed=false,attempts=0;
  const db={
    objectStoreNames:{contains:()=>true},close(){closed=true;},
    transaction(name,mode){
      assert.equal(name,'matrices');const tx={};
      const request=fn=>{
        const operation={};queueMicrotask(()=>{
          if(mode==='readwrite'&&failWrites){attempts++;tx.onabort?.();return;}
          operation.result=fn();operation.onsuccess?.();tx.oncomplete?.();
        });return operation;
      };
      tx.objectStore=()=>({get:key=>request(()=>structuredClone(records.get(key))),
        put:(value,key)=>request(()=>records.set(key,structuredClone(value))),
        delete:key=>request(()=>records.delete(key)),clear:()=>request(()=>records.clear())});
      return tx;
    },
  };
  const indexedDB={open(){const request={};queueMicrotask(()=>{request.result=db;request.onsuccess?.();});return request;}};
  const cache=await createPreparedModelCache({indexedDB});
  assert(cache);assert.equal(cache.writable,true);
  assert.equal(await cache.get('missing'),undefined);
  const value={words:new Uint32Array([1,2,3])};
  assert.equal(await cache.set('key',value),true);assert.deepEqual(await cache.get('key'),value);
  assert.equal(await cache.delete('key'),true);assert.equal(await cache.get('key'),undefined);
  await cache.set('key',value);assert.equal(await cache.clear(),true);assert.equal(records.size,0);
  failWrites=true;assert.equal(await cache.set('key',value),null);assert.equal(cache.writable,false);
  assert.equal(await cache.set('again',value),false);assert.equal(attempts,1);
  cache.close();assert.equal(closed,true);
});
