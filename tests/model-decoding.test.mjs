import test from 'node:test';
import assert from 'node:assert/strict';
import {Model,half,e4,packedIndex,inverseInput,halfIndex} from '../src/model.js';

test('Table decoding preserves every half value including signed zero and NaNs',()=>{
  const bytes=new Uint8Array(65536*2),view=new DataView(bytes.buffer);
  for(let i=0;i<65536;i++)view.setUint16(i*2,i,true);
  const model=new Model({},new Map([['values',bytes]])),values=model.vector('values',0,65536);
  for(let i=0;i<65536;i++)assert(Object.is(values[i],half(i)),`half ${i}`);
});

test('Packed upload preserves original codes, batches, offsets, and padded half columns',()=>{
  for(const halfMode of [false,true])for(const N of halfMode?[4,32]:[32,128]){
    const K=32,batches=2,offset=8,length=batches*K*(halfMode?Math.ceil(N/16)*16*2:N);
    const bytes=Uint8Array.from({length:offset+length},(_,i)=>(i*73+17)%256),view=new DataView(bytes.buffer);
    const model=new Model({},new Map([['values',bytes]])),words=model.packedMatrix('values',offset,K,N,{batches,halfMode});
    const lanes=halfMode?2:4,bits=halfMode?16:8;
    assert.equal(words.byteLength,batches*K*N*(halfMode?2:1));
    for(let b=0;b<batches;b++)for(let k=0;k<K;k++)for(let n=0;n<N;n++){
      const i=(b*K+k)*N+n,actual=(words[Math.floor(i/lanes)]>>>((i%lanes)*bits))&(halfMode?65535:255);
      const index=offset+(halfMode?2*(b*K*Math.ceil(N/16)*16+halfIndex(k,n,N)):packedIndex(b*K+inverseInput(k),n,N));
      assert.equal(actual,halfMode?view.getUint16(index,true):bytes[index]);
    }
    assert.throws(()=>model.packedMatrix('values',offset+4,K,N,{batches,halfMode}),/Tensor bounds/);
  }
});

test('Packed matrices retain every FP8 code and half layout after table decoding',()=>{
  const bytes=Uint8Array.from({length:4096},(_,i)=>i%256),model=new Model({},new Map([['values',bytes]]));
  const values=model.matrix('values',0,32,128);
  for(let k=0;k<32;k++)for(let n=0;n<128;n++)assert(Object.is(values[k*128+n],e4(bytes[packedIndex(inverseInput(k),n,128)])));
  const fp16=model.matrix('values',0,32,32,{halfMode:true}),view=new DataView(bytes.buffer);
  for(let k=0;k<32;k++)for(let n=0;n<32;n++)assert(Object.is(fp16[k*32+n],half(view.getUint16(2*halfIndex(k,n,32),true))));
});
