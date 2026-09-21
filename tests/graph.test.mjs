import test from 'node:test';
import assert from 'node:assert/strict';
import {createGraph} from '../src/graph.js';
import {geometry,layout} from '../src/geometry.js';
import {Model,packedIndex,inverseInput,halfIndex,half,e4} from '../src/model.js';
test('Native geometry examples and small-size rejection',()=>{
  for(const [w,h,fw,fh] of [[512,512,576,512],[768,768,832,768],[644,768,768,768],[1920,1080,1920,1152],[3840,2160,3840,2176]]) {
    const g=geometry(w,h);assert.deepEqual([g.fullWidth,g.fullHeight],[fw,fh]);
  }
  for(const x of [0,32,NaN,Infinity,4.5])assert.throws(()=>geometry(x,512));
});
test('Full graph covers 70 block boundaries and five encoder transitions',()=>{
  const graph=createGraph(512,512),names=Object.keys(graph.boundaries);
  assert.equal(names.length,75);
  for(let b=0;b<70;b++)assert.ok(names.includes('block-'+b));
  for(const pair of ['4-5','8-9','14-15','22-23','30-31'])assert.ok(names.includes('transition-'+pair));
  assert.equal(graph.ops.filter(x=>x.entry==='nr_normalize'&&x.scalars.globalMode).length,8);
  assert.equal(graph.ops.filter(x=>x.entry==='nr_normalize'&&!x.scalars.globalMode).length,62);
  assert.equal(graph.ops.at(-1).scalars.halfMode,1);
});
test('All graph reads have a previous producer; no storage output aliases input',()=>{
  const g=createGraph(512,512),available=new Set([g.features]);
  for(const op of g.ops) {
    const outs=op.entry==='nr_scores'?['scores']:op.entry==='nr_softmax'?['weights','inverse']:['raw','output'];
    for(const [key,v] of Object.entries(op.bindings))if(typeof v==='string'&&!outs.includes(key))assert.ok(available.has(v),`${op.entry}.${key}: ${v}`);
    for(const key of outs)if(typeof op.bindings[key]==='string'){assert.ok(!available.has(op.bindings[key]));available.add(op.bindings[key]);}
  }
});
test('Decoder window phases continue encoder counts',()=>{
  const g=createGraph(512,512),scores=g.ops.filter(x=>x.entry==='nr_scores'&&!x.scalars.globalMode);
  assert.deepEqual([scores[0].scalars.shiftX,scores[0].scalars.shiftY],[0,0]);
  assert.deepEqual([scores.at(-1).scalars.shiftX,scores.at(-1).scalars.shiftY],[4,4]);
  const l2=g.geometry.levels[2],stage=scores.filter(x=>x.scalars.width===l2.width&&x.scalars.height===l2.height);
  assert.deepEqual([stage[6].scalars.shiftX,stage[6].scalars.shiftY],[4,0]);
});
test('Packed weight indexing is a bijection at every channel shape',()=>{
  for(const N of [16,32,64,96,128,192,256,512,1024,3072,4096]) {
    const seen=new Set();for(let k=0;k<64;k++)for(let n=0;n<N;n++)seen.add(packedIndex(k,n,N));
    assert.equal(seen.size,64*N);assert.equal([...seen].reduce((a,b)=>Math.max(a,b),0),64*N-1);
  }
  assert.equal(new Set(Array.from({length:64},(_,k)=>inverseInput(k))).size,64);
  for(const N of [4,32])assert.equal(new Set(Array.from({length:32*N},(_,i)=>halfIndex(Math.floor(i/N),i%N,N))).size,32*N);
});
test('Known native fused layouts',()=>{
  assert.equal(layout(32).ffn,8208);assert.equal(layout(32).end,20656);
  assert.equal(layout(32,'up').up,8192);assert.equal(layout(32,'up').transition,10336);
  assert.equal(layout(32,'pre').adapter,8208);assert.equal(layout(32,'post').head,20784);
});
test('Model loader rejects missing stages, incorrect hashes and short slices',async()=>{
  const bytes=new Uint8Array(4),digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
  const manifest={totals:{blockCount:71},stages:[{id:'a',file:'a.bin',packedByteLength:4,sha256:digest}],tensors:[{name:'block0.layer0.layer',stage:'a',stageOffset:0,byteLength:4}]};
  const read=async p=>p==='manifest.json'?new TextEncoder().encode(JSON.stringify(manifest)):bytes;
  const model=await Model.load(read);assert.equal(model.tensor(0),'block0.layer0.layer');
  manifest.tensors[0].byteLength=5;await assert.rejects(Model.load(read),/slice/);manifest.tensors[0].byteLength=4;
  manifest.stages[0].sha256='0'.repeat(64);await assert.rejects(Model.load(read),/SHA-256/);
  manifest.stages[0].file='../secret';await assert.rejects(Model.load(read),/stage/);
});
test('Scalar model decoding preserves signed zero and E4 maximum',()=>{
  assert.ok(Object.is(half(32768),-0));assert.ok(Object.is(e4(128),-0));assert.equal(e4(126),448);assert.equal(e4(127),0);
});
