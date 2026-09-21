import test from 'node:test';
import assert from 'node:assert/strict';
import {loadFixture,runParity,compareBits,encodeBoundary} from '../src/parity.js';
import {createGraph} from '../src/graph.js';
import {e4} from '../src/model.js';
test('Fixture replay preserves legacy conditioning independently of interactive defaults',async()=>{
  const calls=[],engine={run:async input=>{calls.push(input.conditioning);return {head:new Float32Array(4)};}};
  const fixture={manifest:{conditioning:{style:2,localTone:0.25},autoMask:0},width:33,height:33,boundaries:new Map()};
  assert.equal((await runParity(engine,fixture,{repeat:2})).passed,true);
  assert.deepEqual(calls,[{localTone:0.25,localStructure:0,style:2,autoMask:0},{localTone:0.25,localStructure:0,style:2,autoMask:0}]);
});
test('Boundary encoding retains all finite E4 byte codes including negative zero',()=>{
  const codes=Array.from({length:256},(_,i)=>i).filter(i=>(i&127)!==127);
  assert.deepEqual(Array.from(encodeBoundary(Float32Array.from(codes,e4))),codes);
});
test('Parity fails on signed-zero differences and mismatched length',()=>{
  const result=compareBits(new Float32Array([0]),new Float32Array([-0]));assert.equal(result.mismatches,1);assert.equal(result.signedZeros,1);
  assert.throws(()=>compareBits(new Float32Array(2),new Float32Array(1)),/size/);
});
test('Fixture preflight refuses incomplete coverage, bad shapes, orphan checks and short files',async()=>{
  const g=createGraph(33,33),dims=[g.geometry.fullWidth,g.geometry.fullHeight],features=new Uint8Array(dims[0]*dims[1]*64);
  const manifest={sourceDimensions:[33,33],fullDimensions:dims,inputFeatures:{file:'features.bin'},checks:['head'],referenceHead:{file:'head.bin'}};
  const files={'features.bin':features,'head.bin':new Uint8Array(dims[0]*dims[1]*16)};
  const read=async p=>{if(p==='manifest.json')return new TextEncoder().encode(JSON.stringify(manifest));if(!files[p])throw Error('Missing '+p);return files[p];};
  await loadFixture(read);
  manifest.checks=['head','boundaries'];await assert.rejects(loadFixture(read),/Unaccounted boundary/);
  manifest.omittedBoundaries=Object.fromEntries(Object.keys(g.boundaries).map(n=>[n,'Not captured']));
  await assert.rejects(loadFixture(read),/no references/);
  delete manifest.omittedBoundaries['block-0'];const shape=g.resources.get(g.boundaries['block-0']);
  manifest.blocks=[{block:0,width:shape.width,height:shape.height,channels:shape.channels,file:'block.bin'}];files['block.bin']=new Uint8Array(shape.rows*shape.channels);
  await loadFixture(read);
  manifest.blocks[0].width++;await assert.rejects(loadFixture(read),/shape/);manifest.blocks[0].width--;
  files['block.bin']=new Uint8Array(4);await assert.rejects(loadFixture(read),/length/);
  manifest.checks=[];await assert.rejects(loadFixture(read),/checks/);
});
