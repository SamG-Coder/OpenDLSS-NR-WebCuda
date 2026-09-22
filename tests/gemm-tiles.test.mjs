import test from 'node:test';
import assert from 'node:assert/strict';
import {wideTiles,preferredWideTile,wideTileForEntry,selectWideGemm,wideDispatchGroups} from '../src/gemm-tiles.js';
import {NeuralRenderer} from '../src/engine.js';
import {GpuRuntime} from '../vendor/webcuda/runtime/runtime.js';

const base='nr_gemm_tile8x8_compact_s32_96_1_32_0_0_0_0_0_1_1_0_0_1_0';
const limits={maxComputeWorkgroupStorageSize:16384,maxComputeInvocationsPerWorkgroup:256,maxComputeWorkgroupSizeX:256};
const manifest=Object.fromEntries(Object.values(wideTiles).map(t=>[base+t.suffix,base+t.suffix+'.json']));

test('Wide tiles respect memory, invocation and X limits and missing artifacts',()=>{
  for(const tile of Object.values(wideTiles)){
    assert.equal(wideTileForEntry(base+tile.suffix),tile);
    assert.equal(selectWideGemm(base,manifest,limits,tile.name).tile,tile);
  }
  assert.equal(wideTileForEntry(base+'_half'),undefined);
  assert.equal(wideTiles['64x32'].sharedBytes,12544);
  assert.equal(wideTiles['32x64'].sharedBytes,12800);
  for(const reduced of [{maxComputeWorkgroupStorageSize:12000},{maxComputeInvocationsPerWorkgroup:128},{maxComputeWorkgroupSizeX:128}]){
    assert.equal(selectWideGemm(base,manifest,{...limits,...reduced},'64x32').tile.name,'32x32');
  }
  assert.equal(selectWideGemm(base,manifest,{...limits,maxComputeWorkgroupStorageSize:8447},'64x32'),null);
  assert.equal(selectWideGemm(base,{[base+'_wide_half']:true},limits,'32x64').tile.name,'32x32');
  assert.equal(selectWideGemm(base,{},limits,'32x64'),null);
});

test('Wide dispatches cover partial row and column tiles in every batch',()=>{
  const s={rows:65,N:96,batches:3};
  assert.equal(wideDispatchGroups(wideTiles['32x32'],s),27);
  assert.equal(wideDispatchGroups(wideTiles['64x32'],s),18);
  assert.equal(wideDispatchGroups(wideTiles['32x64'],s),18);
});

test('Renderer compiles only the selected supported wide variant per shape',async()=>{
  await assert.rejects(NeuralRenderer.create({},{gemmTile:'16x16'}),/GEMM tile/);
  const originalCreate=GpuRuntime.create,originalFetch=globalThis.fetch,loaded=[];
  const device={features:new Set(['shader-f16']),limits:{...limits}};
  GpuRuntime.create=async()=>({device,kernel:async a=>{loaded.push(a.entry);return {};},dispose:()=>{}});
  globalThis.fetch=async url=>({ok:true,json:async()=>String(url).endsWith('/manifest.json')?manifest:{entry:String(url).split('/').pop().replace('.json','')}});
  try{
    for(const gemmTile of Object.keys(wideTiles)){
      loaded.length=0;await NeuralRenderer.create({},{gemmTile});assert.deepEqual(loaded,[base+wideTiles[gemmTile].suffix]);
    }
    loaded.length=0;device.limits.maxComputeWorkgroupSizeX=128;
    await NeuralRenderer.create({},{gemmTile:'64x32'});assert.deepEqual(loaded,[base+'_wide_half']);
    for(const options of [{wideGemm:false},{nativeHalf:false},{specializeGemm:false}]){
      loaded.length=0;await NeuralRenderer.create({},options);assert.deepEqual(loaded,[]);
    }
    device.features.clear();loaded.length=0;await NeuralRenderer.create({});assert.deepEqual(loaded,[]);
  }finally{GpuRuntime.create=originalCreate;globalThis.fetch=originalFetch;}
});


test('Automatic tiles keep deep and narrow matrices on their measured shapes',()=>{
  for(const [K,N,expected] of [[32,32,'64x32'],[32,64,'32x64'],[32,128,'32x64'],[32,96,'32x32'],[64,128,'64x32'],[64,64,'32x64'],[64,192,'32x64'],[128,384,'32x64'],[256,768,'32x64'],[128,32,'32x32'],[128,128,'32x32'],[512,512,'32x32'],[4096,1024,'32x32']]){
    assert.equal(preferredWideTile(base.replace('_s32_96_',`_s${K}_${N}_`)),expected);
  }
  assert.equal(preferredWideTile('unknown'),'32x32');
});
