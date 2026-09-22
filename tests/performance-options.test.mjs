import test from 'node:test';
import assert from 'node:assert/strict';
import {NeuralRenderer} from '../src/engine.js';
import {GpuRuntime} from '../vendor/webcuda/runtime/runtime.js';
import {gemmKernel} from '../src/kernel-selection.js';
import {GpuProfile} from '../src/gpu-profile.js';

test('Renderer rejects invalid cache budgets and GEMM modes before requesting a device',async()=>{
  for(const workspaceCacheBytes of [-1,NaN,Infinity,0.5])await assert.rejects(NeuralRenderer.create({}, {workspaceCacheBytes}),/cache budget/);
  await assert.rejects(NeuralRenderer.create({}, {gemmMode:'unknown'}),/GEMM mode/);
  await assert.rejects(NeuralRenderer.create({}, {attentionMode:'unknown'}),/attention mode/);
  for(const maxInFlightBatches of [0,9,NaN,1.5])await assert.rejects(NeuralRenderer.create({}, {maxInFlightBatches}),/In-flight/);
  await assert.rejects(NeuralRenderer.create({}, {cacheNoise:1}),/cacheNoise/);
  await assert.rejects(NeuralRenderer.create({}, {specializeGemm:1}),/specializeGemm/);
  await assert.rejects(NeuralRenderer.create({}, {nativeHalf:1}),/nativeHalf/);
  for(const graphBatchSize of [0,65,NaN,1.5])await assert.rejects(NeuralRenderer.create({}, {graphBatchSize}),/batch size/);
  await assert.rejects(NeuralRenderer.create({}, {activationStorage:'unknown'}),/activation storage/);
  await assert.rejects(NeuralRenderer.create({}, {executionMode:'unknown'}),/execution plan/);
  for(const planCacheBytes of [-1,NaN,Infinity,0.5])await assert.rejects(NeuralRenderer.create({}, {planCacheBytes}),/execution plan/);
});

test('Profiling explicitly reports unsupported timestamps without inventing GPU timings',async()=>{
  const profile=new GpuProfile({device:{features:new Set()}});
  assert.equal(profile.batch('test',{}),null);
  const result=await profile.read();assert.equal(result.supported,false);assert.match(result.reason,/timestamp-query/);assert.equal(result.totalGpuMs,undefined);
  profile.dispose();
});


test('Automatic selection keeps the proven kernels; multi-output modes are explicit',()=>{
  const base={rows:100000,K:32,N:128,batches:1,halfMode:0};
  assert.equal(gemmKernel(base),'nr_gemm_tile8x8');
  assert.equal(gemmKernel({...base,K:128,N:32}),'nr_gemm_tile8x16');
  assert.equal(gemmKernel(base,'multi-auto'),'nr_gemm_multi32x32');
  assert.equal(gemmKernel({...base,rows:100},'multi-auto'),'nr_gemm_tile8x16');
  assert.equal(gemmKernel({...base,halfMode:1},'multi32x32'),'nr_gemm_packed');
});

test('Default setup skips experimental pipelines and opt-in loads only the selected family',async()=>{
  const originalCreate=GpuRuntime.create,originalFetch=globalThis.fetch,loaded=[];
  const manifest={nr_gemm_tile8x8_compact_s0:'nr_gemm_tile8x8_compact_s0.json',nr_numeric:'nr_numeric.json',nr_gemm_multi8x32:'nr_gemm_multi8x32.json',nr_gemm_multi8x32_compact:'nr_gemm_multi8x32_compact.json',nr_gemm_multi32x32:'nr_gemm_multi32x32.json',nr_gemm_multi32x32_compact:'nr_gemm_multi32x32_compact.json'};
  GpuRuntime.create=async()=>({kernel:async artifact=>{loaded.push(artifact.entry);return {};},dispose:()=>{}});
  globalThis.fetch=async url=>({ok:true,json:async()=>String(url).endsWith('/manifest.json')?manifest:{entry:String(url).split('/').pop().replace('.json','')}});
  try {
    await NeuralRenderer.create({});assert.deepEqual(loaded,['nr_gemm_tile8x8_compact_s0','nr_numeric']);loaded.length=0;
    await NeuralRenderer.create({},{specializeGemm:false});assert.deepEqual(loaded,['nr_numeric']);loaded.length=0;
    await NeuralRenderer.create({},{activationStorage:'float'});assert.deepEqual(loaded,['nr_numeric']);loaded.length=0;
    await NeuralRenderer.create({},{gemmMode:'multi32x32'});assert.deepEqual(loaded,['nr_numeric','nr_gemm_multi32x32','nr_gemm_multi32x32_compact']);
  }finally{GpuRuntime.create=originalCreate;globalThis.fetch=originalFetch;}
});

test('Half pipelines require both the device feature and renderer option',async()=>{
  const originalCreate=GpuRuntime.create,originalFetch=globalThis.fetch,loaded=[],features=new Set();
  GpuRuntime.create=async()=>({device:{features},kernel:async a=>{loaded.push(a.entry);return {};},dispose:()=>{}});
  const base='nr_gemm_tile8x8_compact_s1';
  globalThis.fetch=async url=>({ok:true,json:async()=>String(url).endsWith('/manifest.json')?{[base]:base+'.json',[base+'_half']:base+'_half.json'}:{entry:String(url).split('/').pop().replace('.json','')}});
  try{
    await NeuralRenderer.create({});assert.deepEqual(loaded,[base]);loaded.length=0;
    features.add('shader-f16');await NeuralRenderer.create({});assert.deepEqual(loaded,[base,base+'_half']);loaded.length=0;
    await NeuralRenderer.create({},{nativeHalf:false});assert.deepEqual(loaded,[base]);
  }finally{GpuRuntime.create=originalCreate;globalThis.fetch=originalFetch;}
});
