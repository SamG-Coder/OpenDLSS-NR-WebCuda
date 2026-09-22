import test from 'node:test';
import assert from 'node:assert/strict';
import {NeuralRenderer} from '../src/engine.js';
import {GpuProfile} from '../src/gpu-profile.js';

test('Renderer rejects invalid cache budgets and GEMM modes before requesting a device',async()=>{
  for(const workspaceCacheBytes of [-1,NaN,Infinity,0.5])await assert.rejects(NeuralRenderer.create({}, {workspaceCacheBytes}),/cache budget/);
  await assert.rejects(NeuralRenderer.create({}, {gemmMode:'unknown'}),/GEMM mode/);
  await assert.rejects(NeuralRenderer.create({}, {attentionMode:'unknown'}),/attention mode/);
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
