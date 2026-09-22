import test from 'node:test';
import assert from 'node:assert/strict';
import {NeuralRenderer} from '../src/engine.js';
import {GpuRuntime} from '../vendor/webcuda/runtime/runtime.js';
import {specializedGemm} from '../src/gemm-specialization.js';
import {preparedBackends,preparedBackendForEntry,supportsPreparedBackend,preparedGemmEntry} from '../src/prepared-gemm.js';
import {createGraph} from '../src/graph.js';
import {preparedMatrixKey} from '../src/model-preparation.js';

const limits={maxComputeWorkgroupStorageSize:16384,maxComputeInvocationsPerWorkgroup:256,maxComputeWorkgroupSizeX:256,maxComputeWorkgroupsPerDimension:65535};
const scalars={rows:65,K:32,N:96,batches:3,inputStride:96,inputBatchStride:32,partition:0,halfMode:0,silu:0,hasResidual:0,quantize:1,rawEnabled:0};
const formats={input:1,residual:0,raw:0,output:1};
const base=specializedGemm('nr_gemm_tile8x8_compact',{...scalars,...Object.fromEntries(Object.entries(formats).map(([k,v])=>[k+'Format',v]))});

test('Prepared options reject incompatible layouts before touching a GPU',async()=>{
  for(const options of [{gemmBackend:'other'},{modelCache:0},{onPrepareProgress:null}])await assert.rejects(NeuralRenderer.create({},options),/backend|preparation/);
  for(const options of [{specializeGemm:false},{activationStorage:'float'},{gemmMode:'scalar'}])await assert.rejects(NeuralRenderer.create({},{gemmBackend:'prepared-integer',...options}),/require packed activations/);
});

test('Prepared integer backend supports devices without half arithmetic',()=>{
  const device={limits,features:new Set()};
  assert.equal(supportsPreparedBackend('prepared-half',device),false);
  assert.equal(supportsPreparedBackend('prepared-integer',device),true);
  device.features.add('shader-f16');
  for(const [mode,backend] of Object.entries(preparedBackends)){
    assert.equal(preparedBackendForEntry(base+backend.suffix),mode);
    assert.equal(preparedGemmEntry(base,mode),base+backend.suffix);
    assert.equal(supportsPreparedBackend(mode,device),true);
    for(const reduced of [{maxComputeWorkgroupStorageSize:backend.sharedBytes-1},{maxComputeInvocationsPerWorkgroup:127},{maxComputeWorkgroupSizeX:127}])assert.equal(supportsPreparedBackend(mode,{...device,limits:{...limits,...reduced}}),false);
  }
});

test('Standard setup never loads or prepares experimental model formats',async()=>{
  const originalCreate=GpuRuntime.create,originalFetch=globalThis.fetch,loaded=[];
  const manifest=Object.fromEntries([base,...Object.values(preparedBackends).map(v=>base+v.suffix)].map(name=>[name,name+'.json']));
  GpuRuntime.create=async()=>({device:{limits,features:new Set(['shader-f16'])},kernel:async a=>{loaded.push(a.entry);return {};},dispose(){}});
  globalThis.fetch=async url=>({ok:true,json:async()=>String(url).endsWith('/manifest.json')?manifest:{entry:String(url).split('/').pop().replace('.json','')}});
  try{
    const renderer=await NeuralRenderer.create({});
    assert.equal(renderer.gemmBackend,'half');assert.equal(renderer.preparedModel,undefined);
    assert.deepEqual(loaded,[base]);
  }finally{GpuRuntime.create=originalCreate;globalThis.fetch=originalFetch;}
});

test('Prepared buffers dispatch only matching kernels with complete partial tiles',()=>{
  for(const mode of Object.keys(preparedBackends)){
    const entry=preparedGemmEntry(base,mode),calls=[];
    const kernel=name=>({bind:(bindings,scalars)=>({name,bindings,scalars})});
    const runtime={device:{limits},batch:()=>({dispatch(...args){calls.push(args);return this;},submit(){}})};
    const engine=new NeuralRenderer(runtime,{[base]:kernel(base),[entry]:kernel(entry)},{cache:new Map()});
    engine.gemmBackend=mode;engine.lookup={metadata:{},siluTable:{}};
    const weights={preparedBackend:mode,boundedHalf:true};
    engine.dispatch('nr_gemm_tile8x8',{weights},scalars,1,undefined,formats);
    assert.equal(calls[0][0].name,entry);assert.equal(calls[0][0].bindings.weights,weights);
    assert.deepEqual(calls[0][0].scalars,{rows:65});assert.deepEqual(calls[0][1],[27,1,1]);
    delete engine.kernels[entry];
    assert.throws(()=>engine.dispatch('nr_gemm_tile8x8',{weights},scalars,1,undefined,formats),/matching GEMM/);
    engine.specializeGemm=false;
    assert.throws(()=>engine.dispatch('nr_gemm_tile8x8',{weights},scalars,1,undefined,formats),/unspecialized GEMM/);
  }
});

test('Unsupported or unbounded matrices retain their original row-major path',()=>{
  const entry=preparedGemmEntry(base,'prepared-half'),calls=[];
  const kernel=name=>({bind:()=>name});
  const runtime={device:{limits},batch:()=>({dispatch(...args){calls.push(args);return this;},submit(){}})};
  const engine=new NeuralRenderer(runtime,{[base]:kernel(base),[entry]:kernel(entry)},{cache:new Map()});
  engine.gemmBackend='prepared-half';
  engine.dispatch('nr_gemm_tile8x8',{weights:{boundedHalf:false}},scalars,1,undefined,formats);
  assert.equal(calls[0][0],base);
});

test('Full graph uses prepared weights in all 358 FP8 dispatches and reuses warm bindings',async()=>{
  const graph=createGraph(128,96,{activationStorage:'packed',fuseLocalAttention:true,fuseNormalization:true});
  let decoded=0,preparedUploads=0,bindingsCreated=0;
  const preparedWords=new Uint32Array([0x12345678]);
  const runtime={device:{limits:{...limits,maxStorageBufferBindingSize:2**30}},
    createBuffer:data=>{if(data===preparedWords)preparedUploads++;return {size:typeof data==='number'?data:data.byteLength};},
    destroyBuffer(){},write(){},async idle(){},dispose(){},
    batch:()=>({dispatch(){return this;},submit(){},discard(){}})};
  const kernels=new Proxy({}, {get:(_,name)=>({bind:(bindings,scalars)=>{
    bindingsCreated++;
    if(bindings.weights?.preparedBackend)assert.ok(String(name).endsWith('_prepared_integer'),'prepared storage must never reach a row-major kernel');
    return {bindings,scalars};
  }})});
  const model={cache:new Map(),packedMatrix(){decoded++;return new Uint32Array(1);},vector:(_n,_o,count)=>new Float32Array(count),prior:(_n,_o,heads)=>new Float32Array(heads*4096)};
  const engine=new NeuralRenderer(runtime,kernels,model);
  engine.gemmBackend='prepared-integer';engine.preparedModel={matrices:new Map()};
  for(const op of graph.ops)for(const spec of Object.values(op.bindings))if(spec?.kind==='matrix'&&!spec.halfMode)engine.preparedModel.matrices.set(preparedMatrixKey(spec),{words:preparedWords,boundedHalf:true});
  const inputFeatures=new Float32Array(graph.geometry.fullWidth*graph.geometry.fullHeight*16);
  const run=()=>engine.run({width:128,height:96,inputFeatures,readHead:false});
  const cold=await run(),coldBindings=bindingsCreated;
  assert.equal(cold.gemmBackend.prepared,358);assert.equal(cold.gemmBackend.fallback,2);
  assert.equal(decoded,2,'only the two half endpoints need row-major decoding');
  assert.equal(preparedUploads,358);
  const warm=await run();assert.deepEqual(warm.gemmBackend,cold.gemmBackend);
  assert.equal(decoded,2);assert.equal(preparedUploads,358);assert.equal(bindingsCreated,coldBindings);
  engine.clearWeightCache();await run();assert.equal(preparedUploads,716,'clearing GPU weights preserves CPU preparation for re-upload');
  engine.specializeGemm=false;
  const fallback=await run();assert.equal(fallback.gemmBackend.prepared,0);assert.equal(fallback.gemmBackend.fallback,360);
  engine.dispose();assert.equal(engine.preparedModel,null);
});
