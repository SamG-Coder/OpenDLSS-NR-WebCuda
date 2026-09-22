import test from 'node:test';
import assert from 'node:assert/strict';
import {NeuralRenderer} from '../src/engine.js';
import {GpuRuntime} from '../vendor/webcuda/runtime/runtime.js';
import {createGraph} from '../src/graph.js';
import {gemmKernel} from '../src/kernel-selection.js';
import {specializedGemm} from '../src/gemm-specialization.js';
import {wideTiles,selectWideGemm,wideDispatchGroups} from '../src/gemm-tiles.js';
import {precomputedTiles,precomputedTileForEntry,selectPrecomputedGemm} from '../src/precomputed-gemm.js';
import {preparedMatrixKey} from '../src/model-preparation.js';
import {PRECOMPUTED_FORMAT_VERSION} from '../src/precomputed-model.js';

const limits={maxComputeWorkgroupStorageSize:32768,maxComputeInvocationsPerWorkgroup:1024,maxComputeWorkgroupSizeX:1024,maxComputeWorkgroupsPerDimension:65535};
const scalars={rows:65,K:32,N:96,batches:3,inputStride:96,inputBatchStride:32,partition:0,halfMode:0,silu:0,hasResidual:0,quantize:1,rawEnabled:0};
const formats={input:1,residual:0,raw:0,output:1};
const base=specializedGemm('nr_gemm_tile8x8_compact',{...scalars,...Object.fromEntries(Object.entries(formats).map(([key,value])=>[key+'Format',value]))});
const availableFor=name=>Object.fromEntries([...Object.values(wideTiles),...Object.values(precomputedTiles)].map(tile=>[name+tile.suffix,true]));

test('Precomputed operands reject incompatible layouts before requesting a GPU',async()=>{
  for(const options of [{specializeGemm:false},{activationStorage:'float'},{gemmMode:'scalar'}])
    await assert.rejects(NeuralRenderer.create({},{gemmBackend:'precomputed-half',...options}),/require packed activations/);
});

test('Precomputed operands use the same auto and explicit tile policy as standard across the graph',()=>{
  const graph=createGraph(1280,720,{activationStorage:'packed',fuseLocalAttention:true});
  let operations=0;
  for(const op of graph.ops)if(op.entry==='nr_gemm'&&!op.scalars.halfMode){
    operations++;
    const signature=gemmKernel(op.scalars)+'_compact_s'+op.scalars.K+'_'+op.scalars.N+'_';
    const available=availableFor(signature);
    for(const mode of ['auto',...Object.keys(wideTiles)]){
      const original=selectWideGemm(signature,available,limits,mode),precomputed=selectPrecomputedGemm(signature,available,limits,mode);
      assert.equal(precomputed.tile.name,original.tile.name);
      assert.equal(precomputed.tile.sharedBytes,original.tile.sharedBytes);
      assert.equal(precomputed.tile.threads,original.tile.threads);
      assert.equal(precomputedTileForEntry(precomputed.entry),precomputed.tile);
      assert.equal(wideDispatchGroups(precomputed.tile,op.scalars),wideDispatchGroups(original.tile,op.scalars));
    }
  }
  assert.equal(operations,358);
  assert.equal(precomputedTileForEntry(base+'_wide_half'),undefined);
});

test('Precomputed tiles fall back to 32 by 32 when larger tiles or limits are unavailable',()=>{
  const signature='nr_gemm_tile8x8_compact_s64_256_',available=availableFor(signature);
  assert.equal(selectPrecomputedGemm(signature,available,limits).tile.name,'32x64');
  for(const restricted of [{maxComputeWorkgroupStorageSize:12799},{maxComputeInvocationsPerWorkgroup:128},{maxComputeWorkgroupSizeX:128}])
    assert.equal(selectPrecomputedGemm(signature,available,{...limits,...restricted}).tile.name,'32x32');
  delete available[signature+precomputedTiles['32x64'].suffix];
  assert.equal(selectPrecomputedGemm(signature,available,limits).tile.name,'32x32');
  for(const restricted of [{maxComputeWorkgroupStorageSize:8447},{maxComputeInvocationsPerWorkgroup:127},{maxComputeWorkgroupSizeX:127}])
    assert.equal(selectPrecomputedGemm(signature,available,{...limits,...restricted}),null);
  assert.equal(selectPrecomputedGemm(signature,{},limits),null);
  assert.equal(selectPrecomputedGemm(null,available,limits),null);
});

test('Standard, missing shader-f16 and disabled native-half setup skip precomputation',async()=>{
  const originalCreate=GpuRuntime.create,originalFetch=globalThis.fetch,loaded=[];
  const manifest=Object.fromEntries([base,...Object.values(precomputedTiles).map(tile=>base+tile.suffix)].map(name=>[name,name+'.json']));
  const device={limits,features:new Set(['shader-f16'])};
  GpuRuntime.create=async()=>({device,kernel:async artifact=>{loaded.push(artifact.entry);return {};},dispose(){}});
  globalThis.fetch=async url=>({ok:true,json:async()=>String(url).endsWith('/manifest.json')?manifest:{entry:String(url).split('/').pop().replace('.json','')}});
  try{
    for(const options of [{},{gemmBackend:'precomputed-half',nativeHalf:false}]){
      const engine=await NeuralRenderer.create({},options);
      assert.equal(engine.preparedModel,undefined);assert.deepEqual(loaded,[base]);loaded.length=0;
    }
    device.features.clear();
    const engine=await NeuralRenderer.create({},{gemmBackend:'precomputed-half'});
    assert.equal(engine.preparedModel,undefined);assert.deepEqual(loaded,[base]);
  }finally{GpuRuntime.create=originalCreate;globalThis.fetch=originalFetch;}
});

test('Precomputed buffers dispatch their matching tile and never a row-major kernel',()=>{
  for(const mode of Object.keys(precomputedTiles)){
    const entry=base+precomputedTiles[mode].suffix,calls=[];
    const kernel=name=>({bind:(bindings,scalars)=>({name,bindings,scalars})});
    const runtime={device:{limits},batch:()=>({dispatch(...args){calls.push(args);return this;},submit(){}})};
    const engine=new NeuralRenderer(runtime,{[base]:kernel(base),[entry]:kernel(entry)},{cache:new Map()},{gemmTile:mode});
    engine.gemmBackend='precomputed-half';engine.lookup={metadata:{},siluTable:{}};
    const weights={preparedBackend:'precomputed-half',boundedHalf:true};
    engine.dispatch('nr_gemm_tile8x8',{weights},scalars,1,undefined,formats);
    assert.equal(calls[0][0].name,entry);assert.equal(calls[0][0].bindings.weights,weights);
    assert.deepEqual(calls[0][0].scalars,{rows:65});
    assert.deepEqual(calls[0][1],[wideDispatchGroups(precomputedTiles[mode],scalars),1,1]);
    delete engine.kernels[entry];
    assert.throws(()=>engine.dispatch('nr_gemm_tile8x8',{weights},scalars,1,undefined,formats),/matching GEMM/);
    engine.specializeGemm=false;
    assert.throws(()=>engine.dispatch('nr_gemm_tile8x8',{weights},scalars,1,undefined,formats),/unspecialized GEMM/);
  }
});

test('Full graph precomputes all 358 FP8 dispatches and reuses weights across warm renders and tile changes',async()=>{
  const graph=createGraph(128,96,{activationStorage:'packed',fuseLocalAttention:true,fuseNormalization:true});
  let decoded=0,operandUploads=0,bindingsCreated=0;
  const operandWords=new Uint32Array([0x3c003c00,0x00000000]);
  const runtime={device:{limits:{...limits,maxStorageBufferBindingSize:2**30}},
    createBuffer:data=>{if(data===operandWords)operandUploads++;return {size:typeof data==='number'?data:data.byteLength};},
    destroyBuffer(){},write(){},async idle(){},dispose(){},
    batch:()=>({dispatch(){return this;},submit(){},discard(){}})};
  const kernels=new Proxy({}, {get:(_,name)=>({bind:(bindings,scalars)=>{
    bindingsCreated++;
    if(bindings.weights?.preparedBackend)assert.ok(precomputedTileForEntry(String(name)),'precomputed storage must only reach a precomputed kernel');
    return {bindings,scalars};
  }})});
  const model={cache:new Map(),packedMatrix(){decoded++;return new Uint32Array(1);},vector:(_name,_offset,count)=>new Float32Array(count),prior:(_name,_offset,heads)=>new Float32Array(heads*4096)};
  const engine=new NeuralRenderer(runtime,kernels,model);
  engine.gemmBackend='precomputed-half';engine.preparedModel={matrices:new Map(),formatVersion:PRECOMPUTED_FORMAT_VERSION};
  for(const op of graph.ops)for(const spec of Object.values(op.bindings))if(spec?.kind==='matrix'&&!spec.halfMode)
    engine.preparedModel.matrices.set(preparedMatrixKey(spec),{words:operandWords,boundedHalf:true});
  const inputFeatures=new Float32Array(graph.geometry.fullWidth*graph.geometry.fullHeight*16);
  const run=()=>engine.run({width:128,height:96,inputFeatures,readHead:false});
  const cold=await run(),coldBindings=bindingsCreated;
  assert.equal(cold.gemmBackend.prepared,358);assert.equal(cold.gemmBackend.fallback,2);
  assert.equal(decoded,2);assert.equal(operandUploads,358);
  const warm=await run();assert.deepEqual(warm.gemmBackend,cold.gemmBackend);
  assert.equal(decoded,2);assert.equal(operandUploads,358);assert.equal(bindingsCreated,coldBindings);
  engine.gemmTile='32x32';
  const retiled=await run();assert.deepEqual(retiled.gemmBackend,cold.gemmBackend);
  assert.equal(operandUploads,358,'one operand representation supports all tile shapes');
  engine.clearWeightCache();await run();assert.equal(operandUploads,716,'GPU cache clearing reuses retained CPU operands');
  engine.nativeHalf=false;
  const fallback=await run();assert.equal(fallback.gemmBackend.prepared,0);assert.equal(fallback.gemmBackend.fallback,360);
  engine.dispose();assert.equal(engine.preparedModel,null);
});
