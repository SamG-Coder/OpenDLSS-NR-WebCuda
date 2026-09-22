import {wideTiles,wideTileForEntry,selectWideGemm,wideDispatchGroups} from './gemm-tiles.js';
import {preparedBackendForEntry,supportsPreparedBackend,preparedGemmEntry} from './prepared-gemm.js';
import {prepareModel,preparedMatrixKey,PREPARED_FORMAT_VERSION} from './model-preparation.js';
import {specializedGemm,dynamicGemmScalars,boundedHalfWeights} from './gemm-specialization.js';
import {GpuRuntime} from '../vendor/webcuda/runtime/runtime.js';
import {createGraph} from './graph.js';
import {planLayout,createExecutionPlan} from './execution-plan.js';
import {unpackActivations} from './model.js';
import {activationBindings} from './activation-bindings.js';
import {GpuProfile} from './gpu-profile.js';
import {gemmKernel,dispatchGroups,dispatchGrid} from './kernel-selection.js';
export class NeuralRenderer {
  static async create(model,options={}) {
    const {workspaceCacheBytes=256*1024*1024,gemmMode='auto',attentionMode='fused',activationStorage='packed',executionMode='prepared',planCacheBytes=1024*1024*1024,graphBatchSize=32,maxInFlightBatches=4,cacheNoise=true,specializeGemm=true,nativeHalf=true,wideGemm=true,gemmTile='auto',normalizeAttention=true,gemmBackend='half',modelCache=true,onPrepareProgress=()=>{},...runtimeOptions}=options;
    if(!['half','prepared-half','prepared-integer'].includes(gemmBackend))throw Error('Invalid GEMM backend.');
    if(typeof modelCache!=='boolean'||typeof onPrepareProgress!=='function')throw Error('Invalid model preparation options.');
    if(gemmBackend!=='half'&&(!specializeGemm||activationStorage!=='packed'||!['auto','tile8x8','tile8x16'].includes(gemmMode)))throw Error('Prepared GEMM backends require packed activations and GEMM specialization in auto, tile8x8 or tile8x16 mode.');
    if(!Number.isInteger(maxInFlightBatches)||maxInFlightBatches<1||maxInFlightBatches>8)throw Error('In-flight batch limit must be 1 through 8.');
    if(typeof normalizeAttention!=='boolean')throw Error('normalizeAttention must be boolean.');
    if(gemmTile!=='auto'&&!Object.hasOwn(wideTiles,gemmTile))throw Error('Invalid GEMM tile.');
    if(typeof wideGemm!=='boolean')throw Error('wideGemm must be boolean.');
    if(typeof nativeHalf!=='boolean')throw Error('nativeHalf must be boolean.');
    if(typeof specializeGemm!=='boolean')throw Error('specializeGemm must be boolean.');
    if(typeof cacheNoise!=='boolean')throw Error('cacheNoise must be boolean.');
    if(!Number.isInteger(graphBatchSize)||graphBatchSize<1||graphBatchSize>64)throw Error('Graph batch size must be 1 through 64.');
    if(!Number.isSafeInteger(workspaceCacheBytes)||workspaceCacheBytes<0)throw Error('Invalid workspace cache budget.');
    if(!['auto','multi-auto','tiled','scalar','tile8x8','tile8x16','multi8x32','multi16x16','multi16x32','multi4x32','multi32x32','multi16x64'].includes(gemmMode))throw Error('Invalid GEMM mode.');
    if(!['fused','tiled','scalar'].includes(attentionMode))throw Error('Invalid attention mode.');
    if(!['prepared','streamed'].includes(executionMode)||!Number.isSafeInteger(planCacheBytes)||planCacheBytes<0)throw Error('Invalid execution plan options.');
    if(!['float','packed'].includes(activationStorage))throw Error('Invalid activation storage mode.');
    const runtime=await GpuRuntime.create({useAdapterBufferLimits:true,useAdapterWorkgroupLimits:normalizeAttention,...runtimeOptions}),kernels={};
    try {
      const response=await fetch(new URL('../generated/manifest.json',import.meta.url));if(!response.ok)throw Error('Run npm run build before starting.');
      const manifest=await response.json();
      for(const [entry,file] of Object.entries(manifest)) {
        const preparedBackend=preparedBackendForEntry(entry);
        if(preparedBackend&&(preparedBackend!==gemmBackend||!supportsPreparedBackend(preparedBackend,runtime.device)||(preparedBackend==='prepared-half'&&!nativeHalf)))continue;
        if(entry==='nr_lookup_tables'&&!wideGemm&&gemmBackend==='half')continue;
        if(entry==='nr_local_attention_normalized'&&(!normalizeAttention||!runtime.device.features.has('shader-f16')||runtime.device.limits.maxComputeWorkgroupStorageSize<31680||runtime.device.limits.maxComputeInvocationsPerWorkgroup<512||runtime.device.limits.maxComputeWorkgroupSizeX<512||activationStorage!=='packed'||attentionMode!=='fused'))continue;
        const tile=wideTileForEntry(entry);
        if(tile&&(!wideGemm||selectWideGemm(entry.slice(0,-tile.suffix.length),manifest,runtime.device.limits,gemmTile)?.entry!==entry))continue;
        if(entry.endsWith('_half')&&(!nativeHalf||!runtime.device.features.has('shader-f16')))continue;
        if(/_compact_s[0-9]/.test(entry)&&(!specializeGemm||activationStorage!=='packed'||!['auto','tile8x8','tile8x16'].includes(gemmMode)))continue;
        if(entry.startsWith('nr_gemm_multi')&&!entry.startsWith('nr_gemm_'+(gemmMode==='multi-auto'?'multi32x32':gemmMode)))continue;
        const r=await fetch(new URL('../generated/'+file,import.meta.url));if(!r.ok)throw Error('Missing kernel '+file);kernels[entry]=await runtime.kernel(await r.json());}
      const engine=new NeuralRenderer(runtime,kernels,model,{workspaceCacheBytes,gemmMode,attentionMode,activationStorage,executionMode,planCacheBytes,graphBatchSize,maxInFlightBatches,cacheNoise,specializeGemm,nativeHalf,wideGemm,gemmTile,normalizeAttention});
      engine.gemmBackend=gemmBackend;
      if(Object.keys(kernels).some(entry=>preparedBackendForEntry(entry)))engine.preparedModel=await prepareModel(model,{cache:modelCache,onProgress:onPrepareProgress});
      return engine;
    } catch(e) {runtime.dispose();throw e;}
  }
  constructor(runtime,kernels,model,{workspaceCacheBytes=256*1024*1024,gemmMode='auto',attentionMode='fused',activationStorage='packed',executionMode='prepared',planCacheBytes=1024*1024*1024,graphBatchSize=32,maxInFlightBatches=4,cacheNoise=true,specializeGemm=true,nativeHalf=true,wideGemm=true,gemmTile='auto',normalizeAttention=true}={}) {this.runtime=runtime;this.kernels=kernels;this.model=model;this.busy=false;this.weightCache=new Map();this.weightCacheBytes=0;this.workspace=new Map();this.workspaceBytes=0;this.workspaceLimit=workspaceCacheBytes;this.gemmMode=gemmMode;this.attentionMode=attentionMode;this.activationStorage=activationStorage;this.executionMode=executionMode;this.planLimit=planCacheBytes;this.graphBatchSize=graphBatchSize;this.plan=null;this.graphCache=null;this.maxInFlightBatches=maxInFlightBatches;this.cacheNoise=cacheNoise;this.specializeGemm=specializeGemm;this.nativeHalf=nativeHalf;this.wideGemm=wideGemm;this.gemmTile=gemmTile;this.normalizeAttention=normalizeAttention;this.lookup=null;this.noise=null;}
  clearWeightCache() {
    if(this.busy)throw Error('Cancel and await inference before clearing weights.');
    this.#releasePlan();this.#releaseLookup();
    for(const b of this.weightCache.values())this.runtime.destroyBuffer(b);
    this.weightCache.clear();this.weightCacheBytes=0;
  }
  clearWorkspace() {
    if(this.busy)throw Error('Cancel and await inference before clearing working buffers.');
    this.#releaseWorkspace();
  }
  #releasePlan() {
    if(this.plan)for(const b of this.plan.buffers)this.runtime.destroyBuffer(b);
    this.plan=null;
  }
  #releaseLookup() {
    if(this.lookup)for(const buffer of Object.values(this.lookup))this.runtime.destroyBuffer(buffer);
    this.lookup=null;
  }
  #releaseWorkspace() {
    if(this.noise)this.runtime.destroyBuffer(this.noise.buffer);this.noise=null;
    this.#releasePlan();this.#releaseLookup();
    for(const list of this.workspace.values())for(const b of list)this.runtime.destroyBuffer(b);
    this.workspace.clear();this.workspaceBytes=0;
  }
  dispatch(entry,bindings,scalars,count,batch,formats,planIndex) {
    if(this.attentionMode!=='scalar'&&(entry==='nr_scores'||entry==='nr_attend'))entry+='_tiled';
    let groups=dispatchGroups(entry,scalars,count),limit=this.runtime.device.limits.maxComputeWorkgroupsPerDimension;
    if(formats&&activationBindings[entry]){scalars={...scalars,...Object.fromEntries(activationBindings[entry].map(b=>[b+'Format',formats[b]??0]))};entry+='_compact';}
    const profileScalars=scalars;
    if(bindings.weights?.preparedBackend&&!this.specializeGemm)throw Error('Prepared weights cannot use an unspecialized GEMM.');
    if(this.specializeGemm){const specialized=specializedGemm(entry,scalars);if(specialized&&this.kernels[specialized]){
      if(bindings.weights?.preparedBackend){
        entry=preparedGemmEntry(specialized,bindings.weights.preparedBackend);
        if(!this.kernels[entry])throw Error('Prepared weights require a matching GEMM pipeline.');
        groups=wideDispatchGroups(wideTiles['32x32'],scalars);
      }else{
        entry=this.nativeHalf&&bindings.weights?.boundedHalf&&this.kernels[specialized+'_half']?specialized+'_half':specialized;
        if(this.wideGemm&&entry.endsWith('_half')){const selected=selectWideGemm(specialized,this.kernels,this.runtime.device.limits,this.gemmTile);if(selected){entry=selected.entry;groups=wideDispatchGroups(selected.tile,scalars);}}
      }
      scalars=dynamicGemmScalars(scalars);
    }else if(bindings.weights?.preparedBackend)throw Error('Prepared weights cannot use an unspecialized GEMM.');}
    if(this.gemmDispatches&&entry.startsWith('nr_gemm'))this.gemmDispatches[preparedBackendForEntry(entry)?'prepared':'fallback']++;
    if(wideTileForEntry(entry)||preparedBackendForEntry(entry)){
      if(!this.lookup){
        const metadata=this.runtime.createBuffer(256*8),siluTable=this.runtime.createBuffer(65536*4);
        this.runtime.batch().dispatch(this.kernels.nr_lookup_tables.bind({metadata,silu:siluTable}),[1024,1,1]).submit();
        this.lookup={metadata,siluTable};
      }
      bindings={...bindings,...this.lookup};
    }
    const timed=this.profile?.batch(entry,profileScalars);
    const commands=timed??batch??this.runtime.batch();
    let invocation=planIndex===undefined?null:this.plan.invocations.get(planIndex);
    if(!invocation){invocation=this.kernels[entry].bind(bindings,scalars);if(planIndex!==undefined)this.plan.invocations.set(planIndex,invocation);}
    commands.dispatch(invocation,dispatchGrid(groups,limit));
    if(timed||!batch)commands.submit();
  }
  async run({width,height,proxy,inputFeatures,history,motion,seed=0,conditioning={},onProgress=()=>{},capture,signal,geometryOverride,profile=false,readHead=true}={}) {
    if(typeof readHead!=='boolean')throw Error('readHead must be boolean.');
    const started=performance.now(),timings={setupMs:0,encodeMs:0,waitMs:0,completionMs:0,totalMs:0};
    if(this.busy)throw Error('An inference is already running.');this.busy=true;
    this.profile=null;this.gemmDispatches={prepared:0,fallback:0};
    const runtime=this.runtime,model=this.model,live=new Map(),pool=this.workspace,owned=new Set();
    let poolBytes=this.workspaceBytes;const poolLimit=this.workspaceLimit;
    let inFlight=0,noiseReused=false;
    let batch=null,queuedOps=0,retiredBytes=0;const retired=[];
    const retire=b=>{retired.push(b);retiredBytes+=b.size;owned.add(b);};
    // Evict the oldest free size bucket. Defer destruction until queued users finish.
    const cacheBuffer=(b,evict)=>{
      if(b.size>poolLimit)return false;
      while(poolBytes+b.size>poolLimit){
        const [size,list]=pool.entries().next().value,old=list.shift();
        if(!list.length)pool.delete(size);poolBytes-=old.size;evict(old);
      }
      const list=pool.get(b.size)||[];list.push(b);pool.delete(b.size);pool.set(b.size,list);poolBytes+=b.size;owned.delete(b);return true;
    };
    const flush=async(wait=true)=>{
      if(batch){batch.submit();batch=null;inFlight++;}
      queuedOps=0;
      // Queue writes and submissions remain ordered, including the shared uniform arena.
      // Only prepared plans retain every graph buffer until all submitted work completes.
      if(!wait&&inFlight<this.maxInFlightBatches)return;
      const t=performance.now();await runtime.idle();timings.waitMs+=performance.now()-t;inFlight=0;
      for(const b of retired){runtime.destroyBuffer(b);owned.delete(b);}
      retired.length=0;retiredBytes=0;queuedOps=0;
    };
    let createdBuffers=0,reusedBuffers=0;
    const alloc=data=>{
      // Host writes must not overwrite a buffer referenced by unsubmitted commands.
      const size=typeof data==='number'?data:data.byteLength,b=(typeof data==='number'||!batch)?pool.get(size)?.pop():undefined;
      if(b){if(!pool.get(size).length)pool.delete(size);poolBytes-=b.size;owned.add(b);reusedBuffers++;if(typeof data!=='number')runtime.write(b,data);return b;}
      const next=runtime.createBuffer(data);owned.add(next);createdBuffers++;return next;
    };
    const release=b=>{if(!cacheBuffer(b,retire))retire(b);};
    const take=alloc;
    try {
      this.profile=profile?new GpuProfile(this.runtime):null;
      const graphKey=JSON.stringify([width,height,geometryOverride,this.activationStorage,this.gemmMode,this.attentionMode,this.specializeGemm,this.nativeHalf,this.wideGemm,this.gemmTile,this.normalizeAttention,this.gemmBackend]);
      if(this.graphCache?.key!==graphKey)this.graphCache={key:graphKey,graph:createGraph(width,height,{geometryOverride,activationStorage:this.activationStorage,fuseLocalAttention:this.attentionMode==='fused',fuseNormalization:this.attentionMode==='fused'&&this.normalizeAttention&&Boolean(this.kernels.nr_local_attention_normalized)&&this.activationStorage==='packed'})};
      const graph=this.graphCache.graph,g=graph.geometry;
      const workspaceKey=graphKey;
      if(this.workspaceKey!==workspaceKey){this.#releaseWorkspace();poolBytes=0;this.workspaceKey=workspaceKey;}
      model.validateGraph?.(graph);
      for(const r of graph.resources.values())if(r.bytes>runtime.device.limits.maxStorageBufferBindingSize)throw Error('Resolution requires a buffer larger than the device limit.');
      const pixels=width*height,featureCount=g.fullWidth*g.fullHeight*16;
      if(inputFeatures&&(!(inputFeatures instanceof Float32Array)||inputFeatures.length!==featureCount))throw Error('Input features do not match the padded geometry.');
      if(!inputFeatures&&(!(proxy instanceof Float32Array)||proxy.length!==pixels*4))throw Error('Supply RGBA Float32Array proxy or padded input features.');
      if(history&&(!(history instanceof Float32Array)||history.length!==pixels*4))throw Error('History must be an already reprojected RGBA Float32Array.');
      if(proxy&&(!(proxy instanceof Float32Array)||proxy.length!==pixels*4))throw Error('Proxy must be source-sized RGBA Float32Array.');
      if(motion&&(!history||!proxy||!(motion instanceof Float32Array)||motion.length!==pixels*4))throw Error('Motion requires proxy, history, and source-sized RGBA UV displacements with validity in z.');
      if(!this.plan&&this.executionMode==='prepared'&&typeof model.packedMatrix==='function') {
        const layout=planLayout(graph);
        if(layout.bytes+4<=this.planLimit){this.plan=createExecutionPlan(runtime,graph,layout);createdBuffers+=this.plan.buffers.length;}
      }
      const plan=this.plan;
      const zero=plan?.zero??alloc(new Float32Array(1)),proxyBuffer=proxy?alloc(proxy):zero;
      let historyBuffer=history?alloc(history):proxyBuffer;
      if(motion) {
        const reprojected=alloc(pixels*16),motionBuffer=alloc(motion);
        this.dispatch('nr_reproject',{history:historyBuffer,motion:motionBuffer,proxy:proxyBuffer,output:reprojected},{width,height},pixels);
        historyBuffer=reprojected;
      }
      const featureBuffer=plan?.resources.get(graph.features)??(inputFeatures?alloc(inputFeatures):alloc(featureCount*4));
      if(plan&&inputFeatures)runtime.write(featureBuffer,inputFeatures);
      live.set(graph.features,featureBuffer);
      if(!inputFeatures) {
        let noise;
        if(this.cacheNoise){
          const key=JSON.stringify([g.fullWidth,g.fullHeight,seed]);
          noiseReused=this.noise?.key===key;
          if(!noiseReused){
            if(this.noise)runtime.destroyBuffer(this.noise.buffer);this.noise=null;
            const buffer=runtime.createBuffer(g.fullWidth*g.fullHeight*8);
            try{this.dispatch('nr_noise',{noise:buffer},{fullWidth:g.fullWidth,fullHeight:g.fullHeight,seed},g.fullWidth*g.fullHeight);}
            catch(error){runtime.destroyBuffer(buffer);throw error;}
            this.noise={key,buffer};
          }
          noise=this.noise.buffer;
        }
        this.dispatch(noise?'nr_preprocess_cached':'nr_preprocess',{...(noise?{noise}:{}),proxy:proxyBuffer,history:historyBuffer,features:featureBuffer},{width,height,fullWidth:g.fullWidth,fullHeight:g.fullHeight,seed,autoMask:conditioning.autoMask??1,localTone:conditioning.localTone??1,localStructure:conditioning.localStructure??1,skinStructure:conditioning.skinStructure??-1,style:conditioning.style??0,useHistory:history?1:0},g.fullWidth*g.fullHeight);
      }
      const lastUse=new Map();graph.ops.forEach((op,i)=>Object.values(op.bindings).forEach(v=>{if(typeof v==='string')lastUse.set(v,i);}));lastUse.set(graph.head,graph.ops.length);
      const boundaryById=new Map(Object.entries(graph.boundaries).map(([name,id])=>[id,name]));
      timings.setupMs=performance.now()-started;const graphStarted=performance.now();
      for(let index=0;index<graph.ops.length;index++) {
        if(signal?.aborted)throw new DOMException('Inference cancelled','AbortError');
        const op=graph.ops[index],bindings={},weights=[];
        const formats=this.activationStorage==='packed'?Object.fromEntries(Object.entries(op.bindings).filter(([,id])=>typeof id==='string').map(([key,id])=>[key,graph.resources.get(id).format])):undefined;
        const preparedEntry=this.preparedModel&&this.specializeGemm&&this.activationStorage==='packed'&&(this.gemmBackend!=='prepared-half'||this.nativeHalf)&&op.entry==='nr_gemm'?preparedGemmEntry(specializedGemm(gemmKernel(op.scalars,this.gemmMode)+'_compact',{...op.scalars,...Object.fromEntries(['input','residual','raw','output'].map(key=>[key+'Format',formats?.[key]??0]))}),this.gemmBackend):null;
        for(const [key,spec] of Object.entries(op.bindings)) {
          if(typeof spec==='string') {if(!live.has(spec))live.set(spec,plan?.resources.get(spec)??take(graph.resources.get(spec).bytes));bindings[key]=live.get(spec);}
          else if(spec.kind==='zero')bindings[key]=zero;
          else {
            const persistent=typeof model.packedMatrix==='function';
            const candidate=key==='weights'&&spec.kind==='matrix'&&preparedEntry&&this.kernels[preparedEntry]?this.preparedModel.matrices.get(preparedMatrixKey(spec)):null;
            const prepared=candidate&&(this.gemmBackend!=='prepared-half'||candidate.boundedHalf)?candidate:null;
            const cacheKey=(prepared?'prepared-v'+PREPARED_FORMAT_VERSION+':'+this.gemmBackend+':':'')+JSON.stringify(spec);
            const cached=persistent&&this.weightCache.get(cacheKey);
            if(cached){bindings[key]=cached;continue;}
            let data;
            if(spec.kind==='matrix')data=prepared?.words??(persistent?model.packedMatrix(spec.name,spec.offset,spec.K,spec.N,spec):model.matrix(spec.name,spec.offset,spec.K,spec.N,spec));
            if(spec.kind==='vector')data=model.vector(spec.name,spec.offset,spec.count,spec.type);
            if(spec.kind==='prior')data=model.prior(spec.name,spec.offset,spec.heads);
            const b=persistent?runtime.createBuffer(data):alloc(data);bindings[key]=b;
            if(prepared){b.preparedBackend=this.gemmBackend;b.boundedHalf=prepared.boundedHalf;}
            else if(this.nativeHalf&&persistent&&spec.kind==='matrix'&&!spec.halfMode)b.boundedHalf=boundedHalfWeights(data);
            if(persistent){this.weightCache.set(cacheKey,b);this.weightCacheBytes+=b.size;}else weights.push(b);
          }
        }
        batch??=runtime.batch();
        this.dispatch(op.entry==='nr_gemm'&&typeof model.packedMatrix==='function'?gemmKernel(op.scalars,this.gemmMode):op.entry,bindings,op.scalars,op.count,batch,formats,plan?index:undefined);queuedOps++;
        for(const b of weights)retire(b);model.cache.clear();
        // Keep intermediates queue-ordered, and retain resources until submitted work finishes.
        // Bound both cancellation latency and transient uploads, rather than waiting every op.
        const boundary=capture&&boundaryById.get(op.bindings.output);
        if(boundary){await flush();await capture(boundary,await this.readActivation(live.get(op.bindings.output),graph.resources.get(op.bindings.output)),graph.resources.get(op.bindings.output));}
        for(const [id,b] of live)if(lastUse.get(id)===index){live.delete(id);if(!plan)release(b);}
        if(queuedOps>=(plan?this.graphBatchSize:8)||retiredBytes>=64*1024*1024)await flush(!plan||!!capture||retiredBytes>=64*1024*1024);
        onProgress({index:index+1,total:graph.ops.length,label:op.label});
      }
      await flush();
      timings.encodeMs=performance.now()-graphStarted-timings.waitMs;
      if(signal?.aborted)throw new DOMException('Inference cancelled','AbortError');
      let head=null,output=null;const readbackStarted=performance.now();
      if(proxy) {
        const blend=model.vector(model.tensor(70,0,'blend_scale'),0,1)[0],out=alloc(pixels*16);
        this.dispatch('nr_compose',{proxy:proxyBuffer,head:live.get(graph.head),history:historyBuffer,output:out},{width,height,fullWidth:g.fullWidth,blendScale:blend,useHistory:history?1:0},pixels,undefined,this.activationStorage==='packed'?{head:graph.resources.get(graph.head).format}:undefined);
        output=await runtime.read(out);
      }
      if(readHead)head=await this.readActivation(live.get(graph.head),graph.resources.get(graph.head));
      timings.completionMs=performance.now()-readbackStarted;timings.totalMs=performance.now()-started;
      return {head,output,timings,gemmBackend:{requested:this.gemmBackend??'half',...this.gemmDispatches},noiseCache:{reused:noiseReused,bytes:this.noise?.buffer.size??0},geometry:g,dispatches:graph.ops.length,profile:await this.profile?.read(),workingBuffers:{created:createdBuffers,reused:reusedBuffers,prepared:!!plan,planBytes:plan?.bytes??0}};
    } finally {batch?.discard();await runtime.idle().catch(()=>{});for(const b of [...owned]){
      if(!cacheBuffer(b,old=>runtime.destroyBuffer(old)))runtime.destroyBuffer(b);
    }
    this.workspaceBytes=poolBytes;this.profile?.dispose();this.profile=null;this.model.cache.clear();this.busy=false;timings.totalMs=performance.now()-started;}
  }
  async readActivation(buffer,resource) {
    const count=resource.rows*resource.channels;
    if(!resource.format)return this.runtime.read(buffer,Float32Array,count*4);
    const words=await this.runtime.read(buffer,Uint32Array,resource.bytes);
    return unpackActivations(words,resource.format,count);
  }
  dispose(){if(this.busy)throw Error('Cancel and await inference before disposal.');this.weightCache.clear();this.weightCacheBytes=0;this.workspace.clear();this.workspaceBytes=0;this.plan=null;this.graphCache=null;this.noise=null;this.lookup=null;this.preparedModel=null;this.runtime.dispose();}
}
