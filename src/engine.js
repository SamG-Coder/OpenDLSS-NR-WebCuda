import {GpuRuntime} from '../vendor/webcuda/runtime/runtime.js';
import {createGraph} from './graph.js';
import {planLayout,createExecutionPlan} from './execution-plan.js';
import {unpackActivations} from './model.js';
import {activationBindings} from './activation-bindings.js';
import {GpuProfile} from './gpu-profile.js';
import {gemmKernel,dispatchGroups} from './kernel-selection.js';
export class NeuralRenderer {
  static async create(model,options={}) {
    const {workspaceCacheBytes=256*1024*1024,gemmMode='auto',attentionMode='fused',activationStorage='packed',executionMode='prepared',planCacheBytes=1024*1024*1024,graphBatchSize=32,...runtimeOptions}=options;
    if(!Number.isInteger(graphBatchSize)||graphBatchSize<1||graphBatchSize>64)throw Error('Graph batch size must be 1 through 64.');
    if(!Number.isSafeInteger(workspaceCacheBytes)||workspaceCacheBytes<0)throw Error('Invalid workspace cache budget.');
    if(!['auto','multi-auto','tiled','scalar','tile8x8','tile8x16','multi8x32','multi16x16','multi16x32','multi4x32','multi32x32','multi16x64'].includes(gemmMode))throw Error('Invalid GEMM mode.');
    if(!['fused','tiled','scalar'].includes(attentionMode))throw Error('Invalid attention mode.');
    if(!['prepared','streamed'].includes(executionMode)||!Number.isSafeInteger(planCacheBytes)||planCacheBytes<0)throw Error('Invalid execution plan options.');
    if(!['float','packed'].includes(activationStorage))throw Error('Invalid activation storage mode.');
    const runtime=await GpuRuntime.create({useAdapterBufferLimits:true,...runtimeOptions}),kernels={};
    try {
      const response=await fetch(new URL('../generated/manifest.json',import.meta.url));if(!response.ok)throw Error('Run npm run build before starting.');
      const manifest=await response.json();
      for(const [entry,file] of Object.entries(manifest)) {
        if(entry.startsWith('nr_gemm_multi')&&!entry.startsWith('nr_gemm_'+(gemmMode==='multi-auto'?'multi32x32':gemmMode)))continue;
        const r=await fetch(new URL('../generated/'+file,import.meta.url));if(!r.ok)throw Error('Missing kernel '+file);kernels[entry]=await runtime.kernel(await r.json());}
      return new NeuralRenderer(runtime,kernels,model,{workspaceCacheBytes,gemmMode,attentionMode,activationStorage,executionMode,planCacheBytes,graphBatchSize});
    } catch(e) {runtime.dispose();throw e;}
  }
  constructor(runtime,kernels,model,{workspaceCacheBytes=256*1024*1024,gemmMode='auto',attentionMode='fused',activationStorage='packed',executionMode='prepared',planCacheBytes=1024*1024*1024,graphBatchSize=32}={}) {this.runtime=runtime;this.kernels=kernels;this.model=model;this.busy=false;this.weightCache=new Map();this.weightCacheBytes=0;this.workspace=new Map();this.workspaceBytes=0;this.workspaceLimit=workspaceCacheBytes;this.gemmMode=gemmMode;this.attentionMode=attentionMode;this.activationStorage=activationStorage;this.executionMode=executionMode;this.planLimit=planCacheBytes;this.graphBatchSize=graphBatchSize;this.plan=null;this.graphCache=null;}
  clearWeightCache() {
    if(this.busy)throw Error('Cancel and await inference before clearing weights.');
    this.#releasePlan();
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
  #releaseWorkspace() {
    this.#releasePlan();
    for(const list of this.workspace.values())for(const b of list)this.runtime.destroyBuffer(b);
    this.workspace.clear();this.workspaceBytes=0;
  }
  dispatch(entry,bindings,scalars,count,batch,formats,planIndex) {
    if(this.attentionMode!=='scalar'&&(entry==='nr_scores'||entry==='nr_attend'))entry+='_tiled';
    const groups=dispatchGroups(entry,scalars,count),limit=this.runtime.device.limits.maxComputeWorkgroupsPerDimension;
    if(groups>limit*limit)throw Error(`${entry}: dispatch exceeds device limits.`);
    if(formats&&activationBindings[entry]){scalars={...scalars,...Object.fromEntries(activationBindings[entry].map(b=>[b+'Format',formats[b]??0]))};entry+='_compact';}
    const timed=this.profile?.batch(entry,scalars);
    const commands=timed??batch??this.runtime.batch();
    let invocation=planIndex===undefined?null:this.plan.invocations.get(planIndex);
    if(!invocation){invocation=this.kernels[entry].bind(bindings,scalars);if(planIndex!==undefined)this.plan.invocations.set(planIndex,invocation);}
    commands.dispatch(invocation,[Math.min(groups,limit),Math.ceil(groups/limit),1]);
    if(timed||!batch)commands.submit();
  }
  async run({width,height,proxy,inputFeatures,history,motion,seed=0,conditioning={},onProgress=()=>{},capture,signal,geometryOverride,profile=false}={}) {
    if(this.busy)throw Error('An inference is already running.');this.busy=true;
    this.profile=null;
    const runtime=this.runtime,model=this.model,live=new Map(),pool=this.workspace,owned=new Set();
    let poolBytes=this.workspaceBytes;const poolLimit=this.workspaceLimit;
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
    const flush=async()=>{
      if(batch){batch.submit();batch=null;}
      await runtime.idle();
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
      const graphKey=JSON.stringify([width,height,geometryOverride,this.activationStorage,this.gemmMode,this.attentionMode]);
      if(this.graphCache?.key!==graphKey)this.graphCache={key:graphKey,graph:createGraph(width,height,{geometryOverride,activationStorage:this.activationStorage,fuseLocalAttention:this.attentionMode==='fused'})};
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
        this.dispatch('nr_preprocess',{proxy:proxyBuffer,history:historyBuffer,features:featureBuffer},{width,height,fullWidth:g.fullWidth,fullHeight:g.fullHeight,seed,autoMask:conditioning.autoMask??1,localTone:conditioning.localTone??1,localStructure:conditioning.localStructure??1,skinStructure:conditioning.skinStructure??-1,style:conditioning.style??0,useHistory:history?1:0},g.fullWidth*g.fullHeight);
      }
      const lastUse=new Map();graph.ops.forEach((op,i)=>Object.values(op.bindings).forEach(v=>{if(typeof v==='string')lastUse.set(v,i);}));lastUse.set(graph.head,graph.ops.length);
      const boundaryById=new Map(Object.entries(graph.boundaries).map(([name,id])=>[id,name]));
      for(let index=0;index<graph.ops.length;index++) {
        if(signal?.aborted)throw new DOMException('Inference cancelled','AbortError');
        const op=graph.ops[index],bindings={},weights=[];
        for(const [key,spec] of Object.entries(op.bindings)) {
          if(typeof spec==='string') {if(!live.has(spec))live.set(spec,plan?.resources.get(spec)??take(graph.resources.get(spec).bytes));bindings[key]=live.get(spec);}
          else if(spec.kind==='zero')bindings[key]=zero;
          else {
            const persistent=typeof model.packedMatrix==='function',cacheKey=JSON.stringify(spec);
            const cached=persistent&&this.weightCache.get(cacheKey);
            if(cached){bindings[key]=cached;continue;}
            let data;
            if(spec.kind==='matrix')data=persistent?model.packedMatrix(spec.name,spec.offset,spec.K,spec.N,spec):model.matrix(spec.name,spec.offset,spec.K,spec.N,spec);
            if(spec.kind==='vector')data=model.vector(spec.name,spec.offset,spec.count,spec.type);
            if(spec.kind==='prior')data=model.prior(spec.name,spec.offset,spec.heads);
            const b=persistent?runtime.createBuffer(data):alloc(data);bindings[key]=b;
            if(persistent){this.weightCache.set(cacheKey,b);this.weightCacheBytes+=b.size;}else weights.push(b);
          }
        }
        const formats=this.activationStorage==='packed'?Object.fromEntries(Object.entries(op.bindings).filter(([,id])=>typeof id==='string').map(([key,id])=>[key,graph.resources.get(id).format])):undefined;
        batch??=runtime.batch();
        this.dispatch(op.entry==='nr_gemm'&&typeof model.packedMatrix==='function'?gemmKernel(op.scalars,this.gemmMode):op.entry,bindings,op.scalars,op.count,batch,formats,plan?index:undefined);queuedOps++;
        for(const b of weights)retire(b);model.cache.clear();
        // Keep intermediates queue-ordered, and retain resources until submitted work finishes.
        // Bound both cancellation latency and transient uploads, rather than waiting every op.
        const boundary=capture&&boundaryById.get(op.bindings.output);
        if(boundary){await flush();await capture(boundary,await this.readActivation(live.get(op.bindings.output),graph.resources.get(op.bindings.output)),graph.resources.get(op.bindings.output));}
        for(const [id,b] of live)if(lastUse.get(id)===index){live.delete(id);if(!plan)release(b);}
        if(queuedOps>=(plan?this.graphBatchSize:8)||retiredBytes>=64*1024*1024)await flush();
        onProgress({index:index+1,total:graph.ops.length,label:op.label});
      }
      await flush();
      if(signal?.aborted)throw new DOMException('Inference cancelled','AbortError');
      const head=await this.readActivation(live.get(graph.head),graph.resources.get(graph.head));let output=null;
      if(proxy) {
        const blend=model.vector(model.tensor(70,0,'blend_scale'),0,1)[0],out=alloc(pixels*16);
        this.dispatch('nr_compose',{proxy:proxyBuffer,head:live.get(graph.head),history:historyBuffer,output:out},{width,height,fullWidth:g.fullWidth,blendScale:blend,useHistory:history?1:0},pixels,undefined,this.activationStorage==='packed'?{head:graph.resources.get(graph.head).format}:undefined);
        output=await runtime.read(out);
      }
      return {head,output,geometry:g,dispatches:graph.ops.length,profile:await this.profile?.read(),workingBuffers:{created:createdBuffers,reused:reusedBuffers,prepared:!!plan,planBytes:plan?.bytes??0}};
    } finally {batch?.discard();await runtime.idle().catch(()=>{});for(const b of [...owned]){
      if(!cacheBuffer(b,old=>runtime.destroyBuffer(old)))runtime.destroyBuffer(b);
    }
    this.workspaceBytes=poolBytes;this.profile?.dispose();this.profile=null;this.model.cache.clear();this.busy=false;}
  }
  async readActivation(buffer,resource) {
    const count=resource.rows*resource.channels;
    if(!resource.format)return this.runtime.read(buffer,Float32Array,count*4);
    const words=await this.runtime.read(buffer,Uint32Array,resource.bytes);
    return unpackActivations(words,resource.format,count);
  }
  dispose(){if(this.busy)throw Error('Cancel and await inference before disposal.');this.weightCache.clear();this.weightCacheBytes=0;this.workspace.clear();this.workspaceBytes=0;this.plan=null;this.graphCache=null;this.runtime.dispose();}
}
