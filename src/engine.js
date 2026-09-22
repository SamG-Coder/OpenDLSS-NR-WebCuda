import {GpuRuntime} from '../vendor/webcuda/runtime/runtime.js';
import {createGraph} from './graph.js';
import {GpuProfile} from './gpu-profile.js';
import {gemmKernel,dispatchGroups} from './kernel-selection.js';
export class NeuralRenderer {
  static async create(model,options={}) {
    const {workspaceCacheBytes=256*1024*1024,gemmMode='auto',attentionMode='tiled',...runtimeOptions}=options;
    if(!Number.isSafeInteger(workspaceCacheBytes)||workspaceCacheBytes<0)throw Error('Invalid workspace cache budget.');
    if(!['auto','tiled','scalar','tile8x8','tile8x16'].includes(gemmMode))throw Error('Invalid GEMM mode.');
    if(!['tiled','scalar'].includes(attentionMode))throw Error('Invalid attention mode.');
    const runtime=await GpuRuntime.create({useAdapterBufferLimits:true,...runtimeOptions}),kernels={};
    try {
      const response=await fetch(new URL('../generated/manifest.json',import.meta.url));if(!response.ok)throw Error('Run npm run build before starting.');
      const manifest=await response.json();
      for(const [entry,file] of Object.entries(manifest)) {const r=await fetch(new URL('../generated/'+file,import.meta.url));if(!r.ok)throw Error('Missing kernel '+file);kernels[entry]=await runtime.kernel(await r.json());}
      return new NeuralRenderer(runtime,kernels,model,{workspaceCacheBytes,gemmMode,attentionMode});
    } catch(e) {runtime.dispose();throw e;}
  }
  constructor(runtime,kernels,model,{workspaceCacheBytes=256*1024*1024,gemmMode='auto',attentionMode='tiled'}={}) {this.runtime=runtime;this.kernels=kernels;this.model=model;this.busy=false;this.weightCache=new Map();this.weightCacheBytes=0;this.workspace=new Map();this.workspaceBytes=0;this.workspaceLimit=workspaceCacheBytes;this.gemmMode=gemmMode;this.attentionMode=attentionMode;}
  clearWeightCache() {
    if(this.busy)throw Error('Cancel and await inference before clearing weights.');
    for(const b of this.weightCache.values())this.runtime.destroyBuffer(b);
    this.weightCache.clear();this.weightCacheBytes=0;
  }
  clearWorkspace() {
    if(this.busy)throw Error('Cancel and await inference before clearing working buffers.');
    this.#releaseWorkspace();
  }
  #releaseWorkspace() {
    for(const list of this.workspace.values())for(const b of list)this.runtime.destroyBuffer(b);
    this.workspace.clear();this.workspaceBytes=0;
  }
  dispatch(entry,bindings,scalars,count,batch) {
    if(this.attentionMode==='tiled'&&(entry==='nr_scores'||entry==='nr_attend'))entry+='_tiled';
    const groups=dispatchGroups(entry,scalars,count),limit=this.runtime.device.limits.maxComputeWorkgroupsPerDimension;
    if(groups>limit*limit)throw Error(`${entry}: dispatch exceeds device limits.`);
    const timed=this.profile?.batch(entry,scalars);
    const commands=timed??batch??this.runtime.batch();
    commands.dispatch(this.kernels[entry].bind(bindings,scalars),[Math.min(groups,limit),Math.ceil(groups/limit),1]);
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
      const graph=createGraph(width,height,{geometryOverride}),g=graph.geometry;
      const workspaceKey=JSON.stringify(g);
      if(this.workspaceKey!==workspaceKey){this.#releaseWorkspace();poolBytes=0;this.workspaceKey=workspaceKey;}
      model.validateGraph?.(graph);
      for(const r of graph.resources.values())if(r.bytes>runtime.device.limits.maxStorageBufferBindingSize)throw Error('Resolution requires a buffer larger than the device limit.');
      const pixels=width*height,featureCount=g.fullWidth*g.fullHeight*16;
      if(inputFeatures&&(!(inputFeatures instanceof Float32Array)||inputFeatures.length!==featureCount))throw Error('Input features do not match the padded geometry.');
      if(!inputFeatures&&(!(proxy instanceof Float32Array)||proxy.length!==pixels*4))throw Error('Supply RGBA Float32Array proxy or padded input features.');
      if(history&&(!(history instanceof Float32Array)||history.length!==pixels*4))throw Error('History must be an already reprojected RGBA Float32Array.');
      if(proxy&&(!(proxy instanceof Float32Array)||proxy.length!==pixels*4))throw Error('Proxy must be source-sized RGBA Float32Array.');
      if(motion&&(!history||!proxy||!(motion instanceof Float32Array)||motion.length!==pixels*4))throw Error('Motion requires proxy, history, and source-sized RGBA UV displacements with validity in z.');
      const zero=alloc(new Float32Array(1)),proxyBuffer=proxy?alloc(proxy):zero;
      let historyBuffer=history?alloc(history):proxyBuffer;
      if(motion) {
        const reprojected=alloc(pixels*16),motionBuffer=alloc(motion);
        this.dispatch('nr_reproject',{history:historyBuffer,motion:motionBuffer,proxy:proxyBuffer,output:reprojected},{width,height},pixels);
        historyBuffer=reprojected;
      }
      const featureBuffer=inputFeatures?alloc(inputFeatures):alloc(featureCount*4);live.set(graph.features,featureBuffer);
      if(!inputFeatures) {
        this.dispatch('nr_preprocess',{proxy:proxyBuffer,history:historyBuffer,features:featureBuffer},{width,height,fullWidth:g.fullWidth,fullHeight:g.fullHeight,seed,autoMask:conditioning.autoMask??1,localTone:conditioning.localTone??1,localStructure:conditioning.localStructure??1,skinStructure:conditioning.skinStructure??-1,style:conditioning.style??0,useHistory:history?1:0},g.fullWidth*g.fullHeight);
      }
      const lastUse=new Map();graph.ops.forEach((op,i)=>Object.values(op.bindings).forEach(v=>{if(typeof v==='string')lastUse.set(v,i);}));lastUse.set(graph.head,graph.ops.length);
      const boundaryById=new Map(Object.entries(graph.boundaries).map(([name,id])=>[id,name]));
      for(let index=0;index<graph.ops.length;index++) {
        if(signal?.aborted)throw new DOMException('Inference cancelled','AbortError');
        const op=graph.ops[index],bindings={},weights=[];
        for(const [key,spec] of Object.entries(op.bindings)) {
          if(typeof spec==='string') {if(!live.has(spec))live.set(spec,take(graph.resources.get(spec).bytes));bindings[key]=live.get(spec);}
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
        batch??=runtime.batch();
        this.dispatch(op.entry==='nr_gemm'&&typeof model.packedMatrix==='function'?gemmKernel(op.scalars,this.gemmMode):op.entry,bindings,op.scalars,op.count,batch);queuedOps++;
        for(const b of weights)retire(b);model.cache.clear();
        // Keep intermediates queue-ordered, and retain resources until submitted work finishes.
        // Bound both cancellation latency and transient uploads, rather than waiting every op.
        const boundary=capture&&boundaryById.get(op.bindings.output);
        if(boundary){await flush();await capture(boundary,await runtime.read(live.get(op.bindings.output)),graph.resources.get(op.bindings.output));}
        for(const [id,b] of live)if(lastUse.get(id)===index){live.delete(id);release(b);}
        if(queuedOps>=8||retiredBytes>=64*1024*1024)await flush();
        onProgress({index:index+1,total:graph.ops.length,label:op.label});
      }
      await flush();
      if(signal?.aborted)throw new DOMException('Inference cancelled','AbortError');
      const head=await runtime.read(live.get(graph.head));let output=null;
      if(proxy) {
        const blend=model.vector(model.tensor(70,0,'blend_scale'),0,1)[0],out=alloc(pixels*16);
        this.dispatch('nr_compose',{proxy:proxyBuffer,head:live.get(graph.head),history:historyBuffer,output:out},{width,height,fullWidth:g.fullWidth,blendScale:blend,useHistory:history?1:0},pixels);
        output=await runtime.read(out);
      }
      return {head,output,geometry:g,dispatches:graph.ops.length,profile:await this.profile?.read(),workingBuffers:{created:createdBuffers,reused:reusedBuffers}};
    } finally {batch?.discard();await runtime.idle().catch(()=>{});for(const b of [...owned]){
      if(!cacheBuffer(b,old=>runtime.destroyBuffer(old)))runtime.destroyBuffer(b);
    }
    this.workspaceBytes=poolBytes;this.profile?.dispose();this.profile=null;this.model.cache.clear();this.busy=false;}
  }
  dispose(){if(this.busy)throw Error('Cancel and await inference before disposal.');this.weightCache.clear();this.weightCacheBytes=0;this.workspace.clear();this.workspaceBytes=0;this.runtime.dispose();}
}
