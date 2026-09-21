import {GpuRuntime} from '../vendor/webcuda/runtime/runtime.js';
import {createGraph} from './graph.js';
export class NeuralRenderer {
  static async create(model,options={}) {
    const runtime=await GpuRuntime.create(options),kernels={};
    try {
      const response=await fetch(new URL('../generated/manifest.json',import.meta.url));if(!response.ok)throw Error('Run npm run build before starting.');
      const manifest=await response.json();
      for(const [entry,file] of Object.entries(manifest)) {const r=await fetch(new URL('../generated/'+file,import.meta.url));if(!r.ok)throw Error('Missing kernel '+file);kernels[entry]=await runtime.kernel(await r.json());}
      return new NeuralRenderer(runtime,kernels,model);
    } catch(e) {runtime.dispose();throw e;}
  }
  constructor(runtime,kernels,model) {this.runtime=runtime;this.kernels=kernels;this.model=model;this.busy=false;}
  dispatch(entry,bindings,scalars,count) {
    const groups=Math.ceil(count/64),limit=this.runtime.device.limits.maxComputeWorkgroupsPerDimension;
    if(groups>limit*limit)throw Error(`${entry}: dispatch exceeds device limits.`);
    this.runtime.batch().dispatch(this.kernels[entry].bind(bindings,scalars),[Math.min(groups,limit),Math.ceil(groups/limit),1]).submit();
  }
  async run({width,height,proxy,inputFeatures,history,motion,seed=0,conditioning={},onProgress=()=>{},capture,signal,geometryOverride}={}) {
    if(this.busy)throw Error('An inference is already running.');this.busy=true;
    const runtime=this.runtime,model=this.model,live=new Map(),pool=new Map(),owned=new Set();
    const alloc=data=>{const b=runtime.createBuffer(data);owned.add(b);return b;};
    const release=b=>{const list=pool.get(b.size)||[];list.push(b);pool.set(b.size,list);};
    const take=size=>pool.get(size)?.pop()||alloc(size);
    try {
      const graph=createGraph(width,height,{geometryOverride}),g=graph.geometry;
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
            let data;
            if(spec.kind==='matrix')data=model.matrix(spec.name,spec.offset,spec.K,spec.N,spec);
            if(spec.kind==='vector')data=model.vector(spec.name,spec.offset,spec.count,spec.type);
            if(spec.kind==='prior')data=model.prior(spec.name,spec.offset,spec.heads);
            const b=alloc(data);bindings[key]=b;weights.push(b);
          }
        }
        this.dispatch(op.entry,bindings,op.scalars,op.count);
        // Bound submissions and memory. This is a correctness backend, not a fused realtime backend.
        await runtime.idle();
        for(const b of weights){runtime.destroyBuffer(b);owned.delete(b);}model.cache.clear();
        if(capture)for(const id of [op.bindings.output])if(boundaryById.has(id))await capture(boundaryById.get(id),await runtime.read(live.get(id)),graph.resources.get(id));
        for(const [id,b] of live)if(lastUse.get(id)===index){live.delete(id);release(b);}
        onProgress({index:index+1,total:graph.ops.length,label:op.label});
      }
      const head=await runtime.read(live.get(graph.head));let output=null;
      if(proxy) {
        const blend=model.vector(model.tensor(70,0,'blend_scale'),0,1)[0],out=alloc(pixels*16);
        this.dispatch('nr_compose',{proxy:proxyBuffer,head:live.get(graph.head),history:historyBuffer,output:out},{width,height,fullWidth:g.fullWidth,blendScale:blend,useHistory:history?1:0},pixels);
        output=await runtime.read(out);
      }
      return {head,output,geometry:g,dispatches:graph.ops.length};
    } finally {await runtime.idle().catch(()=>{});for(const b of owned)runtime.destroyBuffer(b);this.model.cache.clear();this.busy=false;}
  }
  dispose(){if(this.busy)throw Error('Cancel and await inference before disposal.');this.runtime.dispose();}
}
