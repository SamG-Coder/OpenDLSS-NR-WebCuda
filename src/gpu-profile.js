// Diagnostic mode uses one timestamped pass per dispatch. It changes submission
// overhead, so GPU durations must not be presented as normal frame wall time.
export class GpuProfile {
  constructor(runtime) {
    this.runtime=runtime;this.records=[];this.supported=runtime.device.features.has('timestamp-query');
    if(this.supported)this.queries=runtime.device.createQuerySet({type:'timestamp',count:2048});
  }
  batch(entry,scalars) {
    if(!this.supported)return null;
    const index=this.records.length;if(index>=1024)throw Error('GPU profile query capacity exceeded.');
    this.records.push({entry,...scalars});
    return this.runtime.batch({timestampWrites:{querySet:this.queries,beginningOfPassWriteIndex:index*2,endOfPassWriteIndex:index*2+1}});
  }
  async read() {
    if(!this.supported)return {supported:false,reason:'Device does not expose timestamp-query.'};
    const device=this.runtime.device,count=this.records.length*2,bytes=count*8;
    if(!count)return {supported:true,dispatches:[],byKernel:{}};
    const resolve=device.createBuffer({size:bytes,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
    const readback=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try {
      const encoder=device.createCommandEncoder();encoder.resolveQuerySet(this.queries,0,count,resolve,0);encoder.copyBufferToBuffer(resolve,0,readback,0,bytes);device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);const timestamps=new BigUint64Array(readback.getMappedRange());
      const dispatches=this.records.map((r,i)=>({...r,gpuMs:Number(timestamps[i*2+1]-timestamps[i*2])/1e6})),byKernel={};
      for(const r of dispatches)byKernel[r.entry]=(byKernel[r.entry]??0)+r.gpuMs;
      readback.unmap();return {supported:true,dispatches,byKernel,totalGpuMs:dispatches.reduce((n,r)=>n+r.gpuMs,0)};
    }finally{resolve.destroy();readback.destroy();}
  }
  dispose(){this.queries?.destroy();}
}
