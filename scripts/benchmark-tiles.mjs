// Isolated GPU timings for the packed FP8 GEMMs that occur in the real graph.
// Uses synthetic finite data; no model DLL or extracted weights are required.
import {chromium} from 'playwright';
import {mkdir,writeFile} from 'node:fs/promises';
import {dirname} from 'node:path';
import {createServer} from './serve.mjs';

const integer=(name,fallback,min,max)=>{
  const n=Number(process.env[name]??fallback);
  if(!Number.isSafeInteger(n)||n<min||n>max)throw Error(`${name} must be an integer from ${min} to ${max}.`);
  return n;
};
if(Boolean(process.env.NR_WIDTH)!==Boolean(process.env.NR_HEIGHT))throw Error('Set both NR_WIDTH and NR_HEIGHT.');
const dimensions=(process.env.NR_RESOLUTIONS||(process.env.NR_WIDTH?`${process.env.NR_WIDTH}x${process.env.NR_HEIGHT}`:'1280x720,1920x1080')).split(',').map(value=>{
  const match=/^(\d+)x(\d+)$/.exec(value.trim());
  if(!match)throw Error('NR_RESOLUTIONS must contain comma-separated dimensions, e.g. 1280x720,1920x1080.');
  const [width,height]=match.slice(1).map(Number);
  if(![width,height].every(n=>Number.isSafeInteger(n)&&n>=33&&n<=8192))throw Error('Benchmark dimensions must be between 33 and 8192.');
  return {width,height};
});
const samples=integer('NR_RUNS',7,3,99),dispatches=integer('NR_TILE_DISPATCHES',3,1,32),warmups=integer('NR_TILE_WARMUP',2,1,16);
const filter=process.env.NR_TILE_FILTER||null;
if(filter)new RegExp(filter);
const report=process.env.NR_REPORT||'reports/gemm-tiles.json';
const server=createServer();await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));let browser;
try {
  browser=await chromium.launch({headless:true,...(process.env.NR_BROWSER?{executablePath:process.env.NR_BROWSER}:{}),args:['--enable-unsafe-webgpu']});
  const page=await browser.newPage(),errors=[];
  page.on('pageerror',error=>errors.push(error.message));page.on('console',message=>console.log(message.text()));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const result=await page.evaluate(async({dimensions,samples,dispatches,warmups,filter})=>{
    const {GpuRuntime}=await import('/vendor/webcuda/runtime/runtime.js');
    const {createGraph}=await import('/src/graph.js');
    const {gemmKernel,dispatchGrid}=await import('/src/kernel-selection.js');
    const {specializedGemm,dynamicGemmScalars}=await import('/src/gemm-specialization.js');
    const {GpuProfile}=await import('/src/gpu-profile.js');
    const runtime=await GpuRuntime.create({useAdapterBufferLimits:true,useAdapterWorkgroupLimits:true});
    const variants=[
      {name:'32x32',suffix:'_wide_half',rows:32,cols:32,threads:128,sharedBytes:8448},
      {name:'64x32',suffix:'_wide64x32_half',rows:64,cols:32,threads:256,sharedBytes:12544},
      {name:'32x64',suffix:'_wide32x64_half',rows:32,cols:64,threads:256,sharedBytes:12800},
    ];
    const kernels=new Map(),artifacts=[],resolutions=[];
    const getKernel=async entry=>{
      if(kernels.has(entry))return kernels.get(entry);
      const response=await fetch(`/generated/${entry}.json`);
      if(!response.ok)throw Error(`Missing ${entry}. Run npm run build first.`);
      const source=await response.text(),artifact=JSON.parse(source);
      const sha256=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(source))),b=>b.toString(16).padStart(2,'0')).join('');
      const kernel=await runtime.kernel(artifact);kernels.set(entry,kernel);
      artifacts.push({entry,sha256,workgroupSize:artifact.metadata.workgroupSize,workgroupStorageBytes:artifact.metadata.workgroupStorageBytes});
      return kernel;
    };
    // Each lane is finite. FP8 magnitudes are at most 8, inside the native-half
    // product path's weight bound; FP16 residuals are small normal values.
    const packed=(count,format,seed)=>{
      const lanes=format===1?4:2,bits=format===1?8:16,result=new Uint32Array(Math.ceil(count/lanes));
      let state=seed>>>0;
      for(let word=0;word<result.length;word++){
        let value=0;
        for(let lane=0;lane<lanes;lane++){
          state=(Math.imul(state,1664525)+1013904223)>>>0;
          const code=format===1?((state>>>8)%73)|((state>>>24)&128):(0x2000+((state>>>8)&0x17ff))|((state>>>16)&0x8000);
          if(word*lanes+lane<count)value|=code<<(lane*bits);
        }
        result[word]=value;
      }
      return result;
    };
    const median=values=>{const sorted=values.slice().sort((a,b)=>a-b),i=Math.floor(sorted.length/2);return sorted.length%2?sorted[i]:(sorted[i-1]+sorted[i])/2;};
    const stats=values=>({medianMs:median(values),minMs:Math.min(...values),maxMs:Math.max(...values),samplesMs:values});
    const compare=(expected,actual,label)=>{
      if(actual.length!==expected.length)throw Error(`${label}: word count differs.`);
      for(let i=0;i<expected.length;i++)if(expected[i]!==actual[i])throw Error(`${label}: packed word ${i} differs (${expected[i]} versus ${actual[i]}).`);
    };
    try {
      if(!runtime.device.features.has('shader-f16')||!runtime.device.features.has('timestamp-query'))throw Error('Tile benchmark requires shader-f16 and timestamp-query.');
      for(const v of variants)if(runtime.device.limits.maxComputeWorkgroupStorageSize<v.sharedBytes||runtime.device.limits.maxComputeInvocationsPerWorkgroup<v.threads||runtime.device.limits.maxComputeWorkgroupSizeX<v.threads)throw Error(`Device cannot run the ${v.name} tile.`);
      const metadata=runtime.createBuffer(256*8),siluTable=runtime.createBuffer(65536*4);
      const lookup=await getKernel('nr_lookup_tables');
      runtime.batch().dispatch(lookup.bind({metadata,silu:siluTable}),[1024,1,1]).submit();
      await runtime.idle();
      for(const {width,height} of dimensions){
        const graph=createGraph(width,height,{activationStorage:'packed',fuseLocalAttention:true,fuseNormalization:true});
        const shapes=new Map();let fp8GraphDispatches=0,excludedDispatches=0;
        for(const op of graph.ops){
          if(op.entry!=='nr_gemm'||op.scalars.halfMode)continue;
          fp8GraphDispatches++;
          const format=binding=>typeof op.bindings[binding]==='string'?graph.resources.get(op.bindings[binding]).format:0;
          const scalars={...op.scalars,inputFormat:format('input'),residualFormat:format('residual'),rawFormat:format('raw'),outputFormat:format('output')};
          const specialized=specializedGemm(gemmKernel(scalars)+'_compact',scalars);
          if(!specialized)throw Error(`Graph GEMM ${op.label} is not specialized at ${width}x${height}.`);
          const key=`${specialized}:rows=${scalars.rows}`;
          if(filter&&!new RegExp(filter).test(key)){excludedDispatches++;continue;}
          if(scalars.inputFormat!==1)throw Error(`Wide GEMM input must be FP8: ${key}`);
          let shape=shapes.get(key);
          if(!shape){shape={key,specialized,scalars,occurrences:0,labels:[]};shapes.set(key,shape);}
          shape.occurrences++;shape.labels.push(op.label);
        }
        if(!shapes.size)throw Error(`No shapes selected at ${width}x${height}.`);
        const results=[];
        for(const shape of shapes.values()){
          const s=shape.scalars,count=s.rows*s.N*s.batches,owned=[];
          const buffer=value=>{const b=runtime.createBuffer(value);owned.push(b);return b;};
          const bindings={metadata,siluTable,
            input:buffer(packed(s.rows*s.inputStride,s.inputFormat,0x12345678)),
            weights:buffer(packed(s.K*s.N*s.batches,1,0x23456789)),
            residual:buffer(s.hasResidual?packed(count,s.residualFormat,0x3456789a):new Uint32Array(1)),
            scales:buffer(new Float32Array(s.N*s.batches).fill(.5)),
            raw:buffer(s.rawEnabled?count*2:4),output:buffer(count*(s.outputFormat===1?1:2)),
          };
          const profile=new GpuProfile(runtime);
          try {
            const invocations=[];
            for(const variant of variants){
              const kernel=await getKernel(shape.specialized+variant.suffix);
              const totalGroups=Math.ceil(s.rows/variant.rows)*s.batches*Math.ceil(s.N/variant.cols),limit=runtime.device.limits.maxComputeWorkgroupsPerDimension;
              if(totalGroups>limit*limit)throw Error(`Dispatch grid exceeds device limits: ${shape.key}`);
              invocations.push({variant,invocation:kernel.bind(bindings,dynamicGemmScalars(s)),totalGroups,groups:dispatchGrid(totalGroups,limit)});
            }
            const run=(item,commands,repeats=1)=>{for(let i=0;i<repeats;i++)commands.dispatch(item.invocation,item.groups);commands.submit();};
            // Compilation, uploads, output checks and readback are outside all
            // timestamped passes. The first tile provides the exact reference.
            run(invocations[0],runtime.batch());
            const expected=await runtime.read(bindings.output,Uint32Array),expectedRaw=s.rawEnabled?await runtime.read(bindings.raw,Uint32Array):null;
            for(const item of invocations.slice(1)){
              run(item,runtime.batch());
              compare(expected,await runtime.read(bindings.output,Uint32Array),`${shape.key} ${item.variant.name} output`);
              if(expectedRaw)compare(expectedRaw,await runtime.read(bindings.raw,Uint32Array),`${shape.key} ${item.variant.name} raw`);
            }
            for(let warmup=0;warmup<warmups;warmup++)for(let i=0;i<invocations.length;i++)run(invocations[(warmup+i)%invocations.length],runtime.batch(),dispatches);
            await runtime.idle();
            for(let sample=0;sample<samples;sample++)for(let i=0;i<invocations.length;i++){
              const item=invocations[(sample+i)%invocations.length];
              run(item,profile.batch(item.variant.name,{sample}),dispatches);
            }
            await runtime.idle();
            const timings=await profile.read(),times={};
            for(const variant of variants)times[variant.name]=stats(timings.dispatches.filter(row=>row.entry===variant.name).map(row=>row.gpuMs/dispatches));
            const winner=variants.reduce((best,v)=>times[v.name].medianMs<times[best].medianMs?v.name:best,variants[0].name);
            const grids=Object.fromEntries(invocations.map(item=>[item.variant.name,{dimensions:item.groups,requiredWorkgroups:item.totalGroups,dispatchedWorkgroups:item.groups[0]*item.groups[1]*item.groups[2]}]));
            results.push({...shape,times,grids,winner,exactOutput:true,exactRaw:s.rawEnabled?true:null});
            console.log(JSON.stringify({resolution:`${width}x${height}`,key:shape.key,occurrences:shape.occurrences,medianMs:Object.fromEntries(variants.map(v=>[v.name,times[v.name].medianMs])),winner}));
          }finally{profile.dispose();for(const b of owned)runtime.destroyBuffer(b);}
        }
        const weightedGraphGpuMs=Object.fromEntries(variants.map(v=>[v.name,results.reduce((total,shape)=>total+shape.occurrences*shape.times[v.name].medianMs,0)]));
        weightedGraphGpuMs.bestPerShape=results.reduce((total,shape)=>total+shape.occurrences*shape.times[shape.winner].medianMs,0);
        const summary={width,height,fp8GraphDispatches,excludedDispatches,measuredDispatches:results.reduce((n,shape)=>n+shape.occurrences,0),uniqueShapes:results.length,weightedGraphGpuMs,results};
        resolutions.push(summary);console.log(JSON.stringify({...summary,results:undefined}));
      }
      return {configuration:{dimensions,samples,dispatches,warmups,filter,data:'Deterministic synthetic bounded FP8; finite FP16 residuals; seed fixed in script.'},measurement:'GPU timestamps per dispatch, from warm repeated-dispatch passes. Weighted totals sum isolated shape medians × graph occurrence counts; they are not measured whole-frame or whole-graph times.',adapter:runtime.describe(),artifacts,resolutions};
    }finally{runtime.dispose();}
  },{dimensions,samples,dispatches,warmups,filter});
  if(errors.length)throw Error(errors.join('; '));
  result.browser=browser.version();result.createdAt=new Date().toISOString();
  await mkdir(dirname(report),{recursive:true});await writeFile(report,JSON.stringify(result,null,2));
  console.log(`Tile benchmark saved to ${report}`);
}finally{await browser?.close();await new Promise(resolve=>server.close(resolve));}
