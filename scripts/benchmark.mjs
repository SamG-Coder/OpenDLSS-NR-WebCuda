import {chromium} from 'playwright';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {createServer} from './serve.mjs';

const width=Number(process.env.NR_WIDTH||1280),height=Number(process.env.NR_HEIGHT||720),runs=Number(process.env.NR_RUNS||3);
if(![width,height,runs].every(n=>Number.isSafeInteger(n)&&n>0)||width<33||height<33)throw Error('Invalid benchmark dimensions or run count (minimum 33 × 33).');
if(!process.env.NR_DLL)throw Error('Set NR_DLL to your local NR DLL. It is never uploaded.');
const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
try {
  browser=await chromium.launch({headless:true,...(process.env.NR_BROWSER?{executablePath:process.env.NR_BROWSER}:{}),args:['--enable-unsafe-webgpu']});
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>console.log(m.text()));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  await page.evaluate(()=>{const input=document.createElement('input');input.type='file';input.id='benchmark-dll';document.body.append(input);});
  await page.locator('#benchmark-dll').setInputFiles(process.env.NR_DLL);
  const result=await page.evaluate(async({width,height,runs,profile,gemmMode,attentionMode,workspaceCacheBytes,activationStorage,executionMode,graphBatchSize,maxInFlightBatches,cacheNoise,specializeGemm,readHead,captureBoundaries})=>{
    const {modelFromDll}=await import('/src/dll-model.js'),{NeuralRenderer}=await import('/src/engine.js');
    const model=await modelFromDll(document.querySelector('#benchmark-dll').files[0]);
    const engine=await NeuralRenderer.create(model,{gemmMode,attentionMode,workspaceCacheBytes,activationStorage,executionMode,graphBatchSize,maxInFlightBatches,cacheNoise,specializeGemm}),runtime=engine.runtime,measurements=[];
    const proxy=new Float32Array(width*height*4);
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){const p=(y*width+x)*4;proxy[p]=x/(width-1);proxy[p+1]=y/(height-1);proxy[p+2]=0.4;proxy[p+3]=1;}
    let decodeMs=0,waitMs=0,waits=0;
    for(const name of ['matrix','packedMatrix','vector','prior']){if(!model[name])continue;const original=model[name].bind(model);model[name]=(...args)=>{const t=performance.now();try{return original(...args);}finally{decodeMs+=performance.now()-t;}};}
    const idle=runtime.idle.bind(runtime);runtime.idle=async()=>{const t=performance.now();waits++;try{return await idle();}finally{waitMs+=performance.now()-t;}};
    const hash=async data=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',data)),b=>b.toString(16).padStart(2,'0')).join('');
    try {
      for(let run=0;run<runs;run++){
        decodeMs=0;waitMs=0;waits=0;const before={...runtime.stats},start=performance.now();
        const boundaries=captureBoundaries?{}:undefined;
        const output=await engine.run({width,height,proxy,profile,readHead,capture:captureBoundaries?async(name,data)=>{boundaries[name]=await hash(data);}:undefined});
        const elapsedMs=performance.now()-start;
        if(!output.output.every(Number.isFinite)||(output.head&&!output.head.every(Number.isFinite)))throw Error('Non-finite benchmark output.');
        const measurement={run:run+1,boundaries,timings:output.timings,noiseCache:output.noiseCache,readbackBytes:runtime.stats.readbackBytes-before.readbackBytes,workingBuffers:output.workingBuffers,workspaceBytes:engine.workspaceBytes,profile:output.profile,elapsedMs,decodeMs,waitMs,waits,weightCacheBytes:engine.weightCacheBytes,bindGroupsCreated:runtime.stats.bindGroupsCreated-before.bindGroupsCreated,dataBytesUploaded:runtime.stats.dataBytesUploaded-before.dataBytesUploaded,submissions:runtime.stats.submissions-before.submissions,headSha256:output.head?await hash(output.head):null,outputSha256:await hash(output.output)};
        if(run>0&&engine.weightCacheBytes>0&&measurement.dataBytesUploaded!==proxy.byteLength+(output.workingBuffers.prepared?0:4))throw Error('Warm render unexpectedly uploaded model data.');
        measurements.push(measurement);console.log(JSON.stringify({...measurement,boundaries:boundaries?Object.keys(boundaries).length:undefined,profile:measurement.profile?{supported:measurement.profile.supported,byKernel:measurement.profile.byKernel}:undefined}));
      }
      return {width,height,configuration:{profile,gemmMode,attentionMode,workspaceCacheBytes,activationStorage,executionMode,graphBatchSize,maxInFlightBatches,cacheNoise,specializeGemm,readHead,captureBoundaries},adapter:runtime.describe(),measurements};
    }finally{engine.dispose();}
  },{width,height,runs,specializeGemm:process.env.NR_SPECIALIZE!=='0',maxInFlightBatches:Number(process.env.NR_INFLIGHT??4),cacheNoise:process.env.NR_NOISE_CACHE!=='0',readHead:process.env.NR_READ_HEAD!=='0',graphBatchSize:Number(process.env.NR_BATCH??32),executionMode:process.env.NR_EXECUTION||'prepared',activationStorage:process.env.NR_ACTIVATIONS||'packed',profile:process.env.NR_PROFILE==='1',captureBoundaries:process.env.NR_CAPTURE==='1',gemmMode:process.env.NR_GEMM||'auto',attentionMode:process.env.NR_ATTENTION||'fused',workspaceCacheBytes:Number(process.env.NR_WORKSPACE_MIB??256)*1024*1024});
  if(errors.length)throw Error(errors.join('; '));
  result.browser=browser.version();
  if(process.env.NR_COMPARE){const baseline=JSON.parse(await readFile(process.env.NR_COMPARE,'utf8'));if(baseline.width!==width||baseline.height!==height)throw Error('Baseline dimensions differ.');for(const run of result.measurements)if((run.headSha256!==null&&run.headSha256!==baseline.measurements[0].headSha256)||run.outputSha256!==baseline.measurements[0].outputSha256)throw Error('Output differs from baseline.');if(baseline.measurements[0].boundaries){for(const run of result.measurements)if(JSON.stringify(run.boundaries)!==JSON.stringify(baseline.measurements[0].boundaries))throw Error('Intermediate boundaries differ from baseline.');}result.matchesBaseline=true;}
  await mkdir('reports',{recursive:true});await writeFile(process.env.NR_REPORT||'reports/performance.json',JSON.stringify(result,null,2));
}finally{await browser?.close();await new Promise(r=>server.close(r));}
