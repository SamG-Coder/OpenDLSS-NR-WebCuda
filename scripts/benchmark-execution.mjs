import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {createServer} from './serve.mjs';
const runs=Number(process.env.NR_RUNS||6);
if(!Number.isSafeInteger(runs)||runs<2||runs>100)throw Error('NR_RUNS must be an integer from 2 to 100.');
const gemmBackend=process.env.NR_GEMM_BACKEND||'half',baselineGemmBackend=process.env.NR_BASELINE_GEMM_BACKEND||'half';
const compareBackends=Boolean(process.env.NR_GEMM_BACKEND||process.env.NR_BASELINE_GEMM_BACKEND);
for(const backend of [gemmBackend,baselineGemmBackend])if(!['half','prepared-half','prepared-integer'].includes(backend))throw Error('GEMM backend must be half, prepared-half, or prepared-integer.');
if(compareBackends&&(process.env.NR_PREVIOUS_GENERATED||process.env.NR_PREVIOUS_ENGINE))throw Error('Backend comparisons require the same current generated shaders and renderer for both modes.');
if(!process.env.NR_DLL)throw Error('Set NR_DLL to a local compatible DLL.');
const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
try {
  browser=await chromium.launch({headless:true,...(process.env.NR_BROWSER?{executablePath:process.env.NR_BROWSER}:{}),args:['--enable-unsafe-webgpu']});
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  await page.evaluate(()=>{const input=document.createElement('input');input.type='file';input.id='local-model';document.body.append(input);});
  await page.locator('#local-model').setInputFiles(process.env.NR_DLL);
  const enhanced=compareBackends||process.env.NR_ENHANCED==='1',normalizeAttention=enhanced||process.env.NR_NORMALIZE==='1',wideGemm=normalizeAttention||process.env.NR_WIDE==='1';
  const previousGenerated=process.env.NR_PREVIOUS_GENERATED||null;
  const configuration={runs,gemmBackend,baselineGemmBackend,compareBackends,previousEngine:process.env.NR_PREVIOUS_ENGINE||null,gemmTile:process.env.NR_TILE||'auto',previousGenerated,enhanced,normalizeAttention,wideGemm,nativeHalf:wideGemm||process.env.NR_HALF==='1',specialization:process.env.NR_SPECIALIZE==='1'};
  const results=await page.evaluate(async({runs,gemmBackend,baselineGemmBackend,compareBackends,previousEngine,gemmTile,specialization,nativeHalf,wideGemm,normalizeAttention,enhanced,previousGenerated})=>{
    const {modelFromDll}=await import('/src/dll-model.js'),{NeuralRenderer}=await import('/src/engine.js');
    const BaselineRenderer=previousEngine?(await import('/'+previousEngine)).NeuralRenderer:NeuralRenderer;
    const model=await modelFromDll(document.querySelector('#local-model').files[0]);
    const hash=async a=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',a)),v=>v.toString(16).padStart(2,'0')).join('');
    const cases=[];
    for(const [width,height] of [[1280,720],[1920,1080]]) {
      const setupStarted=performance.now();
      const originalFetch=globalThis.fetch;
      if(previousGenerated)globalThis.fetch=(url,...args)=>{const target=new URL(url,location.href);if(target.origin===location.origin&&target.pathname.startsWith('/generated/'))target.pathname='/'+previousGenerated.replace(/^\/+|\/+$/g,'')+'/'+target.pathname.slice('/generated/'.length);return originalFetch(target,...args);};
      let baseline;
      try { baseline=await BaselineRenderer.create(model,compareBackends?{gemmBackend:baselineGemmBackend,gemmTile,specializeGemm:true,nativeHalf,wideGemm,normalizeAttention}:previousGenerated?{gemmTile:'32x32',nativeHalf:true,wideGemm:true,normalizeAttention:true}:enhanced?{nativeHalf:true,wideGemm:false,normalizeAttention:false}:normalizeAttention?{nativeHalf:true,wideGemm:true,normalizeAttention:false}:wideGemm?{nativeHalf:true,wideGemm:false,normalizeAttention:false}:nativeHalf?{nativeHalf:false,wideGemm:false,normalizeAttention:false}:specialization?{specializeGemm:false,nativeHalf:false,wideGemm:false,normalizeAttention:false}:{maxInFlightBatches:1,cacheNoise:false,nativeHalf:false,wideGemm:false,normalizeAttention:false}); }finally{globalThis.fetch=originalFetch;}const baselineSetupMs=performance.now()-setupStarted;
      const candidateStarted=performance.now(),candidate=await NeuralRenderer.create(model,{gemmBackend,gemmTile,specializeGemm:true,nativeHalf,wideGemm,normalizeAttention}),candidateSetupMs=performance.now()-candidateStarted;
      const engines=[baseline,candidate],samples=[[],[]];
      const proxy=Float32Array.from({length:width*height*4},(_,i)=>{const p=i>>2;return i%4===3?1:i%4===0?(p%width)/(width-1):i%4===1?Math.floor(p/width)/(height-1):0.4;});
      let expected,adapter,preparation;
      try {
        for(let round=-1;round<runs;round++)for(const mode of round%2===0?[1,0]:[0,1]) {
          const engine=engines[mode],before={...engine.runtime.stats},start=performance.now();
          const result=await engine.run({width,height,proxy,readHead:!compareBackends&&!specialization&&!nativeHalf&&mode===0}),elapsedMs=performance.now()-start;
          const requestedBackend=mode===0?baselineGemmBackend:gemmBackend;
          if(compareBackends&&requestedBackend!=='half'&&!(result.gemmBackend?.prepared>0))throw Error(`The ${requestedBackend} comparison did not dispatch any prepared kernels.`);
          const outputHash=await hash(result.output);expected??=outputHash;if(expected!==outputHash)throw Error('Output mismatch');
          if(round>=0)samples[mode].push({elapsedMs,timings:result.timings,gemmBackend:result.gemmBackend,noiseCache:result.noiseCache,readbackBytes:engine.runtime.stats.readbackBytes-before.readbackBytes,workingBuffers:result.workingBuffers});
        }
        adapter=engines[1].runtime.describe();
        preparation=engines.map(engine=>engine.preparedModel?{...engine.preparedModel.stats}:null);
      }finally{engines.forEach(e=>e.dispose());}
      cases.push({width,height,baselineGemmBackend,candidateGemmBackend:gemmBackend,baselineSetupMs,candidateSetupMs,preparation,outputHash:expected,adapter,baseline:samples[0],candidate:samples[1]});
    }
    return cases;
  },configuration);
  assert.deepEqual(errors,[]);
  await writeFile(process.env.NR_REPORT||'reports/execution-comparison.json',JSON.stringify({...configuration,browser:browser.version(),cases:results},null,2));
  for(const r of results){const median=values=>{const a=values.map(v=>v.elapsedMs).sort((a,b)=>a-b);const mid=Math.floor(a.length/2);return a.length%2?a[mid]:(a[mid-1]+a[mid])/2;};console.log(JSON.stringify({width:r.width,height:r.height,baselineMs:median(r.baseline),candidateMs:median(r.candidate),baseline:r.baseline.map(v=>v.elapsedMs),candidate:r.candidate.map(v=>v.elapsedMs),outputHash:r.outputHash}));}
}finally{await browser?.close();await new Promise(r=>server.close(r));}
