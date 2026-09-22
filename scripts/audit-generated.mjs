// Diagnostic only: compare emitted WGSL edits without changing production artifacts.
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {chromium} from 'playwright';
import {createServer} from './serve.mjs';
if(!process.env.NR_DLL)throw Error('Set NR_DLL to a local compatible DLL.');
const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
try {
  browser=await chromium.launch({headless:true,...(process.env.NR_BROWSER?{executablePath:process.env.NR_BROWSER}:{}),args:['--enable-unsafe-webgpu']});
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  await page.evaluate(()=>{const input=document.createElement('input');input.type='file';input.id='local-model';document.body.append(input);});
  await page.locator('#local-model').setInputFiles(process.env.NR_DLL);
  const experiment=process.env.NR_AUDIT||'bitscan';if(!['bitscan','barriers'].includes(experiment))throw Error('NR_AUDIT must be bitscan or barriers');
  const results=await page.evaluate(async({specialization,nativeHalf,experiment})=>{
    const {modelFromDll}=await import('/src/dll-model.js'),{NeuralRenderer}=await import('/src/engine.js');
    const model=await modelFromDll(document.querySelector('#local-model').files[0]);
    const hash=async a=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',a)),v=>v.toString(16).padStart(2,'0')).join('');
    const cases=[];
    for(const [width,height] of [[1280,720],[1920,1080]]) {
      const setupStarted=performance.now();
      const baseline=await NeuralRenderer.create(model,{nativeHalf:true});const baselineSetupMs=performance.now()-setupStarted;
      const candidateStarted=performance.now(),originalFetch=globalThis.fetch;
      let changed=0;
      globalThis.fetch=async(...args)=>{const r=await originalFetch(...args);if(String(args[0]).endsWith('_half.json')){const a=await r.json();const scan=/var v_msb: u32 = 0u;\s*var v_t: u32 = v_mag;\s*\{\s*loop \{\s*if \(!\(v_t > 1u\)\) \{ break; \}\s*v_t = \(v_t >> 1u\);\s*v_msb = \(v_msb \+ 1u\);\s*\}\s*\}/;if(experiment==='barriers'&&a.wgsl.includes('storageBarrier();')){a.wgsl=a.wgsl.replaceAll('storageBarrier();','');changed++;}else if(experiment==='bitscan'&&scan.test(a.wgsl)){a.wgsl=a.wgsl.replace(scan,'var v_msb: u32 = firstLeadingBit(v_mag);');changed++;}return new Response(JSON.stringify(a),{headers:{'Content-Type':'application/json'}});}return r;};
      let candidate;try{candidate=await NeuralRenderer.create(model,{nativeHalf:true});}finally{globalThis.fetch=originalFetch;}
      if(!changed)throw Error('No eligible emitted shaders changed');
      const candidateSetupMs=performance.now()-candidateStarted;
      const engines=[baseline,candidate],samples=[[],[]];
      const proxy=Float32Array.from({length:width*height*4},(_,i)=>{const p=i>>2;return i%4===3?1:i%4===0?(p%width)/(width-1):i%4===1?Math.floor(p/width)/(height-1):0.4;});
      let expected,adapter;
      try {
        for(let round=-1;round<6;round++)for(const mode of round%2===0?[1,0]:[0,1]) {
          const engine=engines[mode],before={...engine.runtime.stats},start=performance.now();
          const result=await engine.run({width,height,proxy,readHead:!specialization&&!nativeHalf&&mode===0}),elapsedMs=performance.now()-start;
          const outputHash=await hash(result.output);expected??=outputHash;if(expected!==outputHash)throw Error('Output mismatch');
          if(round>=0)samples[mode].push({elapsedMs,timings:result.timings,noiseCache:result.noiseCache,readbackBytes:engine.runtime.stats.readbackBytes-before.readbackBytes,workingBuffers:result.workingBuffers});
        }
        adapter=engines[1].runtime.describe();
      }finally{engines.forEach(e=>e.dispose());}
      cases.push({width,height,baselineSetupMs,candidateSetupMs,outputHash:expected,adapter,baseline:samples[0],candidate:samples[1]});
    }
    return cases;
  },{specialization:false,nativeHalf:true,experiment});
  assert.deepEqual(errors,[]);
  await writeFile(process.env.NR_REPORT||'reports/generated-'+experiment+'.json',JSON.stringify({experiment,nativeHalf:true,browser:browser.version(),cases:results},null,2));
  for(const r of results){const median=values=>{const a=values.map(v=>v.elapsedMs).sort((a,b)=>a-b);return(a[2]+a[3])/2;};console.log(JSON.stringify({width:r.width,height:r.height,baselineMs:median(r.baseline),candidateMs:median(r.candidate),baseline:r.baseline.map(v=>v.elapsedMs),candidate:r.candidate.map(v=>v.elapsedMs),outputHash:r.outputHash}));}
}finally{await browser?.close();await new Promise(r=>server.close(r));}
