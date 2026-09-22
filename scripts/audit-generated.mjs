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
  const experiment=process.env.NR_AUDIT||'bitscan';if(!['bitscan','barriers','publication'].includes(experiment))throw Error('NR_AUDIT must be bitscan, barriers or publication');
  const results=await page.evaluate(async({specialization,nativeHalf,experiment})=>{
    const {modelFromDll}=await import('/src/dll-model.js'),{NeuralRenderer}=await import('/src/engine.js');
    const model=await modelFromDll(document.querySelector('#local-model').files[0]);
    const hash=async a=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',a)),v=>v.toString(16).padStart(2,'0')).join('');
    const cases=[];
    for(const [width,height] of [[1280,720],[1920,1080]]) {
      const create=async edited=>{
        const originalFetch=globalThis.fetch;let changed=0;
        if(edited)globalThis.fetch=async(...args)=>{
          const response=await originalFetch(...args),url=String(args[0]);
          if(!url.endsWith('.json')||url.endsWith('/manifest.json'))return response;
          const artifact=await response.json();
          if(typeof artifact.wgsl==='string'){
            if(experiment==='barriers'&&url.endsWith('_half.json')&&artifact.wgsl.includes('storageBarrier();')){
              artifact.wgsl=artifact.wgsl.replaceAll('storageBarrier();','');changed++;
            }else if(experiment==='publication'&&/_1_[01]_half.json$/.test(url)){
              const publication=/(fn f_nr_quant\([^]*?\) -> f32 \{)[^]*?\n\}/;
              if(publication.test(artifact.wgsl)){artifact.wgsl=artifact.wgsl.replace(publication,'$1\n  return cw_arg_v;\n}');changed++;}
            }else if(experiment==='bitscan'){
              const scan=/var v_msb: u32 = [^;]*countLeadingZeros[^;]*;/;
              if(scan.test(artifact.wgsl)){
                artifact.wgsl=artifact.wgsl.replace(scan,'var v_msb: u32 = 0u; var v_t: u32 = v_mag; loop { if(v_t <= 1u){break;} v_t >>= 1u; v_msb += 1u; }');changed++;
              }
            }
          }
          return new Response(JSON.stringify(artifact),{headers:{'Content-Type':'application/json'}});
        };
        try{
          const renderer=await NeuralRenderer.create(model,{nativeHalf:true});
          if(edited&&!changed){renderer.dispose();throw Error('No eligible emitted shaders changed');}
          return renderer;
        }finally{globalThis.fetch=originalFetch;}
      };
      const setupStarted=performance.now();
      // Bit-scan comparison restores the old loop only in the baseline; candidate is production CUDA.
      const baseline=await create(experiment==='bitscan'),baselineSetupMs=performance.now()-setupStarted;
      const candidateStarted=performance.now(),candidate=await create(experiment!=='bitscan'),candidateSetupMs=performance.now()-candidateStarted;
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
