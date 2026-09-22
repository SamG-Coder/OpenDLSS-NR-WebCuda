// Separate comparison harness. An external upstream checkout and local extracted
// model are required; neither is copied into the app or publication artifact.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import http from 'node:http';
import path from 'node:path';
import {readFile,writeFile} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {chromium} from 'playwright';
import {createServer} from './serve.mjs';

if(!process.env.NR_UPSTREAM||!process.env.NR_MODEL)throw Error('Set NR_UPSTREAM to the upstream repository and NR_MODEL to your local extracted model directory.');
const upstream=path.resolve(process.env.NR_UPSTREAM),model=path.resolve(process.env.NR_MODEL);
const oursRevision=execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim();
const sourceHashes=Object.fromEntries(await Promise.all(['kernels/numeric.cuh','kernels/gemm-half.cu','vendor/webcuda/compiler/compiler.js','vendor/webcuda/compiler/cpu-oracle.js'].map(async file=>[file,createHash('sha256').update(await readFile(file)).digest('hex')])));
const oursWorkingTreeDirty=Boolean(execFileSync('git',['status','--porcelain','--untracked-files=no'],{encoding:'utf8'}).trim());
const revision=execFileSync('git',['-C',upstream,'rev-parse','HEAD'],{encoding:'utf8'}).trim();
const upstreamDirty=execFileSync('git',['-C',upstream,'status','--porcelain'],{encoding:'utf8'}).trim();
if(upstreamDirty)throw Error('Use an unmodified upstream checkout.');
const application=createServer(),mounts={'/comparison-upstream/':path.join(upstream,'ports/browser-webgpu'),'/comparison-model/':model};
const server=http.createServer(async(req,res)=>{
  try {
    const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname),mount=Object.keys(mounts).find(p=>pathname.startsWith(p));
    if(!mount){application.emit('request',req,res);return;}
    const root=mounts[mount],file=path.resolve(root,pathname.slice(mount.length));
    if(!file.startsWith(root+path.sep)||path.relative(root,file).split(path.sep).some(p=>p.startsWith('.'))){res.writeHead(403).end();return;}
    const bytes=await readFile(file);res.writeHead(200,{'Content-Type':file.endsWith('.js')?'text/javascript':file.endsWith('.json')?'application/json':'application/octet-stream','Cache-Control':'no-store'});res.end(bytes);
  }catch{res.writeHead(404).end();}
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
try {
  browser=await chromium.launch({headless:true,...(process.env.NR_BROWSER?{executablePath:process.env.NR_BROWSER}:{}),args:['--enable-unsafe-webgpu']});
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());else if(m.text().startsWith('COMPARE '))console.log(m.text());});
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const results=await page.evaluate(async({profile})=>{
    const {Network}=await import('/comparison-upstream/src/network.js');
    const {Model}=await import('/src/model.js'),{NeuralRenderer}=await import('/src/engine.js'),{geometry}=await import('/src/geometry.js');
    const hash=async a=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',a)),v=>v.toString(16).padStart(2,'0')).join('');
    const modelStart=performance.now(),model=await Model.load(async p=>{const r=await fetch('/comparison-model/'+p);if(!r.ok)throw Error('Missing local model file');return r.arrayBuffer();}),modelLoadMs=performance.now()-modelStart;
    const results=[];
    for(const [width,height] of [[512,512],[1280,720],[1920,1080]]){
      console.log(`COMPARE setup ${width}x${height}`);
      let ours,upstream;
      try {
        const start=performance.now();ours=await NeuralRenderer.create(model);const oursSetupMs=performance.now()-start;
        const shaderModules=[],pipelineDescriptions=[],moduleIds=new WeakMap();
        const originalModule=GPUDevice.prototype.createShaderModule,originalPipeline=GPUDevice.prototype.createComputePipelineAsync;
        if(profile){GPUDevice.prototype.createShaderModule=function(d){const module=originalModule.call(this,d);moduleIds.set(module,shaderModules.length);shaderModules.push({label:d.label??'',code:d.code});return module;};GPUDevice.prototype.createComputePipelineAsync=function(d){pipelineDescriptions.push({label:d.label??'',module:moduleIds.get(d.compute.module),entryPoint:d.compute.entryPoint,constants:d.compute.constants??{}});return originalPipeline.call(this,d);};}
        const upStart=performance.now();try{upstream=await Network.create({weights:'/comparison-model',width,height});}finally{GPUDevice.prototype.createShaderModule=originalModule;GPUDevice.prototype.createComputePipelineAsync=originalPipeline;}const upstreamSetupMs=performance.now()-upStart;
        const g=geometry(width,height);if(g.fullWidth!==upstream.geometry.fullWidth||g.fullHeight!==upstream.geometry.fullHeight)throw Error('Padded geometry differs');
        const proxy=Float32Array.from({length:width*height*4},(_,i)=>{const p=i>>2;return i%4===3?1:i%4===0?(p%width)/(width-1):i%4===1?Math.floor(p/width)/(height-1):0.4;});
        const runtime=ours.runtime,source=runtime.createBuffer(proxy),featureBuffer=runtime.createBuffer(g.fullWidth*g.fullHeight*64);
        let features;
        try {
          ours.dispatch('nr_preprocess',{proxy:source,history:source,features:featureBuffer},{width,height,fullWidth:g.fullWidth,fullHeight:g.fullHeight,seed:0,autoMask:1,localTone:1,localStructure:1,skinStructure:-1,style:0,useHistory:0},g.fullWidth*g.fullHeight);
          features=await runtime.read(featureBuffer);
        }finally{runtime.destroyBuffer(source);runtime.destroyBuffer(featureBuffer);}
        const inputSha256=await hash(features),samples={ours:[],upstream:[]},cold={},heads={};
        let mismatchCount=0,maxAbsError=0;
        for(let round=-1;round<6;round++){
          for(const mode of round%2===0?['upstream','ours']:['ours','upstream']){
            const t=performance.now();let head;
            if(mode==='ours')head=(await ours.run({width,height,inputFeatures:features})).head;
            else {upstream.writeFeatures(features);await upstream.run();head=await upstream.readHead();}
            const elapsedMs=performance.now()-t;
            if(!head.every(Number.isFinite))throw Error('Non-finite '+mode+' output');
            if(round<0)cold[mode]=elapsedMs;else samples[mode].push(elapsedMs);
            heads[mode]=head;
          }
          if(heads.ours.length!==heads.upstream.length)throw Error('Head dimensions differ');
          const a=new Uint32Array(heads.ours.buffer,heads.ours.byteOffset,heads.ours.length),b=new Uint32Array(heads.upstream.buffer,heads.upstream.byteOffset,heads.upstream.length);
          let count=0,maximum=0;for(let i=0;i<a.length;i++){if(a[i]!==b[i])count++;maximum=Math.max(maximum,Math.abs(heads.ours[i]-heads.upstream[i]));}
          mismatchCount=Math.max(mismatchCount,count);maxAbsError=Math.max(maxAbsError,maximum);
        }
        let profiles;
        if(profile){
          if(!upstream.recorder.enableProfiling())throw Error('Upstream timestamp-query unavailable');
          profiles={ours:[],upstream:[]};
          for(let round=0;round<3;round++){
            const result=await ours.run({width,height,inputFeatures:features,profile:true});
            if(!result.profile.supported)throw Error('Renderer timestamp-query unavailable');
            if(await hash(result.head)!==await hash(heads.ours))throw Error('Profiled renderer output differs');
            profiles.ours.push(result.profile);
            upstream.writeFeatures(features);await upstream.run();
            if(await hash(await upstream.readHead())!==await hash(heads.upstream))throw Error('Profiled upstream output differs');
            profiles.upstream.push(await upstream.recorder.readProfile());
          }
        }
        const dispatchInventory=profile?upstream.recorder.passes.filter(p=>p.kind==='dispatch').map(p=>({entry:p.entryPoint,label:p.label,x:p.x,y:p.y,z:p.z})):undefined;
        const result={width,height,profiles,shaderModules:profile?shaderModules:undefined,pipelineDescriptions:profile?pipelineDescriptions:undefined,dispatchInventory,paddedWidth:g.fullWidth,paddedHeight:g.fullHeight,inputSha256,modelLoadMs,oursSetupMs,upstreamSetupMs,cold,samples,headHashes:{ours:await hash(heads.ours),upstream:await hash(heads.upstream)},mismatchCount,maxAbsError,headElements:heads.ours.length,dispatches:{ours:ours.graphCache.graph.ops.length,upstream:upstream.recorder.dispatchCount},memory:{upstreamActivations:upstream.tensors.total,upstreamWeights:upstream.model.bytesUploaded,oursPlan:ours.plan?.bytes,oursWeights:ours.weightCacheBytes},adapter:{ours:runtime.describe(),upstream:{vendor:upstream.adapterInfo.vendor,architecture:upstream.adapterInfo.architecture,features:[...upstream.device.features],workgroupStorage:upstream.device.limits.maxComputeWorkgroupStorageSize}}};
        results.push(result);console.log('COMPARE '+JSON.stringify({...result,profiles:undefined,shaderModules:undefined,pipelineDescriptions:undefined,dispatchInventory:undefined}));
      }finally{upstream?.destroy();ours?.dispose();}
    }
    return results;
  },{profile:process.env.NR_PROFILE==='1'});
  assert.deepEqual(errors,[]);
  for(const result of results)assert.equal(result.mismatchCount,0,'Network output differs at '+result.width+'x'+result.height);
  await writeFile(process.env.NR_REPORT||'reports/upstream-comparison.json',JSON.stringify({oursRevision,oursWorkingTreeDirty,sourceHashes,upstreamRevision:revision,browser:browser.version(),scope:'Identical precomputed features: upload + graph execution + float32 head readback; no preprocessing, composition, or UI.',results},null,2));
}finally{await browser?.close();await new Promise(r=>server.close(r));}
