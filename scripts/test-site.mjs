import {chromium} from 'playwright';
import {readFile} from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import assert from 'node:assert/strict';
const root=path.resolve('site'),prefix='/OpenDLSS-NR-WebCuda/';
const types={'.js':'text/javascript','.html':'text/html','.json':'application/json','.css':'text/css','.wasm':'application/wasm'};
const server=http.createServer(async(req,res)=>{try{const url=new URL(req.url,'http://local');if(!url.pathname.startsWith(prefix)){res.writeHead(404).end();return;}const file=path.resolve(root,decodeURIComponent(url.pathname.slice(prefix.length))||'index.html');if(!file.startsWith(root+path.sep))throw Error('Outside root');const data=await readFile(file);res.writeHead(200,{'Content-Type':types[path.extname(file)]||'application/octet-stream'});res.end(data);}catch{res.writeHead(404).end();}});
await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
try{
 browser=await chromium.launch({headless:true,...(process.env.NR_BROWSER?{executablePath:process.env.NR_BROWSER}:{}),args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 const page=await browser.newPage(),errors=[],badRequests=[];page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)badRequests.push(r.url());});
 await page.goto(`http://127.0.0.1:${server.address().port}${prefix}`);await page.locator('h1').waitFor();
 assert.equal(await page.locator('#gpu-status').textContent()!=='Checking WebGPU…',true);
 await page.locator('#dll').setInputFiles({name:'bad.dll',buffer:Buffer.alloc(64)});await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('not a Windows DLL'));
 const png=await page.evaluate(()=>{const c=document.createElement('canvas');c.width=160;c.height=96;c.getContext('2d').fillRect(0,0,160,96);return c.toDataURL().split(',')[1];});
 await page.locator('#image').setInputFiles({name:'test.png',buffer:Buffer.from(png,'base64')});await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('160 × 96'));assert(await page.locator('#run').isDisabled());
 await page.locator('#mode-object').click();
 const data=Buffer.from(new Float32Array([-1,-1,0,1,-1,0,0,1,0]).buffer);
 const gltf={asset:{version:'2.0'},scene:0,scenes:[{nodes:[0]}],nodes:[{mesh:0}],meshes:[{primitives:[{attributes:{POSITION:0}}]}],buffers:[{uri:'data:application/octet-stream;base64,'+data.toString('base64'),byteLength:36}],bufferViews:[{buffer:0,byteLength:36}],accessors:[{bufferView:0,componentType:5126,count:3,type:'VEC3',min:[-1,-1,0],max:[1,1,0]}]};
 await page.locator('#object').setInputFiles({name:'triangle.gltf',buffer:Buffer.from(JSON.stringify(gltf))});await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('Input ready: 512 × 512'));
 assert(await page.locator('#resolution-error').isHidden());assert(await page.locator('#input-empty').isHidden());
 const assets=await page.evaluate(async()=>{const base=new URL('.',location.href);const manifest=await(await fetch(new URL('generated/manifest.json',base))).json();const paths=['node_modules/three/examples/jsm/libs/draco/gltf/draco_decoder.wasm','node_modules/three/examples/jsm/libs/basis/basis_transcoder.wasm',...Object.values(manifest).map(f=>'generated/'+f)];return Promise.all(paths.map(async p=>({p,ok:(await fetch(new URL(p,base))).ok})));});assert(assets.every(a=>a.ok),JSON.stringify(assets));
 assert.deepEqual(errors,[]);assert.deepEqual(badRequests,[]);console.log('Pages smoke test passed at repository subpath: module graph, CSS, DLL rejection, exact image dimensions, local glTF rendering, decoder assets, and all 12 generated kernels. No proprietary test data used.');
}finally{await browser?.close();await new Promise(r=>server.close(r));}
