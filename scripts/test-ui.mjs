import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {createServer} from './serve.mjs';
const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
function fixture(binary=false,external=false){const positions=new Float32Array([-1,-1,0,1,-1,0,0,1,0,0,0,1]),indices=new Uint16Array([0,1,2,0,3,1,1,3,2,2,3,0]),buffer=Buffer.concat([Buffer.from(positions.buffer),Buffer.from(indices.buffer)]);const doc={asset:{version:'2.0'},scene:0,scenes:[{nodes:[0]}],nodes:[{mesh:0}],meshes:[{primitives:[{attributes:{POSITION:0},indices:1,material:0}]}],materials:[{pbrMetallicRoughness:{baseColorFactor:[.7,.35,.15,1],metallicFactor:.1,roughnessFactor:.65},doubleSided:true}],buffers:[{byteLength:buffer.length,...(!binary?{uri:external?'mesh.bin':'data:application/octet-stream;base64,'+buffer.toString('base64')}:{})}],bufferViews:[{buffer:0,byteOffset:0,byteLength:positions.byteLength},{buffer:0,byteOffset:positions.byteLength,byteLength:indices.byteLength}],accessors:[{bufferView:0,componentType:5126,count:4,type:'VEC3',min:[-1,-1,0],max:[1,1,1]},{bufferView:1,componentType:5123,count:12,type:'SCALAR'}]};if(!binary)return {doc:Buffer.from(JSON.stringify(doc)),buffer};let j=Buffer.from(JSON.stringify(doc));j=Buffer.concat([j,Buffer.alloc((4-j.length%4)%4,32)]);const bin=Buffer.concat([buffer,Buffer.alloc((4-buffer.length%4)%4)]),header=Buffer.alloc(12),jc=Buffer.alloc(8),bc=Buffer.alloc(8);header.writeUInt32LE(0x46546c67);header.writeUInt32LE(2,4);header.writeUInt32LE(12+8+j.length+8+bin.length,8);jc.writeUInt32LE(j.length);jc.writeUInt32LE(0x4e4f534a,4);bc.writeUInt32LE(bin.length);bc.writeUInt32LE(0x004e4942,4);return Buffer.concat([header,jc,j,bc,bin]);}
try{
 browser=await chromium.launch({headless:true,...(process.env.NR_BROWSER?{executablePath:process.env.NR_BROWSER}:{}),args:['--enable-unsafe-webgpu']});
 const page=await browser.newPage({viewport:{width:1500,height:1100}}),errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(r.method()!=='GET')requests.push(r.url());});page.setDefaultTimeout(15000);
 await page.goto(`http://127.0.0.1:${server.address().port}`);await page.locator('h1').waitFor();
 assert.equal(await page.locator('#tone').inputValue(),'1');assert.equal(await page.locator('#structure').inputValue(),'1');assert(await page.locator('#auto-mask').isChecked());assert(await page.locator('#run').isDisabled());
 await page.locator('#dll').setInputFiles({name:'invalid.dll',buffer:Buffer.alloc(64)});await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('not a Windows DLL'));
 const png=await page.evaluate(()=>{const c=document.createElement('canvas');c.width=320;c.height=192;const x=c.getContext('2d'),g=x.createLinearGradient(0,0,320,192);g.addColorStop(0,'#e2b991');g.addColorStop(1,'#3e668b');x.fillStyle=g;x.fillRect(0,0,320,192);return c.toDataURL().split(',')[1];});
 await page.locator('#image').setInputFiles({name:'landscape.png',buffer:Buffer.from(png,'base64')});await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('320 × 192'));
 assert.equal(await page.locator('#width').inputValue(),'320');assert.equal(await page.locator('#height').inputValue(),'192');assert(await page.locator('#width').isDisabled());
 await page.locator('#resolution-mode').selectOption('custom');await page.locator('#width').fill('160');await page.locator('#width').dispatchEvent('change');assert.equal(await page.locator('#height').inputValue(),'96');
 await page.locator('#width').fill('16384');await page.locator('#width').dispatchEvent('change');assert(await page.locator('#resolution-error').isVisible());assert(await page.locator('#run').isDisabled());
 await page.locator('#resolution-mode').selectOption('original');assert.equal(await page.locator('#width').inputValue(),'320');
 await page.locator('#skin-mode').selectOption('custom');assert(await page.locator('#skin').isVisible());await page.locator('#reset').click();assert.equal(await page.locator('#skin-mode').inputValue(),'follow');
 await page.locator('#mode-object').click();await page.locator('#object').setInputFiles({name:'tetra.glb',buffer:fixture(true)});await page.waitForFunction(()=>document.querySelector('#source-name').textContent==='tetra.glb');await page.waitForFunction(()=>!document.body.classList.contains('busy'));
  assert(await page.locator('#viewport canvas').isVisible());
  assert(await page.locator('#resolution-error').isHidden(),'512-square 3D preview must work before model setup');
  assert.deepEqual(await page.locator('#input').evaluate(c=>[c.width,c.height]),[512,512]);
  assert(await page.locator('#input-empty').isHidden(),'The source must be captured, not merely an empty canvas');
 await page.locator('#width').fill('192');await page.locator('#width').dispatchEvent('change');await page.locator('#height').fill('128');await page.locator('#height').dispatchEvent('change');
 const f=fixture(false,true);await page.locator('#object').setInputFiles([{name:'tetra.gltf',buffer:f.doc},{name:'mesh.bin',buffer:f.buffer}]);await page.waitForFunction(()=>document.querySelector('#source-name').textContent==='tetra.gltf');await page.waitForFunction(()=>!document.body.classList.contains('busy'));
 await page.locator('#object').setInputFiles({name:'broken.gltf',buffer:f.doc});await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('Missing local glTF asset'));assert.equal(await page.locator('#source-name').textContent(),'tetra.gltf');
 await page.locator('#lighting').selectOption('side');await page.locator('#frame-object').click();
 const variance=await page.locator('#input').evaluate(c=>{const p=c.getContext('2d').getImageData(0,0,c.width,c.height).data;return new Set(p).size;});assert(variance>30,'3D capture should contain a lit object');
 if(process.env.NR_DLL){
  await page.locator('#dll').setInputFiles(process.env.NR_DLL);await page.waitForFunction(()=>document.querySelector('#model-status').textContent.includes('153 tensors'),{},{timeout:120000});
  await page.locator('#width').fill('512');await page.locator('#width').dispatchEvent('change');await page.locator('#height').fill('512');await page.locator('#height').dispatchEvent('change');
  assert(await page.locator('#resolution-error').isHidden(),'Actual NR device supports the 54 MiB packed buffer');
  await page.locator('#run').click();await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Completed in '),{},{timeout:600000});
  assert.deepEqual(await page.locator('#output').evaluate(c=>[c.width,c.height]),[512,512]);
  const timing=await page.locator('#output').evaluate(c=>JSON.parse(c.dataset.timings));assert(timing.totalMs>0&&timing.inputMs>=0&&timing.presentationMs>=0&&timing.pngMs>=0);assert(timing.engine.totalMs<=timing.totalMs);
  console.log('Verified 512 × 512 3D preview before model setup and real NR output after loading DLL.');
  await page.locator('#width').fill('192');await page.locator('#width').dispatchEvent('change');await page.locator('#height').fill('128');await page.locator('#height').dispatchEvent('change');
  await page.locator('#run').click();await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Completed in '),{},{timeout:600000});
  assert.deepEqual(await page.locator('#output').evaluate(c=>[c.width,c.height]),[192,128]);assert(await page.locator('#download').isVisible());
  const downloadPromise=page.waitForEvent('download');await page.locator('#download').click();assert.match((await downloadPromise).suggestedFilename(),/192x128\.png$/);
  await page.locator('#view-wipe').click();assert(await page.locator('#wipe-control').isVisible());await page.locator('#view-pair').click();
  await page.screenshot({path:'reports/object-workspace.png',fullPage:true});
  await page.locator('#mode-image').click();assert(await page.locator('#download').isHidden());
  await page.locator('#run').click();await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Completed in '),{},{timeout:600000});assert.deepEqual(await page.locator('#output').evaluate(c=>[c.width,c.height]),[320,192]);
  const square=await page.evaluate(()=>{const c=document.createElement('canvas');c.width=c.height=256;const x=c.getContext('2d');x.fillStyle='#7aa8bf';x.fillRect(0,0,256,256);return c.toDataURL().split(',')[1];});
  await page.locator('#image').setInputFiles({name:'square.png',buffer:Buffer.from(square,'base64')});await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('256 × 256'));
  await page.locator('#run').click();assert(await page.locator('#width').isDisabled());assert(await page.locator('#mode-object').isDisabled());await page.locator('#cancel').click();await page.waitForFunction(()=>document.querySelector('#status').textContent==='Render cancelled.');assert(await page.locator('#download').isHidden());
  await page.locator('#run').click();await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Completed in '),{},{timeout:600000});assert.deepEqual(await page.locator('#output').evaluate(c=>[c.width,c.height]),[256,256]);
  await page.screenshot({path:'reports/browser-runner.png',fullPage:true});
  await page.locator('#dll').setInputFiles({name:'invalid.dll',buffer:Buffer.alloc(64)});await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('not a Windows DLL'));assert(!await page.locator('#run').isDisabled());
  await page.locator('summary').filter({hasText:'Temporal inputs'}).click();await page.locator('#motion').setInputFiles({name:'bad.f32',buffer:Buffer.alloc(16)});await page.locator('#run').click();await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('require a previous'));await page.locator('#clear-temporal').click();
  await page.locator('summary').filter({hasText:'Network input override'}).click();await page.locator('#features').setInputFiles({name:'bad.f32',buffer:Buffer.alloc(16)});await page.locator('#run').click();await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('Features requires exactly'));await page.locator('#clear-features').click();
  await page.locator('summary').filter({hasText:'Network input override'}).click();await page.locator('summary').filter({hasText:'Temporal inputs'}).click();
  const settingsPromise=page.waitForEvent('download');await page.locator('#save-settings').click();assert.equal((await settingsPromise).suggestedFilename(),'nr-settings.json');
 }
 await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'No mobile horizontal overflow');await page.screenshot({path:'reports/mobile-workspace.png',fullPage:true});
 assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);console.log('UI passed: native defaults, exact image resolution, aspect ratio, GPU limits, GLB and glTF loading, missing assets, real image/3D inference when DLL supplied, export, wipe, invalid auxiliary inputs, mobile layout, no page errors or uploads.');
}finally{await browser?.close();await new Promise(r=>server.close(r));}
