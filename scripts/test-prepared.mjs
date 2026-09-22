import assert from 'node:assert/strict';
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
  const results=await page.evaluate(async()=>{
    const {modelFromDll}=await import('/src/dll-model.js'),{NeuralRenderer}=await import('/src/engine.js'),{geometry}=await import('/src/geometry.js');
    const model=await modelFromDll(document.querySelector('#local-model').files[0]);
    const reference=await NeuralRenderer.create(model,{activationStorage:'float',executionMode:'streamed',attentionMode:'tiled'}),prepared=await NeuralRenderer.create(model);
    const width=97,height=65,proxy=Float32Array.from({length:width*height*4},(_,i)=>i%4===3?1:(i%251)/251),history=proxy.slice(),motion=new Float32Array(proxy.length),g=geometry(width,height);
    for(let i=0;i<motion.length;i+=4){motion[i]=1/width;motion[i+1]=-1/height;motion[i+2]=i%8?1:0;}
    const inputFeatures=Float32Array.from({length:g.fullWidth*g.fullHeight*16},(_,i)=>(i%31-15)/16),results=[];
    const hash=async a=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',a)),v=>v.toString(16).padStart(2,'0')).join('');
    try {
      for(const [name,extra] of [
        ['initial',{}],
        ['changed seed, controls, history and motion',{seed:219,conditioning:{autoMask:0,localTone:0.5,localStructure:0.25,style:0.6},history,motion}],
        ['feature override and aligned history',{inputFeatures,history}],
        ['return to preprocessing without history',{seed:13}],
      ]) {
        if(results.length)for(let i=0;i<proxy.length;i++)if(i%4!==3)proxy[i]=1-proxy[i];
        const args={width,height,proxy,...extra},expected=await reference.run(args),actual=await prepared.run(args);
        const matches=(a,b)=>{const aa=new Uint32Array(a.buffer),bb=new Uint32Array(b.buffer);return aa.length===bb.length&&aa.every((v,i)=>v===bb[i]);};
        if(!matches(expected.head,actual.head)||!matches(expected.output,actual.output))throw Error('Prepared output mismatch: '+name);
        results.push({name,head:await hash(actual.head),output:await hash(actual.output),prepared:actual.workingBuffers.prepared});
      }
      return results;
    }finally{reference.dispose();prepared.dispose();}
  });
  assert.deepEqual(errors,[]);assert(results.every(r=>r.prepared));assert.equal(new Set(results.map(r=>r.output)).size,results.length);
  console.log('Prepared and reference paths match bit for bit for all changing-input scenarios:',results.map(r=>r.name).join('; '));
}finally{await browser?.close();await new Promise(r=>server.close(r));}
