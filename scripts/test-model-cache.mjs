import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {createServer} from './serve.mjs';

// This test imports only CPU model modules. It never creates a GPU adapter/device.
const server=createServer();await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const origin=`http://127.0.0.1:${server.address().port}`;let browser;
try {
  browser=await chromium.launch({headless:true,...(process.env.NR_BROWSER?{executablePath:process.env.NR_BROWSER}:{}),args:['--disable-gpu','--disable-webgpu']});
  const page=await browser.newPage(),errors=[],unexpectedRequests=[];
  page.on('pageerror',error=>errors.push(error.message));
  page.on('request',request=>{if(request.method()!=='GET'||!request.url().startsWith(origin+'/'))unexpectedRequests.push(request.url());});
  await page.route(origin+'/',route=>route.fulfill({contentType:'text/html',body:'<!doctype html><meta charset="utf-8"><title>CPU model cache test</title><input type="file" id="dll">'}));
  await page.addInitScript(()=>{
    globalThis.gpuRequests=0;
    Object.defineProperty(navigator,'gpu',{value:{requestAdapter(){globalThis.gpuRequests++;throw Error('GPU use is forbidden in the CPU model-cache test.');}}});
  });
  await page.goto(origin+'/');
  const results=[];
  for(const phase of ['cold','warm','corrupt']) {
    if(phase!=='cold')await page.reload();
    if(process.env.NR_DLL)await page.locator('#dll').setInputFiles(process.env.NR_DLL);
    const result=await page.evaluate(async({phase,fromDll})=>{
      const {Model}=await import('/src/model.js');
      const {createGraph}=await import('/src/graph.js');
      const {prepareModel,clearPreparedModelCache,PREPARED_FORMAT_VERSION}=await import('/src/model-preparation.js');
      const {createPreparedModelCache,sha256}=await import('/src/prepared-model-cache.js');
      let model;
      if(fromDll) {
        const {modelFromDll}=await import('/src/dll-model.js');
        model=await modelFromDll(document.querySelector('#dll').files[0]);
      } else {
        const lengths=new Map();
        for(const op of createGraph(1280,720).ops)for(const spec of Object.values(op.bindings)) {
          if(!spec||typeof spec!=='object'||spec.kind==='zero')continue;
          const length=spec.kind==='matrix'?spec.batches*spec.K*(spec.halfMode?Math.ceil(spec.N/16)*32:spec.N):
            spec.kind==='prior'?spec.heads*8192:spec.count*(spec.type==='float'?4:2);
          lengths.set(spec.name,Math.max(lengths.get(spec.name)||0,spec.offset+length));
        }
        // Synthetic codes are finite and bounded for every native tensor layout.
        const tensors=new Map(Array.from(lengths,([name,length],tensor)=>[name,Uint8Array.from({length},(_,i)=>(i*17+tensor*13)%82)]));
        model=new Model({totals:{blockCount:71}},tensors);
      }
      if(phase==='cold'&&!await clearPreparedModelCache())throw Error('Browser IndexedDB cache is unavailable.');
      if(phase==='corrupt') {
        const db=await new Promise((resolve,reject)=>{const request=indexedDB.open('opendlss-nr-prepared-models',1);request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);});
        const keys=await new Promise((resolve,reject)=>{const request=db.transaction('matrices','readonly').objectStore('matrices').getAllKeys();request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);});
        db.close();
        if(keys.length!==358)throw Error('Expected all 358 prepared cache records.');
        const cache=await createPreparedModelCache();
        try {
          const changed=await cache.get(keys[0]);changed.words[0]^=1;
          if(!await cache.set(keys[0],changed))throw Error('Could not corrupt the test record.');
          const outdated=await cache.get(keys[1]);outdated.formatVersion=PREPARED_FORMAT_VERSION+1;
          if(!await cache.set(keys[1],outdated))throw Error('Could not change the test record version.');
        } finally {cache.close();}
      }
      const started=performance.now(),prepared=await prepareModel(model),elapsedMs=performance.now()-started;
      const hashes=[];
      for(const [key,matrix] of prepared.matrices)hashes.push([key,await sha256(new Uint8Array(matrix.words.buffer,matrix.words.byteOffset,matrix.words.byteLength))]);
      const digest=await sha256(new TextEncoder().encode(JSON.stringify(hashes)));
      if(phase==='corrupt') {
        if(!await clearPreparedModelCache())throw Error('Could not clear prepared browser data.');
        const cache=await createPreparedModelCache();
        try {
          for(const [key] of hashes)if(await cache.get(`${PREPARED_FORMAT_VERSION}/${prepared.modelHash}/${key}`)!==undefined)throw Error('Prepared data survived cache clearing.');
        } finally {cache.close();}
      }
      return {phase,source:fromDll?'local DLL':'synthetic',elapsedMs,modelHash:prepared.modelHash,digest,
        stats:prepared.stats,gpuRequests:globalThis.gpuRequests};
    },{phase,fromDll:!!process.env.NR_DLL});
    assert.equal(result.stats.preparedCount,358);assert.equal(result.stats.skippedCount,2);
    assert.equal(result.stats.cacheAvailable,true);assert.equal(result.stats.preparedBytes,152820480);
    assert.equal(result.gpuRequests,0);results.push(result);
  }
  const [cold,warm,corrupt]=results;
  assert.equal(cold.stats.cacheHits,0);assert.equal(cold.stats.cacheWrites,358);
  assert.equal(warm.stats.cacheHits,358);assert.equal(warm.stats.cacheWrites,0);assert.equal(warm.stats.cacheRejected,0);
  assert.equal(corrupt.stats.cacheHits,356);assert.equal(corrupt.stats.cacheRejected,2);assert.equal(corrupt.stats.cacheWrites,2);
  assert.equal(warm.modelHash,cold.modelHash);assert.equal(corrupt.modelHash,cold.modelHash);
  assert.equal(warm.digest,cold.digest);assert.equal(corrupt.digest,cold.digest);
  assert.deepEqual(errors,[]);assert.deepEqual(unexpectedRequests,[]);
  console.log(JSON.stringify({test:'CPU browser model cache',source:cold.source,preparedMatrices:358,preparedBytes:cold.stats.preparedBytes,gpuRequests:0,
    phases:results.map(({phase,elapsedMs,stats})=>({phase,elapsedMs,cacheHits:stats.cacheHits,cacheWrites:stats.cacheWrites,cacheRejected:stats.cacheRejected})),
    contentDigest:cold.digest,cleared:true},null,2));
} finally {await browser?.close();await new Promise(resolve=>server.close(resolve));}
