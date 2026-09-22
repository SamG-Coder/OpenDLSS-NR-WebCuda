import {chromium} from 'playwright';
import {writeFile} from 'node:fs/promises';
import {createServer} from './serve.mjs';
const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
try{
  browser=await chromium.launch({headless:true,executablePath:process.env.NR_BROWSER||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',args:['--enable-unsafe-webgpu']});
  const page=await browser.newPage();page.on('console',m=>console.log(m.text()));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const results=await page.evaluate(async()=>{
    const {Model,unpackActivations}=await import('/src/model.js'),{NeuralRenderer}=await import('/src/engine.js'),{createGraph}=await import('/src/graph.js');
    const model=await Model.load(async p=>{const r=await fetch('/models/nr/'+p);if(!r.ok)throw Error(p);return new Uint8Array(await r.arrayBuffer());});
    const results=[];
    for(const [width,height] of [[1280,720],[1920,1080]]){
      const graph=createGraph(width,height,{activationStorage:'packed',fuseLocalAttention:true,fuseNormalization:true});
      const g=graph.geometry,inputFeatures=new Float32Array(g.fullWidth*g.fullHeight*16);
      for(let i=0;i<inputFeatures.length;i++)inputFeatures[i]=Math.sin(i*0.0017)*0.125;
      const renderer=await NeuralRenderer.create(model),actual=await renderer.run({width,height,inputFeatures});
      const r=await fetch(`/build/native-benchmark/head-${width}.bin`);if(!r.ok)throw Error('Native output missing');
      const bytes=await r.arrayBuffer(),resource=graph.resources.get(graph.head);
      const expected=resource.format?unpackActivations(new Uint32Array(bytes),resource.format,resource.rows*resource.channels):new Float32Array(bytes);
      const a=new Uint32Array(actual.head.buffer),b=new Uint32Array(expected.buffer);let mismatches=0,nonfinite=0,maxError=0;
      if(a.length!==b.length)throw Error('Head length mismatch');
      for(let i=0;i<a.length;i++){if(a[i]!==b[i])mismatches++;if(!Number.isFinite(expected[i]))nonfinite++;maxError=Math.max(maxError,Math.abs(actual.head[i]-expected[i]));}
      const samples=[];for(let i=0;i<5;i++){const start=performance.now();await renderer.run({width,height,inputFeatures,readHead:false});samples.push(performance.now()-start);}
      const result={width,height,elements:a.length,mismatches,nonfinite,maxError,browserWallMs:samples};results.push(result);console.log(JSON.stringify(result));renderer.dispose();
    }
    return results;
  });
  await writeFile('reports/our-native-comparison.json',JSON.stringify(results,null,2));
  if(results.some(x=>x.mismatches||x.nonfinite))throw Error('Native/browser head comparison failed');
}finally{await browser?.close();await new Promise(r=>server.close(r));}
