import {chromium} from 'playwright';
import {writeFile} from 'node:fs/promises';
import {createServer} from './serve.mjs';
const width=Number(process.env.NR_WIDTH||33),height=Number(process.env.NR_HEIGHT||33);
const gemmBackend=process.env.NR_GEMM_BACKEND||'half';
if(![width,height].every(n=>Number.isInteger(n)&&n>=33))throw Error('Invalid NR_WIDTH/NR_HEIGHT');
if(!['half','prepared-half','prepared-integer','precomputed-half'].includes(gemmBackend))throw Error('Invalid NR_GEMM_BACKEND');
if(!process.env.NR_DLL)throw Error('Set NR_DLL to your nvngx_dlssnr.dll');
const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
try {
  browser=await chromium.launch({headless:true,...(process.env.NR_BROWSER?{executablePath:process.env.NR_BROWSER}:{}),args:['--enable-unsafe-webgpu']});
  const errors=[],page=await browser.newPage({viewport:{width:1440,height:1100}});page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>console.log(m.text()));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.locator('summary').filter({hasText:'Model preparation'}).click();
  await page.locator('#gemm-backend').selectOption(gemmBackend);
  await page.locator('#dll').setInputFiles(process.env.NR_DLL);
  await page.waitForFunction(()=>document.querySelector('#model-status').textContent.includes('153 tensors'),{},{timeout:120000});
  const png=await page.evaluate(({width,height})=>{const c=document.createElement('canvas');c.width=width;c.height=height;const x=c.getContext('2d');for(let y=0;y<height;y++)for(let a=0;a<width;a++){x.fillStyle=`rgb(${Math.round(a*230/(width-1))},${Math.round(y*230/(height-1))},100)`;x.fillRect(a,y,1,1);}return c.toDataURL().split(',')[1];},{width,height});
  await page.locator('#image').setInputFiles({name:'gradient.png',mimeType:'image/png',buffer:Buffer.from(png,'base64')});
  await page.waitForFunction(()=>!document.querySelector('#run').disabled);
  await page.evaluate(()=>{new MutationObserver(()=>{const s=document.querySelector('#progress-text').textContent,m=s.match(/^(\d+) \/ /);if(m&&Number(m[1])%50===0)console.log(s);}).observe(document.querySelector('#progress-text'),{childList:true});});
  const started=performance.now();await page.locator('#run').click();
  await page.waitForFunction(()=>!document.querySelector('#run').disabled,{},{timeout:600000});
  const result=await page.evaluate(()=>{
    const status=document.querySelector('#status').textContent,c=document.querySelector('#output'),p=c.getContext('2d').getImageData(0,0,c.width,c.height).data,src=document.querySelector('#input').getContext('2d').getImageData(0,0,c.width,c.height).data;
    let different=0,min=255,max=0;for(let i=0;i<p.length;i++)if(i%4!==3){if(p[i]!==src[i])different++;min=Math.min(min,p[i]);max=Math.max(max,p[i]);}
    return {status,downloadReady:!document.querySelector('#download').hidden,width:c.width,height:c.height,differentChannels:different,min,max};
  });
  result.elapsedSeconds=(performance.now()-started)/1000;
  result.gemmBackend=gemmBackend;
  await page.screenshot({path:'reports/real-model-browser.png',fullPage:true});
  await writeFile('reports/real-model.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
  if(errors.length||!result.status.startsWith('Completed in ')||!result.downloadReady||result.width!==width||result.height!==height||!result.differentChannels||result.min===result.max)throw Error('Real-model rendering failed: '+result.status+' '+errors.join('; '));
} finally {await browser?.close();await new Promise(r=>server.close(r));}
