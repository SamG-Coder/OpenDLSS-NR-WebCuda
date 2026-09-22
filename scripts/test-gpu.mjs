import {chromium} from 'playwright';
import {writeFile} from 'node:fs/promises';
import {createServer} from './serve.mjs';
const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
try {
  browser=await chromium.launch({headless:true,...(process.env.NR_BROWSER?{executablePath:process.env.NR_BROWSER}:{}),args:['--enable-unsafe-webgpu']});
  const page=await browser.newPage();page.on('pageerror',e=>console.error(e));
  await page.goto(`http://127.0.0.1:${server.address().port}/tests/gpu.html`);
  await page.waitForFunction(()=>window.report!==undefined,{},{timeout:300000});
  const report=await page.evaluate(()=>window.report);console.log(JSON.stringify(report,null,2));
  await writeFile('reports/gpu.json',JSON.stringify(report,null,2));if(report.failed)process.exitCode=1;
} finally {await browser?.close();await new Promise(r=>server.close(r));}
