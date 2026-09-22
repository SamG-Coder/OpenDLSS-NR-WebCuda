import {spawn} from 'node:child_process';
import {readFile,writeFile} from 'node:fs/promises';
const runs=Number(process.env.NR_RUNS||10);
if(!Number.isInteger(runs)||runs<2||runs>100)throw Error('NR_RUNS must be 2 through 100');
function run(exe){return new Promise((resolve,reject)=>{let output='';const p=spawn(exe,[String(runs)],{stdio:['ignore','pipe','pipe']});p.stdout.on('data',s=>output+=s);p.stderr.on('data',s=>output+=s);p.on('error',reject);p.on('exit',code=>code?reject(Error(output)):resolve(output));});}
const result={runs,cases:[]};
for(const [name,dir] of [['baseline','build/native-benchmark'],['integerDot','build/native-intdot']]){
  const output=await run(dir+'/runner.exe');await writeFile(`reports/intdot-${name}-timings.txt`,output);
  for(const width of [1280,1920]){
    const samples=[...output.matchAll(new RegExp(`${width}x\\d+ frame \\d+: ([\\d.]+) ms`,'g'))].map(m=>Number(m[1])).sort((a,b)=>a-b);
    if(samples.length!==runs)throw Error('Incomplete timing result');
    const median=(samples[Math.floor((runs-1)/2)]+samples[Math.floor(runs/2)])/2;
    let item=result.cases.find(c=>c.width===width);if(!item){item={width};result.cases.push(item);}item[name]={medianMs:median,minMs:samples[0],samples};
    console.log(`${name} ${width}: ${median.toFixed(3)} ms median`);
  }
}
for(const c of result.cases){const a=await readFile(`build/native-benchmark/head-${c.width}.bin`),b=await readFile(`build/native-intdot/head-${c.width}.bin`);c.exact=a.equals(b);c.headBytes=a.length;c.changePercent=100*(c.integerDot.medianMs/c.baseline.medianMs-1);}
await writeFile('reports/intdot-comparison.json',JSON.stringify(result,null,2));
console.log(JSON.stringify(result.cases.map(({width,exact,changePercent})=>({width,exact,changePercent})),null,2));
if(result.cases.some(c=>!c.exact))throw Error('Native head mismatch');
