import {spawn} from 'node:child_process';
import {readFile} from 'node:fs/promises';
const out=process.env.NR_NATIVE_BUILD||'build/native-benchmark',nvcc=process.env.NVCC||'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe';
const manifest=JSON.parse(await readFile(out+'/manifest.json','utf8'));
const files=manifest.kernels.map((_,i)=>`kernel${i}.cu`);files.push('runner.cu');
const flags=['-O2','--fmad=false','-std=c++17','-arch='+ (process.env.NR_CUDA_ARCH||'native'),'-allow-unsupported-compiler'];
function run(args){return new Promise((resolve,reject)=>{const p=spawn(nvcc,args,{stdio:'inherit'});p.on('error',reject);p.on('exit',n=>n?reject(Error('NVCC failed '+n)):resolve());});}
let next=0,done=0;
if(!process.argv.includes('--link-only'))await Promise.all(Array.from({length:4},async()=>{while(next<files.length){const f=files[next++];await run([...flags,'-c',out+'/'+f,'-o',out+'/'+f+'.obj']);console.log(`Compiled ${++done}/${files.length}: ${f}`);}}));
await run([...flags,...files.map(f=>out+'/'+f+'.obj'),'-o',out+'/runner.exe']);
console.log('Native benchmark executable ready.');
