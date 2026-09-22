import path from 'node:path';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {e4,half} from '../src/model.js';
if(!process.env.NR_NATIVE_SOURCE)throw Error('Set NR_NATIVE_SOURCE to the src folder of a separate OpenDLSS-NR checkout. Reference sources are not distributed here.');
await mkdir('build/native',{recursive:true});
await writeFile('build/native/native-numeric.h',await readFile(path.join(process.env.NR_NATIVE_SOURCE,'numeric.h')));
let seed=424242;const rand=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed;};
const f8=n=>Float32Array.from({length:n},()=>e4((rand()%64)+((rand()&1)*128)));
const f16=n=>Float32Array.from({length:n},()=>half((rand()%4096)+8192+((rand()&1)*32768)));
const cases=[];
function add(name,entry,scalars,count,buffers,outputs){cases.push({name,entry,scalars,count,buffers,outputs});}
for(const halfMode of [0,1])for(const partition of [0,32]) {
  const rows=7,K=64,N=32,batches=2;
  add(`gemm-${halfMode}-${partition}`,'nr_gemm',{rows,K,N,batches,inputStride:K*batches,inputBatchStride:K,partition,halfMode,silu:0,hasResidual:1,quantize:1},rows*N*batches,
    {input:(halfMode?f16:f8)(rows*K*batches),weights:(halfMode?f16:f8)(K*N*batches),residual:f16(rows*N*batches),scales:f16(N*batches),raw:new Float32Array(rows*N*batches),output:new Float32Array(rows*N*batches)},['raw','output']);
}
const patterns=Float32Array.from({length:65536},(_,i)=>half(i));
add('numeric-all-half','nr_numeric',{count:65536},65536,{input:patterns,output:new Float32Array(65536*3),codes:new Uint32Array(65536)},['output','codes']);
add('numeric-random-f32','nr_numeric',{count:8192},8192,{input:Float32Array.from({length:8192},()=>((rand()/2**32)-0.5)*150000),output:new Float32Array(8192*3),codes:new Uint32Array(8192)},['output','codes']);
for(const globalMode of [0,1]) {
  add(`normalize-${globalMode}`,'nr_normalize',{rows:13,heads:2,globalMode},26,{qkv:f16(13*192),scales:new Float32Array([2.125,3.3321]),output:new Float32Array(13*192)},['output']);
  for(const shift of [0,4]) {
    const width=9,height=7,heads=2,padded=64,rows=width*height,keys=64;
    add(`scores-${globalMode}-${shift}`,'nr_scores',{width,height,heads,shiftX:shift,shiftY:shift,padded,globalMode},rows*heads*keys,{qkv:f8(rows*heads*96),prior:f16(heads*4096),scores:new Float32Array(rows*heads*keys)},['scores']);
    add(`attend-${globalMode}-${shift}`,'nr_attend',{width,height,heads,shiftX:shift,shiftY:shift,keys,globalMode},rows*heads*32,{qkv:f8(rows*heads*96),weights:f8(rows*heads*keys),inverse:new Float32Array(rows*heads).fill(0.125),output:new Float32Array(rows*heads*32)},['output']);
  }
  add(`softmax-${globalMode}`,'nr_softmax',{rows:63,heads:2,keys:128,globalMode},126,{scores:Float32Array.from({length:126*128},()=>half(8192+rand()%4096)),weights:new Float32Array(126*128),inverse:new Float32Array(126)},['weights','inverse']);
}
add('pool-padding','nr_pool',{iw:6,ih:6,ow:4,oh:4,channels:32},512,{input:f16(6*6*32),output:new Float32Array(512)},['output']);
for(const post of [0,1])add(`merge-${post}`,'nr_merge',{iw:4,ow:8,oh:6,channels:32,post},1536,{low:f8(4*3*32),skip:f8(1536),scaleA:f16(32),scaleB:f16(32),raw:new Float32Array(1536),output:new Float32Array(1536)},['raw','output']);
add('compose','nr_compose',{width:9,height:7,fullWidth:12,blendScale:0.75,useHistory:1},63,{proxy:Float32Array.from({length:63*4},()=>rand()/2**32),head:f16(12*7*4),history:Float32Array.from({length:63*4},()=>rand()/2**32),output:new Float32Array(63*4)},['output']);
add('preprocess','nr_preprocess',{width:35,height:33,fullWidth:40,fullHeight:40,seed:1234,autoMask:1,localTone:0.25,localStructure:0.4,skinStructure:-1,style:3,useHistory:1},1600,{proxy:Float32Array.from({length:35*33*4},()=>rand()/2**32),history:Float32Array.from({length:35*33*4},()=>rand()/2**32),features:new Float32Array(1600*16)},['features']);
add('reproject','nr_reproject',{width:9,height:7},63,{history:Float32Array.from({length:63*4},()=>rand()/2**32),proxy:Float32Array.from({length:63*4},()=>rand()/2**32),motion:Float32Array.from({length:63*4},(_,i)=>i%4===2?(i%3?1:0):((rand()/2**32)-0.5)*0.5),output:new Float32Array(63*4)},['output']);
let cpp=`#include <cuda_runtime.h>\n#include <cstdio>\n#include <cstdlib>\n#include <vector>\n#include <fstream>\n#include <string>\n#include <algorithm>\n#include "native-numeric.h"\n`;
const ref=await readFile(path.join(process.env.NR_NATIVE_SOURCE,'reference.cpp'),'utf8');
cpp+='namespace original { using num::roundF16;\n'+ref.slice(ref.indexOf('int normalExponent'),ref.indexOf('uint32_t inverseTiledToken'))+ref.slice(ref.indexOf('float adaFp8Fdpa16'),ref.indexOf('float gemmFp8Element'))+ref.slice(ref.indexOf('float mpCubicSilu'),ref.indexOf('std::vector<uint16_t> siluTable'))+'}\n';
cpp+=(await readFile('kernels/numeric.cuh','utf8')).replace(/^#pragma once\s*/m,'');
cpp+=(await readFile('kernels/packed.cuh','utf8')).replace(/^#pragma once\s*/m,'');
for(const f of ['ops','gemm','attention','frame'])cpp+='\n'+(await readFile(`kernels/${f}.cu`,'utf8')).replace(/^#include "numeric.cuh"\s*/m,'').replace(/^#include "packed.cuh"\s*/m,'');
cpp+='\nvoid ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\\n",cudaGetErrorString(e));exit(1);}}\nint main(){\n';
for(const c of cases) {
  cpp+='{\n';const args=[];
  const bufferInfo={};
  for(const [name,data] of Object.entries(c.buffers)) {
    const file=`${c.name}-${name}.bin`,type=data instanceof Uint32Array?'unsigned':'float';
    await writeFile('build/native/'+file,new Uint8Array(data.buffer));
    bufferInfo[name]={file,type,bytes:data.byteLength};
    cpp+=`std::vector<${type}> h_${name}(${data.length});${type}* ${name};ck(cudaMalloc(&${name},${data.byteLength}));{std::ifstream f("build/native/${file}",std::ios::binary);f.read((char*)h_${name}.data(),${data.byteLength});ck(cudaMemcpy(${name},h_${name}.data(),${data.byteLength},cudaMemcpyHostToDevice));}\n`;
  }
  // Follow source parameter order, never JSON object order assumptions.
  const family=['ops','gemm','attention','frame'];let params;
  for(const f of family){const src=await readFile(`kernels/${f}.cu`,'utf8');const m=src.match(new RegExp('__global__ void '+c.entry+'\\(([^)]*)\\)'));if(m)params=m[1].split(',').map(p=>p.trim().split(/\s+/).at(-1));}
  for(const name of params)args.push(name in bufferInfo?name:Number.isInteger(c.scalars[name])?String(c.scalars[name]):String(c.scalars[name])+'f');
  cpp+=`${c.entry}<<<${Math.ceil(c.count/64)},64>>>(${args.join(',')});ck(cudaGetLastError());ck(cudaDeviceSynchronize());\n`;
  if(c.entry==='nr_gemm') {
    const {K,N,batches,rows,halfMode,partition}=c.scalars,group=halfMode?8:16;
    cpp+=`ck(cudaMemcpy(h_raw.data(),raw,h_raw.size()*4,cudaMemcpyDeviceToHost));
      for(unsigned row=0;row<${rows};row++)for(unsigned batch=0;batch<${batches};batch++)for(unsigned col=0;col<${N};col++){
        unsigned i=(row*${batches}+batch)*${N}+col;
        float acc=num::roundF16(h_residual[i]*h_scales[batch*${N}+col]),total=0;
        for(unsigned kb=0;kb<${K};kb+=${group}){
          float a[${group}],b[${group}];for(unsigned j=0;j<${group};j++){a[j]=h_input[(row*${batches}+batch)*${K}+kb+j];b[j]=h_weights[(batch*${K}+kb+j)*${N}+col];}
          acc=original::${halfMode?'adaF16Fdpa8':'adaFp8Fdpa16'}(a,b,${group},acc);
          ${partition?`if((kb+${group})%${partition}==0){total=kb<${partition}?acc:num::roundF16(total+acc);acc=0;}`:''}
        }
        ${partition?'acc=total;':''}
        if(num::f32Bits(acc)!=num::f32Bits(h_raw[i])){fprintf(stderr,"Original CPU reference mismatch: ${c.name} at %u\\n",i);return 2;}
      }\n`;
  }
  if(c.entry==='nr_numeric')cpp+=`ck(cudaMemcpy(h_output.data(),output,h_output.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(h_codes.data(),codes,h_codes.size()*4,cudaMemcpyDeviceToHost));
    for(unsigned i=0;i<h_input.size();i++){
      float expected=original::mpCubicSilu(num::roundF16(h_input[i]));
      if(!std::isnan(expected)&&num::f32Bits(expected)!=num::f32Bits(h_output[i*3+2])){fprintf(stderr,"Original SiLU reference mismatch at %u\\n",i);return 3;}
      if(num::e4m3FromF32(h_input[i])!=h_codes[i]){fprintf(stderr,"Original E4 reference mismatch at %u\\n",i);return 4;}
    }\n`;
  for(const name of c.outputs)cpp+=`{std::vector<char> v(${bufferInfo[name].bytes});ck(cudaMemcpy(v.data(),${name},v.size(),cudaMemcpyDeviceToHost));std::ofstream f("build/native/${c.name}-${name}-expected.bin",std::ios::binary);f.write(v.data(),v.size());}\n`;
  for(const name of Object.keys(bufferInfo))cpp+=`ck(cudaFree(${name}));\n`;
  cpp+='}\n';c.buffers=bufferInfo;
}
cpp+='puts("Native CUDA fixtures complete");}\n';
await writeFile('build/native/runner.cu',cpp);
await writeFile('build/native/cases.json',JSON.stringify(cases,null,2));
console.log(`Prepared ${cases.length} native CUDA cases`);
