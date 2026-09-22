// Replay the production renderer's dispatches using its generated CUDA sources.
// Model bytes and generated runner stay in ignored build/; no upstream kernels.
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {Model} from '../src/model.js';
import {NeuralRenderer} from '../src/engine.js';
import {geometry} from '../src/geometry.js';
import {integerDotVariant} from './integer-dot-variant.mjs';
const out=process.env.NR_NATIVE_BUILD||'build/native-benchmark';await mkdir(out,{recursive:true});
const modelDir=process.env.NR_MODEL||'models/nr';
const model=await Model.load(p=>readFile(modelDir+'/'+p));
const manifest=JSON.parse(await readFile('generated/manifest.json','utf8'));
const artifacts=Object.fromEntries(await Promise.all(Object.entries(manifest).map(async([k,f])=>[k,JSON.parse(await readFile('generated/'+f,'utf8'))])));
const used=new Map(),cases=[];
for(const [width,height] of [[1280,720],[1920,1080]]){
  const buffers=[],commands=[],reads=[],pending=[];
  const runtime={device:{limits:{maxStorageBufferBindingSize:2**31,maxComputeWorkgroupsPerDimension:65535,maxComputeWorkgroupStorageSize:49152,maxComputeInvocationsPerWorkgroup:1024,maxComputeWorkgroupSizeX:1024}},
    createBuffer(data){const b={id:buffers.length,size:typeof data==='number'?data:data.byteLength};buffers.push(b);if(typeof data!=='number')this.write(b,data);return b;},
    write(b,data){b.file=`${width}-${b.id}.bin`;pending.push(writeFile(out+'/'+b.file,new Uint8Array(data.buffer,data.byteOffset,data.byteLength)));},
    destroyBuffer(){},idle:async()=>{},
    read:async(b,type=Float32Array,size=b.size)=>{reads.push({id:b.id,size});return new type(size/type.BYTES_PER_ELEMENT);},
    batch(){const list=[];return {dispatch(inv,grid){list.push({...inv,grid});return this;},submit(){commands.push(...list);},discard(){}};}
  };
  const kernels=Object.fromEntries(Object.keys(manifest).map(entry=>[entry,{bind:(bindings,scalars={})=>({entry,bindings,scalars})}]));
  const renderer=new NeuralRenderer(runtime,kernels,model);
  const g=geometry(width,height),inputFeatures=new Float32Array(g.fullWidth*g.fullHeight*16);
  for(let i=0;i<inputFeatures.length;i++)inputFeatures[i]=Math.sin(i*0.0017)*0.125;
  await renderer.run({width,height,inputFeatures});await Promise.all(pending);
  for(const c of commands)if(!used.has(c.entry))used.set(c.entry,used.size);
  cases.push({width,height,buffers,commands,head:reads.at(-1),geometry:g,featureId:renderer.plan.resources.get(renderer.graphCache.graph.features).id,featureBytes:inputFeatures.byteLength});
  console.log(`${width}x${height}: ${commands.length} dispatches, ${buffers.length} buffers`);
}
const header='#include <cstdio>\n#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <cmath>\n';
const types={f32:'float',u32:'unsigned',i32:'int'};
let cpp=header+'#include <cstdio>\n#include <cstdlib>\n#include <vector>\n#include <fstream>\n#include <algorithm>\n#include <string>\nvoid ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\\n",cudaGetErrorString(e));exit(1);}}\n';
for(const [entry,id] of used){
  const metadata=artifacts[entry].metadata;
  let source;try{source=await readFile('generated/'+entry+'.cu','utf8');}catch{
    for(const family of ['ops','gemm','attention','frame']){const s=await readFile('generated/'+family+'.cu','utf8');if(s.includes('__global__ void '+entry+'(')){source=s;break;}}
  }
  if(!source)throw Error('No CUDA source for '+entry);
  if(process.env.NR_INTEGER_DOT==='1')source=integerDotVariant(source,{audit:process.env.NR_DOT_AUDIT==='1'});
  const args=[...metadata.bindings.map((b,i)=>`(${types[b.elementType]}*)p[${i}]`),...metadata.scalars.map((s,i)=>`(${types[s.type]})s[${i}]`)];
  const macros=[...source.matchAll(/^#define\s+(\w+)/gm)].map(m=>m[1]);
  let unit=header+`namespace kernel${id}{\n`+source.replace(/^#include <cuda_fp16.h>\s*/gm,'')+`\n}\nextern "C" void launch${id}(void**p,double*s,dim3 grid){kernel${id}::${entry}<<<grid,dim3(${metadata.workgroupSize.join(',')})>>>(${args.join(',')});}\n`+macros.map(m=>'#undef '+m).join('\n');
  if(source.includes('nr_dot_stats')){
    unit+=`\nextern "C" void audit${id}(){unsigned long long counts[4];cudaMemcpyFromSymbol(counts,kernel${id}::nr_dot_stats,sizeof(counts));printf("DOT_AUDIT ${id} %llu %llu %llu %llu\\n",counts[0],counts[1],counts[2],counts[3]);}\n`;
    cpp+=`extern "C" void audit${id}();\n`;
  }
  await writeFile(`${out}/kernel${id}.cu`,unit);
  cpp+=`extern "C" void launch${id}(void**,double*,dim3);\n`;
}
cpp+='int main(int argc,char**argv){int runs=argc>1?atoi(argv[1]):10;cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));printf("Device: %s\\n",prop.name);\n';
for(const c of cases){
  cpp+=`{printf("Loading ${c.width}x${c.height}\\n");fflush(stdout);std::vector<void*> b(${c.buffers.length});\n`;
  for(const b of c.buffers){cpp+=`ck(cudaMalloc(&b[${b.id}],${b.size}));ck(cudaMemset(b[${b.id}],0,${b.size}));\n`;if(b.file)cpp+=`{std::ifstream f("${out}/${b.file}",std::ios::binary|std::ios::ate);if(!f)return 2;std::vector<char> v((size_t)f.tellg());f.seekg(0);if(v.size()>${b.size}||!f.read(v.data(),v.size()))return 2;ck(cudaMemcpy(b[${b.id}],v.data(),v.size(),cudaMemcpyHostToDevice));}\n`;}
  const calls=c.commands.map(cmd=>{const m=artifacts[cmd.entry].metadata;return `{void*p[]={${m.bindings.map(x=>'b['+cmd.bindings[x.name].id+']').join(',')}};double s[]={${m.scalars.length?m.scalars.map(x=>{if(!Number.isFinite(cmd.scalars[x.name]))throw Error(cmd.entry+' missing '+x.name);return cmd.scalars[x.name];}).join(','):'0'}};launch${used.get(cmd.entry)}(p,s,dim3(${cmd.grid.join(',')}));ck(cudaGetLastError());}`;});
  const setup=c.commands.map((x,i)=>x.entry==='nr_lookup_tables'?calls[i]:'').join('\n');
  const frame=c.commands.map((x,i)=>x.entry!=='nr_lookup_tables'?calls[i]:'').join('\n');
  cpp+=`void* fixedFeatures;ck(cudaMalloc(&fixedFeatures,${c.featureBytes}));ck(cudaMemcpy(fixedFeatures,b[${c.featureId}],${c.featureBytes},cudaMemcpyDeviceToDevice));auto resetInput=[&](){ck(cudaMemcpyAsync(b[${c.featureId}],fixedFeatures,${c.featureBytes},cudaMemcpyDeviceToDevice));};\n`;
  cpp+=setup+'\nck(cudaDeviceSynchronize());auto frame=[&](){\n'+frame+'\n};\n';
  cpp+=`for(int i=0;i<3;i++){resetInput();frame();ck(cudaDeviceSynchronize());}cudaEvent_t start,end;ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));std::vector<float> times;for(int i=0;i<runs;i++){resetInput();ck(cudaEventRecord(start));frame();ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,start,end));times.push_back(ms);printf("${c.width}x${c.height} frame %d: %.3f ms\\n",i,ms);fflush(stdout);}std::sort(times.begin(),times.end());printf("RESULT ${c.width}x${c.height} median %.3f min %.3f ms (%d samples)\\n",times[times.size()/2],times.front(),runs);\n`;
  cpp+=`{std::vector<char> v(${c.head.size});ck(cudaMemcpy(v.data(),b[${c.head.id}],v.size(),cudaMemcpyDeviceToHost));std::ofstream f("${out}/head-${c.width}.bin",std::ios::binary);f.write(v.data(),v.size());}ck(cudaFree(fixedFeatures));for(void*p:b)ck(cudaFree(p));ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));}\n`;
}
if(process.env.NR_DOT_AUDIT==='1')for(const [entry,id] of used)if(entry.includes('_wide'))cpp+=`audit${id}();\n`;
cpp+='}\n';await writeFile(out+'/runner.cu',cpp);
await writeFile(out+'/manifest.json',JSON.stringify({kernels:[...used.keys()],cases:cases.map(({width,height,geometry,head,commands})=>({width,height,geometry,head,dispatches:commands.length}))},null,2));
console.log(`Generated ${used.size} native CUDA translation units in ${out}`);
