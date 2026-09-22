import {createGraph} from '../src/graph.js';
import {gemmKernel} from '../src/kernel-selection.js';
import {specializedGemm} from '../src/gemm-specialization.js';
import {specializeGemmSource} from './specialize-gemm.mjs';
import {compactKernel} from './activation-variants.mjs';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {compile,serializableArtifact} from '../vendor/webcuda/compiler/compiler.js';
export const families = ['ops','gemm','attention','frame'];
export async function build() {
  await mkdir('generated',{recursive:true});
  const header = (await readFile('kernels/numeric.cuh','utf8')).replace(/^#pragma once\s*/m,'');
  const packed = (await readFile('kernels/packed.cuh','utf8')).replace(/^#pragma once\s*/m,'');
  const activationHeader=(await readFile('kernels/activations.cuh','utf8')).replace(/^#pragma once\s*/m,'');
  const manifest = {};
  async function compact(source,entry,workgroupSize){
    const cuda=compactKernel(source,entry,activationHeader);if(!cuda)return;
    const name=entry+'_compact',artifact=compile(cuda,{entry:name,workgroupSize});
    await writeFile('generated/'+name+'.cu',cuda);
    await writeFile('generated/'+name+'.json',JSON.stringify(serializableArtifact(artifact)));
    manifest[name]=name+'.json';
  }
  for (const family of families) {
    const source = header + '\n' + (await readFile(`kernels/${family}.cu`,'utf8')).replace(/^#include "numeric.cuh"\s*/m,'').replace(/^#include "packed.cuh"\s*/m,packed+'\n');
    await writeFile(`generated/${family}.cu`,source);
    for (const match of source.matchAll(/__global__ void (\w+)/g)) {
      const entry = match[1];
      const workgroupSize=[entry==='nr_local_attention'?128:64,1,1];
      const artifact = compile(source,{entry,workgroupSize});
      await writeFile(`generated/${entry}.json`,JSON.stringify(serializableArtifact(artifact)));
      await writeFile(`generated/${entry}.wgsl`,artifact.wgsl);
      manifest[entry] = `${entry}.json`;
      await compact(source,entry,workgroupSize);
      console.log(`${entry}: ${artifact.wgsl.length} bytes WGSL`);
    }
  }
  const template=await readFile('kernels/gemm-tiled.cu','utf8');
  for(const [entry,rows,cols] of [['nr_gemm_tiled',4,16],['nr_gemm_tile8x8',8,8],['nr_gemm_tile8x16',8,16]]) {
    const source=header+'\n'+template.replace(/^#include "numeric.cuh"\s*/m,'').replace(/^#include "packed.cuh"\s*/m,packed+'\n').replace('#define NR_TILE_ROWS 4','#define NR_TILE_ROWS '+rows).replace('#define NR_TILE_COLS 16','#define NR_TILE_COLS '+cols).replace('#define NR_TILE_ENTRY nr_gemm_tiled','#define NR_TILE_ENTRY '+entry);
    const artifact=compile(source,{entry,workgroupSize:[rows*cols,1,1]});
    await writeFile('generated/'+entry+'.cu',source);
    await writeFile('generated/'+entry+'.json',JSON.stringify(serializableArtifact(artifact)));
    await writeFile('generated/'+entry+'.wgsl',artifact.wgsl);
    await compact(source,entry,[rows*cols,1,1]);
    manifest[entry]=entry+'.json';console.log(entry+': '+artifact.wgsl.length+' bytes WGSL');
  }
  const multiTemplate=await readFile('kernels/gemm-multi.cu','utf8');
  for(const [rows,cols] of [[8,32],[16,16],[16,32],[4,32],[32,32],[16,64]]) {
    const entry=`nr_gemm_multi${rows}x${cols}`,workgroupSize=[rows*cols/4,1,1];
    const source=header+'\n'+multiTemplate.replace(/^#include "numeric.cuh"\s*/m,'').replace(/^#include "packed.cuh"\s*/m,packed+'\n').replace('#define NR_MULTI_ROWS 8','#define NR_MULTI_ROWS '+rows).replace('#define NR_MULTI_COLS 32','#define NR_MULTI_COLS '+cols).replace('#define NR_MULTI_ENTRY nr_gemm_multi8x32','#define NR_MULTI_ENTRY '+entry);
    const artifact=compile(source,{entry,workgroupSize});
    await writeFile('generated/'+entry+'.cu',source);
    await writeFile('generated/'+entry+'.json',JSON.stringify(serializableArtifact(artifact)));
    manifest[entry]=entry+'.json';await compact(source,entry,workgroupSize);
  }
  const graph=createGraph(1280,720,{activationStorage:'packed',fuseLocalAttention:true});
  for(const op of graph.ops)if(op.entry==='nr_gemm'&&!op.scalars.halfMode){
    const scalars={...op.scalars,...Object.fromEntries(['input','residual','raw','output'].map(key=>[key+'Format',typeof op.bindings[key]==='string'?graph.resources.get(op.bindings[key]).format:0]))};
    const entry=gemmKernel(scalars)+'_compact',name=specializedGemm(entry,scalars);
    if(!name||manifest[name])continue;
    const base=JSON.parse(await readFile('generated/'+entry+'.json','utf8'));
    const source=specializeGemmSource(await readFile('generated/'+entry+'.cu','utf8'),entry,name,scalars);
    const artifact=compile(source,{entry:name,workgroupSize:base.metadata.workgroupSize});
    await writeFile('generated/'+name+'.cu',source);
    await writeFile('generated/'+name+'.json',JSON.stringify(serializableArtifact(artifact)));manifest[name]=name+'.json';
  }
  await writeFile('generated/manifest.json',JSON.stringify(manifest,null,2));
  return manifest;
}
if (process.argv[1]?.endsWith('build.mjs')) await build();
