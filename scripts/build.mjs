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
  await writeFile('generated/manifest.json',JSON.stringify(manifest,null,2));
  return manifest;
}
if (process.argv[1]?.endsWith('build.mjs')) await build();
