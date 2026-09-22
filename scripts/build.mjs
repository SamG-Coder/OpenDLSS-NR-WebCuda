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
  const fastHalf=(await readFile('kernels/fast-half.cuh','utf8')).replace(/^#pragma once\s*/m,'');
  const vectorF13=(await readFile('kernels/vector-f13.cuh','utf8')).replace(/^#pragma once\s*/m,'');
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
  {
    const source=(await readFile('kernels/attention-normalized.cu','utf8')).replace(/^#include "fast-half.cuh"\s*/m,fastHalf+'\n').replace(/^#include "vector-f13.cuh"\s*/m,vectorF13+'\n').replace(/^#include "attention.cu"\s*/m,await readFile('generated/attention.cu','utf8')).replace(/^#include "activations.cuh"\s*/m,activationHeader+'\n');
    const entry='nr_local_attention_normalized',artifact=compile(source.replace(/^#include <cuda_fp16.h>\s*/m,''),{entry,workgroupSize:[512,1,1]});
    await writeFile('generated/'+entry+'.cu',source);await writeFile('generated/'+entry+'.json',JSON.stringify(serializableArtifact(artifact)));manifest[entry]=entry+'.json';
  }
  {
    const source=header+'\n'+(await readFile('kernels/lookup.cu','utf8')).replace(/^#include "numeric.cuh"\s*/m,'');
    const entry='nr_lookup_tables',artifact=compile(source,{entry,workgroupSize:[64,1,1]});
    await writeFile('generated/'+entry+'.cu',source);await writeFile('generated/'+entry+'.json',JSON.stringify(serializableArtifact(artifact)));manifest[entry]=entry+'.json';
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
  const wideTemplate=await readFile('kernels/gemm-wide.cu','utf8');
  const precomputedTemplate=await readFile('kernels/gemm-precomputed.cu','utf8');
  const preparedTemplate=await readFile('kernels/gemm-prepared.cu','utf8');
  const halfTemplate=await readFile('kernels/gemm-half.cu','utf8');
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
    if(scalars.inputFormat===1){
      const baseEntry=entry.replace('_compact',''),cols=baseEntry.endsWith('8x16')?16:8;
      const halfCuda=header+'\n'+halfTemplate.replace(/^#include "numeric.cuh"\s*/m,'').replace(/^#include "packed.cuh"\s*/m,packed+'\n').replace('#define NR_TILE_ROWS 4','#define NR_TILE_ROWS 8').replace('#define NR_TILE_COLS 16','#define NR_TILE_COLS '+cols).replace('#define NR_TILE_ENTRY nr_gemm_tiled','#define NR_TILE_ENTRY '+baseEntry);
      const halfName=name+'_half',halfSource=specializeGemmSource(compactKernel(halfCuda,baseEntry,activationHeader),entry,halfName,scalars);
      const halfArtifact=compile(halfSource.replace(/^#include <cuda_fp16.h>\s*/m,''),{entry:halfName,workgroupSize:base.metadata.workgroupSize});
      await writeFile('generated/'+halfName+'.cu',halfSource);await writeFile('generated/'+halfName+'.json',JSON.stringify(serializableArtifact(halfArtifact)));manifest[halfName]=halfName+'.json';
      if([scalars.K,scalars.N,scalars.inputStride,scalars.inputBatchStride].every(n=>n%4===0)&&[1,2].includes(scalars.outputFormat)&&(!scalars.rawEnabled||scalars.rawFormat===2)){
        const wideCuda=header+'\n'+wideTemplate.replace(/^#include "fast-half.cuh"\s*/m,fastHalf+'\n').replace(/^#include "vector-f13.cuh"\s*/m,vectorF13+'\n').replace(/^#include "numeric.cuh"\s*/m,'').replace(/^#include "packed.cuh"\s*/m,packed+'\n').replace(/^#include "activations.cuh"\s*/m,activationHeader+'\n');
        for(const [tileRows,tileCols] of [[32,32],[64,32],[32,64]]){
          const suffix=tileRows===32&&tileCols===32?'_wide_half':`_wide${tileRows}x${tileCols}_half`;
          const defines={ROWS:tileRows,COLS:tileCols,THREADS:tileRows*tileCols/8,COLUMN_GROUPS:tileCols/4,A_PAIRS:tileRows*16,B_PAIRS:tileCols*17,INPUT_WORDS:tileRows*8,WEIGHT_PAIRS:tileCols*4};
          let tileCuda=wideCuda;for(const [key,value] of Object.entries(defines))tileCuda=tileCuda.replace(new RegExp('#define NR_WIDE_'+key+' \\d+'),'#define NR_WIDE_'+key+' '+value);
          const wideName=name+suffix,wideSource=specializeGemmSource(tileCuda,'nr_gemm_wide',wideName,scalars);
          const wideArtifact=compile(wideSource.replace(/^#include <cuda_fp16.h>\s*/m,''),{entry:wideName,workgroupSize:[defines.THREADS,1,1]});
          await writeFile('generated/'+wideName+'.cu',wideSource);await writeFile('generated/'+wideName+'.json',JSON.stringify(serializableArtifact(wideArtifact)));manifest[wideName]=wideName+'.json';
        }
        if(scalars.K%32===0&&scalars.N%32===0){
          // Precompute both weight operands and exponents once per model. Keep
          // the established wide-kernel arithmetic and the same tile choices.
          const precomputedCuda=header+'\n'+precomputedTemplate.replace(/^#include "fast-half.cuh"\s*/m,fastHalf+'\n').replace(/^#include "vector-f13.cuh"\s*/m,vectorF13+'\n').replace(/^#include "numeric.cuh"\s*/m,'').replace(/^#include "packed.cuh"\s*/m,packed+'\n').replace(/^#include "activations.cuh"\s*/m,activationHeader+'\n');
          for(const [tileRows,tileCols] of [[32,32],[64,32],[32,64]]){
            const suffix=tileRows===32&&tileCols===32?'_precomputed_half':`_precomputed${tileRows}x${tileCols}_half`;
            const defines={ROWS:tileRows,COLS:tileCols,THREADS:tileRows*tileCols/8,COLUMN_GROUPS:tileCols/4,A_PAIRS:tileRows*16,B_PAIRS:tileCols*17,INPUT_WORDS:tileRows*8,WEIGHT_PAIRS:tileCols*16};
            let tileCuda=precomputedCuda;for(const [key,value] of Object.entries(defines))tileCuda=tileCuda.replace(new RegExp('#define NR_WIDE_'+key+' \\d+'),'#define NR_WIDE_'+key+' '+value);
            const precomputedName=name+suffix,precomputedSource=specializeGemmSource(tileCuda,'nr_gemm_precomputed',precomputedName,scalars);
            const precomputedArtifact=compile(precomputedSource.replace(/^#include <cuda_fp16.h>\s*/m,''),{entry:precomputedName,workgroupSize:[defines.THREADS,1,1]});
            await writeFile('generated/'+precomputedName+'.cu',precomputedSource);await writeFile('generated/'+precomputedName+'.json',JSON.stringify(serializableArtifact(precomputedArtifact)));manifest[precomputedName]=precomputedName+'.json';
          }
          // Older compact experiments retain original FP8 codes and append
          // group exponent metadata. Their tile remains fixed at 32x32.
          const preparedCuda=header+'\n'+preparedTemplate.replace(/^#include "fast-half.cuh"\s*/m,fastHalf+'\n').replace(/^#include "vector-f13.cuh"\s*/m,vectorF13+'\n').replace(/^#include "numeric.cuh"\s*/m,'').replace(/^#include "packed.cuh"\s*/m,packed+'\n').replace(/^#include "activations.cuh"\s*/m,activationHeader+'\n');
          for(const integer of [0,1]){
            const preparedName=name+(integer?'_prepared_integer':'_prepared_half');
            const preparedSource=specializeGemmSource(preparedCuda.replace('#define NR_PREPARED_INTEGER 0','#define NR_PREPARED_INTEGER '+integer),'nr_gemm_prepared',preparedName,scalars);
            const preparedArtifact=compile(preparedSource.replace(/^#include <cuda_fp16.h>\s*/m,''),{entry:preparedName,workgroupSize:[128,1,1]});
            await writeFile('generated/'+preparedName+'.cu',preparedSource);await writeFile('generated/'+preparedName+'.json',JSON.stringify(serializableArtifact(preparedArtifact)));manifest[preparedName]=preparedName+'.json';
          }
        }
      }
    }
  }
  await writeFile('generated/manifest.json',JSON.stringify(manifest,null,2));
  return manifest;
}
if (process.argv[1]?.endsWith('build.mjs')) await build();
