// Read-only full-model audit. Writes aggregate statistics, never weight values.
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {Model} from '../src/model.js';
import {createGraph} from '../src/graph.js';
const model=await Model.load(p=>readFile((process.env.NR_MODEL||'models/nr')+'/'+p));
const graph=createGraph(1280,720,{activationStorage:'packed',fuseLocalAttention:true,fuseNormalization:true});
model.validateGraph(graph);
const ints=new Int32Array(256),exps=new Int32Array(256),tz=new Int32Array(256);
for(let c=0;c<256;c++){
  const e=(c>>>3)&15,m=c&7,v=(c&127)===127?0:e?(m+8)*2**(e-1):m;
  ints[c]=v;exps[c]=v?Math.max(-6,Math.floor(Math.log2(v))-9):-100;
  tz[c]=v?31-Math.clz32((v&-v)>>>0):30;
}
const totals={weights:0,zeros:0,groups:0,zeroGroups:0,int8Groups:0,int8Macs:0,macs:0,bytes:0};
const matrices=[],hist=new Array(256).fill(0),bitsHist=new Array(20).fill(0),macBits=new Array(20).fill(0),digests=new Set();let duplicates=0;
for(const op of graph.ops){
  if(op.entry!=='nr_gemm')continue;
  const spec=op.bindings.weights,{K,N,batches=1,halfMode}=spec;
  if(halfMode)continue;
  const packed=model.packedMatrix(spec.name,spec.offset,K,N,spec),codes=new Uint8Array(packed.buffer),rows=op.scalars.rows;
  const hash=createHash('sha256').update(codes).digest('hex');if(digests.has(hash))duplicates++;digests.add(hash);
  let zeros=0,eligible=0,groups=0,allzero=0;
  for(const c of codes){hist[c]++;if(!ints[c])zeros++;}
  for(let b=0;b<batches;b++)for(let k=0;k<K;k+=16)for(let n=0;n<N;n++){
    let minTZ=30,maxV=0;
    for(let j=0;j<16&&k+j<K;j++){const c=codes[(b*K+k+j)*N+n];minTZ=Math.min(minTZ,tz[c]);maxV=Math.max(maxV,ints[c]);}
    const maxCoeff=maxV?maxV/2**minTZ:0,bits=maxCoeff?Math.floor(Math.log2(maxCoeff))+1:0;
    groups++;bitsHist[bits]++;macBits[bits]+=rows*16;if(!maxV)allzero++;if(maxCoeff<=127)eligible++;
  }
  const macs=rows*batches*K*N;
  matrices.push({name:spec.name,offset:spec.offset,K,N,batches,rows,macs,weights:codes.length,zeroFraction:zeros/codes.length,int8GroupFraction:eligible/groups});
  totals.weights+=codes.length;totals.zeros+=zeros;totals.groups+=groups;totals.zeroGroups+=allzero;totals.int8Groups+=eligible;totals.int8Macs+=eligible*rows*16;totals.macs+=macs;totals.bytes+=packed.byteLength;
}
const family={};for(const m of matrices){const key=`K${m.K} N${m.N} B${m.batches}`;const f=family[key]??={matrices:0,macs:0,weights:0,int8Macs:0};f.matrices++;f.macs+=m.macs;f.weights+=m.weights;f.int8Macs+=m.macs*m.int8GroupFraction;}
const result={tensorCount:model.tensors.size,packedModelBytes:model.manifest.totals.packedByteLength,graphOps:graph.ops.length,fp8Matrices:matrices.length,duplicateMatrices:duplicates,totals,weightCodeHistogram:hist,exactIntegerMagnitudeBitsHistogram:bitsHist,macWeightedMagnitudeBits:macBits,families:Object.entries(family).sort((a,b)=>b[1].macs-a[1].macs),matrices};
await writeFile('reports/model-structure.json',JSON.stringify(result,null,2));
console.log(JSON.stringify({tensorCount:result.tensorCount,modelMiB:result.packedModelBytes/2**20,fp8Matrices:matrices.length,duplicateMatrices:duplicates,weights:totals.weights,zeroPercent:100*totals.zeros/totals.weights,zeroGroupPercent:100*totals.zeroGroups/totals.groups,int8GroupPercent:100*totals.int8Groups/totals.groups,int8MacPercent:100*totals.int8Macs/totals.macs,fp8GMacs:totals.macs/1e9,topFamilies:result.families.slice(0,10)},null,2));
