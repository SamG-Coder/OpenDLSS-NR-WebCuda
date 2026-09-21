import {readFile,mkdir,writeFile,stat} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {weightResource,parseWeights} from '../src/dll-model.js';
const sha=b=>createHash('sha256').update(b).digest('hex');
export async function extractModel(dll,output) {
  try{await stat(output);throw Error('Output already exists; choose a new directory');}catch(e){if(e.code!=='ENOENT')throw e;}
  const source=await readFile(dll),resource=weightResource(source),tensors=parseWeights(resource);
  const ends=[4,8,14,22,30,38,47,55,61,65,70],stages=[],records=[];
  await mkdir(path.join(output,'model'),{recursive:true});
  for(let i=0;i<ends.length;i++) {
    const group=tensors.filter(t=>t.block>(ends[i-1]??-1)&&t.block<=ends[i]).sort((a,b)=>a.block-b.block||a.layer-b.layer||a.name.localeCompare(b.name));
    const id=`stage-${i}`,file=id+'.bin',bytes=Buffer.concat(group.map(t=>t.bytes));let stageOffset=0;
    for(const {bytes:data,...t} of group){records.push({...t,stage:id,stageOffset,byteLength:data.length});stageOffset+=data.length;}
    await writeFile(path.join(output,'model',file),bytes);stages.push({id,file,packedByteLength:bytes.length,sha256:sha(bytes)});
  }
  const manifest={totals:{blockCount:71,tensorCount:records.length,packedByteLength:stages.reduce((a,s)=>a+s.packedByteLength,0)},source:{dll:path.basename(dll),sha256:sha(source),resource:'WEIGHTS_HT',resourceByteLength:resource.length},stages,tensors:records};
  await writeFile(path.join(output,'manifest.json'),JSON.stringify(manifest,null,2));return manifest;
}
if(process.argv[1]===fileURLToPath(import.meta.url)) {
  const [dll,output]=process.argv.slice(2);if(!dll||!output)throw Error('Usage: node scripts/extract-model.mjs <nvngx_dlssnr.dll> <new-output-directory>');
  const m=await extractModel(dll,output);console.log(JSON.stringify({output,source:m.source,totals:m.totals,stages:m.stages.length},null,2));
}

