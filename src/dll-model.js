import {Model} from './model.js';
import {createGraph} from './graph.js';

// Static PE resource parsing only: no DLL loading, native execution, or network requests.
function reader(bytes) {
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  const bounds=(offset,size)=>{
    if(!Number.isSafeInteger(offset)||!Number.isSafeInteger(size)||offset<0||size<0||offset+size>bytes.length)throw Error('The DLL or weight resource is truncated.');
  };
  return {
    bounds,
    u16:o=>{bounds(o,2);return view.getUint16(o,true);},
    u32:o=>{bounds(o,4);return view.getUint32(o,true);},
    u64:o=>{bounds(o,8);const n=Number(view.getBigUint64(o,true));if(!Number.isSafeInteger(n))throw Error('Weight record is too large.');return n;},
    text:(o,n,encoding='utf-8')=>{bounds(o,n);return new TextDecoder(encoding,{fatal:true}).decode(bytes.subarray(o,o+n));},
  };
}

export function weightResource(bytes) {
  const r=reader(bytes);
  if(r.u16(0)!==0x5a4d)throw Error('Choose nvngx_dlssnr.dll: this file is not a Windows DLL.');
  const pe=r.u32(60);
  if(r.u32(pe)!==0x4550||r.u16(pe+4)!==0x8664)throw Error('Expected a 64-bit NVIDIA DLL.');
  const opt=pe+24,optSize=r.u16(pe+20);r.bounds(opt,optSize);
  if(optSize<136||r.u16(opt)!==0x20b||r.u32(opt+108)<3)throw Error('The DLL has no valid resource directory.');
  const sections=[],table=opt+optSize;
  for(let i=0;i<r.u16(pe+6);i++) {
    const s=table+i*40;r.bounds(s,40);
    sections.push({rva:r.u32(s+12),size:r.u32(s+16),offset:r.u32(s+20)});
  }
  function offset(rva,size) {
    const s=sections.find(s=>rva>=s.rva&&rva+size<=s.rva+s.size);
    if(!s)throw Error('The DLL resource points outside its data sections.');
    const p=s.offset+rva-s.rva;r.bounds(p,size);return p;
  }
  const rootSize=r.u32(opt+132),root=offset(r.u32(opt+128),rootSize),found=[];
  function resourceOffset(relative,size) {
    if(relative<0||relative+size>rootSize)throw Error('Invalid DLL resource directory offset.');
    return root+relative;
  }
  function walk(relative,keys,seen=new Set()) {
    if(keys.length>3||seen.has(relative))throw Error('Invalid DLL resource tree.');
    seen=new Set(seen).add(relative);
    const d=resourceOffset(relative,16),count=r.u16(d+12)+r.u16(d+14);resourceOffset(relative+16,count*8);
    for(let i=0;i<count;i++) {
      const p=d+16+i*8,k=r.u32(p),v=r.u32(p+4);let key=k;
      if(k&0x80000000) {
        const rel=k&0x7fffffff,s=resourceOffset(rel,2),length=r.u16(s)*2;
        key=r.text(resourceOffset(rel+2,length),length,'utf-16le');
      }
      if(v&0x80000000)walk(v&0x7fffffff,[...keys,key],seen);
      else {
        const e=resourceOffset(v,16),size=r.u32(e+4),p=offset(r.u32(e),size);
        if(keys[0]===10&&keys[1]==='WEIGHTS_HT')found.push(bytes.subarray(p,p+size));
      }
    }
  }
  walk(0,[]);
  if(found.length!==1)throw Error('This DLL does not contain the supported WEIGHTS_HT model. Choose nvngx_dlssnr.dll (310.8.0), not the Streamline wrapper.');
  return found[0];
}

export function parseWeights(bytes) {
  const r=reader(bytes);
  if(r.u64(0)!==bytes.length)throw Error('Embedded model length does not match its header.');
  let p=8;const records=[],names=new Set();
  while(p<bytes.length) {
    const length=r.u64(p);p+=8;
    if(length<1||length>128)throw Error('Invalid tensor name length.');
    const name=r.text(p,length);p+=length;
    const match=/^block(\d+)\.layer(\d+)\.(layer|blend_scale)$/.exec(name);
    if(!match||names.has(name))throw Error('Invalid or duplicate tensor: '+name);names.add(name);
    const size=r.u64(p);p+=8;const total=r.u64(p),dataLength=r.u64(p+8);r.bounds(p,size);
    if(size!==total||size!==dataLength+40||dataLength<2||dataLength%2||r.u32(p+16)!==1)throw Error('Unsupported tensor encoding: '+name);
    const tail=p+20+dataLength;
    if(r.u64(tail)!==0||r.u64(tail+8)!==1||r.u32(tail+16)*2!==dataLength)throw Error('Unsupported tensor metadata: '+name);
    records.push({name,block:Number(match[1]),layer:Number(match[2]),parameter:match[3],bytes:bytes.subarray(p+20,p+20+dataLength)});p+=size;
  }
  if(records.length!==153||new Set(records.map(t=>t.block)).size!==71||records.some(t=>t.block>70))throw Error('This DLL does not contain the supported 71-block, 153-tensor network.');
  return records;
}

export async function modelFromDll(file,{onProgress=()=>{}}={}) {
  if(!file||typeof file.arrayBuffer!=='function'||file.size>512*1024*1024)throw Error('Choose an NR DLL smaller than 512 MiB.');
  onProgress('Reading DLL from your device…');
  const bytes=new Uint8Array(await file.arrayBuffer());
  onProgress('Extracting embedded NR weights…');
  const resource=weightResource(bytes),records=parseWeights(resource);
  // Copy just the tensor payloads so the unused executable sections can be collected.
  const tensors=new Map(records.map(t=>[t.name,t.bytes.slice()]));
  const manifest={totals:{blockCount:71,tensorCount:records.length,packedByteLength:records.reduce((n,t)=>n+t.bytes.length,0)},source:{file:file.name,resource:'WEIGHTS_HT'}};
  const model=new Model(manifest,tensors);
  onProgress('Validating all network tensor layouts…');
  model.validateGraph(createGraph(33,33));
  const digest=await crypto.subtle.digest('SHA-256',resource);
  manifest.source.resourceSha256=Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,'0')).join('');
  return model;
}
