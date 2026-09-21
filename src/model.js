// Model decoding is reconstructed from native nr_model.cpp; no browser-port code.
export function half(h) {
  const s=h&32768?-1:1,e=(h>>>10)&31,m=h&1023;
  return e===0?s*m*2**-24:e===31?(m?NaN:s*Infinity):s*(1+m/1024)*2**(e-15);
}
export function e4(b) { const s=b&128?-1:1,e=(b>>>3)&15,m=b&7;return (b&127)===127?0:e===0?s*m/512:s*(1+m/8)*2**(e-7); }
export function packedIndex(k,n,N) {return (k>>>5)*N*32+(n>>>7)*4096+((n&127)>>>6)*2048+((n&63)>>>4)*512+(((n&7)*4+((k&15)>>>2))*16)+((n&15)>>>3)*8+((k&31)>>>4)*4+(k&3);}
export function inverseInput(k) {return (k&~31)+(k&17)+((k&2)<<1)+((k&4)<<1)+((k&8)>>>2);}
export function halfIndex(k,n,N) {return ((k>>>4)*Math.ceil(N/16)+(n>>>4))*256+((n&7)*4+((k&7)>>>1))*8+((n>>>3)&1)*4+((k&15)>=8?2:0)+(k&1);}
const safePath=p=>typeof p==='string'&&p.length>0&&!p.includes('\\')&&!p.startsWith('/')&&!p.split('/').some(x=>x==='..'||x==='.')&&!p.includes(':');
export class Model {
  static async load(read,{verify=true}={}) {
    const manifest=JSON.parse(new TextDecoder().decode(await read('manifest.json')));
    if(manifest.totals?.blockCount!==71||!Array.isArray(manifest.stages)||!Array.isArray(manifest.tensors))throw Error('Expected the 71-block NR model manifest.');
    const stages=new Map(),tensors=new Map();
    for(const s of manifest.stages) {
      if(!safePath(s.file)||stages.has(s.id)||!Number.isSafeInteger(s.packedByteLength)||s.packedByteLength<0)throw Error('Invalid or duplicate stage.');
      const bytes=new Uint8Array(await read('model/'+s.file));
      if(bytes.length!==s.packedByteLength)throw Error('Stage length mismatch: '+s.id);
      if(verify) {
        const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
        if(digest!==s.sha256)throw Error('Stage SHA-256 mismatch: '+s.id);
      }
      stages.set(s.id,bytes);
    }
    for(const t of manifest.tensors) {
      const s=stages.get(t.stage);
      if(!s||tensors.has(t.name)||![t.stageOffset,t.byteLength].every(v=>Number.isSafeInteger(v)&&v>=0)||t.stageOffset+t.byteLength>s.length)throw Error('Invalid tensor slice: '+t.name);
      tensors.set(t.name,new Uint8Array(s.buffer,s.byteOffset+t.stageOffset,t.byteLength));
    }
    return new Model(manifest,tensors);
  }
  constructor(manifest,tensors) {this.manifest=manifest;this.tensors=tensors;this.cache=new Map();}
  validateGraph(graph) {
    for(const op of graph.ops)for(const s of Object.values(op.bindings))if(s&&typeof s==='object'&&s.kind!=='zero') {
      const length=s.kind==='matrix'?s.batches*s.K*(s.halfMode?Math.ceil(s.N/16)*16*2:s.N):s.kind==='prior'?s.heads*8192:s.count*(s.type==='float'?4:2);
      this.bytes(s.name,s.offset,length);
    }
  }
  tensor(block,layer=0,parameter='layer') { const name=`block${block}.layer${layer}.${parameter}`;if(!this.tensors.has(name))throw Error('Missing tensor '+name);return name; }
  bytes(name,offset,length) {const t=this.tensors.get(name);if(!t||offset<0||offset+length>t.length)throw Error('Tensor bounds: '+name);return new DataView(t.buffer,t.byteOffset+offset,length);}
  vector(name,offset,count,type='half') {const v=this.bytes(name,offset,count*(type==='half'?2:4));return Float32Array.from({length:count},(_,i)=>type==='half'?half(v.getUint16(i*2,true)):v.getFloat32(i*4,true));}
  matrix(name,offset,K,N,{batches=1,halfMode=false}={}) {
    const key=[name,offset,K,N,batches,halfMode].join('/');if(this.cache.has(key))return this.cache.get(key);
    const out=new Float32Array(batches*K*N),t=this.tensors.get(name);if(!t)throw Error('Missing tensor '+name);
    const v=new DataView(t.buffer,t.byteOffset,t.byteLength);
    for(let batch=0;batch<batches;batch++)for(let k=0;k<K;k++)for(let n=0;n<N;n++) {
      const idx=offset+(halfMode?2*halfIndex(k,n,N):packedIndex(batch*K+inverseInput(k),n,N));
      if(idx+(halfMode?2:1)>t.length)throw Error('Matrix exceeds tensor '+key);
      out[(batch*K+k)*N+n]=halfMode?half(v.getUint16(idx,true)):e4(t[idx]);
    }
    this.cache.set(key,out);return out;
  }
  prior(name,offset,heads) {
    const v=this.bytes(name,offset,heads*8192),out=new Float32Array(heads*4096);
    for(let h=0;h<heads;h++)for(let q=0;q<64;q++)for(let k=0;k<64;k++) {
      const x=q%8,y=q>>>3,tq=(y>>>2)*32+(x>>>2)*16+(y&3)*4+(x&3),m=tq&15,n=k&15;
      const index=(tq>>>4)*1024+(k>>>4)*256+(((m&7)<<2)|((n&7)>>>1))*8+(n>>>3)*4+(m>=8?2:0)+(n&1);
      out[(h*64+q)*64+k]=half(v.getUint16(h*8192+index*2,true));
    }
    return out;
  }
}
export function directoryReader(files) {
  const map=new Map();
  for(const f of files) { const parts=(f.webkitRelativePath||f.name).split('/');if(parts.length>1)parts.shift();map.set(parts.join('/'),f); }
  return async path=>{const f=map.get(path);if(!f)throw Error('Missing model file '+path);return f.arrayBuffer();};
}
