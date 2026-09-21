import {createGraph} from './graph.js';
const safe=p=>typeof p==='string'&&p.length&&!p.includes('\\')&&!p.startsWith('/')&&!p.includes(':')&&!p.split('/').some(x=>x==='..'||x==='.');
export function compareBits(actual,expected) {
  if(actual.byteLength!==expected.byteLength)throw Error('Comparison size mismatch');
  const a=new Uint32Array(actual.buffer,actual.byteOffset,actual.byteLength/4),b=new Uint32Array(expected.buffer,expected.byteOffset,expected.byteLength/4);
  let mismatches=0,signedZeros=0,first=-1;
  for(let i=0;i<a.length;i++)if(a[i]!==b[i]){mismatches++;if(first<0)first=i;if((a[i]&0x7fffffff)===0&&(b[i]&0x7fffffff)===0)signedZeros++;}
  return {verdict:mismatches===0?'bit-exact':mismatches===signedZeros?'equal only up to sign of zero':'mismatch',mismatches,signedZeros,first};
}
// Graph activations are exact decoded E4 values in f32 storage. Re-encode without
// doing any arithmetic on them, preserving -0 for boundary comparison.
export function encodeBoundary(values) {
  const words=new Uint32Array(values.buffer,values.byteOffset,values.length),out=new Uint8Array(values.length);
  for(let i=0;i<values.length;i++) {
    const v=Math.abs(values[i]),sign=words[i]>>>24&128;
    if(!Number.isFinite(v)||v>448)throw Error('Invalid E4 boundary value');
    if(v<1/64)out[i]=sign|Math.round(v*512);
    else {const e=Math.floor(Math.log2(v)),m=Math.round((v/2**e-1)*8);out[i]=sign|((e+7)<<3)|m;}
  }
  return out;
}
export async function loadFixture(read) {
  const manifest=JSON.parse(new TextDecoder().decode(await read('manifest.json'))),dims=manifest.sourceDimensions;
  if(!Array.isArray(dims)||dims.length!==2)throw Error('sourceDimensions must be [width,height]');
  const [width,height]=dims,graph=createGraph(width,height),g=graph.geometry;
  if(JSON.stringify(manifest.fullDimensions)!==JSON.stringify([g.fullWidth,g.fullHeight]))throw Error('Fixture padded dimensions differ from native geometry');
  const checks=manifest.checks;
  if(!Array.isArray(checks)||!checks.length||new Set(checks).size!==checks.length||checks.some(c=>!['head','boundaries','output'].includes(c)))throw Error('Invalid fixture checks');
  async function binary(entry,length) {if(!safe(entry?.file))throw Error('Invalid fixture file');const bytes=new Uint8Array(await read(entry.file));if(bytes.byteLength!==length)throw Error('Fixture length mismatch: '+entry.file);return bytes;}
  const asFloat=b=>new Float32Array(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));
  if(!!manifest.proxy===!!manifest.inputFeatures)throw Error('Exactly one proxy or inputFeatures required');
  let proxy,inputFeatures;
  if(manifest.proxy) {
    const p=manifest.proxy;
    if(p.width!==width||p.height!==height)throw Error('Proxy fixture must match the source dimensions');
    if(!manifest.conditioning||manifest.seed===undefined||manifest.autoMask===undefined)throw Error('Proxy fixture needs conditioning, seed and autoMask');
    proxy=asFloat(await binary(p,width*height*16));
  } else inputFeatures=asFloat(await binary(manifest.inputFeatures,g.fullWidth*g.fullHeight*64));
  const boundaries=new Map(),omitted=manifest.omittedBoundaries||{};
  if(checks.includes('boundaries')) {
    for(const [name,entry] of [...(manifest.blocks||[]).map(e=>['block-'+e.block,e]),...(manifest.transitions||[]).map(e=>['transition-'+e.id,e])]) {
      const id=graph.boundaries[name],shape=graph.resources.get(id);
      if(!shape||boundaries.has(name)||name in omitted)throw Error('Invalid/duplicate boundary '+name);
      if(entry.width!==shape.width||entry.height!==shape.height||entry.channels!==shape.channels)throw Error('Boundary shape mismatch: '+name);
      boundaries.set(name,await binary(entry,shape.rows*shape.channels));
    }
    for(const [name,reason] of Object.entries(omitted))if(!(name in graph.boundaries)||typeof reason!=='string'||!reason.trim())throw Error('Invalid omission '+name);
    for(const name of Object.keys(graph.boundaries))if(!boundaries.has(name)&&!(name in omitted))throw Error('Unaccounted boundary '+name);
    if(boundaries.size===0)throw Error('Boundary check has no references');
  } else if(manifest.blocks?.length||manifest.transitions?.length||manifest.omittedBoundaries)throw Error('Boundary data without boundary check');
  if(checks.includes('head')!==!!manifest.referenceHead||checks.includes('output')!==!!manifest.nativeOutput)throw Error('Declared checks and references do not agree');
  const head=manifest.referenceHead?asFloat(await binary(manifest.referenceHead,g.fullWidth*g.fullHeight*16)):null;
  let output=null;
  if(manifest.nativeOutput) {
    const o=manifest.nativeOutput;if(o.width!==width||o.height!==height||!['u8','f32'].includes(o.dtype))throw Error('Invalid nativeOutput');
    if(o.dtype==='f32'&&!proxy)throw Error('f32 output reference requires a proxy');
    output=await binary(o,width*height*(o.dtype==='f32'?16:4));if(o.dtype==='f32')output=asFloat(output);
  }
  return {manifest,width,height,graph,proxy,inputFeatures,boundaries,omitted,head,output};
}
export async function runParity(engine,fixture,{repeat=3,onProgress,signal}={}) {
  if(!Number.isInteger(repeat)||repeat<2)throw Error('Parity needs at least two production runs');
  const f=fixture,results=[],input={width:f.width,height:f.height,proxy:f.proxy,inputFeatures:f.inputFeatures,seed:f.manifest.seed??0,conditioning:{localTone:0,localStructure:0,...f.manifest.conditioning,autoMask:f.manifest.autoMask??0},onProgress,signal};
  const checkOutput=run=>{
    if(!f.output)return;
    if(f.manifest.nativeOutput.dtype==='f32') {
      // Native gates RGB; alpha is an output surface convention.
      const a=new Float32Array(f.width*f.height*3),b=new Float32Array(a.length);
      for(let i=0;i<f.width*f.height;i++)for(let c=0;c<3;c++){a[i*3+c]=run.output[i*4+c];b[i*3+c]=f.output[i*4+c];}
      results.push({name:'output',...compareBits(a,b)});
    } else {
      let mismatches=0,tolerated=0;
      for(let y=0;y<f.height;y++)for(let x=0;x<f.width;x++)for(let c=0;c<3;c++) {
        const i=y*f.width+x,h=(y*run.geometry.fullWidth+x)*4+c;
        let value=run.output?.[i*4+c];
        if(value===undefined) {
          const centered=f.inputFeatures[(y*run.geometry.fullWidth+x)*16+4+c];
          value=Math.min(1,Math.max(0,Math.fround(Math.fround(run.head[h]*0.03125+centered)*8+0.5)));
          // Positive [0,1] truncation to half: quantize toward zero on its exponent grid.
          const step=value<2**-14?2**-24:2**(Math.floor(Math.log2(value))-10);value=Math.floor(value/step)*step;
        }
        const delta=Math.abs(Math.floor(value*255+0.5)-f.output[i*4+c]);if(delta>1)mismatches++;else if(delta)tolerated++;
      }
      results.push({name:'output-u8',verdict:mismatches?'mismatch':tolerated?'within one code':'bit-exact',mismatches,tolerated});
    }
  };
  let production;
  for(let i=0;i<repeat;i++) {
    const run=await engine.run(input);
    if(production)results.push({name:`production-repeat-${i}`, ...compareBits(run.head,production.head)});else production=run;
    if(f.head)results.push({name:`head-${i}`,...compareBits(run.head,f.head)});checkOutput(run);
  }
  if(f.boundaries.size) {
    const captured=await engine.run({...input,capture:(name,data)=>{
      const reference=f.boundaries.get(name);if(!reference)return;
      const actual=encodeBoundary(data);let mismatches=0,first=-1;
      for(let i=0;i<actual.length;i++)if(actual[i]!==reference[i]){mismatches++;if(first<0)first=i;}
      results.push({name,verdict:mismatches?'mismatch':'bit-exact',mismatches,first});
    }});
    results.push({name:'instrumented-vs-production',...compareBits(captured.head,production.head)});
  }
  return {passed:results.every(r=>r.mismatches===0),checks:results,omitted:f.omitted};
}
