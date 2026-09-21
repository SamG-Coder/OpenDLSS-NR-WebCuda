import {GpuRuntime} from '../vendor/webcuda/runtime/runtime.js';
import {half,e4} from '../src/model.js';
import {NeuralRenderer} from '../src/engine.js';
const report={passed:0,failed:0,checks:[]};let runtime;
const check=(name,ok,detail='')=>{report[ok?'passed':'failed']++;report.checks.push({name,ok,detail});};
try {
  runtime=await GpuRuntime.create({onError:e=>{throw e;}});report.adapter=runtime.describe();
  const manifest=await (await fetch('../generated/manifest.json')).json(),kernels={};
  for(const [name,file] of Object.entries(manifest))kernels[name]=await runtime.kernel(await(await fetch('../generated/'+file)).json());
  check('All CUDA-generated shaders create GPU pipelines',true,Object.keys(kernels).length+' kernels');
  const dispatch=(entry,bindings,scalars,count)=>runtime.batch().dispatch(kernels[entry].bind(bindings,scalars),[Math.ceil(count/64),1,1]).submit();
  // Exhaust every finite half bit pattern, plus NaNs/infinities, without assuming float32 NaN payloads survive.
  const values=Float32Array.from({length:65536},(_,i)=>half(i));
  const input=runtime.createBuffer(values),output=runtime.createBuffer(values.length*12),codes=runtime.createBuffer(values.length*4);
  dispatch('nr_numeric',{input,output,codes},{count:values.length},values.length);
  const actual=await runtime.read(output),actualCodes=await runtime.read(codes,Uint32Array);
  let halfErrors=0,quantErrors=0;const word=new DataView(new ArrayBuffer(4));
  const bits=x=>{word.setFloat32(0,x,true);return word.getUint32(0,true);};
  const positive=Float32Array.from({length:127},(_,i)=>e4(i));
  function nearest(x) {
    if(Number.isNaN(x))return 0;const sign=x<0||Object.is(x,-0)?128:0,v=Math.min(Math.abs(x),448);let best=0;
    for(let c=1;c<127;c++){const d=Math.abs(positive[c]-v),b=Math.abs(positive[best]-v);if(d<b||d===b&&c%2===0)best=c;}
    return sign|best;
  }
  for(let i=0;i<values.length;i++) {
    if(!Number.isNaN(values[i])&&bits(actual[i*3])!==bits(values[i]))halfErrors++;
    if(actualCodes[i]!==nearest(values[i]))quantErrors++;
  }
  check('65,536 half patterns: half publication',halfErrors===0,halfErrors+' mismatches');
  check('65,536 half patterns: E4M3 RNE, saturation, NaN, signed zero',quantErrors===0,quantErrors+' mismatches');
  {
    const count=65535*64+3,source=runtime.createBuffer(new Float32Array(count).fill(1.25)),target=runtime.createBuffer(new Float32Array(count+8).fill(777));
    runtime.batch().dispatch(kernels.nr_publish.bind({input:source,output:target},{count,quantize:0}),[65535,2,1]).submit();
    const data=await runtime.read(target);
    check('2D dispatch crosses the 65,535-workgroup boundary and preserves tail guards',data.subarray(0,count).every(v=>v===1.25)&&data.subarray(count).every(v=>v===777));
    runtime.destroyBuffer(source);runtime.destroyBuffer(target);
  }
  const a=runtime.createBuffer(new Float32Array(32).fill(1)),b=runtime.createBuffer(new Float32Array(32*16).fill(0.5)),residual=runtime.createBuffer(new Float32Array(16).fill(2)),scales=runtime.createBuffer(new Float32Array(16).fill(0.5)),raw=runtime.createBuffer(64),out=runtime.createBuffer(64);
  dispatch('nr_gemm',{input:a,weights:b,residual,scales,raw,output:out},{rows:1,K:32,N:16,batches:1,inputStride:32,inputBatchStride:32,partition:0,halfMode:0,silu:0,hasResidual:1,quantize:0},16);
  const gemm=await runtime.read(out);check('FP8 GEMM seeds the residual accumulator',gemm.every(x=>x===17),Array.from(gemm.slice(0,4)).join(','));
  const z=runtime.createBuffer(new Float32Array(64*96)),prior=runtime.createBuffer(new Float32Array(4096)),scores=runtime.createBuffer(64*64*4),weights=runtime.createBuffer(64*64*4),inverse=runtime.createBuffer(64*4),attended=runtime.createBuffer(64*32*4);
  dispatch('nr_scores',{qkv:z,prior,scores},{width:8,height:8,heads:1,shiftX:4,shiftY:4,padded:64,globalMode:0},4096);
  dispatch('nr_softmax',{scores,weights,inverse},{rows:64,heads:1,keys:64,globalMode:0},64);
  dispatch('nr_attend',{qkv:z,weights,inverse,output:attended},{width:8,height:8,heads:1,shiftX:4,shiftY:4,keys:64,globalMode:0},2048);
  const attention=await runtime.read(attended);check('Shifted zero windows are finite zeros',attention.every(x=>x===0));
  const nativeResponse=await fetch('../build/native/cases.json');
  if(nativeResponse.ok) {
    for(const c of await nativeResponse.json()) {
      const bindings={};
      try {
        for(const [name,b] of Object.entries(c.buffers)) {
          const r=await fetch('../build/native/'+b.file);if(!r.ok)throw Error('Missing native input '+b.file);
          bindings[name]=runtime.createBuffer(new Uint32Array(await r.arrayBuffer()));
        }
        dispatch(c.entry,bindings,c.scalars,c.count);
        for(const name of c.outputs) {
          const r=await fetch(`../build/native/${c.name}-${name}-expected.bin`);if(!r.ok)throw Error('Run native fixture executable first');
          const expected=new Uint32Array(await r.arrayBuffer()),actual=await runtime.read(bindings[name],Uint32Array);
          let mismatches=0,first='';
          for(let i=0;i<actual.length;i++) {
            const isNaNBits=x=>(x&0x7f800000)===0x7f800000&&(x&0x007fffff)!==0;
            // NaN payloads are not portable between CUDA and WGSL.
            if(actual[i]!==expected[i]&&!(c.buffers[name].type==='float'&&isNaNBits(actual[i])&&isNaNBits(expected[i]))) {mismatches++;if(!first)first=` at ${i}: ${actual[i].toString(16)} vs ${expected[i].toString(16)}`;}
          }
          check('Native CUDA parity: '+c.name+'/'+name,mismatches===0,`${actual.length} words, ${mismatches} mismatches${first}`);
        }
      } finally {for(const b of Object.values(bindings))runtime.destroyBuffer(b);}
    }
  } else report.nativeFixtures='Not supplied; run scripts/test-native.ps1';
  const synthetic={cache:new Map(),matrix:(name,offset,K,N,{batches=1})=>new Float32Array(K*N*batches),vector:(name,offset,n)=>new Float32Array(n).fill(1),prior:(name,offset,heads)=>new Float32Array(heads*4096)};
  const engine=new NeuralRenderer(runtime,kernels,synthetic),boundaries=[];
  const run=await engine.run({width:2,height:2,inputFeatures:new Float32Array(64),geometryOverride:{width:2,height:2,fullWidth:2,fullHeight:2,levels:Array.from({length:6},()=>({width:1,height:1}))},capture:(name,data)=>{if(!data.every(x=>x===0))throw Error('Nonzero synthetic boundary '+name);boundaries.push(name);}});
  check('Complete graph executes all 75 comparable boundaries (synthetic weights, tiny test geometry)',boundaries.length===75&&new Set(boundaries).size===75&&run.head.every(x=>x===0),`${run.dispatches} dispatches, ${boundaries.length} boundaries`);
  const args={width:2,height:2,inputFeatures:new Float32Array(64),geometryOverride:{width:2,height:2,fullWidth:2,fullHeight:2,levels:Array.from({length:6},()=>({width:1,height:1}))}};
  const resourcesBefore=runtime.buffers.size,submissionsBefore=runtime.stats.submissions;
  const batched=await engine.run(args);
  check('Batched graph preserves captured output and releases all inference buffers',batched.head.every((x,i)=>Object.is(x,run.head[i]))&&runtime.buffers.size===resourcesBefore&&runtime.stats.submissions-submissionsBefore<run.dispatches);
  const abort=new AbortController();let cancelled=false;
  try{await engine.run({...args,signal:abort.signal,onProgress:({index})=>{if(index===3)abort.abort();}});}catch(e){cancelled=e.name==='AbortError';}
  check('Cancellation discards pending commands and releases inference buffers',cancelled&&!engine.busy&&runtime.buffers.size===resourcesBefore);
  const restarted=await engine.run(args);
  check('Renderer can restart after cancellation with identical output',restarted.head.every((x,i)=>Object.is(x,run.head[i]))&&runtime.buffers.size===resourcesBefore);
} catch(error) {check('GPU execution',false,String(error.stack||error));}
finally {runtime?.dispose();window.report=report;document.querySelector('#result').textContent=JSON.stringify(report,null,2);}
