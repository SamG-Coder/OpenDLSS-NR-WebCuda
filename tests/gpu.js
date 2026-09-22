import {createGraph} from '../src/graph.js';
import {gemmKernel} from '../src/kernel-selection.js';
import {specializedGemm,dynamicGemmScalars} from '../src/gemm-specialization.js';
import {GpuRuntime} from '../vendor/webcuda/runtime/runtime.js';
import {half,e4,unpackActivations} from '../src/model.js';
import {NeuralRenderer} from '../src/engine.js';
import {dispatchGroups} from '../src/kernel-selection.js';
const report={passed:0,failed:0,checks:[]};let runtime;
const check=(name,ok,detail='')=>{report[ok?'passed':'failed']++;report.checks.push({name,ok,detail});};
try {
  runtime=await GpuRuntime.create({onError:e=>{throw e;}});report.adapter=runtime.describe();
  const manifest=await (await fetch('../generated/manifest.json')).json(),kernels={};
  for(const [name,file] of Object.entries(manifest))kernels[name]=await runtime.kernel(await(await fetch('../generated/'+file)).json());
  check('All CUDA-generated shaders create GPU pipelines',true,Object.keys(kernels).length+' kernels');
  const packedSource=await(await fetch('../generated/gemm.cu')).text();
  const decoder=await runtime.kernel(packedSource+`\n__global__ void decode_packed_test(const unsigned* input, float* output, unsigned count, int halfMode) { unsigned i=blockIdx.x*blockDim.x+threadIdx.x; if(i<count) output[i]=nr_packed_weight(input[i/(halfMode!=0?2u:4u)],i,halfMode); }`,{entry:'decode_packed_test',workgroupSize:[64,1,1]});
  for(const halfMode of [0,1]) {
    const count=halfMode?65536:256,lanes=halfMode?2:4,packed=new Uint32Array(count/lanes);
    for(let i=0;i<count;i++)packed[Math.floor(i/lanes)]|=i<<((i%lanes)*(halfMode?16:8));
    const input=runtime.createBuffer(packed),output=runtime.createBuffer(count*4);
    runtime.batch().dispatch(decoder.bind({input,output},{count,halfMode}),[Math.ceil(count/64),1,1]).submit();
    const decoded=await runtime.read(output);let mismatches=0;
    for(let i=0;i<count;i++)if(!Object.is(decoded[i],halfMode?half(i):e4(i)))mismatches++;
    check(`Packed GPU decoding: all ${count} ${halfMode?'half':'FP8'} patterns`,mismatches===0,`${mismatches} mismatches`);
    runtime.destroyBuffer(input);runtime.destroyBuffer(output);
  }
  if(runtime.device.features.has('shader-f16')){
    const pairs=[];for(let a=0;a<256;a++)if((a&127)!==127)for(let b=0;b<256;b++)if((b&127)<=81)pairs.push(e4(a),e4(b));
    const source=`__global__ void half_products(const float* input,float* output,unsigned count){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){float a=input[2*i]*4.0f;float b=input[2*i+1]*4.0f;__half2 p=__hmul2(__floats2half2_rn(a,b),__floats2half2_rn(b,a));float2 v=__half22float2(p);output[2*i]=v.x;output[2*i+1]=v.y;}}`;
    const kernel=await runtime.kernel(source,{workgroupSize:[64,1,1]}),input=runtime.createBuffer(new Float32Array(pairs)),output=runtime.createBuffer(pairs.length*4),count=pairs.length/2;
    runtime.batch().dispatch(kernel.bind({input,output},{count}),[Math.ceil(count/64),1,1]).submit();
    const actual=await runtime.read(output);let mismatches=0;
    for(let i=0;i<count;i++){const expected=pairs[2*i]*pairs[2*i+1]*16;if(actual[2*i]!==expected||actual[2*i+1]!==expected)mismatches++;}
    check('Native half2: every finite E4 activation times every signed bounded weight',mismatches===0,`${count} pairs, ${mismatches} mismatches`);
    runtime.destroyBuffer(input);runtime.destroyBuffer(output);
  }
  const dispatch=(entry,bindings,scalars,count)=>runtime.batch().dispatch(kernels[entry].bind(bindings,scalars),[Math.ceil(count/64),1,1]).submit();
  // Separate cached noise must preserve every feature lane, including masked history and padding.
  for(const seed of [0,219,4294967295]) {
    const width=37,height=35,fullWidth=64,fullHeight=64,n=fullWidth*fullHeight;
    const proxy=runtime.createBuffer(Float32Array.from({length:width*height*4},(_,i)=>(i%251)/251));
    const history=runtime.createBuffer(Float32Array.from({length:width*height*4},(_,i)=>i%8===3?0:(i%197)/197));
    const noise=runtime.createBuffer(n*8),reference=runtime.createBuffer(n*64),cached=runtime.createBuffer(n*64);
    const scalars={width,height,fullWidth,fullHeight,seed,autoMask:seed?0:1,localTone:0.5,localStructure:0.25,skinStructure:-1,style:0.7,useHistory:1};
    dispatch('nr_preprocess',{proxy,history,features:reference},scalars,n);
    dispatch('nr_noise',{noise},{fullWidth,fullHeight,seed},n);
    dispatch('nr_preprocess_cached',{noise,proxy,history,features:cached},scalars,n);
    const a=await runtime.read(reference,Uint32Array),b=await runtime.read(cached,Uint32Array);
    check(`Cached noise preserves all feature bits for seed ${seed}`,a.every((v,i)=>v===b[i]));
    for(const buffer of [proxy,history,noise,reference,cached])runtime.destroyBuffer(buffer);
  }
  {
    const graph=createGraph(1280,720,{activationStorage:'packed',fuseLocalAttention:true}),tested=new Set();
    const packed=(count,format)=>{const lanes=format===1?4:2,bits=format===1?8:16,a=new Uint32Array(Math.ceil(count/lanes));for(let i=0;i<count;i++)a[Math.floor(i/lanes)]|=(format===1?(24+(i*17)%48)|(i%3?0:128):0x3000+(i%31)|(i%3?0:32768))<<((i%lanes)*bits);return a;};
    for(const op of graph.ops)if(op.entry==='nr_gemm'&&!op.scalars.halfMode){
      const s={...op.scalars,...Object.fromEntries(['input','residual','raw','output'].map(key=>[key+'Format',typeof op.bindings[key]==='string'?graph.resources.get(op.bindings[key]).format:0]))};
      const base=gemmKernel(s),name=specializedGemm(base+'_compact',s);if(!name||tested.has(name))continue;tested.add(name);s.rows=9;
      const count=s.rows*s.N*s.batches,bindings={input:runtime.createBuffer(packed(s.rows*s.inputStride,s.inputFormat)),weights:runtime.createBuffer(packed(s.K*s.N*s.batches,1)),residual:runtime.createBuffer(s.hasResidual?packed(count,s.residualFormat):new Uint32Array(1)),scales:runtime.createBuffer(new Float32Array(s.N*s.batches).fill(.5)),raw:runtime.createBuffer(new Uint32Array((s.rawEnabled?count/2:1)+2).fill(0xdeadbeef)),output:runtime.createBuffer(new Uint32Array(count*(s.outputFormat===1?1:2)/4+2).fill(0xdeadbeef))};
      try {
        runtime.batch().dispatch(kernels.nr_gemm_packed_compact.bind(bindings,s),[Math.ceil(count/64),1,1]).submit();
        const expected=await runtime.read(bindings.output,Uint32Array),raw=await runtime.read(bindings.raw,Uint32Array);
        for(const variant of [name,...(kernels[name+'_half']?[name+'_half']:[])]){tested.add(variant);
        runtime.write(bindings.output,new Uint32Array(expected.length).fill(0xdeadbeef));runtime.write(bindings.raw,new Uint32Array(raw.length).fill(0xdeadbeef));
        // Deliberate surplus tile exercises row guards with all lanes reaching barriers.
        runtime.batch().dispatch(kernels[variant].bind(bindings,dynamicGemmScalars(s)),[dispatchGroups(base,s,count)+1,1,1]).submit();
        const actual=await runtime.read(bindings.output,Uint32Array),actualRaw=await runtime.read(bindings.raw,Uint32Array);
        check('Specialized GEMM tail and publication: '+variant,actual.every((v,i)=>v===expected[i])&&actualRaw.every((v,i)=>v===raw[i]));
        }
      }finally{for(const b of Object.values(bindings))runtime.destroyBuffer(b);}
    }
    check('Every built specialization is covered',Object.keys(kernels).filter(k=>/_compact_s[0-9]/.test(k)).every(k=>tested.has(k)));
  }
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
  for(const shape of [{rows:7,K:64,N:19,batches:2,halfMode:0,partition:0},{rows:9,K:64,N:32,batches:2,halfMode:0,partition:32},{rows:5,K:16,N:4,batches:1,halfMode:1,partition:8}]) {
    const {rows,K,N,batches,halfMode}=shape,count=rows*batches*N,lanes=halfMode?2:4,packed=new Uint32Array(Math.ceil(K*N*batches/lanes));
    for(let i=0;i<K*N*batches;i++){const code=halfMode?8192+(i*97)%4096+((i%2)*32768):((i*17)%96)+((i%2)*128);packed[Math.floor(i/lanes)]|=code<<((i%lanes)*(halfMode?16:8));}
    const bindings={input:runtime.createBuffer(Float32Array.from({length:rows*K*batches},(_,i)=>e4((i*7)%64))),weights:runtime.createBuffer(packed),residual:runtime.createBuffer(new Float32Array(count).fill(.5)),scales:runtime.createBuffer(new Float32Array(N*batches).fill(.25)),raw:runtime.createBuffer(new Float32Array(count+17).fill(777)),output:runtime.createBuffer(new Float32Array(count+17).fill(777))};
    const scalars={...shape,inputStride:K*batches,inputBatchStride:shape.partition?0:K,silu:1,hasResidual:1,quantize:1};
    try {
      dispatch('nr_gemm_packed',bindings,scalars,count);
      const expectedRaw=await runtime.read(bindings.raw,Uint32Array),expected=await runtime.read(bindings.output,Uint32Array);
      for(const [entry,tileRows,tileCols] of [['nr_gemm_tiled',4,16],['nr_gemm_tile8x8',8,8],['nr_gemm_tile8x16',8,16],['nr_gemm_multi8x32',8,32],['nr_gemm_multi16x16',16,16],['nr_gemm_multi16x32',16,32],['nr_gemm_multi4x32',4,32],['nr_gemm_multi32x32',32,32],['nr_gemm_multi16x64',16,64]]){
        runtime.write(bindings.raw,new Float32Array(count+17).fill(777));runtime.write(bindings.output,new Float32Array(count+17).fill(777));
        const groups=Math.ceil(rows/tileRows)*batches*Math.ceil(N/tileCols);
        runtime.batch().dispatch(kernels[entry].bind(bindings,scalars),[2,Math.ceil(groups/2),1]).submit();
        const actualRaw=await runtime.read(bindings.raw,Uint32Array),actual=await runtime.read(bindings.output,Uint32Array);
        check(entry+' matches scalar: tails, 2D dispatch, half='+halfMode+', partition='+shape.partition,actual.every((v,i)=>v===expected[i])&&actualRaw.every((v,i)=>v===expectedRaw[i]));
        if(entry.startsWith('nr_gemm_multi'))for(const outputFormat of [1,2]) {
          const rawWords=Math.ceil(count/2),outWords=Math.ceil(count/(outputFormat===1?4:2)),sentinel=0x5a5a5a5a;
          const compactRaw=runtime.createBuffer(new Uint32Array(rawWords+4).fill(sentinel)),compactOutput=runtime.createBuffer(new Uint32Array(outWords+4).fill(sentinel));
          try {
            // quantize=1 publishes FP8 values; storing those in half must also be exact.
            runtime.batch().dispatch(kernels[entry+'_compact'].bind({...bindings,raw:compactRaw,output:compactOutput},{...scalars,inputFormat:0,residualFormat:0,rawFormat:2,outputFormat,rawEnabled:1}),[2,Math.ceil(groups/2),1]).submit();
            const rw=await runtime.read(compactRaw,Uint32Array),ow=await runtime.read(compactOutput,Uint32Array),rv=unpackActivations(rw,2,count),ov=unpackActivations(ow,outputFormat,count),er=new Float32Array(expectedRaw.buffer),eo=new Float32Array(expected.buffer);
            check(entry+' compact matches scalar, format='+outputFormat+', half='+halfMode+', N='+N+', partition='+shape.partition,rv.every((v,i)=>Object.is(v,er[i]))&&ov.every((v,i)=>Object.is(v,eo[i]))&&rw.subarray(rawWords).every(v=>v===sentinel)&&ow.subarray(outWords).every(v=>v===sentinel));
          }finally{runtime.destroyBuffer(compactRaw);runtime.destroyBuffer(compactOutput);}
        }

      }
    } finally {for(const b of Object.values(bindings))runtime.destroyBuffer(b);}
  }
  const z=runtime.createBuffer(new Float32Array(64*96)),prior=runtime.createBuffer(new Float32Array(4096)),scores=runtime.createBuffer(64*64*4),weights=runtime.createBuffer(64*64*4),inverse=runtime.createBuffer(64*4),attended=runtime.createBuffer(64*32*4);
  dispatch('nr_scores',{qkv:z,prior,scores},{width:8,height:8,heads:1,shiftX:4,shiftY:4,padded:64,globalMode:0},4096);
  dispatch('nr_softmax',{scores,weights,inverse},{rows:64,heads:1,keys:64,globalMode:0},64);
  dispatch('nr_attend',{qkv:z,weights,inverse,output:attended},{width:8,height:8,heads:1,shiftX:4,shiftY:4,keys:64,globalMode:0},2048);
  const attention=await runtime.read(attended);check('Shifted zero windows are finite zeros',attention.every(x=>x===0));
  for(const [width,height,shiftX,shiftY,globalMode] of [[9,7,0,0,0],[9,7,4,4,0],[9,7,4,0,0],[9,7,0,4,0],[9,7,0,0,1],[11,7,0,0,1]]){
    const rows=width*height,heads=2,keys=globalMode?Math.ceil(rows/64)*64:64;
    const qkv=runtime.createBuffer(Float32Array.from({length:rows*heads*96},(_,i)=>e4((i*17)%40+8+(i%2)*128)));
    const prior=runtime.createBuffer(Float32Array.from({length:heads*4096},(_,i)=>half(8192+i%1024)));
    const weights=runtime.createBuffer(Float32Array.from({length:rows*heads*keys},(_,i)=>e4(8+i%24)));
    const inverse=runtime.createBuffer(new Float32Array(rows*heads).fill(.125));
    try {
      for(const entry of ['nr_scores','nr_attend']){
        const count=rows*heads*(entry==='nr_scores'?keys:32),buffer=runtime.createBuffer(new Float32Array(count+11).fill(777));
        const scalars={width,height,heads,shiftX,shiftY,globalMode,...(entry==='nr_scores'?{padded:keys}:{keys})};
        const bindings=entry==='nr_scores'?{qkv,prior,scores:buffer}:{qkv,weights,inverse,output:buffer};
        try{
          dispatch(entry,bindings,scalars,count);const expected=await runtime.read(buffer,Uint32Array);
          runtime.write(buffer,new Float32Array(count+11).fill(777));
          const groups=dispatchGroups(entry+'_tiled',scalars,count);
          runtime.batch().dispatch(kernels[entry+'_tiled'].bind(bindings,scalars),[3,Math.ceil(groups/3),1]).submit();
          const actual=await runtime.read(buffer,Uint32Array);
          check(`${entry} tiled exact match: ${width}x${height}, shift ${shiftX}/${shiftY}, global ${globalMode}, tail guards`,actual.every((v,i)=>v===expected[i]));
        }finally{runtime.destroyBuffer(buffer);}
      }
    }finally{for(const b of [qkv,prior,weights,inverse])runtime.destroyBuffer(b);}
  }
  // Independent, nonzero local attention comparisons exercise edge windows and
  // surplus 2D groups, including packed input and each output storage format.
  for(const [width,height] of [[1,1],[9,7],[17,11]])for(const [shiftX,shiftY] of [[0,0],[4,4],[4,0],[0,4]]) {
    const heads=2,rows=width*height,count=rows*heads*32,scalars={width,height,heads,shiftX,shiftY};
    const bytes=Uint8Array.from({length:rows*heads*96},(_,i)=>i%19===0?(i%2?128:0):(32+(i*17)%45)|(i%3?0:128));
    const qkv=runtime.createBuffer(Float32Array.from(bytes,e4)),packedQkv=runtime.createBuffer(new Uint32Array(bytes.buffer));
    const prior=runtime.createBuffer(Float32Array.from({length:heads*4096},(_,i)=>half(0x3000+i%31)*(i%3?-1:1)));
    const scores=runtime.createBuffer(rows*heads*64*4),weights=runtime.createBuffer(rows*heads*64*4),inverse=runtime.createBuffer(rows*heads*4),reference=runtime.createBuffer(count*4);
    try {
      dispatch('nr_scores',{qkv,prior,scores},{...scalars,padded:64,globalMode:0},rows*heads*64);
      dispatch('nr_softmax',{scores,weights,inverse},{rows,heads,keys:64,globalMode:0},rows*heads);
      dispatch('nr_attend',{qkv,weights,inverse,output:reference},{...scalars,keys:64,globalMode:0},count);
      const expected=await runtime.read(reference),groups=dispatchGroups('nr_local_attention',scalars,count);
      for(const format of [-1,0,1,2]) {
        const lanes=format===1?4:format===2?2:1,words=count/lanes,sentinel=0x5a5a5a5a;
        const output=runtime.createBuffer(new Uint32Array(words+4).fill(sentinel));
        try {
          const entry='nr_local_attention'+(format<0?'':'_compact');
          runtime.batch().dispatch(kernels[entry].bind({qkv:format<0?qkv:packedQkv,prior,output},{...scalars,...(format<0?{}:{qkvFormat:1,outputFormat:format})}),[3,Math.ceil(groups/3),1]).submit();
          const raw=await runtime.read(output,Uint32Array),actual=format<=0?new Float32Array(raw.buffer,0,count):unpackActivations(raw,format,count);
          check(`Fused attention exact match ${width}x${height}, shift ${shiftX}/${shiftY}, format ${format}`,actual.every((v,i)=>Object.is(v,expected[i]))&&raw.subarray(words).every(v=>v===sentinel));
        }finally{runtime.destroyBuffer(output);}
      }
    }finally{for(const b of [qkv,packedQkv,prior,scores,weights,inverse,reference])runtime.destroyBuffer(b);}
  }
  for(const format of [1,2]) {
    const count=65535,values=Float32Array.from({length:count},(_,i)=>half(i)),input=runtime.createBuffer(values);
    const packed=runtime.createBuffer(new Uint32Array(Math.ceil(count/(format===1?4:2))).fill(0xffffffff)),reference=runtime.createBuffer(count*4);
    try {
      for(let pass=0;pass<2;pass++){
        if(pass){values.reverse();runtime.write(input,values);}
        dispatch('nr_publish',{input,output:reference},{count,quantize:format===1?1:0},count);
        dispatch('nr_publish_compact',{input,output:packed},{count,quantize:format===1?1:0,inputFormat:0,outputFormat:format},count);
        const expected=await runtime.read(reference),words=await runtime.read(packed,Uint32Array),codes=format===1?new Uint8Array(words.buffer):new Uint16Array(words.buffer);
        let errors=0;for(let i=0;i<count;i++)if(!Object.is(expected[i],format===1?e4(codes[i]):half(codes[i])))errors++;
        check(`Compact activation stores ${format===1?'FP8':'FP16'} pass ${pass+1}: signs, NaNs, dirty words, odd tail`,errors===0,`${errors} mismatches`);
      }
    }finally{for(const b of [input,packed,reference])runtime.destroyBuffer(b);}
  }
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
          if(c.entry==='nr_scores'||c.entry==='nr_attend'){
            runtime.write(bindings[name],new Float32Array(actual.length).fill(777));
            new NeuralRenderer(runtime,kernels,{}).dispatch(c.entry,bindings,c.scalars,c.count);
            const tiled=await runtime.read(bindings[name],Uint32Array);
            check('Tiled attention native parity: '+c.name+'/'+name,tiled.every((v,i)=>v===expected[i]));
          }
          check('Native CUDA parity: '+c.name+'/'+name,mismatches===0,`${actual.length} words, ${mismatches} mismatches${first}`);
        }
      } finally {for(const b of Object.values(bindings))runtime.destroyBuffer(b);}
    }
  } else report.nativeFixtures='Not supplied; run scripts/test-native.ps1';
  const synthetic={cache:new Map(),matrix:(name,offset,K,N,{batches=1})=>new Float32Array(K*N*batches),vector:(name,offset,n)=>new Float32Array(n).fill(1),prior:(name,offset,heads)=>new Float32Array(heads*4096)};
  const engine=new NeuralRenderer(runtime,kernels,synthetic,{attentionMode:'tiled',workspaceCacheBytes:0,activationStorage:'float',executionMode:'streamed'}),boundaries=[];
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
  const persistentModel={...synthetic,packedMatrix:(name,offset,K,N,{batches=1,halfMode})=>new Uint32Array(Math.ceil(K*N*batches/(halfMode?2:4)))};
  const persistent=new NeuralRenderer(runtime,kernels,persistentModel,{attentionMode:'tiled',workspaceCacheBytes:0,executionMode:'streamed'});
  await persistent.run(args);
  const cacheBytes=persistent.weightCacheBytes,cacheSize=persistent.weightCache.size,uploaded=runtime.stats.dataBytesUploaded;
  const warm=await persistent.run(args);
  check('Packed model persists without weight uploads on repeated renders',cacheBytes>0&&persistent.weightCacheBytes===cacheBytes&&persistent.weightCache.size===cacheSize&&runtime.stats.dataBytesUploaded-uploaded===args.inputFeatures.byteLength+4&&warm.head.every((x,i)=>Object.is(x,run.head[i])));
  persistent.clearWeightCache();
  const workspaceEngine=new NeuralRenderer(runtime,kernels,persistentModel,{attentionMode:'tiled',workspaceCacheBytes:1024*1024,executionMode:'streamed'});
  const firstWorkspace=await workspaceEngine.run(args),secondWorkspace=await workspaceEngine.run({...args,profile:true});
  check('Persistent workspace reuses allocations within its budget',workspaceEngine.workspaceBytes>0&&workspaceEngine.workspaceBytes<=1024*1024&&secondWorkspace.workingBuffers.created<firstWorkspace.workingBuffers.created&&secondWorkspace.head.every((v,i)=>Object.is(v,firstWorkspace.head[i])));
  check('GPU profiling reports actual per-dispatch timestamps or explicit unsupported status',secondWorkspace.profile.supported?secondWorkspace.profile.dispatches.length===653&&secondWorkspace.profile.dispatches.every(r=>Number.isFinite(r.gpuMs)&&r.gpuMs>=0):!!secondWorkspace.profile.reason);
  const resizeArgs={...args,width:3,height:3,inputFeatures:new Float32Array(144),geometryOverride:{...args.geometryOverride,width:3,height:3,fullWidth:3,fullHeight:3}};
  const resized=await workspaceEngine.run(resizeArgs);
  check('Workspace resets on resolution changes and produces the correct output shape',resized.head.length===36&&resized.head.every(v=>v===0)&&workspaceEngine.workspaceBytes<=1024*1024);
  const cancelWorkspace=new AbortController();let workspaceCancelled=false;
  try{await workspaceEngine.run({...args,signal:cancelWorkspace.signal,onProgress:({index})=>{if(index===3)cancelWorkspace.abort();}});}catch(e){workspaceCancelled=e.name==='AbortError';}
  const resumedWorkspace=await workspaceEngine.run(args);
  check('Workspace remains valid after discarding cancelled work',workspaceCancelled&&resumedWorkspace.head.every((v,i)=>Object.is(v,firstWorkspace.head[i])));
  workspaceEngine.clearWorkspace();workspaceEngine.clearWeightCache();
  check('Explicit workspace release frees all cached GPU resources',workspaceEngine.workspaceBytes===0&&workspaceEngine.workspace.size===0&&runtime.buffers.size===resourcesBefore);
  check('Clearing packed cache releases all model buffers',persistent.weightCacheBytes===0&&persistent.weightCache.size===0&&runtime.buffers.size===resourcesBefore);
  const cancelPacked=new AbortController();let packedCancelled=false;
  try{await persistent.run({...args,signal:cancelPacked.signal,onProgress:({index})=>{if(index===3)cancelPacked.abort();}});}catch(e){packedCancelled=e.name==='AbortError';}
  const partialSize=persistent.weightCache.size;
  const recovered=await persistent.run(args);
  check('Cancelled packed cache can resume without stale resources',packedCancelled&&partialSize>0&&persistent.weightCache.size===cacheSize&&recovered.head.every((x,i)=>Object.is(x,run.head[i])));
  persistent.clearWeightCache();
  const prepared=new NeuralRenderer(runtime,kernels,persistentModel,{workspaceCacheBytes:1024*1024});
  const preparedFirst=await prepared.run(args),beforeWarmBinds=runtime.stats.bindGroupsCreated;
  const preparedWarm=await prepared.run(args);
  check('Prepared packed graph reuses every graph allocation and binding',preparedWarm.workingBuffers.prepared&&preparedWarm.workingBuffers.created===0&&runtime.stats.bindGroupsCreated===beforeWarmBinds&&preparedWarm.head.every((v,i)=>Object.is(v,run.head[i])));
  const changed=await prepared.run({...resizeArgs,profile:true});
  check('Prepared graph resizes safely and supports timestamp profiling',changed.head.length===36&&changed.head.every(v=>v===0)&&(!changed.profile.supported||changed.profile.dispatches.length===529));
  const cancelPlan=new AbortController();let planCancelled=false;
  try{await prepared.run({...resizeArgs,signal:cancelPlan.signal,onProgress:({index})=>{if(index===3)cancelPlan.abort();}});}catch(e){planCancelled=e.name==='AbortError';}
  const resumedPlan=await prepared.run(resizeArgs);
  check('Prepared graph resumes after a partial batch is cancelled',planCancelled&&resumedPlan.head.every(v=>v===0));
  prepared.clearWorkspace();prepared.clearWeightCache();
  check('Clearing the prepared graph releases buffers and cached bindings',prepared.plan===null&&runtime.buffers.size===resourcesBefore);
  const fallback=new NeuralRenderer(runtime,kernels,persistentModel,{attentionMode:'tiled',workspaceCacheBytes:0,planCacheBytes:0});
  const fallbackRun=await fallback.run(args);fallback.clearWeightCache();
  check('Plan budget falls back to streamed execution without changing output',!fallbackRun.workingBuffers.prepared&&fallbackRun.head.every((v,i)=>Object.is(v,run.head[i]))&&runtime.buffers.size===resourcesBefore);
} catch(error) {check('GPU execution',false,String(error.stack||error));}
finally {runtime?.dispose();window.report=report;document.querySelector('#result').textContent=JSON.stringify(report,null,2);}
