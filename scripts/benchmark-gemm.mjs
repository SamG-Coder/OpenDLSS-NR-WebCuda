import {chromium} from 'playwright';
import {writeFile} from 'node:fs/promises';
import {createServer} from './serve.mjs';
const rowCap=Number(process.env.NR_GEMM_ROWS||4096);
if(!Number.isInteger(rowCap)||rowCap<1||rowCap>65536)throw Error('NR_GEMM_ROWS must be an integer from 1 to 65536.');
const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
try {
  browser=await chromium.launch({headless:true,...(process.env.NR_BROWSER?{executablePath:process.env.NR_BROWSER}:{}),args:['--enable-unsafe-webgpu']});
  const page=await browser.newPage();page.on('console',m=>console.log(m.text()));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const result=await page.evaluate(async({rowCap,ablate})=>{
    const {NeuralRenderer}=await import('/src/engine.js'),{createGraph}=await import('/src/graph.js'),{dispatchGroups,gemmKernel}=await import('/src/kernel-selection.js'),{specializedGemm,dynamicGemmScalars}=await import('/src/gemm-specialization.js'),{GpuProfile}=await import('/src/gpu-profile.js');
    const engine=await NeuralRenderer.create({},{specializeGemm:false}),runtime=engine.runtime,graph=createGraph(1280,720,{activationStorage:'packed',fuseLocalAttention:true});
    const shapes=new Map(),results=[];
    for(const op of graph.ops)if(op.entry==='nr_gemm'&&!op.scalars.halfMode){const s=op.scalars,key=[s.K,s.N,s.batches,s.partition,s.silu,s.hasResidual,s.quantize,graph.resources.get(op.bindings.input).format,typeof op.bindings.residual==='string'?graph.resources.get(op.bindings.residual).format:0].join(':');if(!shapes.has(key))shapes.set(key,op);}
    const genericCandidates=['nr_gemm_tile8x8','nr_gemm_tile8x16','nr_gemm_multi8x32','nr_gemm_multi16x16','nr_gemm_multi16x32','nr_gemm_multi4x32','nr_gemm_multi32x32','nr_gemm_multi16x64'];
    for(const entry of genericCandidates){const full=entry+'_compact';if(!engine.kernels[full])engine.kernels[full]=await runtime.kernel(await(await fetch('/generated/'+full+'.json')).json());}
    const packed=(count,format)=>{const lanes=format===1?4:2,bits=format===1?8:16,a=new Uint32Array(Math.ceil(count/lanes));for(let i=0;i<count;i++)a[Math.floor(i/lanes)]|=(format===1?(24+(i*17)%48)|(i%3?0:128):0x3000+(i%31)|(i%3?0:32768))<<((i%lanes)*bits);return a;};
    try {
      for(const [key,op] of shapes) {
        const s={...op.scalars,rows:Math.min(rowCap,op.scalars.rows),inputFormat:graph.resources.get(op.bindings.input).format,residualFormat:typeof op.bindings.residual==='string'?graph.resources.get(op.bindings.residual).format:0,rawFormat:2,outputFormat:graph.resources.get(op.bindings.output).format};
        const base=gemmKernel(op.scalars),specialized=specializedGemm(base+'_compact',s),candidates=[...genericCandidates];
        if(specialized){if(!engine.kernels[specialized])engine.kernels[specialized]=await runtime.kernel(await(await fetch('/generated/'+specialized+'.json')).json());candidates.push('specialized');}
        if(ablate){
          const full=base+'_compact',name='nr_gemm_fma_diagnostic';
          let source=await(await fetch('/generated/'+full+'.cu')).text();
          const start=source.lastIndexOf('    int e = acc'),end=source.indexOf('    if (partition != 0u',start);
          if(start<0||end<0)throw Error('GEMM arithmetic template changed');
          source=source.slice(0,start)+`    for(unsigned j=0u;j<group;j+=1u) { acc=fmaf(tileA[tid / NR_TILE_COLS * 16u + j],tileB[j * NR_TILE_COLS + tid % NR_TILE_COLS],acc); }
    acc=nr_half(acc);
`+source.slice(end);
          source=source.replace('void '+full+'(','void '+name+'(');
          engine.kernels[name]=await runtime.kernel(source,{entry:name,workgroupSize:[base.endsWith('8x16')?128:64,1,1]});
          candidates.push('fma-diagnostic');
        }
        const count=s.rows*s.N*s.batches;
        const bindings={input:runtime.createBuffer(packed(s.rows*s.inputStride,s.inputFormat)),weights:runtime.createBuffer(packed(s.K*s.N*s.batches,1)),residual:runtime.createBuffer(s.hasResidual?packed(count,s.residualFormat):new Uint32Array(1)),scales:runtime.createBuffer(new Float32Array(s.N*s.batches).fill(.5)),raw:runtime.createBuffer(s.rawEnabled?count*2:4),output:runtime.createBuffer(count*(s.outputFormat===1?1:2))};
        const invocations=new Map();
        const run=(entry,profile)=>{const full=entry==='specialized'?specialized:entry==='fma-diagnostic'?'nr_gemm_fma_diagnostic':entry+'_compact';let invocation=invocations.get(full);if(!invocation){invocation=engine.kernels[full].bind(bindings,entry==='specialized'?dynamicGemmScalars(s):s);invocations.set(full,invocation);}const groups=dispatchGroups(['specialized','fma-diagnostic'].includes(entry)?base:entry,s,count),commands=profile?.batch(entry,s)??runtime.batch();commands.dispatch(invocation,[Math.min(groups,65535),Math.ceil(groups/65535),1]).submit();};
        try {
          run('nr_gemm_packed');const expected=await runtime.read(bindings.output,Uint32Array),expectedRaw=s.rawEnabled?await runtime.read(bindings.raw,Uint32Array):null;
          const profile=new GpuProfile(runtime);if(!profile.supported)throw Error('GEMM benchmark requires GPU timestamp queries.');
          try {
            for(const entry of candidates){run(entry);if(entry==='fma-diagnostic'){await runtime.idle();continue;}const actual=await runtime.read(bindings.output,Uint32Array),raw=s.rawEnabled?await runtime.read(bindings.raw,Uint32Array):null;if(!actual.every((v,i)=>v===expected[i])||raw&&!raw.every((v,i)=>v===expectedRaw[i]))throw Error('GEMM mismatch: '+key+' '+entry);}
            // Rotate order across samples so one variant is not always first.
            for(let repeat=0;repeat<3;repeat++)for(let i=0;i<candidates.length;i++)run(candidates[(i+repeat)%candidates.length],profile);
            await runtime.idle();const timings=await profile.read(),times={};
            for(const entry of candidates){const samples=timings.dispatches.filter(d=>d.entry===entry).map(d=>d.gpuMs).sort((a,b)=>a-b);times[entry]=samples[1];}
            results.push({key,scalars:s,originalRows:op.scalars.rows,times});console.log(JSON.stringify({key,rows:s.rows,times}));
          }finally{profile.dispose();}
        }finally{for(const b of Object.values(bindings))runtime.destroyBuffer(b);}
      }
      return {rowCap,ablate,diagnosticWarning:ablate?'FMA diagnostic changes arithmetic, is not parity checked, and cannot be used for inference.':undefined,adapter:runtime.describe(),results};
    }finally{engine.dispose();}
  },{rowCap,ablate:process.env.NR_ABLATE==='1'});
  await writeFile(process.env.NR_REPORT||'reports/gemm-shapes.json',JSON.stringify(result,null,2));
}finally{await browser?.close();await new Promise(r=>server.close(r));}
