import {geometry,layout,phases,align} from './geometry.js';
// Host scheduling only. Every numerical operation is compiled from kernels/*.cu.
export function createGraph(width,height,{geometryOverride,activationStorage='float',fuseLocalAttention=false}={}) {
  const g=geometryOverride||geometry(width,height),ops=[],resources=new Map();let serial=0;
  const phasesUsed=Array(7).fill(0),boundaries={};
  function resource(rows,channels,label) {const id=`${label}:${serial++}`;resources.set(id,{rows,channels,bytes:rows*channels*4});return id;}
  function emit(entry,bindings,scalars,count,label) {ops.push({entry,bindings,scalars,count,label});}
  const tensor=(b,l=0,p='layer')=>`block${b}.layer${l}.${p}`;
  const vector=(name,offset,count,type='half')=>({kind:'vector',name,offset,count,type});
  function gemm(input,name,offset,K,N,{batches=1,broadcast=false,halfMode=0,silu=0,residual=null,scale=null,partition=0,quantize=1,label='gemm'}={}) {
    const {rows,channels}=resources.get(input),raw=resource(rows,N*batches,label+'/raw'),out=resource(rows,N*batches,label+'/out');
    emit('nr_gemm',{input,weights:{kind:'matrix',name,offset,K,N,batches,halfMode:!!halfMode},residual:residual||{kind:'zero'},scales:scale||{kind:'zero'},raw,output:out},
      {rows,K,N,batches,inputStride:channels,inputBatchStride:broadcast?0:K,partition,halfMode,silu,hasResidual:residual?1:0,quantize},rows*N*batches,label);
    return {raw,out};
  }
  function attention(ffn,name,qOffset,priorOffset,scaleOffset,c,w,h,level,globalMode=0) {
    const rows=w*h,heads=c/32,keys=globalMode?align(rows,64):64;
    const [shiftX,shiftY]=globalMode?[0,0]:phases[phasesUsed[level]++%4];
    const q=gemm(ffn,name,qOffset,c,c*3,{quantize:0,partition:globalMode?512:0,label:'qkv'}).out;
    const norm=resource(rows,c*3,'normalized'),scores=resource(rows,heads*keys,'scores'),weights=resource(rows,heads*keys,'softmax'),inverse=resource(rows,heads,'reciprocal'),out=resource(rows,c,'attended');
    emit('nr_normalize',{qkv:q,scales:vector(name,scaleOffset,heads,'float'),output:norm},{rows,heads,globalMode},rows*heads,'normalize');
    emit('nr_scores',{qkv:norm,prior:globalMode?{kind:'zero'}:{kind:'prior',name,offset:priorOffset,heads},scores},{width:w,height:h,heads,shiftX,shiftY,padded:keys,globalMode},rows*heads*keys,'scores');
    emit('nr_softmax',{scores,weights,inverse},{rows,heads,keys,globalMode},rows*heads,'softmax');
    emit('nr_attend',{qkv:norm,weights,inverse,output:out},{width:w,height:h,heads,shiftX,shiftY,keys,globalMode},rows*c,'attend');
    return out;
  }
  function block(state,b,c,w,h,level,l=layout(c),inputRaw=null) {
    const t=tensor(b);let ffn,attended,result;
    if(c===1024) {
      const e=gemm(state,t,0,c,4096,{silu:1,label:`block-${b}/expand`});
      ffn=gemm(e.out,tensor(b,1),0,4096,c,{residual:state,scale:vector(tensor(b,1),4096*c,c),partition:1024,label:'ffn'});
      attended=attention(ffn.out,tensor(b,2),128,0,0,c,w,h,level,1);
      result=gemm(attended,tensor(b,4),0,c,c,{residual:ffn.out,scale:vector(tensor(b,4),c*c,c),partition:256,label:`block-${b}`});
    } else if(c===512) {
      const a=gemm(state,t,0,c,c);
      const e=gemm(a.out,t,c*c,64,256,{batches:8,silu:1});
      const n=gemm(e.out,t,c*c+8*64*256,256,64,{batches:8});
      ffn=gemm(n.out,tensor(b,1),0,c,c,{residual:state,scale:vector(tensor(b,1),c*c,c)});
      attended=attention(ffn.out,tensor(b,2),0,3*c*c,3*c*c+16*8192,c,w,h,level);
      result=gemm(attended,tensor(b,3),0,c,c,{residual:ffn.out,scale:vector(tensor(b,3),c*c,c),label:`block-${b}`});
    } else {
      const batches=c/32;
      const e=gemm(state,t,l.expand,c,128,{batches,broadcast:true,silu:1});
      if(c===32) ffn=gemm(e.out,t,l.contract,128,c,{residual:inputRaw||state,scale:vector(t,l.ffn,c)});
      else {
        const n=gemm(e.out,t,l.contract,128,32,{batches});
        ffn=gemm(n.out,t,l.contract+batches*128*32,c,c,{residual:state,scale:vector(t,l.ffn,c)});
      }
      attended=attention(ffn.out,t,l.qkv,l.prior,l.scale,c,w,h,level);
      result=gemm(attended,t,l.projection,c,c,{residual:c===32?ffn.raw:ffn.out,scale:vector(t,l.attn,c),label:`block-${b}`});
    }
    boundaries[`block-${b}`]=result.out;
    Object.assign(resources.get(result.out),{width:w,height:h});
    return result;
  }
  function pool(raw,c,iw,ih,ow,oh) {const out=resource(ow*oh,c,'pool');emit('nr_pool',{input:raw,output:out},{iw,ih,ow,oh,channels:c},ow*oh*c,'pool');return out;}
  function merge(low,skip,name,offset,c,iw,ow,oh,{post=0,inputOffset=0}={}) {
    const out=resource(ow*oh,c,'merge'),raw=resource(ow*oh,c,'merge/raw');
    emit('nr_merge',{low,skip,scaleA:post?vector(name,inputOffset,c):{kind:'zero'},scaleB:vector(name,offset,c),raw,output:out},{iw,ow,oh,channels:c,post},ow*oh*c,'merge');return {out,raw};
  }
  const features=resource(g.fullWidth*g.fullHeight,16,'features'),rounded=resource(g.fullWidth*g.fullHeight,16,'rounded');
  emit('nr_publish',{input:features,output:rounded},{count:g.fullWidth*g.fullHeight*16,quantize:0},g.fullWidth*g.fullHeight*16,'input-half');
  const pre=layout(32,'pre'),projected=gemm(rounded,tensor(0),pre.adapter,16,32,{halfMode:1});
  const adapter=block(projected.out,0,32,g.fullWidth,g.fullHeight,6,pre,projected.raw);
  const stageDefs=[[1,4,32],[5,8,64],[9,14,128],[15,22,256],[23,30,512]];
  const skips=[];let state=pool(adapter.raw,32,g.fullWidth,g.fullHeight,g.levels[0].width,g.levels[0].height);
  for(let level=0;level<5;level++) {
    const [first,last,c]=stageDefs[level],{width:w,height:h}=g.levels[level];let result;
    for(let b=first;b<=last;b++){result=block(state,b,c,w,h,level);state=result.out;}
    skips[level]=state;
    const next=g.levels[level+1],pooled=pool(result.raw,c,w,h,next.width,next.height);
    state=gemm(pooled,tensor(last,c===512?4:0),c===512?0:layout(c).end,c,c*2).out;
    boundaries[`transition-${last}-${last+1}`]=state;
    Object.assign(resources.get(state),next);
  }
  const bottom=g.levels[5];
  for(let b=31;b<=38;b++)state=block(state,b,1024,bottom.width,bottom.height,5).out;
  let p=gemm(state,tensor(39),0,1024,512,{partition:256,quantize:0}).out;
  const d4=g.levels[4];state=merge(p,skips[4],tensor(39),1024*512,512,bottom.width,d4.width,d4.height).out;
  boundaries['block-39']=state;
  Object.assign(resources.get(state),d4);
  for(let b=40;b<=47;b++)state=block(state,b,512,d4.width,d4.height,4).out;
  for(const [first,last,level,c] of [[48,55,3,256],[56,61,2,128],[62,65,1,64],[66,69,0,32]]) {
    const low=g.levels[level+1],high=g.levels[level],l=layout(c,'up'),t=tensor(first);
    p=gemm(state,t,l.up,c*2,c,{quantize:0}).out;
    const merged=merge(p,skips[level],t,l.transition,c,low.width,high.width,high.height);state=merged.out;
    for(let b=first;b<=last;b++)state=block(state,b,c,high.width,high.height,level,b===first?l:layout(c),b===first&&c===32?merged.raw:null).out;
  }
  const post=layout(32,'post');
  const merged=merge(state,adapter.out,tensor(70),post.adapterScale,32,g.levels[0].width,g.fullWidth,g.fullHeight,{post:1,inputOffset:post.inputScale});
  const last=block(merged.out,70,32,g.fullWidth,g.fullHeight,6,post,merged.raw);
  const head=gemm(last.raw,tensor(70),post.head,32,4,{halfMode:1,quantize:0,label:'head'}).out;
  delete boundaries['block-70'];
  if(fuseLocalAttention)for(let i=0;i<ops.length;i++) {
    const op=ops[i];
    if(op.entry!=='nr_scores'||op.scalars.globalMode)continue;
    const softmax=ops[i+1],attend=ops[i+2];
    for(const id of [op.bindings.scores,softmax.bindings.weights,softmax.bindings.inverse])resources.delete(id);
    ops.splice(i,3,{entry:'nr_local_attention',bindings:{qkv:op.bindings.qkv,prior:op.bindings.prior,output:attend.bindings.output},scalars:Object.fromEntries(Object.entries(op.scalars).filter(([key])=>key!=='padded'&&key!=='globalMode')),count:attend.count,label:'local-attention'});
  }
  if(!['float','packed'].includes(activationStorage))throw Error('Invalid activation storage mode.');
  for(const r of resources.values())r.format=0;
  if(activationStorage==='packed') {
    const mark=(id,format)=>{const r=resources.get(id);r.format=format;r.bytes=Math.ceil(r.rows*r.channels*(format===1?1:2)/4)*4;};
    const readers=new Set([head]);
    for(const op of ops)for(const [key,id] of Object.entries(op.bindings))if(typeof id==='string'&&!['output','raw','scores','weights','inverse'].includes(key))readers.add(id);
    // Softmax and attention read scores/weights/inverse under those same names.
    for(const op of ops)if(op.entry==='nr_softmax')readers.add(op.bindings.scores);else if(op.entry==='nr_attend'){readers.add(op.bindings.weights);readers.add(op.bindings.inverse);}
    for(const op of ops) {
      const b=op.bindings;
      if(b.raw){mark(b.raw,2);op.scalars.rawEnabled=readers.has(b.raw)?1:0;if(!op.scalars.rawEnabled)resources.get(b.raw).bytes=4;}
      if(op.entry==='nr_scores')mark(b.scores,2);
      else if(op.entry==='nr_softmax'){mark(b.weights,1);mark(b.inverse,2);}
      else if(b.output)mark(b.output,op.entry==='nr_publish'||op.entry==='nr_gemm'&&!op.scalars.quantize?2:1);
    }
  }
  return {geometry:g,ops,resources,features,head,boundaries};
}
