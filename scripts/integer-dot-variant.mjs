// Experimental exact fast path. Applied only to isolated native benchmark builds.
export function integerDotVariant(source,{audit=false}={}){
  if(!source.includes('NR_WIDE_A_PAIRS'))return source;
  source=source.replaceAll('\r\n','\n');
  const helper=`
__device__ int nr_dot_quantum(float v){
  unsigned bits=__float_as_uint(v)&0x7fffffffu;
  if(bits==0u)return 30;
  unsigned mantissa=(bits&0x7fffffu)|0x800000u;
  return (int)(bits>>23u)-150+31-__clz(mantissa&(~mantissa+1u));
}
`;
  source=(audit?'__device__ unsigned long long nr_dot_stats[4];\n':'')+helper+source;
  source=source.replace('  unsigned tid=threadIdx.x;',`  __shared__ unsigned dotA[NR_WIDE_ROWS*8];
  __shared__ unsigned dotB[NR_WIDE_COLS*8];
  __shared__ int dotScaleA[NR_WIDE_ROWS*2];
  __shared__ int dotScaleB[NR_WIDE_COLS*2];
  unsigned tid=threadIdx.x;`);
  const pack=`
    // Preserve all original half operands for fallback. The integer representation
    // is lossless or marked ineligible; stored operands already include x4 scaling.
    for(unsigned t=tid;t<(NR_WIDE_ROWS+NR_WIDE_COLS)*2u;t+=NR_WIDE_THREADS){
      bool isA=t<NR_WIDE_ROWS*2u;
      unsigned index=isA?t:t-NR_WIDE_ROWS*2u;
      unsigned base=(index/2u)*(isA?16u:17u)+(index%2u)*8u;
      int q=30;
      for(unsigned j=0;j<8u;j++){
        float2 v=__half22float2(isA?tileA[base+j]:tileB[base+j]);
        q=min(q,min(nr_dot_quantum(v.x),nr_dot_quantum(v.y)));
      }
      if(q==30)q=0;
      bool valid=true;float factor=nr_pow2(-q);
      for(unsigned j=0;j<4u;j++){
        float2 x=__half22float2(isA?tileA[base+j*2u]:tileB[base+j*2u]);
        float2 y=__half22float2(isA?tileA[base+j*2u+1u]:tileB[base+j*2u+1u]);
        int a=(int)(x.x*factor),b=(int)(x.y*factor),c=(int)(y.x*factor),d=(int)(y.y*factor);
        valid=valid&&abs(a)<=127&&abs(b)<=127&&abs(c)<=127&&abs(d)<=127;
        unsigned word=((unsigned)a&255u)|(((unsigned)b&255u)<<8u)|(((unsigned)c&255u)<<16u)|(((unsigned)d&255u)<<24u);
        if(isA)dotA[index*4u+j]=word;else dotB[index*4u+j]=word;
      }
      if(isA)dotScaleA[index]=valid?q:127;else dotScaleB[index]=valid?q:127;
    }
    __syncthreads();
`;
  const staging=/    \/\/ Two ordered native groups share the same staged .* slab\./;
  if(!staging.test(source))throw Error('Integer dot packing insertion point missing');
  source=source.replace(staging,pack+'    // Two ordered native groups share the same staged K slab.');
  const marker='        #pragma unroll\n        for(unsigned j=0u;j<8u;j+=1u){\n          unsigned k=part*8u+j;\n          __half2 a0=tileA';
  if(!source.includes(marker))throw Error('Integer dot insertion point missing');
  let dot='        bool dotEligible=true;\n';
  for(let n=0;n<8;n++){
    const r=n<4?0:1,c=n%4;
    dot+=`        int qa${n}=dotScaleA[(localRow+${r}u)*2u+part],qb${n}=dotScaleB[(tid%NR_WIDE_COLUMN_GROUPS*4u+${c}u)*2u+part];\n        int shift${n}=qa${n}+qb${n}+9-e${n};\n        dotEligible=dotEligible&&qa${n}!=127&&qb${n}!=127&&shift${n}>=0&&shift${n}<=20;\n`;
  }
  if(audit)dot+=`        if(tid==0u){atomicAdd(&nr_dot_stats[0],1ull);if(qa0!=127&&qa4!=127)atomicAdd(&nr_dot_stats[1],1ull);if(qb0!=127&&qb1!=127&&qb2!=127&&qb3!=127)atomicAdd(&nr_dot_stats[2],1ull);if(dotEligible)atomicAdd(&nr_dot_stats[3],1ull);}\n`;
  dot+='        if(dotEligible){\n';
  for(let n=0;n<8;n++){
    const r=n<4?0:1,c=n%4;
    dot+=`          int sum${n}=0;\n          #pragma unroll\n          for(unsigned j=0u;j<4u;j++)sum${n}=__dp4a((int)dotA[((localRow+${r}u)*2u+part)*4u+j],(int)dotB[((tid%NR_WIDE_COLUMN_GROUPS*4u+${c}u)*2u+part)*4u+j],sum${n});\n          sums${r}.${'xyzw'[c]}+=(float)sum${n}*nr_pow2(shift${n});\n`;
  }
  dot+='        }else{\n';
  source=source.replace(marker,dot+marker);
  source=source.replace('        if(isfinite(acc0))acc0=', '        }\n        if(isfinite(acc0))acc0=');
  return source;
}
