import {gemmConstants} from '../src/gemm-specialization.js';

// Specialize CUDA before compilation. Never patch generated WGSL.
export function specializeGemmSource(source,entry,name,scalars) {
  const start=source.indexOf('__global__ void '+entry+'(');
  if(start<0)throw Error('Missing GEMM entry '+entry);
  const signatureEnd=source.indexOf(')',start),bodyStart=source.indexOf('{',signatureEnd);
  let depth=1,end=bodyStart+1;
  for(;depth&&end<source.length;end++){if(source[end]==='{')depth++;if(source[end]==='}')depth--;}
  if(depth)throw Error('Unbalanced GEMM source');
  const params=source.slice(source.indexOf('(',start)+1,signatureEnd).split(',').map(s=>s.trim());
  let body=source.slice(bodyStart,end);
  for(const key of gemmConstants){
    const declaration=params.find(p=>p.split(/\s+/).at(-1)===key);
    if(!declaration||!Number.isSafeInteger(scalars[key])||scalars[key]<0)throw Error('Invalid GEMM constant '+key);
    const value=String(scalars[key])+(declaration.startsWith('unsigned')?'u':'');
    body=body.replace(new RegExp('\\b'+key+'\\b','g'),value);
  }
  const remaining=params.filter(p=>!gemmConstants.includes(p.split(/\s+/).at(-1)));
  return source.slice(0,start)+'__global__ void '+name+'('+remaining.join(', ')+') '+body+source.slice(end);
}
