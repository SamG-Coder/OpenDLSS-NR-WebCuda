// Produce CUDA storage variants from the same authored arithmetic bodies.
// No WGSL rewriting: WebCuda compiles the generated CUDA normally.
import {activationBindings} from '../src/activation-bindings.js';
function closing(source,start,open,close){let depth=1;for(let i=start+1;i<source.length;i++){if(source[i]===open)depth++;if(source[i]===close&&!--depth)return i;}throw Error('Unbalanced CUDA source');}
export function compactKernel(source,entry,helper){
  source=source.replace(/\r\n?/g,'\n');
  const bindings=activationBindings[entry];if(!bindings)return null;
  const name=entry.startsWith('nr_gemm_multi')?'NR_MULTI_ENTRY':entry.startsWith('nr_gemm_tile')?'NR_TILE_ENTRY':entry;
  const start=source.indexOf('__global__ void '+name+'(');if(start<0)throw Error('Missing CUDA entry '+entry);
  const paramsStart=source.indexOf('(',start),paramsEnd=closing(source,paramsStart,'(',')'),bodyStart=source.indexOf('{',paramsEnd),bodyEnd=closing(source,bodyStart,'{','}');
  let signature=source.slice(start,paramsEnd).replace('void '+name+'(','void '+entry+'_compact('),body=source.slice(bodyStart,bodyEnd+1);
  for(const binding of bindings){
    signature=signature.replace('float* '+binding,'unsigned* '+binding);
    const re=new RegExp('\\b'+binding+'\\[','g');let result='',cursor=0,match;
    while((match=re.exec(body))){
      const bracket=match.index+binding.length,end=closing(body,bracket,'[',']'),index=body.slice(bracket+1,end),after=end+1;
      result+=body.slice(cursor,match.index);const assignment=/^\s*=([^=])/.exec(body.slice(after));
      if(assignment){const equal=body.indexOf('=',after),semicolon=body.indexOf(';',equal);if(semicolon<0)throw Error('Missing store terminator');result+=`${binding==='raw'?'if(rawEnabled != 0) ':''}nr_activation_store(${binding}, ${index}, ${body.slice(equal+1,semicolon).trim()}, ${binding}Format)`;cursor=semicolon;}
      else {result+=`nr_activation_load(${binding}, ${index}, ${binding}Format)`;cursor=after;}
      re.lastIndex=cursor;
    }
    body=result+body.slice(cursor);
  }
  if(entry.startsWith('nr_gemm_multi')) {
    const start=body.indexOf('  if (partition != 0u) acc0');
    if(start<0)throw Error('Multi-output GEMM publication template changed');
    const tail=body.slice(start,-1);
    const finalValues=[0,1,2,3].map(i=>`    if (partition != 0u) acc${i} = total${i};\n    if (silu != 0) acc${i} = nr_silu(acc${i});`).join('\n');
    body=body.slice(0,start)+`  if(row < rows && col + 3u < N && (i & 3u) == 0u) {\n${finalValues}\n    if(rawEnabled != 0) nr_activation_store4(raw, i, acc0, acc1, acc2, acc3, rawFormat);\n    nr_activation_store4(output, i, quantize != 0 ? nr_quant(acc0) : acc0, quantize != 0 ? nr_quant(acc1) : acc1, quantize != 0 ? nr_quant(acc2) : acc2, quantize != 0 ? nr_quant(acc3) : acc3, outputFormat);\n  } else {\n${tail}  }\n}`;
  }
  if(entry==='nr_local_attention') {
    const stores=[0,1,2,3].map(i=>`nr_activation_store(output, base${i?' + '+i+'u':''}, queries[source${i?' + '+i+'u':''}], outputFormat);`).join('\n      ');
    if(!body.includes(stores))throw Error('Fused attention packed-store template changed');
    body=body.replace(stores,'nr_activation_store4(output, base, queries[source], queries[source + 1u], queries[source + 2u], queries[source + 3u], outputFormat);');
  }
  return source+'\n'+helper+'\n'+signature+bindings.map(b=>', int '+b+'Format').join('')+(bindings.includes('raw')?', int rawEnabled':'')+') '+body+'\n';
}
