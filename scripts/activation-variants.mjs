// Produce CUDA storage variants from the same authored arithmetic bodies.
// No WGSL rewriting: WebCuda compiles the generated CUDA normally.
import {activationBindings} from '../src/activation-bindings.js';
function closing(source,start,open,close){let depth=1;for(let i=start+1;i<source.length;i++){if(source[i]===open)depth++;if(source[i]===close&&!--depth)return i;}throw Error('Unbalanced CUDA source');}
export function compactKernel(source,entry,helper){
  const bindings=activationBindings[entry];if(!bindings)return null;
  const name=entry.startsWith('nr_gemm_tile')?'NR_TILE_ENTRY':entry;
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
  return source+'\n'+helper+'\n'+signature+bindings.map(b=>', int '+b+'Format').join('')+(bindings.includes('raw')?', int rawEnabled':'')+') '+body+'\n';
}
