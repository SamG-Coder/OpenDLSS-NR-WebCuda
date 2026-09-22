import test from 'node:test';
import assert from 'node:assert/strict';
import {e4,half} from '../src/model.js';
import {prepareMatrix} from '../src/model-preparation.js';

// These tests prove the finite-E4 arithmetic transformation independently of
// GPU execution. The GPU suite separately checks the generated CUDA shaders.
const finiteCodes=Array.from({length:256},(_,i)=>i).filter(i=>(i&127)!==127);
const exponent=code=>(code&127)===0?-128:Math.max((code>>>3)&15,1)-7;
const accumulatorExponent=acc=>acc===0?-21:Number.isFinite(acc)?Math.max(Math.floor(Math.log2(Math.abs(acc))),-14):128;
function integerTerm(a,b,E){
  if((a&127)===0||(b&127)===0)return 0;
  const ea=Math.max((a>>>3)&15,1)-7,eb=Math.max((b>>>3)&15,1)-7;
  const ma=((a>>>3)&15?8:0)+(a&7),mb=((b>>>3)&15?8:0)+(b&7),d=E-ea-eb;
  assert.ok(d>=0);
  const magnitude=d<=7?(ma*mb)<<(7-d):d<15?(ma*mb)>>>(d-7):0;
  return magnitude===0?0:(a^b)&128?-magnitude:magnitude;
}
const canonicalZero=x=>x===0?0:x;
const floatTerm=(a,b,E)=>canonicalZero(Math.trunc(e4(a)*e4(b)*2**(13-E)));
function pack(codes){
  const words=new Uint32Array(Math.ceil(codes.length/4));
  codes.forEach((code,i)=>{words[i>>>2]|=code<<((i&3)*8);});
  return words;
}
function metadataFor(codes){
  const K=32,N=32,source=Array.from({length:K*N},(_,i)=>codes[(i/N|0)%16]);
  const {words}=prepareMatrix(pack(source),{K,N,batches:1});
  return words[K*N/4]&255;
}
function exactGroupExponent(a,b,acc){
  let E=accumulatorExponent(acc);
  for(let k=0;k<16;k++)E=Math.max(E,exponent(a[k])+exponent(b[k]));
  return E;
}
function shortcutExponent(a,b,acc,metadata){
  const seed=accumulatorExponent(acc),ea=Math.max(...a.map(exponent)),eb=(metadata&15)-6;
  if(metadata&16)return seed;
  if(seed>=ea+eb)return seed;
  if((metadata&96)===96)return Math.max(seed,ea+eb);
  return exactGroupExponent(a,b,acc);
}
// Independent nearest-value half rounding: binary search representable positive
// values, using the low significand bit to resolve exact ties.
function nearestHalf(value){
  if(!Number.isFinite(value)||value===0)return value;
  const sign=value<0?-1:1,x=Math.abs(value);
  if(x>=65520)return sign*Infinity;
  let lo=0,hi=31743;
  while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(half(mid)<=x)lo=mid;else hi=mid-1;}
  if(lo===31743)return sign*half(lo);
  const down=x-half(lo),up=half(lo+1)-x;
  const chosen=down<up||down===up&&!(lo&1)?lo:lo+1;
  return sign*half(chosen);
}
function group(a,b,acc,integer){
  const E=integer?shortcutExponent(a,b,acc,metadataFor(b)):exactGroupExponent(a,b,acc);
  if(!Number.isFinite(acc))return acc;
  let sum=Math.trunc(acc*2**(13-E));
  for(let k=0;k<16;k++)sum+=integer?integerTerm(a[k],b[k],E):floatTerm(a[k],b[k],E);
  assert.ok(Number.isSafeInteger(sum)&&Math.abs(sum)<2**24);
  return nearestHalf(sum*2**(E-13));
}

test('integer FP8 products match truncation for every finite signed pair and legal finite-group exponent',()=>{
  let checked=0;
  for(const a of finiteCodes)for(const b of finiteCodes){
    const minimum=Math.max(-21,exponent(a)+exponent(b));
    for(let E=minimum;E<=16;E++){
      assert.equal(integerTerm(a,b,E),floatTerm(a,b,E),`a=${a}, b=${b}, E=${E}`);checked++;
    }
    // Very large shifts must yield zero instead of wrapping at 32 bits.
    for(const E of [31,32,63,127,128])assert.equal(integerTerm(a,b,E),floatTerm(a,b,E));
  }
  assert.ok(checked>1_000_000,`${checked} exhaustive finite-group cases`);
});

test('negative FP8 products truncate toward zero after shifting their unsigned magnitude',()=>{
  assert.equal(integerTerm(0x81,0x01,-6),-2);
  assert.equal(integerTerm(0x81,0x01,-4),0);
  assert.equal(integerTerm(0x81,0x01,-4),floatTerm(0x81,0x01,-4));
  assert.equal((-1)>>1,-1,'an arithmetic shift of the signed product would be incorrect');
});

test('metadata bound skips only exact exponent scans, including zeros and unmatched maxima',()=>{
  const low=8,high=80;
  const a=Array(16).fill(low),b=Array(16).fill(low);a[0]=high;b[1]=high;
  const trueExponent=exactGroupExponent(a,b,0),upperBound=Math.max(...a.map(exponent))+Math.max(...b.map(exponent));
  assert.ok(upperBound>trueExponent,'independent maxima can belong to different K positions');
  assert.equal(shortcutExponent(a,b,0,metadataFor(b)),trueExponent);
  assert.equal(shortcutExponent(a,b,half(0x7000),metadataFor(b)),accumulatorExponent(half(0x7000)));
  const cases=[Array(16).fill(0),Array(16).fill(128),Array(16).fill(56),Array.from({length:16},(_,i)=>56+(i%8)),Array.from({length:16},(_,i)=>i?56:0),Array.from({length:16},(_,i)=>i?56:128)];
  for(const weights of cases)for(const input of [a,Array(16).fill(0),Array(16).fill(128),Array.from({length:16},(_,i)=>i===0?high:0)])for(const acc of [0,-0,2**-24,-(2**-14),32,65504,Infinity,-Infinity]){
    assert.equal(shortcutExponent(input,weights,acc,metadataFor(weights)),exactGroupExponent(input,weights,acc));
  }
  const zeroAtLargest=Array(16).fill(56);zeroAtLargest[0]=0;
  const onlyFirst=Array(16).fill(0);onlyFirst[0]=high;
  assert.ok((metadataFor(zeroAtLargest)&32)!==0&&(metadataFor(zeroAtLargest)&64)===0);
  assert.equal(shortcutExponent(onlyFirst,zeroAtLargest,0,metadataFor(zeroAtLargest)),-21,'constant nonzero weights are insufficient when zero weights mask the maximum activation');
});

test('ordered 16-term groups retain accumulator rounding, cancellation, residual seed and partition publication',()=>{
  let state=0x93e4d712;
  const next=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state;};
  const seeds=[0,-0,half(1),half(0x8001),half(0x0400),half(0x3555),half(0xb555),65504,-65504,Infinity,-Infinity];
  for(let trial=0;trial<140;trial++)for(const partition of [0,16,32,64]){
    let reference=seeds[trial%seeds.length],integer=reference,referenceTotal=0,integerTotal=0;
    for(let start=0;start<64;start+=16){
      const a=Array.from({length:16},()=>finiteCodes[next()%finiteCodes.length]);
      const b=Array.from({length:16},()=>finiteCodes[next()%finiteCodes.length]);
      if(trial%7===0)b.fill(0);
      if(trial%7===1)b.fill(56);
      if(trial%7===2)for(let k=1;k<16;k+=2){a[k]=a[k-1]^128;b[k]=b[k-1];}
      reference=group(a,b,reference,false);integer=group(a,b,integer,true);
      assert.ok(Object.is(integer,reference),`trial ${trial}, K ${start}, partition ${partition}`);
      if(partition&&(start+16)%partition===0){
        referenceTotal=start<partition?reference:nearestHalf(referenceTotal+reference);
        integerTotal=start<partition?integer:nearestHalf(integerTotal+integer);
        assert.ok(Object.is(integerTotal,referenceTotal));reference=integer=0;
      }
    }
  }
});
