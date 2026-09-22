import test from 'node:test';
import assert from 'node:assert/strict';
import {specializedGemm,dynamicGemmScalars,gemmConstants} from '../src/gemm-specialization.js';
import {specializeGemmSource} from '../scripts/specialize-gemm.mjs';

test('GEMM specialization keeps rows dynamic and distinguishes every fixed argument',()=>{
  const s=Object.fromEntries(gemmConstants.map(k=>[k,0]));s.rows=123;
  const entry='nr_gemm_tile8x8_compact',key=specializedGemm(entry,s);
  assert.equal(specializedGemm(entry,{...s,rows:789}),key);
  for(const field of gemmConstants)assert.notEqual(specializedGemm(entry,{...s,[field]:1}),key);
  assert.equal(specializedGemm(entry,{...s,inputFormat:undefined}),null);
  assert.equal(specializedGemm('nr_gemm_packed_compact',s),null);
  assert.equal(specializedGemm('nr_gemm_multi32x32_compact',s),null);
  assert.deepEqual(dynamicGemmScalars(s),{rows:123});
});

test('CUDA specialization replaces only the selected entry and whole identifiers',()=>{
  const s=Object.fromEntries(gemmConstants.map(k=>[k,2]));
  const source='__device__ int helper(int K) { return K; }\n__global__ void entry(unsigned rows, '+gemmConstants.map(k=>'unsigned '+k).join(', ')+') { unsigned Keep = K; if(rows > N) { Keep += inputStride; } }';
  const result=specializeGemmSource(source,'entry','fixed',s);
  assert.match(result,/helper\(int K\) \{ return K; \}/);
  assert.match(result,/void fixed\(unsigned rows\)/);
  assert.match(result,/unsigned Keep = 2u/);
  assert.match(result,/if\(rows > 2u\)/);
  assert.throws(()=>specializeGemmSource(source,'missing','fixed',s),/Missing/);
  assert.throws(()=>specializeGemmSource(source,'entry','fixed',{...s,K:NaN}),/Invalid/);
});
