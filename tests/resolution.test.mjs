import test from 'node:test';
import assert from 'node:assert/strict';
import {assessResolution} from '../src/resolution.js';

test('512-square preview is not rejected using a guessed pre-model GPU limit',()=>{
  const result=assessResolution(512,512);
  assert.equal(result.requiredBufferBytes,54*1048576);
  assert.equal(result.inferenceError,null);
});
test('NR eligibility uses the actual device limit and does not invalidate preview geometry',()=>{
  const limited=assessResolution(512,512,32*1048576);
  assert.match(limited.inferenceError,/source preview is still available/);
  assert.equal(limited.geometry.width,512);
  assert.equal(assessResolution(512,512,256*1048576).inferenceError,null);
});
test('720p and 1080p use exact source dimensions with adapter-sized NR buffers',()=>{
 for(const [w,h] of [[1280,720],[1920,1080]]){
  const r=assessResolution(w,h,2147483644);
  assert.equal(r.inferenceError,null);assert.equal(r.geometry.width,w);assert.equal(r.geometry.height,h);
  assert.equal(r.requiredBufferBytes,(w===1280?189:405)*1048576);
 }
});
