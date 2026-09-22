import test from 'node:test';
import assert from 'node:assert/strict';
import {dispatchGrid} from '../src/kernel-selection.js';

test('Dispatch grid balances large workloads without a mostly empty final row',()=>{
  assert.deepEqual(dispatchGrid(69120,65535),[34560,2,1]);
  assert.deepEqual(dispatchGrid(276480,65535),[55296,5,1]);
  assert.deepEqual(dispatchGrid(65535,65535),[65535,1,1]);
  for(const limit of [1,7,128,65535])for(const groups of new Set([1,limit,Math.min(limit+1,limit*limit),limit*limit-1,limit*limit])){
    if(groups<1)continue;
    const [x,y,z]=dispatchGrid(groups,limit);
    assert.ok(x<=limit&&y<=limit);assert.equal(z,1);
    assert.ok(x*y>=groups&&x*y-groups<y);
  }
  for(const [groups,limit] of [[0,7],[50,7],[1,0],[Infinity,7],[1.5,7]])assert.throws(()=>dispatchGrid(groups,limit),/device limits/);
});
