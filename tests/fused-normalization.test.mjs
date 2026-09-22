import test from 'node:test';
import assert from 'node:assert/strict';
import {createGraph} from '../src/graph.js';
import {dispatchGroups} from '../src/kernel-selection.js';
test('Fused local normalization removes only the 62 local normalization tensors and passes',()=>{
 const base=createGraph(1280,720,{activationStorage:'packed',fuseLocalAttention:true});
 const fused=createGraph(1280,720,{activationStorage:'packed',fuseLocalAttention:true,fuseNormalization:true});
 assert.equal(base.ops.length-fused.ops.length,62);assert.equal(base.resources.size-fused.resources.size,62);
 assert.equal(fused.ops.filter(o=>o.entry==='nr_normalize').length,8);
 assert.equal(fused.ops.filter(o=>o.entry==='nr_local_attention_normalized').length,62);
 for(const op of fused.ops.filter(o=>o.entry==='nr_local_attention_normalized')){assert.equal(fused.resources.get(op.bindings.qkv).format,2);assert.equal(fused.resources.get(op.bindings.output).format,1);assert.equal(dispatchGroups('nr_local_attention',op.scalars,op.count),4*dispatchGroups(op.entry,op.scalars,op.count));}
 assert.deepEqual(fused.boundaries,base.boundaries);
 assert.throws(()=>createGraph(128,96,{fuseNormalization:true}),/requires packed local attention/);
});
