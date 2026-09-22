import test from 'node:test';
import assert from 'node:assert/strict';
import {createGraph} from '../src/graph.js';
import {planLayout,createExecutionPlan} from '../src/execution-plan.js';

test('Prepared slots cover every tensor without overlapping lifetimes or bindings',()=>{
  for(const fuseLocalAttention of [false,true])for(const activationStorage of ['float','packed'])for(const [w,h] of [[128,96],[1280,720],[1920,1080]]) {
    const graph=createGraph(w,h,{activationStorage,fuseLocalAttention}),layout=planLayout(graph);
    assert.equal(layout.assignments.size,graph.resources.size);
    for(const slot of layout.slots)for(const a of slot.ranges) {
      assert(slot.bytes>=graph.resources.get(a.id).bytes);
      for(const b of slot.ranges)if(a!==b)assert(a.last<b.first||b.last<a.first);
    }
    for(const op of graph.ops){const ids=[...new Set(Object.values(op.bindings).filter(v=>typeof v==='string'))];assert.equal(new Set(ids.map(id=>layout.assignments.get(id))).size,ids.length);}
    if(activationStorage==='packed')assert(layout.bytes<1024*1024*1024);
  }
});

test('Only unread raw GEMM/merge outputs omit their full allocation',()=>{
  const graph=createGraph(1280,720,{activationStorage:'packed'});let omitted=0,retained=0;
  for(const [index,op] of graph.ops.entries())if(op.bindings.raw){
    const id=op.bindings.raw,needed=graph.ops.slice(index+1).some(next=>Object.values(next.bindings).includes(id));
    assert.equal(op.scalars.rawEnabled,Number(needed));
    if(needed){retained++;assert(graph.resources.get(id).bytes>4);}else {omitted++;assert.equal(graph.resources.get(id).bytes,4);}
  }
  assert(omitted>100&&retained>0);
});

test('Failed plan allocation releases resources already allocated',()=>{
  let created=0;const destroyed=[];
  const runtime={createBuffer:()=>{if(++created===3)throw Error('out of memory');return created;},destroyBuffer:b=>destroyed.push(b)};
  assert.throws(()=>createExecutionPlan(runtime,{}, {slots:[{bytes:4},{bytes:8},{bytes:16}]}),/out of memory/);
  assert.deepEqual(destroyed,[1,2]);
});
