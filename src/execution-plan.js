// Offline interval allocation. A slot may only serve disjoint inclusive lifetimes:
// inputs and outputs of one dispatch must never alias, even when an output is dead.
export function planLayout(graph) {
  const ranges=new Map();
  graph.ops.forEach((op,index)=>{
    for(const id of Object.values(op.bindings))if(typeof id==='string') {
      if(!ranges.has(id))ranges.set(id,{id,first:index,last:index,bytes:graph.resources.get(id).bytes});
      else ranges.get(id).last=index;
    }
  });
  ranges.get(graph.features).first=-1;ranges.get(graph.head).last=graph.ops.length;
  const slots=[],assignments=new Map();
  // Largest tensors first avoids growing an early small slot to a later large tensor.
  for(const range of [...ranges.values()].sort((a,b)=>b.bytes-a.bytes||a.first-b.first)) {
    let slot=slots.filter(s=>s.ranges.every(r=>range.last<r.first||range.first>r.last)).sort((a,b)=>a.bytes-b.bytes)[0];
    if(!slot){slot={bytes:range.bytes,ranges:[]};slots.push(slot);}
    slot.ranges.push(range);assignments.set(range.id,slots.indexOf(slot));
  }
  return {slots,assignments,bytes:slots.reduce((sum,s)=>sum+s.bytes,0)};
}
export function createExecutionPlan(runtime,graph,layout) {
  const buffers=[];
  try {
    for(const slot of layout.slots)buffers.push(runtime.createBuffer(slot.bytes));
    const zero=runtime.createBuffer(new Float32Array(1));buffers.push(zero);
    return {graph,zero,buffers,bytes:layout.bytes+4,resources:new Map([...layout.assignments].map(([id,slot])=>[id,buffers[slot]])),invocations:new Map()};
  }catch(error){for(const buffer of buffers)runtime.destroyBuffer(buffer);throw error;}
}
