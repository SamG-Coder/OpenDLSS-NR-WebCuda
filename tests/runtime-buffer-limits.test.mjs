import test from 'node:test';
import assert from 'node:assert/strict';
import {GpuRuntime} from '../vendor/webcuda/runtime/runtime.js';

async function requested(options={},maximum=2147483644){
 let descriptor;const stop=new Error('Captured request');
 const adapter={features:new Set(),limits:{maxStorageBuffersPerShaderStage:16,maxComputeInvocationsPerWorkgroup:1024,maxComputeWorkgroupSizeX:1024,maxStorageBufferBindingSize:maximum,maxBufferSize:maximum+4},requestDevice:async d=>{descriptor=d;throw stop;}};
 // The sentinel stops before creating GPU resources; tests the real request path.
 const original=Object.getOwnPropertyDescriptor(globalThis,'navigator');
 Object.defineProperty(globalThis,'navigator',{value:{gpu:{}},configurable:true});
 try{await assert.rejects(GpuRuntime.create({adapter,...options}),e=>e===stop);return descriptor.requiredLimits;}
 finally{if(original)Object.defineProperty(globalThis,'navigator',original);else delete globalThis.navigator;}
}
test('Runtime retains its conservative buffer request unless explicitly opted in',async()=>{
 const limits=await requested();assert.equal(limits.maxStorageBufferBindingSize,256*1024*1024);assert.equal(limits.maxBufferSize,256*1024*1024);
});
test('Large-buffer opt-in requests actual supported limits, including smaller adapters',async()=>{
 const limits=await requested({useAdapterBufferLimits:true});assert.equal(limits.maxStorageBufferBindingSize,2147483644);assert.equal(limits.maxBufferSize,2147483648);
 assert.equal((await requested({useAdapterBufferLimits:true},128*1024*1024)).maxStorageBufferBindingSize,128*1024*1024);
});
test('Large-buffer opt-in rejects invalid options',async()=>{
 await assert.rejects(GpuRuntime.create({useAdapterBufferLimits:'yes'}),/must be a boolean/);
});
