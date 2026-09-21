import test from 'node:test';
import assert from 'node:assert/strict';
import {weightResource,parseWeights,modelFromDll} from '../src/dll-model.js';

function blob(names) {
  const records=names.map(name=>{
    const text=Buffer.from(name),b=Buffer.alloc(8+text.length+8+42);let p=0;
    b.writeBigUInt64LE(BigInt(text.length),p);p+=8;text.copy(b,p);p+=text.length;
    b.writeBigUInt64LE(42n,p);p+=8;b.writeBigUInt64LE(42n,p);b.writeBigUInt64LE(2n,p+8);b.writeUInt32LE(1,p+16);
    b.writeUInt16LE(0x3c00,p+20);b.writeBigUInt64LE(1n,p+30);b.writeUInt32LE(1,p+38);return b;
  });
  const b=Buffer.concat([Buffer.alloc(8),...records]);b.writeBigUInt64LE(BigInt(b.length));return b;
}
const names=[...Array.from({length:71},(_,i)=>`block${i}.layer0.layer`),...Array.from({length:82},(_,i)=>`block0.layer${i+1}.layer`)];
function pe(resource) {
  const b=Buffer.alloc(1024+resource.length),opt=0x98;
  b.write('MZ');b.writeUInt32LE(0x80,60);b.writeUInt32LE(0x4550,0x80);b.writeUInt16LE(0x8664,0x84);b.writeUInt16LE(1,0x86);b.writeUInt16LE(240,0x94);
  b.writeUInt16LE(0x20b,opt);b.writeUInt32LE(16,opt+108);b.writeUInt32LE(0x1000,opt+128);b.writeUInt32LE(b.length-512,opt+132);
  const s=opt+240;b.write('.rsrc',s);b.writeUInt32LE(0x1000,s+12);b.writeUInt32LE(b.length-512,s+16);b.writeUInt32LE(512,s+20);
  const root=512;
  b.writeUInt16LE(1,root+14);b.writeUInt32LE(10,root+16);b.writeUInt32LE(0x80000020,root+20);
  b.writeUInt16LE(1,root+32+12);b.writeUInt32LE(0x80000080,root+48);b.writeUInt32LE(0x80000040,root+52);
  b.writeUInt16LE(1,root+64+14);b.writeUInt32LE(1033,root+80);b.writeUInt32LE(96,root+84);
  b.writeUInt32LE(0x1200,root+96);b.writeUInt32LE(resource.length,root+100);
  const name=Buffer.from('WEIGHTS_HT','utf16le');b.writeUInt16LE(name.length/2,root+128);name.copy(b,root+130);resource.copy(b,1024);return b;
}
test('Browser PE parser finds embedded weights and preserves payload bytes',()=>{
  const data=blob(names),dll=pe(data),resource=weightResource(dll),records=parseWeights(resource);
  assert.deepEqual(new Uint8Array(resource),new Uint8Array(data));assert.equal(records.length,153);assert.equal(records[0].name,names[0]);assert.deepEqual([...records[0].bytes],[0,60]);
});
test('DLL import rejects non-PE files, invalid resource bounds, cycles, and missing NR resource',()=>{
  assert.throws(()=>weightResource(new Uint8Array(64)),/not a Windows DLL/);
  const dll=pe(blob(names));dll.writeUInt32LE(0x80000000,532);assert.throws(()=>weightResource(dll),/tree/);
  const invalid=pe(blob(names));invalid.writeUInt32LE(0xffffffff,512+100);assert.throws(()=>weightResource(invalid),/outside/);
  const wrong=pe(blob(names));wrong.writeUInt32LE(11,528);assert.throws(()=>weightResource(wrong),/WEIGHTS_HT/);
});
test('Weight parser rejects truncation, duplicate names, and incompatible tensor count',()=>{
  assert.throws(()=>parseWeights(blob(names).subarray(0,50)),/length/);
  assert.throws(()=>parseWeights(blob([...names,names[0]])),/duplicate/);
  assert.throws(()=>parseWeights(blob(names.slice(0,71))),/153-tensor/);
});
test('DLL model initialization validates the real graph before accepting a model',async()=>{
  const bytes=pe(blob(names));
  await assert.rejects(modelFromDll({name:'test.dll',size:bytes.length,arrayBuffer:async()=>bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.length)}),/Tensor bounds/);
});
