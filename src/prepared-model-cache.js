// Prepared weights stay in this browser's origin storage; no network requests.
const DATABASE='opendlss-nr-prepared-models',STORE='matrices',SCHEMA=1;
const encoder=new TextEncoder();
const hex=bytes=>Array.from(new Uint8Array(bytes),b=>b.toString(16).padStart(2,'0')).join('');

export async function sha256(bytes) {
  if(!globalThis.crypto?.subtle)throw Error('SHA-256 is unavailable.');
  return hex(await globalThis.crypto.subtle.digest('SHA-256',bytes));
}

// Include the names, lengths and actual tensor contents, independent of manifest claims.
export async function hashModelTensors(model,onProgress=()=>{}) {
  if(!(model.tensors instanceof Map))throw Error('Model tensor data is unavailable.');
  const names=Array.from(model.tensors.keys()).sort(),digests=[];
  for(let i=0;i<names.length;i++) {
    const name=names[i],bytes=model.tensors.get(name);
    if(typeof name!=='string'||!(bytes instanceof Uint8Array))throw Error('Invalid model tensor data.');
    digests.push([name,bytes.byteLength,await sha256(bytes)]);
    onProgress({phase:'hash',completed:i+1,total:names.length});
  }
  return sha256(encoder.encode(JSON.stringify(digests)));
}

function recordHeader(record) {
  const header=[record.schema,record.formatVersion,record.modelHash,record.matrixKey,
    record.K,record.N,record.batches,record.boundedHalf,record.payloadWordLength,
    record.metadataWordOffset,record.metadataByteLength,record.wordsSha256];
  // Leave existing FP8 records byte-for-byte compatible while authenticating the
  // interpretation of every explicitly named layout.
  if(record.layout!==undefined)header.push(record.layout);
  return header;
}

export async function createPreparedRecord(descriptor,{modelHash,matrixKey,formatVersion,layout}) {
  const {K,N,batches,boundedHalf,payloadWordLength,metadataWordOffset,metadataByteLength,words}=descriptor;
  const record={schema:SCHEMA,formatVersion,modelHash,matrixKey,K,N,batches,boundedHalf,
    payloadWordLength,metadataWordOffset,metadataByteLength,words,
    wordsSha256:await sha256(new Uint8Array(words.buffer,words.byteOffset,words.byteLength))};
  if(layout!==undefined)record.layout=layout;
  record.integritySha256=await sha256(encoder.encode(JSON.stringify(recordHeader(record))));
  return record;
}

export async function validatePreparedRecord(record,{modelHash,matrixKey,formatVersion,K,N,batches,layout}) {
  try {
    if(layout!==undefined&&layout!=='half-operands-v1')return null;
    const values=K*N*batches,precomputed=layout==='half-operands-v1';
    const payloadWordLength=precomputed?values/2:values/4,metadataByteLength=precomputed?values*2:values/16;
    if(!record||record.schema!==SCHEMA||record.formatVersion!==formatVersion||record.modelHash!==modelHash||record.matrixKey!==matrixKey||
      record.layout!==layout||
      record.K!==K||record.N!==N||record.batches!==batches||typeof record.boundedHalf!=='boolean'||
      record.payloadWordLength!==payloadWordLength||record.metadataWordOffset!==payloadWordLength||record.metadataByteLength!==metadataByteLength||
      !(record.words instanceof Uint32Array)||record.words.length!==payloadWordLength+metadataByteLength/4||
      !/^[0-9a-f]{64}$/.test(record.wordsSha256)||!/^[0-9a-f]{64}$/.test(record.integritySha256))return null;
    if(await sha256(encoder.encode(JSON.stringify(recordHeader(record))))!==record.integritySha256)return null;
    if(await sha256(new Uint8Array(record.words.buffer,record.words.byteOffset,record.words.byteLength))!==record.wordsSha256)return null;
    return {words:record.words,boundedHalf:record.boundedHalf,K,N,batches,payloadWordLength,
      metadataWordOffset:payloadWordLength,metadataByteLength};
  } catch {return null;}
}

export async function createPreparedModelCache({indexedDB=globalThis.indexedDB}={}) {
  if(!indexedDB?.open)return null;
  const db=await new Promise(resolve=>{
    let request,finished=false;
    const finish=value=>{if(finished){value?.close();return;}finished=true;clearTimeout(timer);resolve(value);};
    const timer=setTimeout(()=>finish(null),2000);
    try {
      request=indexedDB.open(DATABASE,SCHEMA);
      request.onupgradeneeded=()=>{if(!request.result.objectStoreNames.contains(STORE))request.result.createObjectStore(STORE);};
      request.onsuccess=()=>finish(request.result);
      request.onerror=()=>finish(null);
      request.onblocked=()=>finish(null);
    } catch {finish(null);}
  });
  if(!db)return null;
  let writable=true;
  const transact=(mode,operation)=>new Promise(resolve=>{
    let result=null,tx;
    try {
      tx=db.transaction(STORE,mode);
      const request=operation(tx.objectStore(STORE));
      request.onsuccess=()=>{result=mode==='readonly'?request.result:true;};
      tx.oncomplete=()=>resolve(result);
      tx.onabort=tx.onerror=()=>{if(mode==='readwrite')writable=false;resolve(null);};
    } catch {if(mode==='readwrite')writable=false;resolve(null);}
  });
  db.onversionchange=()=>db.close();
  return {
    get:key=>transact('readonly',store=>store.get(key)),
    set:(key,value)=>writable?transact('readwrite',store=>store.put(value,key)):Promise.resolve(false),
    delete:key=>writable?transact('readwrite',store=>store.delete(key)):Promise.resolve(false),
    clear:()=>transact('readwrite',store=>store.clear()),
    close:()=>db.close(),
    get writable(){return writable;},
  };
}

export async function clearPreparedModelCache() {
  const cache=await createPreparedModelCache();
  if(!cache)return false;
  try {return !!await cache.clear();} finally {cache.close();}
}
