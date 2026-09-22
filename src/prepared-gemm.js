// Experimental backends share one model layout and a fixed 32 x 32 output tile.
export const preparedBackends=Object.freeze({
  'prepared-half':Object.freeze({suffix:'_prepared_half',sharedBytes:8704,threads:128,requiresHalf:true}),
  'prepared-integer':Object.freeze({suffix:'_prepared_integer',sharedBytes:2432,threads:128,requiresHalf:false})
});
export function preparedBackendForEntry(entry){
  return Object.entries(preparedBackends).find(([,backend])=>entry.endsWith(backend.suffix))?.[0];
}
export function supportsPreparedBackend(backend,device){
  const spec=preparedBackends[backend],limits=device?.limits;
  return !!spec&&(!spec.requiresHalf||device?.features?.has('shader-f16'))&&
    limits?.maxComputeWorkgroupStorageSize>=spec.sharedBytes&&
    limits?.maxComputeInvocationsPerWorkgroup>=spec.threads&&limits?.maxComputeWorkgroupSizeX>=spec.threads;
}
export function preparedGemmEntry(base,backend){return base&&preparedBackends[backend]?base+preparedBackends[backend].suffix:null;}
