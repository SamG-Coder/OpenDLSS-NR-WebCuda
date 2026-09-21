import {createGraph} from './graph.js';

// Scene/image preview availability is independent of the NR compute device.
// A missing device limit means unknown, not WebGPU's default 128 MiB limit.
export function assessResolution(width,height,storageLimit=null) {
  const graph=createGraph(width,height);
  const requiredBufferBytes=Math.max(...Array.from(graph.resources.values(),r=>r.bytes));
  const inferenceError=storageLimit!==null&&requiredBufferBytes>storageLimit
    ? `NR processing at ${width} × ${height} needs a ${Math.ceil(requiredBufferBytes/1048576)} MiB buffer; this NR device allows ${Math.round(storageLimit/1048576)} MiB per buffer. The source preview is still available. Choose a smaller resolution to run NR.`
    : null;
  return {geometry:graph.geometry,requiredBufferBytes,inferenceError};
}
