import {execFileSync} from 'node:child_process';
import {readFile} from 'node:fs/promises';
const files=execFileSync('git',['ls-files','-z'],{encoding:'utf8'}).split('\0').filter(Boolean);
for(const path of files){
 if(/^(upstream|reference|models|fixtures|examples|build|site)\//i.test(path)||/\.(dll|exe|pdb|lib|safetensors|onnx|pt|pth|bin)$/i.test(path)||/^reports\/.*\.(png|jpg|jpeg)$/i.test(path))throw Error('Private or third-party asset tracked: '+path);
 const data=await readFile(path);
 if(data.length>5*1024*1024)throw Error('Unexpected large tracked file: '+path);
 if(data[0]===0x4d&&data[1]===0x5a)throw Error('Executable binary detected: '+path);
 if(/-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----/.test(data.toString()))throw Error('Private key detected: '+path);
}
console.log(`Public source audit passed: ${files.length} tracked files; no bundled upstream project, reference snapshots, model weights, NVIDIA DLLs, or example media.`);
