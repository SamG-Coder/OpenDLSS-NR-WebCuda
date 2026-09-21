import http from 'node:http';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(fileURLToPath(new URL('../',import.meta.url)));
const types={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.json':'application/json','.css':'text/css','.cu':'text/plain','.wgsl':'text/plain','.wasm':'application/wasm','.glb':'model/gltf-binary','.gltf':'model/gltf+json','.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg','.webp':'image/webp'};
export function createServer() {return http.createServer(async(req,res)=>{
  try {
    const name=decodeURIComponent(new URL(req.url,'http://localhost').pathname),file=path.resolve(root,'.'+(name==='/'?'/web/index.html':name));
    if(!file.startsWith(root+path.sep)||path.relative(root,file).split(path.sep).some(x=>x.startsWith('.'))){res.writeHead(403).end();return;}
    const data=await readFile(file);res.writeHead(200,{'Content-Type':types[path.extname(file)]||'application/octet-stream','Cache-Control':'no-store'});res.end(data);
  } catch {res.writeHead(404).end('Not found');}
});}
if(process.argv[1]===fileURLToPath(import.meta.url))createServer().listen(Number(process.env.PORT||8090),'127.0.0.1',()=>console.log('OpenDLSS-NR: http://127.0.0.1:'+(process.env.PORT||8090)));
