import {cp,mkdir,readFile,writeFile,readdir} from 'node:fs/promises';
import {build} from './build.mjs';
// Deliberate allowlist. Never copy the workspace or user-selected assets.
await build();
await mkdir('site',{recursive:true});
for(const folder of ['web','src','generated','kernels','vendor/webcuda/runtime','vendor/webcuda/compiler'])await cp(folder,'site/'+folder,{recursive:true});
for(const name of ['gpu.html','gpu.js']){await mkdir('site/tests',{recursive:true});await cp('tests/'+name,'site/tests/'+name);}
for(const folder of ['build','examples/jsm'])await cp('node_modules/three/'+folder,'site/node_modules/three/'+folder,{recursive:true});
await cp('node_modules/three/LICENSE','site/node_modules/three/LICENSE');
await cp('vendor/webcuda/LICENSE','site/vendor/webcuda/LICENSE');
for(const name of ['LICENSE','PROVENANCE.md','README.md'])await cp(name,'site/'+name);
const html=await readFile('web/index.html','utf8');
const relative=prefix=>html.replace(/(href|src)="\/(?!\/)/g,`$1="${prefix}`).replaceAll('"/node_modules/',`"${prefix}node_modules/`);
await writeFile('site/index.html',relative('./'));
await writeFile('site/web/index.html',relative('../'));
await writeFile('site/.nojekyll','');
// Prevent stale/accidental sensitive files from riding along in a reused output directory.
async function check(dir){for(const e of await readdir(dir,{withFileTypes:true})){const p=dir+'/'+e.name;if(e.isSymbolicLink())throw Error('Symlink in site: '+p);if(e.isDirectory()){if(['models','fixtures','reference','upstream','reports'].includes(e.name))throw Error('Private directory in site: '+p);await check(p);}else if(/\.(dll|exe|pdb|safetensors|onnx|pt|pth|bin)$/i.test(p))throw Error('Disallowed site asset: '+p);}}
await check('site');console.log('Static Pages build ready in site/ (no model weights or native binaries).');
