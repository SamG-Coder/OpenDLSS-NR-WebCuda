import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {DRACOLoader} from 'three/addons/loaders/DRACOLoader.js';
import {KTX2Loader} from 'three/addons/loaders/KTX2Loader.js';
import {MeshoptDecoder} from 'three/addons/libs/meshopt_decoder.module.js';
function disposeObject(root){const textures=new Set(),materials=new Set(),geometries=new Set();root?.traverse(o=>{if(o.geometry)geometries.add(o.geometry);for(const m of Array.isArray(o.material)?o.material:o.material?[o.material]:[]){materials.add(m);for(const v of Object.values(m))if(v?.isTexture)textures.add(v);}});for(const t of textures){t.source?.data?.close?.();t.dispose();}for(const m of materials)m.dispose();for(const g of geometries)g.dispose();}
export class SceneInput {
 constructor(container,onChange){
  this.onChange=onChange;this.scene=new THREE.Scene();this.scene.background=new THREE.Color('#242a32');
  this.renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});this.renderer.setPixelRatio(1);this.renderer.setSize(512,512,false);this.renderer.outputColorSpace=THREE.SRGBColorSpace;this.renderer.toneMapping=THREE.ACESFilmicToneMapping;container.append(this.renderer.domElement);
  this.camera=new THREE.PerspectiveCamera(40,1,.01,1000);this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.addEventListener('change',()=>{this.render();this.onChange();});
  this.ambient=new THREE.HemisphereLight(0xe4efff,0x6d625b,1.7);this.key=new THREE.DirectionalLight(0xfff1df,3);this.key.position.set(3,4,5);this.fill=new THREE.DirectionalLight(0xc6d8ff,1);this.fill.position.set(-3,1,-2);this.scene.add(this.ambient,this.key,this.fill);
 }
 async load(files){
  const list=Array.from(files),roots=list.filter(f=>/\.(glb|gltf)$/i.test(f.name));if(roots.length!==1)throw Error('Choose exactly one GLB or glTF, together with its textures and buffers.');
  const root=roots[0],urls=[],paths=new Map(),byName=new Map();
  for(const f of list){const p=f.webkitRelativePath||f.name;paths.set(p,f);const same=byName.get(f.name)||[];same.push(f);byName.set(f.name,same);}
  const rootPath=root.webkitRelativePath||root.name,base=rootPath.includes('/')?rootPath.slice(0,rootPath.lastIndexOf('/')+1):'';
  const manager=new THREE.LoadingManager();manager.setURLModifier(url=>{
   if(/^(data:|blob:)/.test(url))return url;
   const decoded=decodeURIComponent(url).replace(/\\/g,'/');
   const relative=new URL(decoded,'https://local.invalid/'+base).pathname.slice(1);
   let file=paths.get(relative)||paths.get(decoded);if(!file){const names=byName.get(relative.split('/').pop());if(names?.length===1)file=names[0];}
   if(!file)throw Error('Missing local glTF asset: '+decoded+'. Select the full folder or include all referenced files.');
   const objectURL=URL.createObjectURL(file);urls.push(objectURL);return objectURL;
  });
  const draco=new DRACOLoader().setDecoderPath(new URL('../node_modules/three/examples/jsm/libs/draco/gltf/',import.meta.url).href);
  const ktx=new KTX2Loader().setTranscoderPath(new URL('../node_modules/three/examples/jsm/libs/basis/',import.meta.url).href).detectSupport(this.renderer);
  try {
   const loader=new GLTFLoader(manager).setDRACOLoader(draco).setKTX2Loader(ktx).setMeshoptDecoder(MeshoptDecoder);
   const gltf=await loader.parseAsync(await root.arrayBuffer(),'');
   const bounds=new THREE.Box3().setFromObject(gltf.scene),size=bounds.getSize(new THREE.Vector3());
   if(bounds.isEmpty()||!Number.isFinite(size.length())||size.length()===0){disposeObject(gltf.scene);throw Error('The object has no visible geometry.');}
   this.mixer?.stopAllAction();if(this.object){this.scene.remove(this.object);disposeObject(this.object);}
   this.object=gltf.scene;this.scene.add(this.object);this.animations=gltf.animations;this.mixer=new THREE.AnimationMixer(this.object);this.frame();return {name:root.name,animations:this.animations.map(a=>({name:a.name,duration:a.duration}))};
  }finally{urls.forEach(u=>URL.revokeObjectURL(u));draco.dispose();ktx.dispose();}
 }
 frame(){if(!this.object)return;const b=new THREE.Box3().setFromObject(this.object),center=b.getCenter(new THREE.Vector3()),radius=b.getSize(new THREE.Vector3()).length()/2;const angle=Math.min(this.camera.fov*Math.PI/180,2*Math.atan(Math.tan(this.camera.fov*Math.PI/360)*this.camera.aspect));const distance=radius/Math.sin(angle/2)*1.12;this.camera.near=Math.max(radius/1000,.0001);this.camera.far=distance+radius*100;this.camera.position.copy(center).add(new THREE.Vector3(.25,.12,1).normalize().multiplyScalar(distance));this.controls.target.copy(center);this.controls.maxDistance=distance*20;this.camera.updateProjectionMatrix();this.controls.update();this.render();this.onChange();}
 configure({width,height,lighting,exposure,background,fov,animation=-1,time=0}){
  this.renderer.setSize(width,height,false);this.camera.aspect=width/height;this.camera.fov=fov;this.camera.updateProjectionMatrix();this.scene.background.set(background);this.renderer.toneMappingExposure=exposure;
  this.ambient.intensity=lighting==='flat'?3:lighting==='side'?.5:1.7;this.key.intensity=lighting==='flat'?1:3;this.key.position.set(lighting==='side'?5:3,lighting==='side'?1:4,3);this.fill.intensity=lighting==='flat'?1:lighting==='side'?.2:1;
  if(this.mixer){this.mixer.stopAllAction();if(animation>=0&&this.animations[animation]){this.mixer.clipAction(this.animations[animation]).play();this.mixer.setTime(time);}}
  this.render();
 }
 render(){this.renderer.render(this.scene,this.camera);}
 capture(canvas){this.render();canvas.width=this.renderer.domElement.width;canvas.height=this.renderer.domElement.height;canvas.getContext('2d').drawImage(this.renderer.domElement,0,0);}
 setEnabled(value){this.controls.enabled=value;}
 dispose(){this.controls.dispose();this.mixer?.stopAllAction();disposeObject(this.object);this.renderer.dispose();this.renderer.domElement.remove();}
}
