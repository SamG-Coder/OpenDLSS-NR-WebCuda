import {Model,directoryReader} from '../src/model.js';
import {NeuralRenderer} from '../src/engine.js';
import {loadFixture,runParity} from '../src/parity.js';
import {modelFromDll} from '../src/dll-model.js';
import {assessResolution} from '../src/resolution.js';
import {clearPreparedModelCache} from '../src/model-preparation.js';
const $=id=>document.getElementById(id);
let engine,bitmap,scene,objectInfo,mode='image',width=512,height=512,abort,fixture,downloadURL,busy=false,valid=false,inferenceError=null,resultReady=false,sourceName='',imageName='' ;
let modelLabel='';
const numeric=(id,min,max,integer=false)=>{const v=Number($(id).value);if(!$(id).value.trim()||!Number.isFinite(v)||v<min||v>max||(integer&&!Number.isInteger(v)))throw Error(`${$(id).closest('label')?.childNodes[0]?.textContent.trim()||id} must be ${integer?'an integer ':''}from ${min} to ${max}.`);return v;};
function status(text,error=false){$('status').textContent=text;$('status').style.color=error?'#ffb3a9':'';}
function hasSource(){return mode==='image'?!!bitmap:!!objectInfo;}
function invalidate(){resultReady=false;$('result-info').textContent='Input and output use the same resolution.';$('download').hidden=true;$('output-empty').hidden=false;$('output-empty').querySelector('strong').textContent='Awaiting render';$('output-empty').querySelector('p').textContent='Render to apply the current input and settings.';$('output-dims').textContent='—';$('progress').value=0;$('progress-text').textContent='';setView('pair');}
function ready(){
 document.body.classList.toggle('busy',busy);
 for(const element of document.querySelectorAll('input,select,button'))element.disabled=busy;
 $('cancel').disabled=!busy||!abort;$('run').disabled=busy||!engine||!valid||!!inferenceError||!hasSource();$('parity').disabled=busy||!engine||!fixture;
 const original=mode==='image'&&$('resolution-mode').value==='original';$('width').disabled=busy||original;$('height').disabled=busy||original;
 $('save-settings').disabled=busy||!valid||!hasSource();$('view-wipe').disabled=busy||!resultReady;$('frame-object').disabled=busy||!objectInfo;
 const override=$('features').files.length>0;for(const id of ['tone','structure','auto-mask','skin-mode','skin','style','seed'])$(id).disabled=busy||override;
 $('skin-mode').disabled=busy||override||!$('auto-mask').checked;$('skin').disabled=busy||override||!$('auto-mask').checked;
 $('skin-field').hidden=$('skin-mode').value!=='custom';scene?.setEnabled(!busy);
}
function conditioning(){return {style:Number($('style').value),localTone:numeric('tone',0,1),localStructure:numeric('structure',0,1),skinStructure:$('skin-mode').value==='follow'?-1:numeric('skin',0,1),autoMask:$('auto-mask').checked?1:0};}
function sceneSettings(){return {width,height,lighting:$('lighting').value,exposure:numeric('exposure',.1,4),background:$('background').value,fov:numeric('fov',15,90),animation:Number($('animation').value||-1),time:numeric('animation-time',0,1e7)};}
function dimensions({inference=false}={}){
 const w=numeric('width',33,16384,true),h=numeric('height',33,16384,true);
 if(w*h>16777216)throw Error('The source preview supports up to 16 megapixels. Choose a smaller custom resolution.');
 if(mode==='object'&&scene&&(w>scene.renderer.capabilities.maxTextureSize||h>scene.renderer.capabilities.maxTextureSize))throw Error('This resolution exceeds the 3D preview device texture limit.');
 const result=assessResolution(w,h,engine?.runtime.device.limits.maxStorageBufferBindingSize??null);
 if(inference&&result.inferenceError)throw Error(result.inferenceError);
 return {w,h,g:result.geometry,inferenceError:result.inferenceError};
}
function updateSource(){
 invalidate();valid=false;inferenceError=null;$('resolution-error').hidden=true;
 try{
  if(mode==='image'&&bitmap&&$('resolution-mode').value==='original'){$('width').value=bitmap.width;$('height').value=bitmap.height;}
  const {w,h,g,inferenceError:error}=dimensions();width=w;height=h;inferenceError=error;
  $('dimension-badge').textContent=`${w} × ${h}`;$('input-dims').textContent=`${w} × ${h}`;
  $('resolution-info').textContent=`Input → output: ${w} × ${h} px${mode==='image'&&bitmap?` · Source ${bitmap.width} × ${bitmap.height}`:''}`;
  $('features-help').textContent=`Float32, 16 channels × ${g.fullWidth} × ${g.fullHeight} padded pixels (${g.fullWidth*g.fullHeight*64} bytes). Overrides preprocessing and neural controls.`;
  if(!hasSource())return;
  const canvas=$('input');canvas.width=w;canvas.height=h;
  if(mode==='image'){const ctx=canvas.getContext('2d');ctx.fillStyle='#000';ctx.fillRect(0,0,w,h);ctx.drawImage(bitmap,0,0,w,h);}
  else {scene.configure(sceneSettings());scene.capture(canvas);}
  $('output').width=w;$('output').height=h;$('input-empty').hidden=true;$('comparison').classList.remove('empty');valid=true;if(inferenceError){$('resolution-error').hidden=false;$('resolution-error').textContent=inferenceError;status('Source preview ready. This resolution exceeds the NR device limit.');}else status(`Input ready: ${w} × ${h}.`);
 }catch(e){$('resolution-error').hidden=false;$('resolution-error').textContent=e.message;$('input-empty').hidden=false;$('input-empty').querySelector('strong').textContent='Check the input resolution';$('input-empty').querySelector('p').textContent='Choose a supported size in the resolution settings.';status(e.message,true);}
 finally{ready();}
}
async function operation(fn,{cancel=false}={}){busy=true;abort=cancel?new AbortController():null;ready();try{await fn();}catch(e){status(e.name==='AbortError'?'Render cancelled.':e.message,true);}finally{busy=false;abort=null;ready();}}
async function loadModel(loader,label){await operation(async()=>{
 try{
  const model=await loader();status('Preparing WebGPU renderer…');
  const next=await NeuralRenderer.create(model,{gemmBackend:$('gemm-backend').value,modelCache:$('model-cache').checked,onPrepareProgress:p=>{status(`${p.phase==='hash'?'Identifying model':'Preparing model'} · ${p.completed} / ${p.total}`);$('progress').value=p.total?p.completed/p.total:0;}});
  engine?.dispose();engine=next;modelLabel=label;
  $('model-status').textContent=`${label} · 71 blocks · ${model.tensors.size} tensors`;
  $('prepared-status').textContent=engine.preparedModel?`${engine.preparedModel.matrices.size} matrices prepared · format v${engine.preparedModel.formatVersion}`:engine.gemmBackend==='half'?'Standard renderer active.':'Prepared kernels unavailable on this device; using the standard renderer.';
  $('gpu-status').textContent='WebGPU ready';updateSource();if((valid&&!inferenceError)||!hasSource())status('NR support ready. Choose your input and render.');
 }catch(error){if(engine)$('gemm-backend').value=engine.gemmBackend??'half';throw error;}
});}
$('gemm-backend').onchange=()=>{if(engine)loadModel(()=>engine.model,modelLabel);};
$('model-cache').onchange=()=>{if(engine&&engine.gemmBackend!=='half')loadModel(()=>engine.model,modelLabel);};
$('clear-model-cache').onclick=()=>operation(async()=>{if(!await clearPreparedModelCache())throw Error('Browser storage is unavailable; saved preparation could not be cleared.');$('prepared-status').textContent='Saved model preparation cleared. The loaded model remains available.';status('Local prepared-model cache cleared.');});
$('dll').onchange=()=>{const file=$('dll').files[0];if(file)loadModel(()=>modelFromDll(file,{onProgress:status}),file.name);};
$('model').onchange=()=>{if($('model').files.length)loadModel(()=>Model.load(directoryReader($('model').files)),'Verified model folder');};
async function loadImage(file){if(!file)return;await operation(async()=>{const next=await createImageBitmap(file);bitmap?.close();bitmap=next;sourceName=file.name;imageName=file.name;$('source-name').textContent=`${file.name} · ${bitmap.width} × ${bitmap.height}`;clearTemporal();updateSource();});}
$('image').onchange=()=>loadImage($('image').files[0]);
async function loadObject(files){if(!files.length)return;await operation(async()=>{status('Loading 3D object…');if(!scene){const {SceneInput}=await import('./scene.js');scene=new SceneInput($('viewport'),()=>{if(mode==='object'&&!busy){invalidate();ready();}});}objectInfo=await scene.load(files);sourceName=objectInfo.name;$('source-name').textContent=objectInfo.name;const select=$('animation');select.replaceChildren(new Option('Rest pose','-1'));objectInfo.animations.forEach((a,i)=>select.add(new Option(a.name||`Animation ${i+1}`,String(i))));$('animation-label').hidden=!objectInfo.animations.length;$('animation-time-label').hidden=!objectInfo.animations.length;clearTemporal();updateSource();scene.frame();});}
$('object').onchange=()=>loadObject($('object').files);$('object-folder').onchange=()=>loadObject($('object-folder').files);
function setMode(value){if(value===mode)return;mode=value;for(const m of ['image','object']){$('mode-'+m).setAttribute('aria-pressed',String(mode===m));$(m+'-fields').hidden=mode!==m;}$('object-view').hidden=mode!=='object';$('scene-fields').hidden=mode!=='object';$('resolution-mode-label').hidden=mode==='object';$('workspace-title').textContent=mode==='image'?'Image workspace':'3D workspace';$('input-empty').hidden=hasSource();$('comparison').classList.toggle('empty',!hasSource());if(!hasSource()){$('input').getContext('2d').clearRect(0,0,$('input').width,$('input').height);$('source-name').textContent='No input selected';}else {sourceName=mode==='image'?imageName:objectInfo.name;$('source-name').textContent=sourceName;}if(mode==='object'){$('width').value=512;$('height').value=512;}clearTemporal();updateSource();}
$('mode-image').onclick=()=>setMode('image');$('mode-object').onclick=()=>setMode('object');
$('resolution-mode').onchange=updateSource;
for(const id of ['width','height'])$(id).onchange=()=>{if(mode==='image'&&bitmap&&$('resolution-mode').value==='custom'){const v=Number($(id).value);$(id==='width'?'height':'width').value=Math.round(id==='width'?v*bitmap.height/bitmap.width:v*bitmap.width/bitmap.height);}clearTemporal();updateSource();};
for(const id of ['lighting','exposure','background','fov','animation','animation-time'])$(id).onchange=updateSource;
$('frame-object').onclick=()=>{scene?.frame();updateSource();};
for(const id of ['tone','structure','skin','style','auto-mask','skin-mode','seed'])$(id).addEventListener('input',()=>{for(const r of ['tone','structure','skin'])$(r+'-value').textContent=Number($(r).value).toFixed(2);invalidate();ready();});
$('reset').onclick=()=>{for(const id of ['tone','structure','skin']){$(id).value=1;$(id+'-value').textContent='1.00';}$('auto-mask').checked=true;$('skin-mode').value='follow';$('style').value='0';$('seed').value=0;invalidate();ready();};
function clearTemporal(){for(const id of ['history','motion'])$(id).value='';$('temporal-status').textContent='First frame · no history';}
$('clear-temporal').onclick=()=>{clearTemporal();invalidate();status('Temporal inputs cleared.');ready();};
for(const id of ['history','motion'])$(id).onchange=()=>{$('temporal-status').textContent=$('history').files[0]?`History selected${$('motion').files.length?' · Motion selected':' · Already aligned'}`:'Motion requires a previous output image';invalidate();ready();};
$('features').onchange=()=>{invalidate();ready();};$('clear-features').onclick=()=>{$('features').value='';invalidate();status('Feature override cleared.');ready();};
async function floats(file,count,label){if(file.size!==count*4)throw Error(`${label} requires exactly ${count*4} bytes for this resolution; got ${file.size}.`);const values=new Float32Array(await file.arrayBuffer());if(values.some(v=>!Number.isFinite(v)))throw Error(`${label} contains non-finite values.`);return values;}
async function extraInputs(){
 const result={};if($('motion').files.length&&!$('history').files.length)throw Error('Motion vectors require a previous output image.');
 if($('history').files.length){const img=await createImageBitmap($('history').files[0]);try{if(img.width!==width||img.height!==height)throw Error(`Previous output must be ${width} × ${height}; it will not be resized.`);const c=document.createElement('canvas');c.width=width;c.height=height;c.getContext('2d').drawImage(img,0,0);result.history=Float32Array.from(c.getContext('2d').getImageData(0,0,width,height).data,v=>v/255);}finally{img.close();}}
 if($('motion').files.length)result.motion=await floats($('motion').files[0],width*height*4,'Motion');
 if($('features').files.length){const {g}=dimensions();result.inputFeatures=await floats($('features').files[0],g.fullWidth*g.fullHeight*16,'Features');}return result;
}
const progress=p=>{$('progress').value=p.index/p.total;$('progress-text').textContent=`${p.index} / ${p.total}`;status(`Rendering · ${p.label}`);};
$('run').onclick=()=>operation(async()=>{
 const frameStarted=performance.now();dimensions({inference:true});const seed=numeric('seed',0,4294967295,true),controls=conditioning(),extra=await extraInputs();if(mode==='object'){scene.configure(sceneSettings());scene.capture($('input'));}
 const proxy=Float32Array.from($('input').getContext('2d').getImageData(0,0,width,height).data,v=>v/255);invalidate();const started=performance.now();
 const result=await engine.run({width,height,proxy,...extra,seed,readHead:false,conditioning:controls,onProgress:progress,signal:abort.signal});const c=$('output');c.width=width;c.height=height;c.getContext('2d').putImageData(new ImageData(Uint8ClampedArray.from(result.output,v=>Math.round(Math.max(0,Math.min(1,v))*255)),width,height),0,0);
 const presented=performance.now();const blob=await new Promise(r=>c.toBlob(r));if(!blob)throw Error('PNG export failed.');if(downloadURL)URL.revokeObjectURL(downloadURL);downloadURL=URL.createObjectURL(blob);Object.assign($('download'),{href:downloadURL,download:`${sourceName.replace(/\.[^.]+$/,'')||'render'}-nr-${width}x${height}.png`,hidden:false});resultReady=true;$('output-empty').hidden=true;$('output-dims').textContent=`${width} × ${height}`;const completed=performance.now(),frameTimings={inputMs:started-frameStarted,engine:result.timings,presentationMs:presented-started-result.timings.totalMs,pngMs:completed-presented,totalMs:completed-frameStarted};c.dataset.timings=JSON.stringify(frameTimings);const seconds=(frameTimings.totalMs/1000).toFixed(1);status(`Completed in ${seconds} seconds.`);$('result-info').title=`Input ${frameTimings.inputMs.toFixed(1)} ms · NR ${result.timings.totalMs.toFixed(1)} ms · Display ${frameTimings.presentationMs.toFixed(1)} ms · PNG ${frameTimings.pngMs.toFixed(1)} ms`;$('result-info').textContent=`${width} × ${height} px · ${seconds} s · Seed ${seed}`;
},{cancel:true});
$('cancel').onclick=()=>abort?.abort();
function setView(view){const wipe=view==='wipe'&&resultReady;$('comparison').classList.toggle('wipe',wipe);$('wipe-control').hidden=!wipe;$('view-pair').setAttribute('aria-pressed',String(!wipe));$('view-wipe').setAttribute('aria-pressed',String(wipe));$('after').style.clipPath=wipe?`inset(0 ${100-Number($('wipe').value)}% 0 0)`:'';}
$('view-pair').onclick=()=>setView('pair');$('view-wipe').onclick=()=>setView('wipe');$('wipe').oninput=()=>setView('wipe');
$('fixture').onchange=()=>operation(async()=>{fixture=null;fixture=await loadFixture(directoryReader($('fixture').files));status('Fixture validated.');});
$('parity').onclick=()=>operation(async()=>{const report=await runParity(engine,fixture,{onProgress:progress,signal:abort.signal});$('report').textContent=JSON.stringify(report,null,2);if(!report.passed)throw Error('Parity failed; see the report.');status('Parity checks passed.');},{cancel:true});
$('save-settings').onclick=()=>{try{const settings={version:1,inputType:mode,source:sourceName,resolution:{width,height},seed:numeric('seed',0,4294967295,true),conditioning:conditioning(),scene:mode==='object'?{...sceneSettings(),cameraPosition:scene?.camera.position.toArray(),cameraTarget:scene?.controls.target.toArray()}:undefined,files:Object.fromEntries(['history','motion','features'].map(id=>[id,$(id).files[0]?.name||null]))};const url=URL.createObjectURL(new Blob([JSON.stringify(settings,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='nr-settings.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){status(e.message,true);}};
for(const [id,loader] of [['image-drop',files=>loadImage(files[0])],['object-drop',loadObject]]){const zone=$(id);zone.ondragover=e=>{e.preventDefault();if(!busy)zone.classList.add('dragover');};zone.ondragleave=()=>zone.classList.remove('dragover');zone.ondrop=e=>{e.preventDefault();zone.classList.remove('dragover');if(!busy)loader(e.dataTransfer.files);};}
$('gpu-status').textContent=navigator.gpu?'WebGPU available':'WebGPU unavailable';if(!navigator.gpu)status('WebGPU is unavailable. Use a supported browser on localhost.',true);ready();
window.addEventListener('beforeunload',()=>{bitmap?.close();scene?.dispose();if(downloadURL)URL.revokeObjectURL(downloadURL);});
