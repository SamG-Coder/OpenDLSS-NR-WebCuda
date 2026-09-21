export const align = (v,a) => Math.ceil(v/a)*a;
export function geometry(width,height) {
  if (![width,height].every(v=>Number.isSafeInteger(v)&&v>=33&&v<=16384)) throw Error('Dimensions must be integers from 33 to 16384.');
  function alignment(v) {
    let n=0;
    for(let l=0;l<6;l++) { const half=align(Math.ceil(v/2),4); if(half<v)n++;if(l===0&&half%8)n++;v=half; }
    return 2**n;
  }
  const aw=alignment(width),ah=alignment(height);
  let fullWidth=Math.max(320,align(width,aw)),fullHeight=Math.max(320,align(height,ah));
  if(fullWidth%(4*aw)===0&&fullHeight%(4*ah)===0)fullWidth+=aw;
  let w=fullWidth,h=fullHeight;
  const levels=Array.from({length:6},()=>{w=align(Math.ceil(w/2),4);h=align(Math.ceil(h/2),4);return {width:w,height:h};});
  if(levels[0].width%8||levels[0].height%8)throw Error('Unsupported level-0 crop.');
  return {width,height,fullWidth,fullHeight,levels};
}
export const phases = [[0,0],[4,4],[4,0],[0,4]];
export function layout(c,kind='standard') {
  if(kind==='pre')return {expand:0,contract:4096,adapter:8208,ffn:9232,qkv:9312,prior:12384,scale:20576,projection:20592,attn:21616,end:21680};
  if(kind==='post')return {expand:0,contract:4096,ffn:8208,inputScale:8272,adapterScale:8336,qkv:8400,prior:11472,scale:19664,projection:19680,attn:20704,head:20784,end:21808};
  const experts=c>=64?c/32:1,expand=experts*c*128,ffnBytes=expand+experts*128*32+(c>=64?c*c:0);
  const l={expand:0,contract:expand};
  if(kind==='up') {l.up=ffnBytes;l.ffn=ffnBytes+2*c*c+(c===32?16:0);l.transition=l.ffn+2*c+(c===32?16:0);l.qkv=l.transition+2*c;}
  else {l.ffn=ffnBytes+16;l.qkv=l.ffn+2*c+16;}
  l.prior=l.qkv+3*c*c;l.scale=l.prior+(c/32)*8192;l.projection=l.scale+align((c/32)*4,16);l.attn=l.projection+c*c;l.end=l.attn+2*c;
  return l;
}
