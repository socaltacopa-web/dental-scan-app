const sweeps = [
  {
    key:"bite_registration",
    title:"Bite registration",
    group:"bridge",
    arrow:"↔",
    hint:"Bring the teeth together gently in your normal bite. Scan across the front so upper and lower teeth appear in the same sequence.",
    overlap:"This sweep is the main upper-to-lower alignment anchor."
  },
  {
    key:"front_arc",
    title:"Front outer arc",
    group:"outer",
    arrow:"→",
    hint:"Open enough to expose the front teeth. Sweep slowly from left canine area across the front to the right canine area.",
    overlap:"Keep the central incisors visible for several frames; they overlap the bite and bridge sweeps."
  },
  {
    key:"right_outer",
    title:"Right outer side",
    group:"outer",
    arrow:"↗",
    hint:"Start near the front teeth, then move along the right-side outer surfaces toward the back molars.",
    overlap:"Begin with 1–2 seconds of the same front teeth captured in the Front outer arc."
  },
  {
    key:"left_outer",
    title:"Left outer side",
    group:"outer",
    arrow:"↖",
    hint:"Start near the front teeth, then move along the left-side outer surfaces toward the back molars.",
    overlap:"Begin with 1–2 seconds of the same front teeth captured in the Front outer arc."
  },
  {
    key:"upper_biting",
    title:"Upper biting surfaces",
    group:"upper",
    arrow:"→",
    hint:"Start at the upper front teeth, tilt toward the biting surfaces, and sweep around the upper arch.",
    overlap:"Keep the upper front incisors in view while changing angle; this helps connect outer and biting scans."
  },
  {
    key:"lower_biting",
    title:"Lower biting surfaces",
    group:"lower",
    arrow:"→",
    hint:"Start at the lower front teeth, tilt toward the biting surfaces, and sweep around the lower arch.",
    overlap:"Keep the lower front incisors in view while changing angle; this helps connect outer and biting scans."
  },
  {
    key:"upper_inside",
    title:"Upper inside surfaces",
    group:"upper",
    arrow:"→",
    hint:"Start behind the upper front teeth, then sweep along the palate-side surfaces toward the back.",
    overlap:"Start on the same upper front incisors seen in Upper bridge."
  },
  {
    key:"lower_inside",
    title:"Lower inside surfaces",
    group:"lower",
    arrow:"→",
    hint:"Lift the tongue when possible. Start behind the lower front teeth and sweep along the tongue-side surfaces.",
    overlap:"Start on the same lower front incisors seen in Lower bridge."
  },
  {
    key:"upper_bridge",
    title:"Upper bridge sweep",
    group:"bridge",
    arrow:"↻",
    hint:"Keep the upper front incisors centered while slowly rolling the phone from the outside/front view toward the biting/inside view.",
    overlap:"This is intentionally redundant: it connects upper outer, biting, and inside point clouds."
  },
  {
    key:"lower_bridge",
    title:"Lower bridge sweep",
    group:"bridge",
    arrow:"↺",
    hint:"Keep the lower front incisors centered while slowly rolling the phone from the outside/front view toward the biting/inside view.",
    overlap:"This is intentionally redundant: it connects lower outer, biting, and inside point clouds."
  }
];

const CAPTURE_SECONDS=7;
const FRAME_INTERVAL_MS=240;
const SELECTED_FRAMES_PER_SWEEP=12;
const EXPORT_MAX_WIDTH=1280;

let stream=null,step=0,capturing=false,qualityLoopId=null,previousPreviewGray=null;
const results={};
const $=id=>document.getElementById(id);
const video=$("video"),overlay=$("overlay"),overlayCtx=overlay.getContext("2d");
const work=$("work"),workCtx=work.getContext("2d",{willReadFrequently:true});

function clamp(v,a,b){return Math.max(a,Math.min(b,v))}
function sleep(ms){return new Promise(r=>setTimeout(r,ms))}

function renderStep(){
  const s=sweeps[step];
  $("stepLabel").textContent=`Sweep ${step+1} of ${sweeps.length}`;
  $("sweepTitle").textContent=s.title;
  $("sweepHint").textContent=s.hint;
  $("overlapHint").textContent=`Alignment tip: ${s.overlap}`;
  $("directionArrow").textContent=s.arrow;
  $("progress").max=sweeps.length;$("progress").value=step+1;

  const done=!!results[s.key];
  $("redoBtn").disabled=!done||capturing;
  $("nextBtn").disabled=!done||capturing;
  $("startSweepBtn").disabled=!stream||capturing;
  updateCoverage();renderThumbs();
}

async function startCamera(){
  try{
    if(stream)stream.getTracks().forEach(t=>t.stop());
    stream=await navigator.mediaDevices.getUserMedia({
      video:{facingMode:{ideal:"environment"},width:{ideal:1920},height:{ideal:1080}},
      audio:false
    });
    video.srcObject=stream;await video.play();
    $("startSweepBtn").disabled=false;
    $("livePrompt").textContent="Hold steady";
    $("quality").textContent="Camera ready. Put the same tooth region in several consecutive frames.";
    startQualityLoop();
  }catch(err){
    console.error(err);
    $("quality").textContent="Camera could not start. Allow camera permission and use HTTPS or localhost.";
    $("livePrompt").textContent="Camera unavailable";
  }
}

function captureSmallFrame(){
  if(!video.videoWidth||!video.videoHeight)return null;
  const W=256,H=Math.round(W*video.videoHeight/video.videoWidth);
  work.width=W;work.height=H;workCtx.drawImage(video,0,0,W,H);
  return workCtx.getImageData(0,0,W,H);
}

function rgbToHsv(r,g,b){
  r/=255;g/=255;b/=255;const mx=Math.max(r,g,b),mn=Math.min(r,g,b),d=mx-mn;let h=0;
  if(d!==0){if(mx===r)h=60*(((g-b)/d)%6);else if(mx===g)h=60*(((b-r)/d)+2);else h=60*(((r-g)/d)+4)}
  if(h<0)h+=360;const s=mx===0?0:d/mx;return[h,s,mx];
}

function analyzeFrame(imageData,prevGray){
  const{data,width,height}=imageData;
  const gray=new Float32Array(width*height),mask=new Uint8Array(width*height);
  let sum=0,sum2=0,toothCount=0,n=0;

  for(let p=0,i=0;i<data.length;i+=4,p++){
    const r=data[i],g=data[i+1],b=data[i+2],y=.2126*r+.7152*g+.0722*b;
    gray[p]=y;sum+=y;sum2+=y*y;
    const[,s,v]=rgbToHsv(r,g,b),redness=r-(g+b)/2;
    if(v>.56&&s<.42&&redness<48){mask[p]=255;toothCount++}
    n++;
  }

  const mean=sum/n,contrast=Math.sqrt(Math.max(0,sum2/n-mean*mean));
  let lapSum2=0,lapN=0;
  for(let y=1;y<height-1;y+=2)for(let x=1;x<width-1;x+=2){
    const i=y*width+x,lap=4*gray[i]-gray[i-1]-gray[i+1]-gray[i-width]-gray[i+width];
    lapSum2+=lap*lap;lapN++;
  }
  const sharpness=Math.sqrt(lapSum2/Math.max(1,lapN));

  let motion=0;
  if(prevGray&&prevGray.length===gray.length){
    let dsum=0,dn=0;
    for(let i=0;i<gray.length;i+=16){dsum+=Math.abs(gray[i]-prevGray[i]);dn++}
    motion=dsum/Math.max(1,dn);
  }

  const lightScore=clamp(100-Math.abs(mean-135)*.9,0,100);
  const sharpScore=clamp((sharpness-8)*4.2,0,100);
  const teethFraction=toothCount/n;
  const teethScore=clamp((teethFraction-.04)*500,0,100);
  const motionScore=prevGray?clamp(100-Math.abs(motion-10)*7,0,100):50;
  const score=.27*lightScore+.29*sharpScore+.29*teethScore+.15*motionScore;

  return{gray,mask,width,height,
    brightness:+mean.toFixed(1),contrast:+contrast.toFixed(1),sharpness:+sharpness.toFixed(1),
    toothFraction:+teethFraction.toFixed(4),motion:+motion.toFixed(2),
    lightScore:+lightScore.toFixed(1),sharpScore:+sharpScore.toFixed(1),
    teethScore:+teethScore.toFixed(1),motionScore:+motionScore.toFixed(1),score:+score.toFixed(1)};
}

function drawMaskOverlay(m){
  if(!m||!video.videoWidth)return;
  overlay.width=video.videoWidth;overlay.height=video.videoHeight;
  const c=document.createElement("canvas");c.width=m.width;c.height=m.height;
  const ctx=c.getContext("2d"),img=ctx.createImageData(m.width,m.height);
  for(let p=0;p<m.mask.length;p++){
    const i=p*4,on=m.mask[p]>0;
    img.data[i]=60;img.data[i+1]=210;img.data[i+2]=120;img.data[i+3]=on?68:0;
  }
  ctx.putImageData(img,0,0);overlayCtx.clearRect(0,0,overlay.width,overlay.height);
  overlayCtx.drawImage(c,0,0,overlay.width,overlay.height);
}

function promptFromMetrics(m){
  if(!m)return"Hold steady";
  if(m.lightScore<40)return"Improve lighting";
  if(m.sharpScore<32)return"Hold steadier";
  if(m.teethScore<28)return"Center more teeth";
  if(m.motion>21)return"Move slower";
  if(m.motion>0&&m.motion<3)return"Move a little";
  return"Good — keep overlap";
}

function updateMeters(m){
  if(!m)return;
  $("lightMeter").value=m.lightScore;$("sharpMeter").value=m.sharpScore;
  $("teethMeter").value=m.teethScore;$("motionMeter").value=m.motionScore;
  $("livePrompt").textContent=promptFromMetrics(m);
  if(!capturing){
    if(m.lightScore<40)$("quality").textContent="Lighting is weak or uneven.";
    else if(m.sharpScore<32)$("quality").textContent="Image is blurry. Hold steadier.";
    else if(m.teethScore<28)$("quality").textContent="Not enough likely tooth area is visible.";
    else $("quality").textContent="Ready. Move slowly enough that neighboring frames share the same teeth.";
  }
}

function startQualityLoop(){
  cancelAnimationFrame(qualityLoopId);
  const loop=()=>{
    if(stream&&!capturing){
      const frame=captureSmallFrame();
      if(frame){
        const m=analyzeFrame(frame,previousPreviewGray);previousPreviewGray=m.gray;
        drawMaskOverlay(m);updateMeters(m);
      }
    }
    qualityLoopId=requestAnimationFrame(loop);
  };
  qualityLoopId=requestAnimationFrame(loop);
}

async function countdown(){
  const box=$("countdown");box.hidden=false;
  for(const n of[3,2,1]){box.textContent=n;await sleep(620)}
  box.textContent="GO";await sleep(420);box.hidden=true;
}

function exportFrame(maxWidth=EXPORT_MAX_WIDTH){
  const srcW=video.videoWidth,srcH=video.videoHeight;if(!srcW||!srcH)return null;
  const scale=Math.min(1,maxWidth/srcW),w=Math.round(srcW*scale),h=Math.round(srcH*scale);
  const c=document.createElement("canvas");c.width=w;c.height=h;
  const ctx=c.getContext("2d");ctx.drawImage(video,0,0,w,h);

  const smallW=256,smallH=Math.round(smallW*h/w);
  work.width=smallW;work.height=smallH;workCtx.drawImage(c,0,0,smallW,smallH);
  const m=analyzeFrame(workCtx.getImageData(0,0,smallW,smallH),previousPreviewGray);previousPreviewGray=m.gray;

  return{dataUrl:c.toDataURL("image/jpeg",.84),width:w,height:h,metrics:{
    brightness:m.brightness,contrast:m.contrast,sharpness:m.sharpness,toothFraction:m.toothFraction,
    motion:m.motion,lightScore:m.lightScore,sharpScore:m.sharpScore,teethScore:m.teethScore,
    motionScore:m.motionScore,score:m.score
  }};
}

function chooseFrames(frames,wanted=SELECTED_FRAMES_PER_SWEEP){
  if(frames.length<=wanted)return frames;
  const chosen=[];
  for(let b=0;b<wanted;b++){
    const start=Math.floor(b*frames.length/wanted),end=Math.max(start+1,Math.floor((b+1)*frames.length/wanted));
    const bin=frames.slice(start,end).sort((a,z)=>(z.metrics?.score||0)-(a.metrics?.score||0));
    if(bin[0])chosen.push(bin[0]);
  }
  chosen.sort((a,b)=>a.index-b.index);return chosen;
}

async function runSweep(){
  if(!stream||capturing)return;
  capturing=true;previousPreviewGray=null;renderStep();await countdown();
  const frames=[],started=performance.now();let index=0;
  $("quality").textContent="Scanning. Preserve overlap: the same tooth should stay visible across several frames.";

  while(performance.now()-started<CAPTURE_SECONDS*1000){
    const frame=exportFrame();
    if(frame){frame.index=index++;frame.tMs=Math.round(performance.now()-started);frames.push(frame);updateMeters({...frame.metrics,mask:new Uint8Array(0),gray:null})}
    await sleep(FRAME_INTERVAL_MS);
  }

  const selected=chooseFrames(frames);
  const avg=k=>selected.reduce((s,f)=>s+(f.metrics?.[k]||0),0)/Math.max(1,selected.length);
  const low=selected.filter(f=>(f.metrics?.score||0)<42).length;
  const coverageConfidence=clamp(.55*avg("teethScore")+.45*avg("motionScore"),0,100);

  results[sweeps[step].key]={
    key:sweeps[step].key,title:sweeps[step].title,group:sweeps[step].group,
    capturedAt:new Date().toISOString(),rawFrameCount:frames.length,selectedFrameCount:selected.length,
    averageQualityScore:+avg("score").toFixed(1),averageTeethScore:+avg("teethScore").toFixed(1),
    averageMotionScore:+avg("motionScore").toFixed(1),coverageConfidence:+coverageConfidence.toFixed(1),
    lowQualityFrameCount:low,frames:selected
  };

  capturing=false;
  $("quality").textContent=(coverageConfidence<45||low>=4)
    ?"Sweep saved, but alignment coverage looks weak. Redoing it may help."
    :"Sweep saved. The frame overlap looks usable for registration.";
  renderStep();
}

function redoSweep(){delete results[sweeps[step].key];$("quality").textContent="Sweep cleared. Reposition and scan again.";renderStep()}
function nextSweep(){if(step<sweeps.length-1){step++;previousPreviewGray=null;$("quality").textContent="Position for the next sweep.";renderStep()}else{$("quality").textContent="Capture set complete. Export the V4 package."}}

function updateCoverage(){
  const done=sweeps.filter(s=>results[s.key]).length;
  $("coverageBadge").textContent=`${Math.round(done/sweeps.length*100)}% complete`;

  const bridgeKeys=["bite_registration","upper_bridge","lower_bridge"];
  const bridges=bridgeKeys.filter(k=>results[k]).length;
  $("bridgeStatus").textContent=bridges===bridgeKeys.length?"All alignment bridges captured":`${bridges}/${bridgeKeys.length} alignment bridges captured`;

  const grid=$("coverageGrid");grid.innerHTML="";
  for(const s of sweeps){
    const item=document.createElement("div");
    item.className=`coverage-item ${results[s.key]?"done":"missing"}`;
    const left=document.createElement("strong");left.textContent=s.title;
    const right=document.createElement("span");
    right.textContent=results[s.key]?`Q ${Math.round(results[s.key].averageQualityScore)}`:"Missing";
    item.append(left,right);grid.appendChild(item);
  }
}

function renderThumbs(){
  const wrap=$("thumbs");wrap.innerHTML="";let total=0;
  for(const s of sweeps){
    const r=results[s.key];if(!r)continue;
    for(const frame of r.frames){
      total++;
      const holder=document.createElement("div");holder.className="thumb-wrap";
      const img=document.createElement("img");img.className="thumb";img.src=frame.dataUrl;img.alt=`${r.title} frame`;
      const tag=document.createElement("span");tag.className="thumb-tag";tag.textContent=`${r.title} · Q${Math.round(frame.metrics?.score||0)}`;
      holder.append(img,tag);wrap.appendChild(holder);
    }
  }
  $("frameCount").textContent=`${total} frame${total===1?"":"s"}`;$("exportBtn").disabled=total===0;
}

function exportPackage(){
  const payload={
    format:"dscan-v4",version:4,createdAt:new Date().toISOString(),
    captureMethod:"overlap-guided smartphone sweeps with explicit cross-sweep bridge scans",
    intendedUse:"experimental local reconstruction, global registration, and dentist review",
    disclaimer:"Not a validated intraoral scanner, diagnostic device, or manufacturing-grade dental impression.",
    captureGraph:{
      anchor:"bite_registration",
      intendedOverlaps:[
        ["bite_registration","front_arc"],
        ["front_arc","right_outer"],["front_arc","left_outer"],
        ["front_arc","upper_bridge"],["front_arc","lower_bridge"],
        ["upper_bridge","upper_biting"],["upper_bridge","upper_inside"],
        ["lower_bridge","lower_biting"],["lower_bridge","lower_inside"],
        ["bite_registration","upper_bridge"],["bite_registration","lower_bridge"]
      ]
    },
    device:{userAgent:navigator.userAgent,platform:navigator.platform||null},
    settings:{captureSeconds:CAPTURE_SECONDS,frameIntervalMs:FRAME_INTERVAL_MS,selectedFramesPerSweep:SELECTED_FRAMES_PER_SWEEP,exportMaxWidth:EXPORT_MAX_WIDTH},
    sweeps:sweeps.map(s=>results[s.key]||{key:s.key,title:s.title,group:s.group,missing:true})
  };
  const blob=new Blob([JSON.stringify(payload)],{type:"application/json"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=`dental-scan-v4-${Date.now()}.dscan.json`;a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href),1000);
}

$("startCameraBtn").addEventListener("click",startCamera);
$("startSweepBtn").addEventListener("click",runSweep);
$("redoBtn").addEventListener("click",redoSweep);
$("nextBtn").addEventListener("click",nextSweep);
$("exportBtn").addEventListener("click",exportPackage);
renderStep();
