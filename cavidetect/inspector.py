"""Inspector — a spatial + temporal review overlay for one sequence (Phase 1, view-only).

Three synchronised, stacked panels like the OSOV / cavicam optical-vulnerability video:
raw tissue, a frame-difference image, and the embolised conduits coloured on black. The
conduits panel colours **by time** (onset frame, a blue->red ramp) or **by tube** (each
rhizoid a distinct hue) and reveals over the timeline. Self-contained HTML, opened from the
GUI's 🔍 Inspect button — one polished part of the cavidetect app.

Built to reuse what the pipeline already computes at ``run()`` exit (``result["onset"]`` /
``["shape"]`` / ``["final"]``) plus a per-rhizoid identity map derived here with
``label_tubes``. It never changes the measurement; it is an output/review layer over any
detection method. Phase 2 (the editing + commit->recompute loop) is specified in
docs/superpowers/specs/2026-11-11-inspector-design.md.
"""
from __future__ import annotations
import os, io as _io, base64, json as _json
import numpy as np


# --------------------------------------------------------------------------- identity

def _relabel_consecutive(seg: np.ndarray) -> np.ndarray:
    """Map an arbitrary label image to 0..K with 0 preserved as background."""
    vals = np.unique(seg)
    out = np.zeros_like(seg, dtype=np.int32)
    k = 0
    for v in vals:
        if v == 0:
            continue
        k += 1
        out[seg == v] = k
    return out


def label_tubes(shape, min_seg_px: int = 4) -> np.ndarray:
    """Split an embolised mask into per-rhizoid segment labels (for identity colouring).

    Same idea as ``rhizoid.tube_segments`` but applied to the committed embolised mask
    instead of a temporal projection: skeletonise -> drop junction pixels so crossing
    bundles separate into branches -> label -> grow each branch label back to the full
    mask width (nearest skeleton segment). Tiny masks that yield no skeleton branch fall
    back to plain connected components so every tube pixel still gets an identity.

    Returns an int32 array, 0 = background, 1..K = rhizoids (consecutive).
    """
    from skimage.morphology import skeletonize
    from skimage.measure import label as sklabel
    from scipy.ndimage import convolve, distance_transform_edt

    shape = np.asarray(shape, dtype=bool)
    if not shape.any():
        return np.zeros(shape.shape, np.int32)

    sk = skeletonize(shape)
    if not sk.any():
        return _relabel_consecutive(sklabel(shape))

    nb = convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - sk.astype(int)
    seg_sk = sklabel(sk & (nb <= 2))             # drop junctions -> separate branches
    if seg_sk.max() > 0:
        sizes = np.bincount(seg_sk.ravel())
        small = sizes < min_seg_px
        small[0] = False
        seg_sk[small[seg_sk]] = 0
    if seg_sk.max() == 0:                         # nothing survived -> component fallback
        return _relabel_consecutive(sklabel(shape))

    _, inds = distance_transform_edt(seg_sk == 0, return_indices=True)
    seg_full = seg_sk[tuple(inds)]
    seg_full[~shape] = 0
    return _relabel_consecutive(seg_full)


# ------------------------------------------------------------------ lossless u16 maps

def _encode_u16_png(values, valid) -> str:
    """Encode a 0..65535 int map + a validity mask as a lossless RGBA PNG data-URL.

    value = R + 256*G ; alpha = 255 where valid else 0. The JS side decodes it the same
    way (``_decode_u16_png`` mirrors it for tests). PNG is lossless so labels / onset
    frames survive exactly — unlike the JPEG used for the greyscale panels.
    """
    from PIL import Image
    v = np.clip(np.asarray(values), 0, 65535).astype(np.uint16)
    valid = np.asarray(valid, dtype=bool)
    Y, X = v.shape
    rgba = np.zeros((Y, X, 4), np.uint8)
    rgba[..., 0] = (v & 0xFF).astype(np.uint8)
    rgba[..., 1] = (v >> 8).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    b = _io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def _decode_u16_png(data_url: str):
    """Inverse of ``_encode_u16_png`` (mirrors the browser's decode). Returns (values, valid)."""
    from PIL import Image
    raw = base64.b64decode(data_url.split(",", 1)[1])
    arr = np.asarray(Image.open(_io.BytesIO(raw)).convert("RGBA"), dtype=np.int32)
    values = arr[..., 0] + 256 * arr[..., 1]
    valid = arr[..., 3] > 0
    return values, valid


# ------------------------------------------------------------------ greyscale panels

def prepare_inspector_frames(s, max_px: int = 820, n_timepoints: int = 90):
    """Pre-render the raw + difference panels at sampled timepoints.

    Returns ``(tps, raw_b64, diff_b64, newsize)`` where ``tps`` are the sampled frame
    indices, ``raw_b64`` / ``diff_b64`` are base64 JPEG data-URLs, and ``newsize`` is the
    (W, H) the maps must also be resized to so every panel aligns. One frame is normalised
    at a time (the ``outputs._grey_u8`` memory-safe pattern), so a full-res 400-frame stack
    never promotes the whole stack to float64.
    """
    from PIL import Image
    T, Y, X = s.shape
    lo, hi = np.percentile(s, [2, 99])
    rng = max(float(hi - lo), 1e-3)
    scale = min(1.0, max_px / max(Y, X))
    newsize = (max(1, int(X * scale)), max(1, int(Y * scale)))

    def _grey_u8(t):
        return (np.clip((s[t] - lo) / rng, 0, 1) * 255).astype(np.uint8)

    def _jpeg(a2d):
        im = Image.fromarray(a2d)
        if scale < 1.0:
            im = im.resize(newsize, Image.BILINEAR)
        b = _io.BytesIO(); im.save(b, format="JPEG", quality=74)
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

    tps = sorted(set(np.linspace(0, T - 1, min(n_timepoints, T)).astype(int).tolist()))

    # symmetric scale for the difference panel: robust 99th pct of |consecutive diff|
    diffs_f32 = []
    prev = None
    for t in tps:
        g = s[t].astype(np.float32)
        diffs_f32.append(None if prev is None else (g - prev))
        prev = g
    finite = [np.abs(d) for d in diffs_f32 if d is not None]
    dscale = max(float(np.percentile(np.concatenate([f.ravel() for f in finite]), 99)), 1.0) \
        if finite else 1.0

    raw_b64, diff_b64 = [], []
    for k, t in enumerate(tps):
        raw_b64.append(_jpeg(_grey_u8(t)))
        d = diffs_f32[k]
        if d is None:
            du8 = np.full(s.shape[1:], 128, np.uint8)
        else:
            du8 = np.clip(128 + d / dscale * 127, 0, 255).astype(np.uint8)
        diff_b64.append(_jpeg(du8))
    return tps, raw_b64, diff_b64, newsize


def _resize_map_nearest(a2d, newsize):
    """Nearest-neighbour resize of an integer/bool map to (W, H) — never interpolate labels."""
    from PIL import Image
    im = Image.fromarray(np.asarray(a2d).astype(np.int32), mode="I")
    if im.size != newsize:
        im = im.resize(newsize, Image.NEAREST)
    return np.asarray(im, dtype=np.int32)


# ----------------------------------------------------------------------------- build

def save_inspector(stack, result: dict, out_dir: str, stem: str,
                   max_px: int = 820, n_timepoints: int = 90) -> str:
    """Write the self-contained inspector HTML (``<stem>_inspect.html``).

    ``result`` is the ``colour_shape.measure`` dict (needs ``onset``, ``shape``, ``final``).
    Raw + difference panels are pre-rendered JPEGs; the onset-frame map and the per-rhizoid
    identity map (``label_tubes``) are embedded as lossless PNGs and coloured live in the
    browser, so the by-time / by-tube / static toggles and the animation are all instant.
    """
    os.makedirs(out_dir, exist_ok=True)
    T = stack.shape[0]
    shape = np.asarray(result["shape"], dtype=bool)
    onset = np.asarray(result["onset"]).astype(np.int32)

    tps, raw_b64, diff_b64, newsize = prepare_inspector_frames(stack, max_px, n_timepoints)

    valid = shape & (onset >= 0)
    onset_small = _resize_map_nearest(np.where(valid, onset, 0), newsize)
    valid_small = _resize_map_nearest(valid.astype(np.int32), newsize) > 0
    onset_png = _encode_u16_png(onset_small, valid_small)

    tubes = label_tubes(shape)
    tubes_small = _resize_map_nearest(tubes, newsize)
    tubes_png = _encode_u16_png(tubes_small, tubes_small > 0)

    html = (_INSPECTOR_HTML
            .replace("__TITLE__", f"{stem} — inspector")
            .replace("__RAW__", _json.dumps(raw_b64))
            .replace("__DIFF__", _json.dumps(diff_b64))
            .replace("__TPS__", _json.dumps(tps))
            .replace("__ONSET__", _json.dumps(onset_png))
            .replace("__TUBES__", _json.dumps(tubes_png))
            .replace("__NFR__", str(T))
            .replace("__NTUBES__", str(int(tubes.max()))))
    path = os.path.join(out_dir, f"{stem}_inspect.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


_INSPECTOR_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 body{background:#0d0d0f;color:#eee;font-family:system-ui,sans-serif;margin:0;padding:14px}
 h2{margin:0 0 4px;font-weight:600;font-size:16px}
 .sub{font-size:12px;color:#8a94a0;margin-bottom:10px}
 #stage{position:relative;overflow:hidden;border:1px solid #333;background:#000;touch-action:none}
 #inner{transform-origin:0 0}
 .panel{position:relative}
 .panel .cap{position:absolute;left:6px;top:4px;font-size:11px;color:#9ab;
   background:rgba(0,0,0,.45);padding:1px 6px;border-radius:3px;pointer-events:none;z-index:2}
 .panel img,.panel canvas{display:block;width:100%;background:#000}
 .controls{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}
 input[type=range]{flex:1;min-width:180px}
 button{padding:6px 12px;background:#243;color:#eee;border:1px solid #365;border-radius:4px;cursor:pointer}
 button.on{background:#2a6;border-color:#4c9}
 .seg{display:inline-flex;border:1px solid #365;border-radius:4px;overflow:hidden}
 .seg button{border:0;border-radius:0}
 #lab{font-variant-numeric:tabular-nums;min-width:150px;text-align:right;color:#bcd}
 .legend{margin-top:8px;font-size:12px;color:#9ab;display:flex;align-items:center;gap:8px}
 .ramp{height:12px;width:220px;border:1px solid #333;
   background:linear-gradient(90deg,#3a4cc0,#1fb6c9,#36c940,#f2e400,#f23030)}
 .hint{color:#7a8690;font-size:12px}
</style></head><body>
<h2>__TITLE__</h2>
<div class="sub">raw tissue · frame difference · embolised conduits — scroll to zoom, drag to pan</div>
<div id="stage">
 <div id="inner">
  <div class="panel"><div class="cap">raw</div><img id="raw"></div>
  <div class="panel"><div class="cap">difference</div><img id="diff"></div>
  <div class="panel"><div class="cap">conduits</div><canvas id="cond"></canvas></div>
 </div>
</div>
<div class="controls">
 <button id="prev">&#9664;</button>
 <button id="play">play</button>
 <button id="next">&#9654;</button>
 <input type="range" id="sld" min="0" value="0">
 <span class="seg"><button id="mTime" class="on">by time</button><button id="mTube">by tube</button></span>
 <button id="mStatic">static</button>
 <span id="lab"></span>
</div>
<div class="legend" id="leg"></div>
<script>
const RAW=__RAW__, DIFF=__DIFF__, TPS=__TPS__, NFR=__NFR__, NTUBES=__NTUBES__;
const ONSET_URL=__ONSET__, TUBES_URL=__TUBES__;
const raw=document.getElementById('raw'), diff=document.getElementById('diff'),
      cond=document.getElementById('cond'), sld=document.getElementById('sld'),
      lab=document.getElementById('lab'), leg=document.getElementById('leg');
sld.max=TPS.length-1;
let mode='time', staticAll=false, W=0, H=0;
let onsetV=null, onsetValid=null, tubeV=null, tubeValid=null, cctx=null, cimg=null;

function decodeMap(url){               // RGBA PNG -> {val:Int32, valid:Uint8, w, h}
 return new Promise(res=>{ const im=new Image(); im.onload=()=>{
   const o=document.createElement('canvas'); o.width=im.width; o.height=im.height;
   const x=o.getContext('2d'); x.drawImage(im,0,0);
   const d=x.getImageData(0,0,im.width,im.height).data, n=im.width*im.height;
   const val=new Int32Array(n), valid=new Uint8Array(n);
   for(let i=0;i<n;i++){const j=i*4; val[i]=d[j]+256*d[j+1]; valid[i]=d[j+3]>0?1:0;}
   res({val,valid,w:im.width,h:im.height}); }; im.src=url; });
}
// blue->red perceptual-ish ramp for onset time
function timeColor(f){
 const stops=[[58,76,192],[31,182,201],[54,201,64],[242,228,0],[242,48,48]];
 const t=Math.max(0,Math.min(1,f/(NFR-1)))*(stops.length-1);
 const i=Math.floor(t), a=stops[i], b=stops[Math.min(i+1,stops.length-1)], u=t-i;
 return [a[0]+(b[0]-a[0])*u, a[1]+(b[1]-a[1])*u, a[2]+(b[2]-a[2])*u];
}
function hsl(h,s,l){ const k=n=>(n+h*12)%12, f=n=>l-s*Math.min(l,1-l)*Math.max(-1,Math.min(k(n)-3,9-k(n),1));
 return [255*f(0),255*f(8),255*f(4)]; }
function tubeColor(id){ return hsl(((id*0.61803398875)%1), 0.72, 0.56); }

function renderCond(frameVal){
 if(!cctx) return;
 const d=cimg.data, n=W*H;
 for(let i=0;i<n;i++){
   const j=i*4; let r=0,g=0,b=0;
   if(mode==='tube'){
     if(tubeValid[i] && (staticAll || (onsetValid[i] && onsetV[i]<=frameVal))){
       const c=tubeColor(tubeV[i]); r=c[0];g=c[1];b=c[2]; }
   } else {
     if(onsetValid[i] && (staticAll || onsetV[i]<=frameVal)){
       const c=timeColor(onsetV[i]); r=c[0];g=c[1];b=c[2]; }
   }
   d[j]=r; d[j+1]=g; d[j+2]=b; d[j+3]=255;
 }
 cctx.putImageData(cimg,0,0);
}
function legend(){
 if(mode==='tube'){ leg.innerHTML='<span>by tube — '+NTUBES+' rhizoids, one hue each</span>'; }
 else { leg.innerHTML='<span class="ramp"></span><span>early</span><span class="hint">→</span><span>late onset (frame)</span>'; }
}
function show(i){
 i=Math.max(0,Math.min(TPS.length-1,i)); sld.value=i;
 const f=TPS[i];
 raw.src=RAW[i]; diff.src=DIFF[i];
 renderCond(f);
 lab.textContent = staticAll ? 'static (all time)' : ('frame '+f+' / '+(NFR-1));
}
sld.oninput=()=>show(+sld.value);
document.getElementById('prev').onclick=()=>show(+sld.value-1);
document.getElementById('next').onclick=()=>show(+sld.value+1);
let timer=null;
document.getElementById('play').onclick=e=>{
 if(timer){clearInterval(timer);timer=null;e.target.textContent='play';}
 else{e.target.textContent='pause'; timer=setInterval(()=>{let v=+sld.value+1; if(v>=TPS.length)v=0; show(v);},120);}
};
function setMode(m){ mode=m;
 document.getElementById('mTime').classList.toggle('on',m==='time');
 document.getElementById('mTube').classList.toggle('on',m==='tube');
 legend(); show(+sld.value); }
document.getElementById('mTime').onclick=()=>setMode('time');
document.getElementById('mTube').onclick=()=>setMode('tube');
document.getElementById('mStatic').onclick=e=>{ staticAll=!staticAll;
 e.target.classList.toggle('on',staticAll); show(+sld.value); };
document.onkeydown=e=>{if(e.key==='ArrowLeft')show(+sld.value-1); if(e.key==='ArrowRight')show(+sld.value+1);};

// shared pan/zoom over all three panels
let zoom=1, px=0, py=0, dragging=false, lx=0, ly=0;
const stage=document.getElementById('stage'), inner=document.getElementById('inner');
function applyT(){ inner.style.transform='translate('+px+'px,'+py+'px) scale('+zoom+')'; }
stage.addEventListener('wheel',e=>{ e.preventDefault(); const f=e.deltaY<0?1.1:1/1.1;
 zoom=Math.max(1,Math.min(12,zoom*f)); if(zoom===1){px=0;py=0;} applyT(); },{passive:false});
stage.addEventListener('mousedown',e=>{dragging=true;lx=e.clientX;ly=e.clientY;});
window.addEventListener('mousemove',e=>{ if(!dragging)return;
 px+=e.clientX-lx; py+=e.clientY-ly; lx=e.clientX; ly=e.clientY; applyT(); });
window.addEventListener('mouseup',()=>dragging=false);

Promise.all([decodeMap(ONSET_URL), decodeMap(TUBES_URL)]).then(([on,tu])=>{
 onsetV=on.val; onsetValid=on.valid; tubeV=tu.val; tubeValid=tu.valid;
 W=on.w; H=on.h; cond.width=W; cond.height=H;
 cctx=cond.getContext('2d'); cimg=cctx.createImageData(W,H);
 legend(); show(0);
});
</script></body></html>
"""
