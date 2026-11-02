"""Write the readouts (§5): event table, per-pixel cumulative embolised-area curve,
temporal montage (raw + embolised-so-far + curve), optional animation, run manifest.

The cumulative embolised AREA (per-pixel) is the headline OVT-style output: the total area
of rhizoid that has embolised by frame t. It is robust to event over-segmentation (it does
not depend on how pixels are grouped into events), unlike the event COUNT."""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd


def embolised_by_frame(det):
    """Per pixel, the frame it embolised (inf if never). Shape [Y,X]."""
    return np.where(det.accepted, det.step_frame, np.iinfo(np.int32).max)


def cumulative_area(det, n_frames: int, scale_um2: float):
    """Cumulative embolised area (µm²) at each frame, from per-pixel step frames."""
    sf = det.step_frame[det.accepted]
    order = np.sort(sf)
    cum_px = np.searchsorted(order, np.arange(n_frames), side="right")
    return cum_px * scale_um2


def write_event_table(df: pd.DataFrame, out_dir: str, stem: str) -> str:
    path = os.path.join(out_dir, f"{stem}_events.csv")
    df.to_csv(path, index=False)
    return path


def write_tube_table(df: pd.DataFrame, out_dir: str, stem: str) -> str:
    """Per-rhizoid outcome: every tube segment, whether/when it cavitated."""
    path = os.path.join(out_dir, f"{stem}_tubes.csv")
    df.to_csv(path, index=False)
    return path


def save_tube_map(proj, seg_full, out_dir: str, stem: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
    ax[0].imshow(proj, cmap="gray"); ax[0].set_title("max projection (tubes visible)"); ax[0].axis("off")
    segcol = np.where(seg_full > 0, seg_full % 20 + 1, np.nan)
    ax[1].imshow(proj, cmap="gray"); ax[1].imshow(segcol, cmap="tab20", alpha=0.7)
    n = int(seg_full.max())
    ax[1].set_title(f"rhizoid tube segments (n={n})"); ax[1].axis("off")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{stem}_tubemap.png")
    plt.savefig(path, dpi=85); plt.close()
    return path


def write_area_curve(det, df: pd.DataFrame, n_frames: int, scale_um2: float,
                     out_dir: str, stem: str) -> tuple[str, np.ndarray]:
    cum = cumulative_area(det, n_frames, scale_um2)
    frames = np.arange(n_frames)
    out = pd.DataFrame({"frame": frames, "cumulative_area_um2": cum})
    out["cumulative_area_frac"] = cum / cum[-1] if cum[-1] > 0 else 0.0
    if len(df):
        counts = np.bincount(df["frame_peak"].clip(0, n_frames - 1).astype(int), minlength=n_frames)
        out["n_events_cumulative"] = np.cumsum(counts)
    else:
        out["n_events_cumulative"] = 0
    path = os.path.join(out_dir, f"{stem}_area_curve.csv")
    out.to_csv(path, index=False)
    return path, cum


def save_temporal_view(s, det, cum, out_dir: str, stem: str,
                       timepoints_frac=(0.03, 0.08, 0.13, 0.20, 0.35, 0.9)) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = s.shape[0]
    ef = embolised_by_frame(det)
    tps = [min(T - 1, int(T * f)) for f in timepoints_frac]
    vmin, vmax = np.percentile(s, [2, 99])
    fig = plt.figure(figsize=(3 * len(tps), 8))
    gs = fig.add_gridspec(3, len(tps), height_ratios=[1, 1, 0.9])
    for i, t in enumerate(tps):
        ax = fig.add_subplot(gs[0, i]); ax.imshow(s[t], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"raw · f{t}", fontsize=10); ax.axis("off")
        ax2 = fig.add_subplot(gs[1, i]); ax2.imshow(s[t], cmap="gray", vmin=vmin, vmax=vmax)
        emb = ef <= t; ov = np.zeros((*emb.shape, 4)); ov[emb] = [1, 0.12, 0.12, 0.85]
        ax2.imshow(ov); ax2.set_title(f"embolised by f{t}", fontsize=10); ax2.axis("off")
    axc = fig.add_subplot(gs[2, :]); axc.plot(np.arange(T), cum, color="#c0392b", lw=2)
    for t in tps:
        axc.axvline(t, color="#888", ls=":", lw=0.8)
    axc.set_title(f"{stem} — cumulative embolised area over time", fontsize=12)
    axc.set_xlabel("frame (time →)"); axc.set_ylabel("embolised area µm²"); axc.margins(x=0)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{stem}_temporal.png")
    plt.savefig(path, dpi=85); plt.close()
    return path


def save_animation(s, det, out_dir: str, stem: str, stride: int = 3, max_px: int = 360) -> str | None:
    """Fast GIF of embolism (red) accumulating over the raw tissue. Uses PIL directly (no matplotlib)."""
    try:
        from PIL import Image
    except Exception:
        return None
    T, Y, X = s.shape
    ef = embolised_by_frame(det)
    lo, hi = np.percentile(s, [2, 99]); rng = max(hi - lo, 1e-3)
    g = np.clip((s - lo) / rng, 0, 1)
    scale = min(1.0, max_px / max(Y, X))
    frames = []
    ts = list(range(0, T, stride))
    for t in ts:
        base = (g[t] * 255).astype(np.uint8)
        rgb = np.stack([base, base, base], -1)
        emb = ef <= t
        rgb[emb] = (0.15 * rgb[emb] + np.array([217, 30, 30]) * 0.85).astype(np.uint8)
        im = Image.fromarray(rgb)
        if scale < 1.0:
            im = im.resize((int(X * scale), int(Y * scale)), Image.NEAREST)
        frames.append(im)
    path = os.path.join(out_dir, f"{stem}_animation.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=80, loop=0)
    return path


def write_embolism_curve(result, n_frames: int, out_dir: str, stem: str) -> str:
    """PRIMARY output: cumulative embolised tissue over time (direct measurement)."""
    frames = np.arange(n_frames)
    out = pd.DataFrame({
        "frame": frames,
        "embolised_area_px": result.cum_area,
        "embolised_area_um2": result.cum_area_um2,
        "frac_tissue_embolised": result.frac_embolised,
    })
    path = os.path.join(out_dir, f"{stem}_embolism_curve.csv")
    out.to_csv(path, index=False)
    return path


def save_embolism_view(s, result, out_dir: str, stem: str) -> str:
    """PRIMARY figure: wet start, dry end, where+when each pixel air-filled, cumulative curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = s.shape[0]
    first, last = s[0], s[-1]
    vmin, vmax = np.percentile(s, [2, 99])
    whenmap = np.where(result.embolised, result.onset_frame.astype(float), np.nan)

    fig = plt.figure(figsize=(18, 5))
    ax = fig.add_subplot(1, 4, 1); ax.imshow(first, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("start (wet)"); ax.axis("off")
    ax = fig.add_subplot(1, 4, 2); ax.imshow(last, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("end (dry)"); ax.axis("off")
    ax = fig.add_subplot(1, 4, 3); ax.imshow(first, cmap="gray", vmin=vmin, vmax=vmax)
    m = ax.imshow(whenmap, cmap="turbo", vmin=0, vmax=T, alpha=0.9)
    ax.set_title("embolised tissue: where + when"); ax.axis("off")
    plt.colorbar(m, ax=ax, fraction=0.046, label="frame of air-filling")
    ax = fig.add_subplot(1, 4, 4)
    ax.plot(np.arange(T), result.frac_embolised * 100, color="#c0392b", lw=2)
    ax.set_xlabel("frame (time →)"); ax.set_ylabel("% tissue embolised")
    ax.set_title(f"{stem} — cumulative embolised tissue"); ax.set_ylim(0, 100); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{stem}_embolism.png")
    plt.savefig(path, dpi=85); plt.close()
    return path


def save_embolism_viewer(s, result, out_dir: str, stem: str,
                         max_px: int = 820, n_timepoints: int = 90) -> str:
    """Interactive self-contained HTML: scrub time; raw | embolised-overlay | cumulative curve.

    Pre-renders raw and red-overlay frames (embolised-so-far) at a set of timepoints, embeds
    them as base64 JPEGs, and draws the cumulative-% curve on a canvas with a moving marker.
    No server, no external files — open the .html in any browser.
    """
    import io as _io, base64
    from PIL import Image
    T, Y, X = s.shape
    lo, hi = np.percentile(s, [2, 99]); rng = max(hi - lo, 1e-3)
    g = np.clip((s - lo) / rng, 0, 1)
    onset, emb = result.onset_frame, result.embolised
    scale = min(1.0, max_px / max(Y, X))
    newsize = (max(1, int(X * scale)), max(1, int(Y * scale)))

    def _b64(rgb):
        im = Image.fromarray(rgb)
        if scale < 1.0:
            im = im.resize(newsize, Image.BILINEAR)
        b = _io.BytesIO(); im.save(b, format="JPEG", quality=72)
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

    tps = sorted(set(np.linspace(0, T - 1, min(n_timepoints, T)).astype(int).tolist()))
    raw_imgs, ov_imgs = [], []
    for t in tps:
        base = (g[t] * 255).astype(np.uint8)
        rgb = np.stack([base, base, base], -1)
        raw_imgs.append(_b64(rgb))
        mask = emb & (onset >= 0) & (onset <= t)
        ov = rgb.copy()
        ov[mask] = (0.25 * ov[mask] + np.array([235, 30, 30]) * 0.75).astype(np.uint8)
        ov_imgs.append(_b64(ov))

    cum_pct = (result.frac_embolised * 100).round(3).tolist()
    import json as _json
    html = _VIEWER_HTML.replace("__TITLE__", f"{stem} — embolism viewer") \
        .replace("__RAW__", _json.dumps(raw_imgs)) \
        .replace("__OV__", _json.dumps(ov_imgs)) \
        .replace("__TPS__", _json.dumps(tps)) \
        .replace("__CUM__", _json.dumps(cum_pct)) \
        .replace("__NFR__", str(T))
    path = os.path.join(out_dir, f"{stem}_viewer.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


_VIEWER_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 body{background:#111;color:#eee;font-family:system-ui,sans-serif;margin:0;padding:14px}
 h2{margin:0 0 10px;font-weight:600;font-size:16px}
 .row{display:flex;gap:12px;align-items:flex-start}
 .panel{flex:1;min-width:0;text-align:center}
 .panel .cap{font-size:12px;color:#9ab;margin-bottom:4px}
 .panel img{width:100%;border:1px solid #333;background:#000}
 canvas{width:100%;height:auto;border:1px solid #333;background:#161616}
 .controls{display:flex;gap:10px;align-items:center;margin-top:12px}
 input[type=range]{flex:1}
 button{padding:6px 12px;background:#243;color:#eee;border:1px solid #365;border-radius:4px;cursor:pointer}
 #lab{font-variant-numeric:tabular-nums;min-width:170px;text-align:right;color:#bcd}
</style></head><body>
<h2>__TITLE__</h2>
<div class="row">
 <div class="panel"><div class="cap">raw</div><img id="raw"></div>
 <div class="panel"><div class="cap">embolised so far (red)</div><img id="ov"></div>
 <div class="panel"><div class="cap">cumulative embolised tissue</div><canvas id="plot" width="520" height="420"></canvas></div>
</div>
<div class="controls">
 <button id="prev">&#9664;</button>
 <button id="play">play</button>
 <button id="next">&#9654;</button>
 <input type="range" id="sld" min="0" value="0">
 <span id="lab"></span>
</div>
<script>
const RAW=__RAW__, OV=__OV__, TPS=__TPS__, CUM=__CUM__, NFR=__NFR__;
const raw=document.getElementById('raw'), ov=document.getElementById('ov'),
      sld=document.getElementById('sld'), lab=document.getElementById('lab'),
      cv=document.getElementById('plot'), ctx=cv.getContext('2d');
sld.max=TPS.length-1;
const PAD=46, W=cv.width, H=cv.height;
function plot(i){
 const f=TPS[i], pct=CUM[f];
 ctx.clearRect(0,0,W,H);
 ctx.strokeStyle='#444'; ctx.fillStyle='#9ab'; ctx.font='11px sans-serif';
 ctx.beginPath(); ctx.moveTo(PAD,10); ctx.lineTo(PAD,H-PAD); ctx.lineTo(W-10,H-PAD); ctx.stroke();
 for(let p=0;p<=100;p+=25){const y=H-PAD-(H-PAD-10)*p/100;
   ctx.fillText(p+'%',6,y+3); ctx.strokeStyle='#222'; ctx.beginPath();ctx.moveTo(PAD,y);ctx.lineTo(W-10,y);ctx.stroke();}
 ctx.fillText('frame',(W-10+PAD)/2-12,H-PAD+28);
 const X=f=>PAD+(W-10-PAD)*f/(NFR-1), Yv=p=>H-PAD-(H-PAD-10)*p/100;
 ctx.strokeStyle='#e74c3c'; ctx.lineWidth=2; ctx.beginPath();
 for(let k=0;k<CUM.length;k++){const x=X(k),y=Yv(CUM[k]); k?ctx.lineTo(x,y):ctx.moveTo(x,y);} ctx.stroke();
 ctx.lineWidth=1; ctx.strokeStyle='#5bf'; ctx.beginPath(); ctx.moveTo(X(f),10); ctx.lineTo(X(f),H-PAD); ctx.stroke();
 ctx.fillStyle='#5bf'; ctx.beginPath(); ctx.arc(X(f),Yv(pct),4,0,7); ctx.fill();
 lab.textContent='frame '+f+' / '+(NFR-1)+'   ·   '+pct.toFixed(1)+'% embolised';
}
function show(i){i=Math.max(0,Math.min(TPS.length-1,i)); sld.value=i; raw.src=RAW[i]; ov.src=OV[i]; plot(i);}
sld.oninput=()=>show(+sld.value);
document.getElementById('prev').onclick=()=>show(+sld.value-1);
document.getElementById('next').onclick=()=>show(+sld.value+1);
let timer=null;
document.getElementById('play').onclick=e=>{
 if(timer){clearInterval(timer);timer=null;e.target.textContent='play';}
 else{e.target.textContent='pause'; timer=setInterval(()=>{let v=+sld.value+1; if(v>=TPS.length)v=0; show(v);},120);}
};
document.onkeydown=e=>{if(e.key==='ArrowLeft')show(+sld.value-1); if(e.key==='ArrowRight')show(+sld.value+1);};
show(0);
</script></body></html>
"""


def save_final_png(final_img, out_dir: str, stem: str) -> str:
    """Write the RAW final (dry) frame as a plain PNG — annotate it in any app if preferred."""
    from PIL import Image
    path = os.path.join(out_dir, f"{stem}_final.png")
    Image.fromarray(np.clip(final_img, 0, 255).astype(np.uint8)).save(path)
    return path


def save_annotator(final_img, out_dir: str, stem: str, max_px: int = 1200,
                   heading: str | None = None, out_name: str | None = None) -> str:
    """Paint directly on the RAW final frame. No tissue mask, no dimming, no processing layer.

    You see the raw image exactly. Paint the air-filled tissue (Add/Erase brush); the optional
    Brightness slider pre-highlights bright pixels as a starting point. Export the mask as PNG.

    ``heading`` overrides the on-page title (e.g. to request SCALE annotation instead of
    embolism); ``out_name`` overrides the output .html filename.
    """
    import base64
    from PIL import Image
    Y, X = final_img.shape
    scale = min(1.0, max_px / max(Y, X))
    fi = Image.fromarray(np.clip(final_img, 0, 255).astype(np.uint8))
    if scale < 1.0:
        fi = fi.resize((max(1, int(X * scale)), max(1, int(Y * scale))), Image.BILINEAR)
    fin = np.asarray(fi, dtype=np.uint8)
    H, W = fin.shape
    fin_b64 = base64.b64encode(fin.tobytes()).decode()
    html = (_ANNOTATOR_HTML.replace("__TITLE__", heading or f"{stem} — raw final frame")
            .replace("__W__", str(W)).replace("__H__", str(H)).replace("__FIN__", fin_b64))
    path = os.path.join(out_dir, out_name or f"{stem}_annotate.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


_ANNOTATOR_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 body{background:#111;color:#eee;font-family:system-ui,sans-serif;margin:0;padding:12px}
 h2{margin:0 0 8px;font-size:16px}
 #wrap{position:relative;display:inline-block;border:1px solid #333}
 canvas{display:block;max-width:100%;cursor:crosshair}
 .controls{margin-top:10px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 .controls label{font-size:13px;color:#bcd}
 button{padding:6px 12px;background:#243;color:#eee;border:1px solid #365;border-radius:4px;cursor:pointer}
 button.on{background:#2a6;border-color:#4c9}
 #read{font-variant-numeric:tabular-nums;color:#9cf}
 input[type=range]{vertical-align:middle}
</style></head><body>
<h2>__TITLE__ &nbsp; <span style="font-size:12px;color:#9ab">this is your raw final image. Paint the air-filled tissue (Add/Erase). Brightness slider is an optional helper. Export PNG, send it to me.</span></h2>
<div id="wrap"><canvas id="c"></canvas></div>
<div class="controls">
 <button id="mAdd" class="on">Add brush</button>
 <button id="mErase">Erase brush</button>
 <label>Brush <input id="bsz" type="range" min="2" max="80" value="22"></label>
 <label>Brightness ≥ <input id="thr" type="range" min="0" max="256" value="256"> <span id="thrv">off</span></label>
 <button id="clear">Clear</button>
 <button id="exp">Export PNG</button>
 <span id="read"></span>
</div>
<script>
const W=__W__, H=__H__;
function dec(s){const b=atob(s),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a;}
const FIN=dec("__FIN__");
const added=new Uint8Array(W*H), removed=new Uint8Array(W*H);
const c=document.getElementById('c'); c.width=W; c.height=H;
const ctx=c.getContext('2d'); const img=ctx.createImageData(W,H);
const thr=document.getElementById('thr'), thrv=document.getElementById('thrv'),
      bsz=document.getElementById('bsz'), read=document.getElementById('read');
let mode='add';
function emb(i,t){ return (added[i] || (t<=255 && FIN[i]>=t && !removed[i])) ? 1:0; }
function draw(){
 const t=+thr.value; thrv.textContent = t>255 ? 'off' : t; let n=0;
 for(let i=0;i<W*H;i++){
   const g=FIN[i]; let r=g,gg=g,b=g;
   if(emb(i,t)){r=Math.min(255,g*0.3+190);gg=g*0.3;b=g*0.3;n++;}
   const j=i*4; img.data[j]=r;img.data[j+1]=gg;img.data[j+2]=b;img.data[j+3]=255;
 }
 ctx.putImageData(img,0,0);
 read.textContent='marked '+(100*n/(W*H)).toFixed(1)+'% of image';
}
thr.oninput=draw;
function setMode(m){mode=m;for(const id of ['mAdd','mErase'])document.getElementById(id).classList.remove('on');
 document.getElementById(m==='add'?'mAdd':'mErase').classList.add('on');}
document.getElementById('mAdd').onclick=()=>setMode('add');
document.getElementById('mErase').onclick=()=>setMode('erase');
document.getElementById('clear').onclick=()=>{added.fill(0);removed.fill(0);draw();};
let painting=false;
function paint(ev){
 const r=c.getBoundingClientRect(), sx=W/r.width, sy=H/r.height;
 const cx=Math.round((ev.clientX-r.left)*sx), cy=Math.round((ev.clientY-r.top)*sy);
 const rad=+bsz.value;
 for(let dy=-rad;dy<=rad;dy++)for(let dx=-rad;dx<=rad;dx++){
   if(dx*dx+dy*dy>rad*rad)continue;
   const x=cx+dx,y=cy+dy; if(x<0||y<0||x>=W||y>=H)continue; const i=y*W+x;
   if(mode==='add'){added[i]=1;removed[i]=0;} else {removed[i]=1;added[i]=0;}
 }
 draw();
}
c.onmousedown=e=>{painting=true;paint(e);};
c.onmousemove=e=>{if(painting)paint(e);};
window.onmouseup=()=>{painting=false;};
document.getElementById('exp').onclick=()=>{
 const t=+thr.value; const o=document.createElement('canvas'); o.width=W;o.height=H;
 const octx=o.getContext('2d'); const im=octx.createImageData(W,H);
 for(let i=0;i<W*H;i++){const v=emb(i,t)?255:0;const j=i*4;im.data[j]=v;im.data[j+1]=v;im.data[j+2]=v;im.data[j+3]=255;}
 octx.putImageData(im,0,0);
 const a=document.createElement('a'); a.download='embolised_mask.png'; a.href=o.toDataURL('image/png'); a.click();
};
setMode('add'); draw();
</script></body></html>
"""


def write_manifest(out_dir: str, stem: str, manifest: dict) -> str:
    path = os.path.join(out_dir, f"{stem}_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path
