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
    scale = min(1.0, max_px / max(Y, X))
    frames = []
    ts = list(range(0, T, stride))
    for t in ts:
        # per-frame uint8, never a full-stack float64 copy (see save_embolism_viewer)
        base = (np.clip((s[t] - lo) / rng, 0, 1) * 255).astype(np.uint8)
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
    # Normalise ONE frame at a time (uint8), never the whole stack: `s` is float32 but
    # `lo` is float64, so `s - lo` would promote the entire (T,Y,X) stack to float64
    # (e.g. 14.6 GiB for 400x1920x2560) even though only the rendered timepoints are used.
    # A single 2D frame is ~tens of MB and released each iteration. Values are identical to
    # the previous `g[t]` (the same expression, sliced per frame).
    def _grey_u8(t):
        return (np.clip((s[t] - lo) / rng, 0, 1) * 255).astype(np.uint8)

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
        base = _grey_u8(t)
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


def save_event_tiffs(onset, shape, n_frames: int, out_dir: str, stem: str,
                     cumulative: bool = True, compress: bool = True) -> dict:
    """Per-frame binary TIFF of *newly* detected pixels, and (optionally) the cumulative
    binary TIFF (v0.2.0 brief §7; shipped early as a diagnostic in §1).

    ``onset`` is the per-pixel onset-frame map (-1 where not embolised), ``shape`` the
    accepted mask. Frame ``t`` of the per-frame stack holds pixels whose onset == t; frame
    ``t`` of the cumulative stack holds pixels with 0 <= onset <= t. Both are ``uint8``
    {0,255}. Deflate-compressed by default because a full-res 400-frame stack is large —
    the masks are sparse so compression is very effective.

    Returns the written paths. The two stacks are exactly consistent by construction: the
    cumulative stack is the running OR of the per-frame stack.
    """
    import tifffile
    onset = np.asarray(onset)
    shape = np.asarray(shape, dtype=bool)
    Y, X = shape.shape
    # zlib-compressed: the masks are sparse so the on-disk size is small even though the
    # in-memory (T,Y,X) uint8 array is allocated (~1 byte/px; halve dims with --bin for
    # a big full-res run on a memory-tight machine).
    kw = {"compression": "zlib"} if compress else {}
    valid = shape & (onset >= 0) & (onset < n_frames)
    ys, xs = np.nonzero(valid)

    per = np.zeros((n_frames, Y, X), np.uint8)
    per[onset[valid], ys, xs] = 255
    per_path = os.path.join(out_dir, f"{stem}_events_per_frame.tif")
    tifffile.imwrite(per_path, per, **kw)
    out = {"per_frame": per_path}

    if cumulative:
        cum = (np.cumsum(per.astype(bool), axis=0) > 0).astype(np.uint8) * 255
        cum_path = os.path.join(out_dir, f"{stem}_events_cumulative.tif")
        tifffile.imwrite(cum_path, cum, **kw)
        out["cumulative"] = cum_path
    return out


def build_onset_table(onset, shape, confidence=None, transition=None,
                      scale_um2: float = 1.0, frame_interval_s: float | None = None):
    """One row per connected event region (v0.2.0 brief §7): the onset-frame marking of a
    complete region, with its area, location, confidence and transition ambiguity.

    A region is a connected component of the accepted ``shape`` mask. Its representative
    ``onset_frame`` is the median of its pixels' onsets; ``onset_spread`` (max-min) and the
    ``ambiguous`` flag expose where the fill is not clean within the permitted transition.
    Returns a ``pandas.DataFrame`` sorted by onset.
    """
    from scipy.ndimage import label
    from skimage.measure import regionprops
    onset = np.asarray(onset)
    shape = np.asarray(shape, dtype=bool)
    lbl, n = label(shape)
    rows = []
    for p in regionprops(lbl):
        coords = tuple(p.coords.T)
        ons = onset[coords]
        ons = ons[ons >= 0]
        if ons.size == 0:
            continue
        onset_med = int(np.median(ons))
        spread = int(ons.max() - ons.min())
        tr = int(np.max(transition[coords])) if transition is not None else spread + 1
        row = {
            "event_id": int(p.label),
            "onset_frame": onset_med,
            "onset_frame_min": int(ons.min()),
            "onset_frame_max": int(ons.max()),
            "onset_spread": spread,
            "area_px": int(p.area),
            "area_um2": float(p.area * scale_um2),
            "centroid_y": float(p.centroid[0]),
            "centroid_x": float(p.centroid[1]),
            "transition_frames": tr,
            "ambiguous": bool(spread > max(tr, 1)),
        }
        if confidence is not None:
            row["mean_confidence"] = float(np.asarray(confidence)[coords].mean())
        if frame_interval_s is not None:
            row["onset_time_s"] = onset_med * float(frame_interval_s)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("onset_frame").reset_index(drop=True) if rows \
        else pd.DataFrame(columns=["event_id", "onset_frame", "area_px"])
    return df


def write_onset_table(onset, shape, out_dir: str, stem: str, confidence=None,
                      transition=None, scale_um2: float = 1.0,
                      frame_interval_s: float | None = None) -> str:
    df = build_onset_table(onset, shape, confidence, transition, scale_um2, frame_interval_s)
    path = os.path.join(out_dir, f"{stem}_onset_table.csv")
    df.to_csv(path, index=False)
    return path


def save_event_lut(onset, shape, final, out_dir: str, stem: str,
                   n_frames: int | None = None) -> str:
    """Event-time coloured overlay (LUT): each accepted pixel coloured by its onset frame
    over the grey final frame, with a colourbar (v0.2.0 brief §7)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    onset = np.asarray(onset, float)
    shape = np.asarray(shape, dtype=bool)
    T = n_frames if n_frames is not None else (int(onset[shape].max()) + 1 if shape.any() else 1)
    vmin, vmax = np.percentile(final, [2, 99])
    whenmap = np.where(shape & (onset >= 0), onset, np.nan)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(final, cmap="gray", vmin=vmin, vmax=vmax)
    m = ax.imshow(whenmap, cmap="turbo", vmin=0, vmax=max(T - 1, 1), alpha=0.9)
    ax.set_title(f"{stem} — event onset time"); ax.axis("off")
    plt.colorbar(m, ax=ax, fraction=0.046, label="onset frame")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{stem}_event_lut.png")
    plt.savefig(path, dpi=100); plt.close()
    return path


def save_evidence_maps(final, evidence: dict, out_dir: str, stem: str, dpi: int = 100) -> dict:
    """Export the fused detector's evidence maps (v0.3.1 hybrid brief, Outputs section).

    ``evidence`` maps a name -> 2-D array. Scalar maps (ridge response, temporal
    confidence, combined confidence) are drawn with a heat colourmap over the grey final
    frame; boolean maps (the artifact/exclusion mask) are drawn as a red overlay. Each is
    written as ``<stem>_<name>.png`` and keeps image/pixel correspondence with the frames.
    Returns the written paths. Only called for methods that provide an ``evidence`` dict
    (hybrid-ridge-temporal); other methods are unaffected.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    final = np.asarray(final, np.float32)
    vmin, vmax = np.percentile(final, [2, 99])
    paths = {}
    for name, arr in (evidence or {}).items():
        arr = np.asarray(arr)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(final, cmap="gray", vmin=vmin, vmax=vmax)
        if arr.dtype == bool:
            ov = np.zeros((*arr.shape, 4)); ov[arr] = [1, 0.15, 0.15, 0.6]
            ax.imshow(ov)
        else:
            m = ax.imshow(np.where(arr > 0, arr, np.nan), cmap="magma", alpha=0.85)
            plt.colorbar(m, ax=ax, fraction=0.046, label=name)
        ax.set_title(f"{stem} — {name}"); ax.axis("off")
        plt.tight_layout()
        path = os.path.join(out_dir, f"{stem}_{name}.png")
        plt.savefig(path, dpi=dpi); plt.close()
        paths[name] = path
    return paths


def write_run_metadata(out_dir: str, stem: str, meta: dict) -> str:
    """Full provenance for a run (v0.2.0 brief §7): calibration, frame interval, method,
    resolved settings, thresholds, model/version. A superset of the older ``_cs_params.json``."""
    path = os.path.join(out_dir, f"{stem}_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return path


def confidence_to_bytes(confidence):
    """Quantise a 0..1 confidence map to uint8 0..255 for compact embedding in the viewer."""
    c = np.clip(np.asarray(confidence, np.float32), 0, 1)
    return (c * 255).round().astype(np.uint8)


def save_confidence_viewer(final, confidence, out_dir: str, stem: str, max_px: int = 820,
                           tissue=None) -> str:
    """Interactive confidence-threshold viewer (v0.2.0 brief §6).

    Drag a slider: pixels with confidence >= threshold show **accepted** (red); a band just
    below shows **uncertain** (amber); the rest are **rejected**. The embolised-% readout
    updates live from the cached per-pixel confidence — no recompute, no server. The score
    is a ranking, **not** a statistical confidence interval (labelled as such in the UI).
    """
    import io as _io, base64, json as _json
    from PIL import Image
    final = np.asarray(final, np.float32)
    Y, X = final.shape
    lo, hi = np.percentile(final, [2, 99]); rng = max(hi - lo, 1e-3)
    g = np.clip((final - lo) / rng, 0, 1)
    scale = min(1.0, max_px / max(Y, X))
    newsize = (max(1, int(X * scale)), max(1, int(Y * scale)))

    cb = confidence_to_bytes(confidence)
    if tissue is not None:
        cb = np.where(np.asarray(tissue, bool), cb, 0).astype(np.uint8)
    # downscale the confidence map alongside the image so indices line up
    gim = Image.fromarray((g * 255).astype(np.uint8))
    cim = Image.fromarray(cb)
    if scale < 1.0:
        gim = gim.resize(newsize, Image.BILINEAR)
        cim = cim.resize(newsize, Image.NEAREST)
    gb = np.asarray(gim, np.uint8)
    conf_small = np.asarray(cim, np.uint8)
    W2, H2 = newsize
    n_tissue = int((conf_small > 0).sum())

    html = (_CONF_HTML
            .replace("__TITLE__", f"{stem} — confidence threshold")
            .replace("__W__", str(W2)).replace("__H__", str(H2))
            .replace("__NTIS__", str(max(n_tissue, 1)))
            .replace("__GREY__", base64.b64encode(gb.tobytes()).decode())
            .replace("__CONF__", base64.b64encode(conf_small.tobytes()).decode()))
    path = os.path.join(out_dir, f"{stem}_confidence.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


_CONF_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 body{background:#111;color:#eee;font-family:system-ui,sans-serif;margin:0;padding:14px}
 h2{margin:0 0 8px;font-size:16px}
 #wrap{position:relative;display:inline-block;border:1px solid #333}
 canvas{display:block;max-width:100%}
 .controls{margin-top:10px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 label{font-size:13px;color:#bcd}
 #read{font-variant-numeric:tabular-nums;color:#9cf}
 .key span{display:inline-block;width:11px;height:11px;margin:0 3px -1px 10px;border:1px solid #555}
</style></head><body>
<h2>__TITLE__ <span style="font-size:12px;color:#9ab">— drag to set the accept threshold. Score is a ranking, not a statistical confidence interval.</span></h2>
<div id="wrap"><canvas id="c"></canvas></div>
<div class="controls">
 <label>Accept ≥ <input id="thr" type="range" min="0" max="255" value="128"> <span id="thrv"></span></label>
 <label>Uncertain band <input id="band" type="range" min="0" max="80" value="25"> <span id="bandv"></span></label>
 <span class="key">key:<span style="background:#e7301e"></span>accepted<span style="background:#e8a200"></span>uncertain</span>
 <span id="read"></span>
</div>
<script>
const W=__W__,H=__H__,NTIS=__NTIS__;
function dec(s){const b=atob(s),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a;}
const GREY=dec("__GREY__"), CONF=dec("__CONF__");
const c=document.getElementById('c');c.width=W;c.height=H;
const ctx=c.getContext('2d'),img=ctx.createImageData(W,H);
const thr=document.getElementById('thr'),band=document.getElementById('band'),
      thrv=document.getElementById('thrv'),bandv=document.getElementById('bandv'),read=document.getElementById('read');
function draw(){
 const t=+thr.value,b=+band.value; thrv.textContent=(t/255).toFixed(2); bandv.textContent=(b/255).toFixed(2);
 let acc=0;
 for(let i=0;i<W*H;i++){
   const gg=GREY[i],cf=CONF[i];let r=gg,g=gg,bl=gg;
   if(cf>=t){r=Math.min(255,gg*0.25+200);g=gg*0.25;bl=gg*0.25;acc++;}
   else if(cf>=t-b && cf>0){r=Math.min(255,gg*0.3+180);g=Math.min(255,gg*0.3+120);bl=gg*0.3;}
   const j=i*4;img.data[j]=r;img.data[j+1]=g;img.data[j+2]=bl;img.data[j+3]=255;
 }
 ctx.putImageData(img,0,0);
 read.textContent='accepted '+(100*acc/NTIS).toFixed(1)+'% of tissue';
}
thr.oninput=draw;band.oninput=draw;draw();
</script></body></html>
"""


def prepare_annotator_frames(stack, n_frames: int = 12, max_px: int = 1000):
    """Pick ~``n_frames`` evenly-spaced frames and return (indices, base64 JPEGs, scale).

    ``scale`` is display/original so the UI can export masks in **original-resolution**
    coordinates. Shared, testable data-prep for the multi-frame annotator (brief §5)."""
    import io as _io, base64
    from PIL import Image
    stack = np.asarray(stack)
    T, Y, X = stack.shape
    idx = sorted(set(np.linspace(0, T - 1, min(n_frames, T)).astype(int).tolist()))
    scale = min(1.0, max_px / max(Y, X))
    newsize = (max(1, int(X * scale)), max(1, int(Y * scale)))
    lo, hi = np.percentile(stack, [2, 99]); rng = max(hi - lo, 1e-3)
    imgs = []
    for t in idx:
        g = np.clip((stack[t].astype(np.float32) - lo) / rng, 0, 1)
        im = Image.fromarray((g * 255).astype(np.uint8))
        if scale < 1.0:
            im = im.resize(newsize, Image.BILINEAR)
        b = _io.BytesIO(); im.save(b, format="JPEG", quality=75)
        imgs.append("data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode())
    return idx, imgs, scale


def save_multiframe_annotator(stack, out_dir: str, stem: str,
                              categories=("lumen", "inter-rhizoid drying", "scale", "tape",
                                          "unknown"),
                              version: str = "?", n_frames: int = 12, max_px: int = 1000) -> str:
    """Multi-frame annotation tool (brief §5): frame slider, add/erase brush, per-action undo,
    labelled categories. Exports a JSON of strokes with **frame id, label, original-resolution
    coordinates and source version** — the versioned training-queue payload for the ML step.

    Live UI is exercised on a desktop (the VM has no browser display); the data-prep
    (:func:`prepare_annotator_frames`) and the export schema are unit-tested."""
    import json as _json
    idx, imgs, scale = prepare_annotator_frames(stack, n_frames, max_px)
    orig_per_disp = 1.0 / scale if scale > 0 else 1.0
    html = (_MULTI_ANN_HTML
            .replace("__TITLE__", f"{stem} — multi-frame annotator")
            .replace("__STEM__", stem)
            .replace("__VERSION__", version)
            .replace("__IMGS__", _json.dumps(imgs))
            .replace("__IDX__", _json.dumps(idx))
            .replace("__CATS__", _json.dumps(list(categories)))
            .replace("__ORIGPERDISP__", repr(float(orig_per_disp))))
    path = os.path.join(out_dir, f"{stem}_multiframe_annotate.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


_MULTI_ANN_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 body{background:#111;color:#eee;font-family:system-ui,sans-serif;margin:0;padding:12px}
 h2{margin:0 0 8px;font-size:16px}
 #wrap{position:relative;display:inline-block;border:1px solid #333}
 #base,#ov{position:absolute;left:0;top:0;max-width:100%}
 #ov{cursor:crosshair}
 .controls{margin-top:10px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 label{font-size:13px;color:#bcd}
 button{padding:6px 12px;background:#243;color:#eee;border:1px solid #365;border-radius:4px;cursor:pointer}
 button.on{background:#2a6;border-color:#4c9}
 select{padding:5px;background:#182;color:#eee;border:1px solid #365;border-radius:4px}
 #read{font-variant-numeric:tabular-nums;color:#9cf}
</style></head><body>
<h2>__TITLE__ <span style="font-size:12px;color:#9ab">— pick a frame, choose a label, paint. Undo per stroke. Export JSON when done.</span></h2>
<div id="wrap"><img id="base"><canvas id="ov"></canvas></div>
<div class="controls">
 <label>Frame <input id="fr" type="range" min="0" value="0"> <span id="frv"></span></label>
 <label>Label <select id="cat"></select></label>
 <button id="mAdd" class="on">Add</button><button id="mErase">Erase</button>
 <label>Brush <input id="bsz" type="range" min="2" max="60" value="16"></label>
 <button id="undo">Undo</button><button id="clearf">Clear frame</button>
 <button id="exp">Export JSON</button>
 <span id="read"></span>
</div>
<script>
const IMGS=__IMGS__, IDX=__IDX__, CATS=__CATS__, ORIGPERDISP=__ORIGPERDISP__,
      STEM="__STEM__", VERSION="__VERSION__";
const base=document.getElementById('base'),ov=document.getElementById('ov'),
      fr=document.getElementById('fr'),frv=document.getElementById('frv'),
      cat=document.getElementById('cat'),bsz=document.getElementById('bsz'),read=document.getElementById('read');
CATS.forEach(c=>{const o=document.createElement('option');o.value=o.textContent=c;cat.appendChild(o);});
fr.max=IMGS.length-1;
let mode='add', painting=false;
// strokes[frameIndex] = array of {label, mode, pts:[[x,y],...], r}
const strokes=IMGS.map(()=>[]);
let cur=null;
function setMode(m){mode=m;document.getElementById('mAdd').classList.toggle('on',m==='add');
 document.getElementById('mErase').classList.toggle('on',m==='erase');}
document.getElementById('mAdd').onclick=()=>setMode('add');
document.getElementById('mErase').onclick=()=>setMode('erase');
function fi(){return +fr.value;}
function redraw(){
 const ctx=ov.getContext('2d');ctx.clearRect(0,0,ov.width,ov.height);
 for(const s of strokes[fi()]){
   ctx.fillStyle = s.mode==='erase' ? 'rgba(20,20,20,0.9)' : 'rgba(231,48,30,0.5)';
   for(const p of s.pts){ctx.beginPath();ctx.arc(p[0],p[1],s.r,0,7);ctx.fill();}
 }
 let n=0; strokes.forEach(fs=>n+=fs.length);
 read.textContent=IMGS.length+' frames · '+n+' strokes total';
}
function show(){const i=fi();base.src=IMGS[i];frv.textContent='t='+IDX[i];
 base.onload=()=>{ov.width=base.width;ov.height=base.height;redraw();};
 if(base.complete){ov.width=base.width;ov.height=base.height;redraw();}}
fr.oninput=show;
function pos(ev){const r=ov.getBoundingClientRect();return [(ev.clientX-r.left)*ov.width/r.width,(ev.clientY-r.top)*ov.height/r.height];}
ov.onmousedown=e=>{painting=true;cur={label:cat.value,mode:mode,r:+bsz.value,pts:[pos(e)]};strokes[fi()].push(cur);redraw();};
ov.onmousemove=e=>{if(painting&&cur){cur.pts.push(pos(e));redraw();}};
window.onmouseup=()=>{painting=false;cur=null;};
document.getElementById('undo').onclick=()=>{strokes[fi()].pop();redraw();};
document.getElementById('clearf').onclick=()=>{strokes[fi()]=[];redraw();};
document.getElementById('exp').onclick=()=>{
 const out={stem:STEM,source_version:VERSION,orig_per_disp:ORIGPERDISP,
   frames:IDX.map((t,i)=>({frame_id:t,strokes:strokes[i].map(s=>({label:s.label,mode:s.mode,
     radius_orig:s.r*ORIGPERDISP, points_orig:s.pts.map(p=>[Math.round(p[0]*ORIGPERDISP),Math.round(p[1]*ORIGPERDISP)])}))}))
   .filter(f=>f.strokes.length)};
 const a=document.createElement('a');a.download=STEM+'_annotations.json';
 a.href='data:application/json,'+encodeURIComponent(JSON.stringify(out,null,1));a.click();
};
show();
</script></body></html>
"""


def write_manifest(out_dir: str, stem: str, manifest: dict) -> str:
    path = os.path.join(out_dir, f"{stem}_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path


# =========================================================================== #
# rhizoid-events outputs (per-rhizoid tracing, count + timing). Written by
# colour_shape.run() when the method returns an ``events`` payload.
# =========================================================================== #

def _distinct_hues(ids):
    """Golden-ratio-spaced HSV hues so adjacent rhizoids get well-separated colours."""
    from matplotlib.colors import hsv_to_rgb
    out = {}
    for i, tid in enumerate(ids):
        h = ((i + 1) * 0.61803398875) % 1.0
        out[tid] = hsv_to_rgb([h, 0.82, 0.98])
    return out


def save_rhizoid_map(tubes, final, out_dir: str, stem: str, dpi: int = 120) -> str:
    """Single image of every traced rhizoid, each a distinct hue over the dimmed
    final frame (the expert's 'single image that shows the traced rhizoids')."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tubes = np.asarray(tubes, np.int32)
    ids = [int(v) for v in np.unique(tubes) if v != 0]
    hue = _distinct_hues(ids)
    f = np.asarray(final, np.float32)
    vmin, vmax = np.percentile(f, [2, 99])
    g = np.clip((f - vmin) / (vmax - vmin + 1e-9), 0, 1)
    rgb = np.stack([g] * 3, -1) * 0.5
    for tid in ids:
        rgb[tubes == tid] = hue[tid]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(rgb); ax.axis("off")
    ax.set_title(f"{stem} — {len(ids)} traced rhizoids")
    path = os.path.join(out_dir, f"{stem}_rhizoids.png")
    plt.tight_layout(); plt.savefig(path, dpi=dpi); plt.close()
    return path


def save_cavitation_series(stack, tubes, events, out_dir: str, stem: str,
                           make_gif: bool = True, max_frames: int | None = None) -> str:
    """Frame-by-frame series highlighting each rhizoid as it cavitates.

    Writes one PNG per frame to ``<stem>_cavitation/`` and an animated GIF. A
    cavitating strand lights at its event frame and stays lit (the spreading
    staircase the expert reads). Memory-safe: one frame rendered at a time.
    """
    from PIL import Image
    tubes = np.asarray(tubes, np.int32)
    stack = np.asarray(stack, np.float32)
    T = stack.shape[0] if max_frames is None else min(int(max_frames), stack.shape[0])
    cav = {e["id"]: e["cavitation_frame"] for e in events
           if e["cls"] == "cavitation" and e["cavitation_frame"] is not None}
    ids = sorted(cav)
    hue = _distinct_hues(ids)
    masks = {tid: (tubes == tid) for tid in ids}
    step = max(1, T // 8)
    vmin, vmax = np.percentile(stack[::step], [2, 99])
    sub = os.path.join(out_dir, f"{stem}_cavitation")
    os.makedirs(sub, exist_ok=True)
    frames = []
    for t in range(T):
        g = np.clip((stack[t] - vmin) / (vmax - vmin + 1e-9), 0, 1)
        rgb = np.stack([g] * 3, -1)
        for tid in ids:
            if cav[tid] <= t:
                m = masks[tid]
                rgb[m] = 0.3 * rgb[m] + 0.7 * np.asarray(hue[tid])
        img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        img.save(os.path.join(sub, f"{stem}_cav_{t:03d}.png"))
        if make_gif:
            frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=64))
    if make_gif and frames:
        frames[0].save(os.path.join(out_dir, f"{stem}_cavitation.gif"), save_all=True,
                       append_images=frames[1:], duration=120, loop=0)
    return sub


def write_rhizoid_table(events, out_dir: str, stem: str, frame_interval_s: float = 15.0,
                        scale_um2: float | None = None) -> str:
    """Per-rhizoid timing table: id, class, cavitation frame/time, size, confidence."""
    rows = []
    for e in sorted(events, key=lambda z: z["id"]):
        f = e["cavitation_frame"]
        row = {"id": e["id"], "class": e["cls"], "cavitation_frame": f,
               "cavitation_time_s": (None if f is None else round(f * frame_interval_s, 2)),
               "length_px": e["length_px"], "bright_frac_peak": e["bright_frac_peak"],
               "confidence": e["confidence"]}
        if scale_um2:
            row["length_um2"] = round(e["length_px"] * scale_um2, 2)
        rows.append(row)
    path = os.path.join(out_dir, f"{stem}_rhizoids.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_rhizoid_count(events, n_frames: int, out_dir: str, stem: str) -> int:
    """Cumulative count of rhizoids cavitated vs frame (the staircase CSV).
    Returns the final total count."""
    T = int(n_frames)
    counts = np.zeros(T, np.int64)
    for e in events:
        if e["cls"] == "cavitation" and e["cavitation_frame"] is not None:
            f = int(e["cavitation_frame"])
            if 0 <= f < T:
                counts[f] += 1
    cum = np.cumsum(counts)
    pd.DataFrame({"frame": np.arange(T), "n_cavitated_cumulative": cum}).to_csv(
        os.path.join(out_dir, f"{stem}_rhizoid_count.csv"), index=False)
    return int(cum[-1]) if T else 0
