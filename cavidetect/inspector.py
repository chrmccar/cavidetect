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
                   max_px: int = 820, n_timepoints: int = 90,
                   tube_ids=None, out_name: str | None = None,
                   seed_edits: dict | None = None) -> str:
    """Write the self-contained inspector HTML (``<stem>_inspect.html``).

    ``result`` is the ``colour_shape.measure`` dict (needs ``onset``, ``shape``, ``final``).
    Raw + difference panels are pre-rendered JPEGs; the onset-frame map and the per-rhizoid
    identity map are embedded as lossless PNGs and coloured live in the browser, so the
    by-time / by-tube / static toggles and the animation are all instant.

    ``tube_ids`` overrides the identity map (used by the corrected inspector after split/merge);
    ``out_name`` overrides the filename; ``seed_edits`` is an exported edits doc embedded so a
    corrected inspector carries the committed corrections forward for further iteration.
    """
    os.makedirs(out_dir, exist_ok=True)
    T = stack.shape[0]
    shape = np.asarray(result["shape"], dtype=bool)
    onset = np.asarray(result["onset"]).astype(np.int32)
    Y, X = shape.shape

    tps, raw_b64, diff_b64, newsize = prepare_inspector_frames(stack, max_px, n_timepoints)

    valid = shape & (onset >= 0)
    onset_small = _resize_map_nearest(np.where(valid, onset, 0), newsize)
    valid_small = _resize_map_nearest(valid.astype(np.int32), newsize) > 0
    onset_png = _encode_u16_png(onset_small, valid_small)

    tubes = label_tubes(shape) if tube_ids is None else np.asarray(tube_ids).astype(np.int32)
    tubes_small = _resize_map_nearest(tubes, newsize)
    tubes_png = _encode_u16_png(tubes_small, tubes_small > 0)

    seed = (seed_edits.get("edits", []) if isinstance(seed_edits, dict) else []) or []
    html = (_INSPECTOR_HTML
            .replace("__TITLE__", f"{stem} — inspector")
            .replace("__RAW__", _json.dumps(raw_b64))
            .replace("__DIFF__", _json.dumps(diff_b64))
            .replace("__TPS__", _json.dumps(tps))
            .replace("__ONSET__", _json.dumps(onset_png))
            .replace("__TUBES__", _json.dumps(tubes_png))
            .replace("__NFR__", str(T))
            .replace("__NTUBES__", str(int(tubes.max())))
            .replace("__OY__", str(Y)).replace("__OX__", str(X))
            .replace("__STEM__", stem)
            .replace("__SEED__", _json.dumps(seed)))
    path = os.path.join(out_dir, out_name or f"{stem}_inspect.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# ============================================================================ Phase 2
# Editing toolkit + commit->recompute loop. The inspector exports an edits JSON (display
# coords); `apply_corrections` re-applies the FULL edit set to the preserved auto baseline
# and recomputes the measurement, so iterating "until satisfactory" never drifts.

def save_inspector_state(out_dir: str, stem: str, onset, shape, tissue, ridge,
                         meta: dict, tubes=None) -> str:
    """Persist the auto run's committed arrays so a correction can recompute without
    re-running detection. Small: onset int16, shape/tissue bool, ridge float16.
    ``tubes`` (the traced per-rhizoid identity map) is stored when supplied so the
    rhizoid-events correction loop edits identities on the real traced map."""
    import numpy as _np
    path = os.path.join(out_dir, f"{stem}_inspect_state.npz")
    arrs = dict(
        onset=_np.asarray(onset).astype(_np.int16),
        shape=_np.asarray(shape, dtype=bool),
        tissue=_np.asarray(tissue, dtype=bool),
        ridge=_np.asarray(ridge, dtype=_np.float16),
        meta=_json.dumps(meta))
    if tubes is not None:
        arrs["tubes"] = _np.asarray(tubes, dtype=_np.int32)
    _np.savez_compressed(path, **arrs)
    return path


def load_inspector_state(path: str) -> dict:
    d = np.load(path, allow_pickle=False)
    out = dict(onset=d["onset"].astype(np.int32), shape=d["shape"].astype(bool),
               tissue=d["tissue"].astype(bool), ridge=d["ridge"].astype(np.float32),
               meta=_json.loads(str(d["meta"])))
    if "tubes" in d.files:
        out["tubes"] = d["tubes"].astype(np.int32)
    return out


def rasterize_stroke(points, radius, hw) -> np.ndarray:
    """Dilate a polyline (list of [row, col]) by ``radius`` into a boolean mask of shape ``hw``.
    A single point stamps one disc; consecutive points are joined by discs along the segment."""
    Y, X = hw
    mask = np.zeros((Y, X), bool)
    pts = [(int(round(p[0])), int(round(p[1]))) for p in points]
    r = max(1, int(round(radius)))

    def disc(cy, cx):
        y0, y1 = max(0, cy - r), min(Y, cy + r + 1)
        x0, x1 = max(0, cx - r), min(X, cx + r + 1)
        if y0 >= y1 or x0 >= x1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r

    if not pts:
        return mask
    if len(pts) == 1:
        disc(*pts[0])
    for (y0, x0), (y1, x1) in zip(pts, pts[1:]):
        n = max(abs(y1 - y0), abs(x1 - x0)) + 1
        for t in np.linspace(0, 1, n):
            disc(int(round(y0 + (y1 - y0) * t)), int(round(x0 + (x1 - x0) * t)))
    return mask


def _split_tube(tube, point, cut_radius: int = 3) -> np.ndarray:
    """Cut the rhizoid under ``point`` ([row, col]) in two: remove a small disc, relabel the
    component's connected halves, then grow the new labels back over the whole component."""
    from skimage.measure import label as sklabel
    from scipy.ndimage import distance_transform_edt
    cy, cx = int(round(point[0])), int(round(point[1]))
    if not (0 <= cy < tube.shape[0] and 0 <= cx < tube.shape[1]):
        return tube
    L = int(tube[cy, cx])
    if L == 0:
        return tube
    comp = tube == L
    cut = comp & ~rasterize_stroke([point], cut_radius, tube.shape)
    lbl = sklabel(cut)
    if lbl.max() <= 1:
        return tube                              # the cut did not separate it
    _, inds = distance_transform_edt(lbl == 0, return_indices=True)
    grown = lbl[tuple(inds)]                      # nearest cut-label everywhere
    out = tube.copy()
    nextid = int(tube.max())
    idmap = {1: L}
    for k in range(2, int(lbl.max()) + 1):
        nextid += 1
        idmap[k] = nextid
    for k, new in idmap.items():
        out[comp & (grown == k)] = new
    return out


def _merge_tubes(tube, points) -> np.ndarray:
    """Merge the two rhizoids under ``points`` ([[row,col],[row,col]]) into one label."""
    (y0, x0), (y1, x1) = [(int(round(p[0])), int(round(p[1]))) for p in points]
    out = tube
    if 0 <= y0 < tube.shape[0] and 0 <= x0 < tube.shape[1] \
            and 0 <= y1 < tube.shape[0] and 0 <= x1 < tube.shape[1]:
        a, b = int(tube[y0, x0]), int(tube[y1, x1])
        if a and b and a != b:
            out = tube.copy()
            out[tube == b] = a
    return out


def recompute_curve(onset, shape, tissue_px: int, T: int):
    """Cumulative embolised-pixel count + fraction-of-tissue per frame, from onset & mask."""
    valid = np.asarray(shape, bool) & (np.asarray(onset) >= 0)
    counts = np.bincount(np.asarray(onset)[valid], minlength=T)[:T]
    cum = np.cumsum(counts)
    frac = cum / max(int(tissue_px), 1)
    return cum, frac


def edits_from_json(doc: dict) -> list:
    """Convert an exported edits doc (display coords, [x, y]) to analysis-space edits
    ([row, col]). ``doc`` carries ``disp_size`` [W, H] and ``orig_size`` [Y, X]."""
    W, H = doc["disp_size"]
    Y, X = doc["orig_size"]
    sx, sy = X / max(W, 1), Y / max(H, 1)
    sr = (sx + sy) / 2
    out = []
    for e in doc.get("edits", []):
        op = e["op"]
        if op in ("erase", "add"):
            pts = [[p[1] * sy, p[0] * sx] for p in e["points"]]
            out.append({"op": op, "points": pts, "radius": e.get("radius", 6) * sr})
        elif op == "split":
            x, y = e["point"]
            out.append({"op": op, "point": [y * sy, x * sx]})
        elif op == "merge":
            out.append({"op": op, "points": [[p[1] * sy, p[0] * sx] for p in e["points"]]})
    return out


def apply_edits(onset, shape, tissue, ridge, edits, stack=None,
                grow_ridge: float = 0.02, rise_frac: float = 0.4, base_tubes=None) -> dict:
    """Apply analysis-space edits to the committed arrays. Returns onset', shape', tube_ids'.

    erase removes pixels (false positives); add inserts a traced rhizoid, snapped to the
    ridge within the drawn corridor, with onset read from ``stack``; split/merge are
    identity-only (resolved by the clicked coordinates on the post-erase/add label map, so
    they never depend on fragile label numbers).

    ``base_tubes`` (rhizoid-events): the persisted traced identity map. When given, identity
    edits apply to it — erases clear those strands, adds get a fresh id — instead of
    re-deriving identity from the embolised mask, so the per-rhizoid count stays faithful."""
    onset = np.asarray(onset).astype(np.int32).copy()
    shape = np.asarray(shape, bool).copy()
    tissue = np.asarray(tissue, bool)
    ridge = np.asarray(ridge, np.float32)
    hw = shape.shape
    tube = np.asarray(base_tubes, np.int32).copy() if base_tubes is not None else None
    add_masks = []
    for e in edits:
        if e["op"] == "erase":
            m = rasterize_stroke(e["points"], e["radius"], hw)
            shape[m] = False
            onset[m] = -1
            if tube is not None:
                tube[m] = 0
        elif e["op"] == "add":
            m = rasterize_stroke(e["points"], e["radius"], hw) & tissue
            snap = m & (ridge >= grow_ridge)
            add_masks.append(snap if snap.any() else m)
    if add_masks:
        if stack is None:
            raise ValueError("add edits need the source stack for onset timing")
        new = np.zeros(hw, bool)
        for m in add_masks:
            new |= m
        new &= ~shape                              # only genuinely new pixels
        if new.any():
            from .colour_shape import onset_start_of_rise
            o = onset_start_of_rise(stack, new, rise_frac=rise_frac)
            shape[new] = True
            onset[new] = o[new]
            if tube is not None:
                tube[new] = int(tube.max()) + 1    # the added strand is a new identity
    if tube is None:
        tube = label_tubes(shape)
    for e in edits:
        if e["op"] == "split":
            tube = _split_tube(tube, e["point"])
        elif e["op"] == "merge":
            tube = _merge_tubes(tube, e["points"])
    return dict(onset=onset, shape=shape, tube_ids=tube)


def apply_corrections(run_dir: str, stem: str, edits_path: str, source: str,
                      bin_xy: int | None = None) -> dict:
    """CLI entry for the commit->recompute loop. Loads the preserved auto baseline state,
    re-applies the full edit set, recomputes the curve, and writes corrected outputs +
    a corrected inspector that carries the edits forward for further iteration."""
    import pandas as pd
    state = load_inspector_state(os.path.join(run_dir, f"{stem}_inspect_state.npz"))
    meta = state["meta"]
    bin_xy = int(bin_xy if bin_xy is not None else meta.get("bin", 1))
    with open(edits_path, encoding="utf-8") as fh:
        doc = _json.load(fh)
    edits = edits_from_json(doc)
    needs_stack = any(e["op"] == "add" for e in edits)
    stack = load_frames_binned(source, bin_xy) if needs_stack else None
    res = apply_edits(state["onset"], state["shape"], state["tissue"], state["ridge"],
                      edits, stack=stack, base_tubes=state.get("tubes"))
    T = int(meta.get("n_frames", int(np.asarray(state["onset"]).max()) + 1))
    tissue_px = int(state["tissue"].sum())
    cum, frac = recompute_curve(res["onset"], res["shape"], tissue_px, T)
    um2 = float(meta.get("um_per_analysis_px", 1.0)) ** 2
    pd.DataFrame({"frame": np.arange(T), "embolised_area_px": cum,
                  "embolised_area_um2": cum * um2,
                  "frac_tissue_embolised": frac}).to_csv(
        os.path.join(run_dir, f"{stem}_corrected_curve.csv"), index=False)
    # the corrected inspector needs the stack for the raw/difference panels
    if stack is None:
        stack = load_frames_binned(source, bin_xy)
    out = dict()
    # rhizoid-events: recompute the per-rhizoid COUNT + timing table from the edited tube
    # map (not just area). No coincident-merge here — it would undo the user's manual splits.
    if meta.get("method") == "rhizoid-events":
        from . import rhizoid_events as _re, outputs as _cout
        params = meta.get("params", {})
        mf = int(params.get("max_frames", 100))
        events, _, _ = _re.recompute_events(res["tube_ids"], stack[:mf], params)
        _cout.write_rhizoid_count(events, T, run_dir, f"{stem}_corrected")
        _cout.write_rhizoid_table(events, run_dir, f"{stem}_corrected",
                                  frame_interval_s=float(meta.get("frame_interval_s", 15.0)),
                                  scale_um2=um2)
        out["n_cavitated"] = sum(1 for e in events if e["cls"] == "cavitation")
        out["rhizoid_table"] = os.path.join(run_dir, f"{stem}_corrected_rhizoids.csv")
    path = save_inspector(stack, {"onset": res["onset"], "shape": res["shape"]},
                          run_dir, stem, tube_ids=res["tube_ids"],
                          out_name=f"{stem}_inspect_corrected.html",
                          seed_edits=doc)
    out.update(inspector=path, frac_final=float(frac[-1]),
               embolised_px=int(res["shape"].sum()),
               curve=os.path.join(run_dir, f"{stem}_corrected_curve.csv"))
    return out


def load_frames_binned(folder: str, bin_xy: int = 1):
    """Thin re-export of colour_shape.load_frames_binned (kept here so inspector is the single
    import for the correction path)."""
    from .colour_shape import load_frames_binned as _lfb
    return _lfb(folder, bin_xy)


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
<div class="controls" style="margin-top:12px;border-top:1px solid #222;padding-top:10px">
 <strong style="font-size:12px;color:#9ab">Correct:</strong>
 <span class="seg">
  <button id="tPan" class="on">pan/zoom</button>
  <button id="tAdd">✎ add tube</button>
  <button id="tErase">erase</button>
  <button id="tSplit">split</button>
  <button id="tMerge">merge</button>
 </span>
 <label style="font-size:12px;color:#bcd">brush <input id="brush" type="range" min="1" max="30" value="5"></label>
 <button id="tUndo">undo</button>
 <button id="tClear">clear edits</button>
 <button id="tExport">⤓ Export corrections</button>
 <span id="editmsg" class="hint"></span>
</div>
<div class="hint" style="margin-top:6px">
 add = trace a missed rhizoid (snaps to the ridge, onset read from the frames) · erase = remove
 false positives · split = cut a tube · merge = click two tubes · then <b>Export corrections</b>
 and, in cavidetect, click <b>Apply corrections…</b> to recompute. The automatic result is kept.
</div>
<script>
const RAW=__RAW__, DIFF=__DIFF__, TPS=__TPS__, NFR=__NFR__, NTUBES=__NTUBES__;
const ONSET_URL=__ONSET__, TUBES_URL=__TUBES__;
const OY=__OY__, OX=__OX__, STEM="__STEM__", SEED=__SEED__;
let tool='pan', pending=[], curStroke=null, mergePt=null;
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
 drawPending();
}
function drawPending(){
 if(!cctx) return;
 const all=pending.concat(curStroke?[curStroke]:[]);
 for(const e of all){
   if(e.op==='add'||e.op==='erase'){
     cctx.fillStyle = e.op==='add' ? 'rgba(60,230,120,0.55)' : 'rgba(240,60,60,0.55)';
     for(const p of e.points){ cctx.beginPath(); cctx.arc(p[0],p[1],e.radius,0,7); cctx.fill(); }
   } else if(e.op==='split'){
     cctx.strokeStyle='#5cf'; cctx.lineWidth=2; cctx.beginPath();
     cctx.arc(e.point[0],e.point[1],6,0,7); cctx.stroke();
   } else if(e.op==='merge'){
     cctx.strokeStyle='#fd5'; cctx.lineWidth=2;
     cctx.beginPath(); cctx.moveTo(e.points[0][0],e.points[0][1]);
     cctx.lineTo(e.points[1][0],e.points[1][1]); cctx.stroke();
     for(const p of e.points){cctx.beginPath();cctx.arc(p[0],p[1],4,0,7);cctx.fillStyle='#fd5';cctx.fill();}
   }
 }
 if(mergePt){cctx.fillStyle='#fd5';cctx.beginPath();cctx.arc(mergePt[0],mergePt[1],4,0,7);cctx.fill();}
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
stage.addEventListener('mousedown',e=>{ if(tool!=='pan')return; dragging=true;lx=e.clientX;ly=e.clientY;});
window.addEventListener('mousemove',e=>{ if(!dragging)return;
 px+=e.clientX-lx; py+=e.clientY-ly; lx=e.clientX; ly=e.clientY; applyT(); });
window.addEventListener('mouseup',()=>dragging=false);

// ---- editing: capture strokes on the conduits canvas (coords in canvas pixels) ----
function canvasXY(ev){ const r=cond.getBoundingClientRect();
 return [ (ev.clientX-r.left)/r.width*W, (ev.clientY-r.top)/r.height*H ]; }
function setTool(t){ tool=t; mergePt=null;
 for(const [id,tt] of [['tPan','pan'],['tAdd','add'],['tErase','erase'],['tSplit','split'],['tMerge','merge']])
   document.getElementById(id).classList.toggle('on',tt===t);
 show(+sld.value); }
for(const [id,t] of [['tPan','pan'],['tAdd','add'],['tErase','erase'],['tSplit','split'],['tMerge','merge']])
  document.getElementById(id).onclick=()=>setTool(t);
document.getElementById('tUndo').onclick=()=>{pending.pop();show(+sld.value);};
document.getElementById('tClear').onclick=()=>{pending=[];mergePt=null;show(+sld.value);};
let drawing=false;
function edMouseDown(ev){ if(tool==='pan')return; ev.stopPropagation();
 const r=+document.getElementById('brush').value, p=canvasXY(ev);
 if(tool==='add'||tool==='erase'){ drawing=true; curStroke={op:tool,points:[p],radius:r}; }
 else if(tool==='split'){ pending.push({op:'split',point:p}); }
 else if(tool==='merge'){ if(!mergePt){mergePt=p;} else {pending.push({op:'merge',points:[mergePt,p]});mergePt=null;} }
 show(+sld.value); }
function edMouseMove(ev){ if(!drawing||!curStroke)return; curStroke.points.push(canvasXY(ev)); show(+sld.value); }
function edMouseUp(){ if(drawing&&curStroke){pending.push(curStroke);} drawing=false; curStroke=null; show(+sld.value); }
cond.addEventListener('mousedown',edMouseDown);
cond.addEventListener('mousemove',edMouseMove);
window.addEventListener('mouseup',edMouseUp);

document.getElementById('tExport').onclick=()=>{
 const doc={stem:STEM, disp_size:[W,H], orig_size:[OY,OX], edits:SEED.concat(pending)};
 const blob=new Blob([JSON.stringify(doc,null,1)],{type:'application/json'});
 const a=document.createElement('a'); a.download=STEM+'_corrections.json';
 a.href=URL.createObjectURL(blob); a.click();
 document.getElementById('editmsg').textContent=
   'saved '+a.download+' — in cavidetect click “Apply corrections…” and pick it';
};

Promise.all([decodeMap(ONSET_URL), decodeMap(TUBES_URL)]).then(([on,tu])=>{
 onsetV=on.val; onsetValid=on.valid; tubeV=tu.val; tubeValid=tu.valid;
 W=on.w; H=on.h; cond.width=W; cond.height=H;
 cctx=cond.getContext('2d'); cimg=cctx.createImageData(W,H);
 legend(); show(0);
});
</script></body></html>
"""
