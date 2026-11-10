"""Colour + shape embolism measurement (calibrated to hand annotation, F1~0.68).

Embolised tissue = (air-COLOUR in the final frame, within plant tissue) AND (TUBE-SHAPED),
timed by the frame each pixel last crossed into the air-colour (read backwards from the end).
Lives in the package so it runs under the same interpreter as the `cavidetect` CLI.
"""
from __future__ import annotations
import glob, os, time
import numpy as np
import tifffile
from . import embolism as cemb, outputs as cout


def load_frames_binned(folder: str, bin_xy: int = 2):
    files = sorted(f for f in glob.glob(os.path.join(folder, "*.tif")) +
                   glob.glob(os.path.join(folder, "*.tiff")) if " - Copy" not in f)
    if not files:
        raise ValueError(f"no .tif frames in {folder}")

    def _g(f):
        a = tifffile.imread(f)
        if a.ndim == 3:
            a = a[..., :3].mean(-1)
        return a[::bin_xy, ::bin_xy].astype(np.float32)

    t = time.time()
    # Preallocate and fill in place: building a Python list of all frames and then
    # np.stack()-ing it holds two full copies of the stack at once (~2x peak RAM, e.g.
    # ~14.6 GiB for 400x1920x2560). One frame is read at a time into the final buffer.
    first = _g(files[0])
    stack = np.empty((len(files), *first.shape), np.float32)
    stack[0] = first
    del first
    for i in range(1, len(files)):
        stack[i] = _g(files[i])
    print(f"loaded {len(files)} frames {stack.shape} in {time.time()-t:.0f}s", flush=True)
    return stack


def tissue_from_final(final, erode=2):
    """Analysis region from the DRY final frame (not the wet one): the bottom rhizoids only
    become distinct once air-filled, so defining tissue from the wet frame excluded them.
    Keeps only the main thallus (largest connected region) so detached background texture in
    the corners is dropped, not analysed."""
    from skimage.filters import threshold_otsu
    from scipy.ndimage import binary_fill_holes, binary_closing, binary_erosion, label
    m = final > threshold_otsu(final)
    m = binary_closing(m, iterations=3)
    m = binary_fill_holes(m)
    lbl, n = label(m)                      # keep the single biggest blob = the thallus
    if n > 1:
        sizes = np.bincount(lbl.ravel()); sizes[0] = 0
        m = lbl == int(sizes.argmax())
    if erode > 0:
        m = binary_erosion(m, iterations=erode)
    return m


def onset_start_of_rise(stack, mask, rise_frac=0.4, persist=3, smooth=3):
    """Per-pixel onset = the START of each pixel's own brightening, not a late global cut.

    For every pixel in `mask`, define its own baseline (an early/low level) and its final
    bright level, and date onset to the first frame it rises past `rise_frac` of the way up
    AND stays there for `persist` frames ("first-and-sticks"). This removes the lag of the
    old "last frame below a high global threshold" rule, which dated fill to the midpoint/end
    of the transition. Only the timing changes; which pixels are embolised (`mask`) is untouched.

    Returns an int32 array: onset frame for masked pixels, -1 elsewhere.
    """
    from scipy.ndimage import uniform_filter1d
    T = stack.shape[0]
    sm = uniform_filter1d(stack, size=max(1, smooth), axis=0)
    lo = np.percentile(sm, 20, axis=0)              # each pixel's own dark baseline
    hi = sm.max(0)                                   # each pixel's own bright level
    thr = lo + rise_frac * (hi - lo)                 # a level early in its own rise
    above = sm >= thr[None]
    del sm                                           # free the full smoothed copy before the bool arrays
    sust = above.copy()                              # "stays above for `persist` frames"
    for s in range(1, max(1, persist)):
        sust[:-s] &= above[s:]
    has = sust.any(0)
    first = np.argmax(sust, axis=0)                  # first sustained crossing
    # fall back to first bare crossing if nothing is sustained (very short/noisy rise)
    first_bare = np.argmax(above, axis=0)
    onset = np.where(has, first, np.where(above.any(0), first_bare, 0)).astype(np.int32)
    onset = np.clip(onset, 0, T - 1)
    onset[~mask] = -1
    return onset


def detect_spatial(final, air=None, vessel_min=0.04, bright_pct=65, tissue=None,
                   ecc_min=0.9, sol_max=0.6, min_area=8):
    """Spatial embolism detection from the dry final frame only (no timing).

    Split out of ``measure`` so the evaluation harness can run it on cached final
    frames without re-loading 400-frame stacks. Returns the embolised mask ``shape``
    plus the intermediates.

    Keeps air-coloured regions that are tube-SHAPED (elongated OR sparse) and contain a
    tubular ridge seed. ``bright_pct``/``sol_max`` tuned against the expert masks
    (2026-11-02): lowering the air percentile 70->65 and loosening the solidity gate
    0.5->0.6 recovers faint rhizoids (mean F1 0.676 -> 0.702; plant-60 recall ~0.53->0.66).
    NOTE: edge SCALES are not yet excluded here (they share brightness+texture with
    rhizoids); that needs expert scale annotation -- see detect_spatial scale work / DESIGN.
    """
    from scipy.ndimage import label
    from skimage.filters import frangi
    from skimage.measure import regionprops
    if tissue is None:
        tissue = tissue_from_final(final, erode=2)            # #2: tissue from the DRY frame
    if air is None:
        air = float(np.percentile(final[tissue], bright_pct))
    colour = tissue & (final >= air)
    v = np.nan_to_num(frangi(final, sigmas=(1, 2, 3, 4), black_ridges=False))
    vn = v / (v[tissue].max() + 1e-9)
    seeds = colour & (vn >= vessel_min)
    lbl, _ = label(colour); keep = []
    for p in regionprops(lbl):
        if p.area < min_area:
            continue
        if ((p.eccentricity >= ecc_min) or (p.solidity <= sol_max)) and seeds[tuple(p.coords.T)].any():
            keep.append(p.label)
    shape = np.isin(lbl, keep)
    # `stages` exposes the intermediate masks for the diagnostic montage (§1 of the
    # v0.2.0 brief). It never changes `shape`/`colour`, so the regression pin is safe.
    return dict(tissue=tissue, air=float(air), colour=colour, shape=shape, ridge=vn,
                stages={"brightness": colour, "frangi": vn, "seed": seeds,
                        "accepted": shape})


def ridge_map(final, tissue, sigmas=(0.5, 1, 1.5, 2, 3, 4)):
    """Normalised Frangi tubeness of the final frame (0..~1 within tissue).

    Finer sigmas than the legacy (1,2,3,4) so thin faint rhizoids register. High on
    tube-like structures, ~0 on flat sheets (scales) and open tissue."""
    from skimage.filters import frangi
    v = np.nan_to_num(frangi(final, sigmas=sigmas, black_ridges=False))
    return v / (v[tissue].max() + 1e-9)


def ridge_evidence(final, tissue, bright_hi_pct=70, bright_lo_pct=45,
                   seed_ridge=0.08, grow_ridge=0.02, sigmas=(0.5, 1, 1.5, 2, 3, 4)):
    """Continuous ridge (anatomical) evidence, with its intermediate masks exposed.

    This is the pre-acceptance half of the ridge-hysteresis detector, split out so the
    hybrid detector can *fuse* the continuous ridge scores and seed/grow masks with temporal
    evidence instead of consuming only the final binary mask (v0.3.1 hybrid brief, Step 3:
    "Retain continuous ridge scores rather than only the final binary accepted mask").

    Returns a dict:

    * ``ridge``     — normalised Frangi tubeness map (continuous, 0..~1 within tissue).
    * ``seed``      — bright (>=hi pct) AND strongly tubular: confident embolised centres.
    * ``grow``      — mildly bright (>=lo pct) AND weakly tubular: faint tube continuations.
    * ``tubes``     — hysteresis result: ``grow`` components that contain a ``seed``.
    * ``edge_dist`` — distance (px) into the tissue, for the border/scale test.
    * ``air_hi`` / ``air_lo`` — the two brightness cuts used.

    No component-size / scale filtering is applied here; callers (``detect_spatial_v2`` and
    the hybrid) impose their own acceptance so ``detect_spatial_v2`` stays byte-identical.
    """
    from scipy.ndimage import label, distance_transform_edt
    vn = ridge_map(final, tissue, sigmas)
    air_hi = float(np.percentile(final[tissue], bright_hi_pct))
    air_lo = float(np.percentile(final[tissue], bright_lo_pct))
    seed = tissue & (final >= air_hi) & (vn >= seed_ridge)
    grow = tissue & (final >= air_lo) & (vn >= grow_ridge)
    lbl, _ = label(grow)                                  # hysteresis grow
    seeded = set(int(x) for x in np.unique(lbl[seed])) - {0}
    tubes = np.isin(lbl, list(seeded)) if seeded else np.zeros_like(tissue)
    edge_dist = distance_transform_edt(tissue)            # px distance into tissue
    return dict(ridge=vn, seed=seed, grow=grow, tubes=tubes, edge_dist=edge_dist,
                air_hi=air_hi, air_lo=air_lo)


def detect_spatial_v2(final, tissue=None, bright_hi_pct=70, bright_lo_pct=45,
                      seed_ridge=0.08, grow_ridge=0.02, sigmas=(0.5, 1, 1.5, 2, 3, 4),
                      min_area=8, scale_area=600, scale_ecc=0.85,
                      border_px=10, erode=2):
    """Ridge-gated hysteresis detector: higher recall on faint tubes, rejects scales.

    1. ``seed``  = bright (>=hi pct) AND strongly tubular  -> confident embolised centres.
    2. ``cand``  = mildly bright (>=lo pct) AND weakly tubular -> faint tube continuations.
    3. hysteresis: keep candidate components that contain a seed -> grows along faint
       rhizoids without admitting flat/dim non-tube tissue.
    4. scale rejection: the edge scales where rhizoids attach brighten like air but are
       rounded SHEETS at the tissue margin, not tubes. Drop a component when it is large
       AND not strongly elongated (low eccentricity -> a blob, whereas a thin rhizoid is
       a line with eccentricity ~1 and is kept) AND touches the tissue border.
    """
    from scipy.ndimage import label
    from skimage.measure import regionprops
    if tissue is None:
        tissue = tissue_from_final(final, erode=erode)
    ev = ridge_evidence(final, tissue, bright_hi_pct, bright_lo_pct,
                        seed_ridge, grow_ridge, sigmas)
    vn, tubes, edge_dist = ev["ridge"], ev["tubes"], ev["edge_dist"]
    lbl2, _ = label(tubes); keep = []
    for p in regionprops(lbl2):
        if p.area < min_area:
            continue
        coords = tuple(p.coords.T)
        touches_border = float(edge_dist[coords].min()) <= border_px
        is_scale = (p.area >= scale_area and p.eccentricity <= scale_ecc
                    and touches_border)
        if not is_scale:
            keep.append(p.label)
    shape = np.isin(lbl2, keep)
    return dict(tissue=tissue, air=ev["air_hi"], colour=tissue & (final >= ev["air_lo"]),
                shape=shape, ridge=vn)


def measure(stack, method: str = "colour-frangi", params: dict | None = None,
            tail_frac: float = 0.12, rise_frac: float = 0.4, bin_xy: int = 1):
    """Measure embolism with the chosen detection method.

    ``params`` are the method's parameters (resolved against its schema). Only
    the spatial-detection stage is method-specific; tissue, onset timing and the
    cumulative curve are shared. ``bin_xy`` folds binning into the pixel scale
    the method sees (one binned pixel spans ``bin_xy`` original pixels) while the
    returned ``params`` keep the true optical ``um_per_px``.
    """
    from . import methods as M
    from . import preprocess_final as pp
    T = stack.shape[0]
    resolved = M.resolve_params(method, params or {})
    for_detect = dict(resolved)
    if "um_per_px" in for_detect:
        for_detect["um_per_px"] = resolved["um_per_px"] * bin_xy   # µm per analysis pixel
    # Build the dry 'final' frame. Default = plain mean of the tail (byte-identical
    # to the old reduction, so every method is unaffected). If the method asks for
    # per-frame CLAHE (the expert's ImageJ macro enhances every frame), CLAHE each
    # tail frame then average, and suppress the method's own contrast step so CLAHE
    # is applied exactly once. The raw stack (onset timing / temporal detectors) is
    # never altered.
    if resolved.get("clahe_scope") == "per-frame" and resolved.get("contrast") == "clahe":
        final = pp.final_from_stack(stack, tail_frac, contrast="clahe",
                                    clahe_blocksize=resolved["clahe_blocksize"],
                                    clahe_bins=resolved["clahe_bins"],
                                    clahe_maxslope=resolved["clahe_maxslope"],
                                    clahe_clip=resolved["clahe_clip"])
        for_detect["contrast"] = "none"
    else:
        final = pp.final_from_stack(stack, tail_frac, contrast="none")
    tissue = tissue_from_final(final, erode=2)
    mth = M.get_method(method)
    if mth.needs_stack:                          # temporal methods see the whole stack
        d = mth.detect(final, tissue, stack=stack, **for_detect)
    else:
        d = mth.detect(final, tissue, **for_detect)
    tissue, shape = d["tissue"], d["shape"]
    # a stack-aware method may supply its own per-pixel onset; otherwise date the start of rise
    onset = d.get("onset")
    if onset is None:
        onset = onset_start_of_rise(stack, shape, rise_frac=rise_frac)
    valid = shape & (onset >= 0)                 # onset is -1 where not embolised
    counts = np.bincount(onset[valid], minlength=T); cum = np.cumsum(counts)
    frac = cum / max(int(tissue.sum()), 1)
    ridge = d.get("ridge")
    if ridge is None:
        ridge = np.zeros(final.shape, np.float32)
    return dict(tissue=tissue, final=final, air=d.get("air"), colour=d["colour"], shape=shape,
                ridge=ridge, onset=onset, cum=cum, frac=frac, tissue_px=int(tissue.sum()),
                method=method, params=resolved, stages=d.get("stages"),
                confidence=d.get("confidence"), transition=d.get("transition"),
                evidence=d.get("evidence"))


def run(folder: str, out_dir: str, bin_xy: int = 2, air=None, tail_frac: float = 0.12,
        method: str = "colour-frangi", params: dict | None = None,
        frame_interval_s: float = 15.0, write_tiffs: bool = True,
        write_diagnostics: bool = True) -> dict:
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from types import SimpleNamespace
    from . import diagnostics as cdiag
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.basename(os.path.normpath(folder))
    stack = load_frames_binned(folder, bin_xy)
    params = dict(params or {})
    if air is not None:                      # back-compat: --air folds into the method params
        params.setdefault("air", air)
    r = measure(stack, method=method, params=params, tail_frac=tail_frac, bin_xy=bin_xy)
    T = stack.shape[0]
    um_per_px = r["params"].get("um_per_px", 2.5)
    print("method=%s" % method, flush=True)
    print("air=%.1f  colour=%.1f%%  colour+shape=%.1f%% of tissue"
          % ((r["air"] if r["air"] is not None else 0.0),
             100 * r["colour"].sum() / max(r["tissue_px"], 1),
             100 * r["shape"].sum() / max(r["tissue_px"], 1)), flush=True)

    # record exactly what produced this result, for reproducibility of an experimental run
    try:
        from .pipeline import __version__ as _ver
    except Exception:
        _ver = "?"
    with open(os.path.join(out_dir, f"{stem}_cs_params.json"), "w", encoding="utf-8") as fh:
        json.dump({"method": method, "params": r["params"], "bin": bin_xy,
                   "tail_frac": tail_frac, "um_per_px": um_per_px, "version": _ver},
                  fh, indent=2)
    # §7 full run metadata (superset of params.json): adds calibration + frame interval
    um_per_analysis_px = um_per_px * bin_xy
    cout.write_run_metadata(out_dir, f"{stem}_cs", {
        "method": method, "params": r["params"], "bin": bin_xy, "tail_frac": tail_frac,
        "um_per_px": um_per_px, "um_per_analysis_px": um_per_analysis_px,
        "frame_interval_s": frame_interval_s, "n_frames": T, "version": _ver})

    fig = plt.figure(figsize=(20, 5))
    for k, (m, title) in enumerate([(None, "raw final"), (r["colour"], "COLOUR only"),
                                    (r["shape"], "COLOUR + SHAPE")]):
        ax = fig.add_subplot(1, 4, k + 1); ax.imshow(r["final"], cmap="gray"); ax.axis("off"); ax.set_title(title)
        if m is not None:
            ov = np.zeros((*m.shape, 4)); ov[m] = [1, 0.15, 0.15, 0.55]; ax.imshow(ov)
    ax = fig.add_subplot(1, 4, 4); ax.plot(r["frac"] * 100, color="#c0392b", lw=2)
    ax.set_ylim(0, 100); ax.grid(alpha=.3); ax.set_xlabel("frame"); ax.set_ylabel("% tissue embolised")
    ax.set_title("cumulative embolised tissue")
    fig.suptitle(f"method: {method}", x=0.01, ha="left", fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{stem}_cs_proof.png"), dpi=95); plt.close()

    pd.DataFrame({"frame": np.arange(T), "embolised_area_px": r["cum"],
                  "frac_tissue_embolised": r["frac"]}).to_csv(
        os.path.join(out_dir, f"{stem}_cs_curve.csv"), index=False)
    res = SimpleNamespace(embolised=r["shape"], onset_frame=r["onset"], frac_embolised=r["frac"],
                          cum_area=r["cum"].astype(np.int64),
                          cum_area_um2=r["cum"] * (um_per_analysis_px ** 2),
                          tissue_px=r["tissue_px"], change_dn=r["final"])
    cout.save_embolism_viewer(stack, res, out_dir, f"{stem}_cs")
    # annotation tool: paint the correct embolised tissue on the raw final frame, export a PNG
    cout.save_annotator(stack[-1], out_dir, f"{stem}_cs")

    # §1 diagnostic montage of every pipeline stage (see where recall is lost)
    if write_diagnostics and r.get("stages"):
        cdiag.save_stage_montage(r, out_dir, stem)
    # §7 event-time LUT + per-region onset table (confidence/transition when the method gives them)
    scale_um2 = um_per_analysis_px ** 2
    cout.save_event_lut(r["onset"], r["shape"], r["final"], out_dir, f"{stem}_cs", n_frames=T)
    cout.write_onset_table(r["onset"], r["shape"], out_dir, f"{stem}_cs",
                           confidence=r.get("confidence"), transition=r.get("transition"),
                           scale_um2=scale_um2, frame_interval_s=frame_interval_s)
    # §6 confidence-threshold viewer (only methods that score candidates, e.g. temporal-evidence)
    if r.get("confidence") is not None:
        cout.save_confidence_viewer(r["final"], r["confidence"], out_dir, f"{stem}_cs",
                                    tissue=r["tissue"])
    # hybrid brief: export the fused evidence maps (ridge / temporal / combined / artifact)
    if r.get("evidence"):
        cout.save_evidence_maps(r["final"], r["evidence"], out_dir, f"{stem}_cs")
    # §7/§1 per-frame + cumulative binary TIFF of newly-detected pixels (streamed, sparse-compressed)
    if write_tiffs:
        cout.save_event_tiffs(r["onset"], r["shape"], T, out_dir, f"{stem}_cs")
    print("wrote", os.path.join(out_dir, f"{stem}_cs_viewer.html"), flush=True)
    return r
