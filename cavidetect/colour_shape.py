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
    stack = np.stack([_g(f) for f in files])
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


def measure(stack, air=None, tail_frac=0.12, vessel_min=0.04, bright_pct=70,
            rise_frac=0.4):
    from scipy.ndimage import label
    from skimage.filters import frangi
    from skimage.measure import regionprops
    T = stack.shape[0]
    final = stack[-max(int(round(T * tail_frac)), 1):].mean(0)
    tissue = tissue_from_final(final, erode=2)                 # #2: tissue from the DRY frame
    if air is None:
        air = float(np.percentile(final[tissue], bright_pct))
    colour = tissue & (final >= air)
    v = np.nan_to_num(frangi(final, sigmas=(1, 2, 3, 4), black_ridges=False))
    vn = v / (v[tissue].max() + 1e-9)
    seeds = colour & (vn >= vessel_min)
    lbl, _ = label(colour); keep = []
    for p in regionprops(lbl):
        if p.area < 8:
            continue
        if ((p.eccentricity >= 0.9) or (p.solidity <= 0.5)) and seeds[tuple(p.coords.T)].any():
            keep.append(p.label)
    shape = np.isin(lbl, keep)
    onset = onset_start_of_rise(stack, shape, rise_frac=rise_frac)   # start-of-rise timing
    counts = np.bincount(onset[shape], minlength=T); cum = np.cumsum(counts)
    frac = cum / max(int(tissue.sum()), 1)
    return dict(tissue=tissue, final=final, air=air, colour=colour, shape=shape,
                onset=onset, cum=cum, frac=frac, tissue_px=int(tissue.sum()))


def run(folder: str, out_dir: str, bin_xy: int = 2, air=None, tail_frac: float = 0.12) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from types import SimpleNamespace
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.basename(os.path.normpath(folder))
    stack = load_frames_binned(folder, bin_xy)
    r = measure(stack, air=air, tail_frac=tail_frac)
    T = stack.shape[0]
    print("air=%.1f  colour=%.1f%%  colour+shape=%.1f%% of tissue"
          % (r["air"], 100 * r["colour"].sum() / max(r["tissue_px"], 1),
             100 * r["shape"].sum() / max(r["tissue_px"], 1)), flush=True)

    fig = plt.figure(figsize=(20, 5))
    for k, (m, title) in enumerate([(None, "raw final"), (r["colour"], "COLOUR only"),
                                    (r["shape"], "COLOUR + SHAPE")]):
        ax = fig.add_subplot(1, 4, k + 1); ax.imshow(r["final"], cmap="gray"); ax.axis("off"); ax.set_title(title)
        if m is not None:
            ov = np.zeros((*m.shape, 4)); ov[m] = [1, 0.15, 0.15, 0.55]; ax.imshow(ov)
    ax = fig.add_subplot(1, 4, 4); ax.plot(r["frac"] * 100, color="#c0392b", lw=2)
    ax.set_ylim(0, 100); ax.grid(alpha=.3); ax.set_xlabel("frame"); ax.set_ylabel("% tissue embolised")
    ax.set_title("cumulative embolised tissue")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{stem}_cs_proof.png"), dpi=95); plt.close()

    pd.DataFrame({"frame": np.arange(T), "embolised_area_px": r["cum"],
                  "frac_tissue_embolised": r["frac"]}).to_csv(
        os.path.join(out_dir, f"{stem}_cs_curve.csv"), index=False)
    res = SimpleNamespace(embolised=r["shape"], onset_frame=r["onset"], frac_embolised=r["frac"],
                          cum_area=r["cum"].astype(np.int64), cum_area_um2=r["cum"] * (2.5 ** 2),
                          tissue_px=r["tissue_px"], change_dn=r["final"])
    cout.save_embolism_viewer(stack, res, out_dir, f"{stem}_cs")
    # annotation tool: paint the correct embolised tissue on the raw final frame, export a PNG
    cout.save_annotator(stack[-1], out_dir, f"{stem}_cs")
    print("wrote", os.path.join(out_dir, f"{stem}_cs_viewer.html"), flush=True)
    return r
