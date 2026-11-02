"""Colour + shape embolism measurement on a folder of raw frames.

Embolised tissue = (air-COLOUR in the final frame, within plant tissue) AND (TUBE-SHAPED).
Timing = the frame each pixel last crossed into the air-colour (read backwards from the end).
Outputs an interactive viewer, the cumulative curve CSV, and a colour-vs-shape proof PNG.

Usage:
    python3 tools/embolism_colour_shape.py FRAMES_FOLDER OUT_DIR [--bin 2] [--air N] [--tail-frac 0.12]
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np
import tifffile
from scipy.ndimage import uniform_filter1d, label
from skimage.filters import threshold_otsu, frangi
from skimage.measure import regionprops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import embolism as cemb, outputs as cout


def load_frames(folder, bin_xy=2):
    files = sorted(f for f in glob.glob(os.path.join(folder, "*.tif")) +
                   glob.glob(os.path.join(folder, "*.tiff")) if " - Copy" not in f)
    if not files:
        sys.exit(f"no .tif frames in {folder}")

    def _g(f):
        a = tifffile.imread(f)
        if a.ndim == 3:
            a = a[..., :3].mean(-1)
        return a[::bin_xy, ::bin_xy].astype(np.float32)

    t = time.time()
    stack = np.stack([_g(f) for f in files])
    print(f"loaded {len(files)} frames {stack.shape} in {time.time()-t:.0f}s", flush=True)
    return stack


def measure(stack, air=None, tail_frac=0.12, vessel_min=0.04, bright_pct=70):
    # calibrated to the user's hand annotation (F1=0.68): brightness pct=70, vessel_min=0.04, grow=True
    T = stack.shape[0]
    tissue = cemb.tissue_mask(stack, base_frames=12, erode=5)
    final = stack[-max(int(round(T * tail_frac)), 1):].mean(0)
    if air is None:
        air = float(np.percentile(final[tissue], bright_pct))
    colour = tissue & (final >= air)                                   # COLOUR
    v = np.nan_to_num(frangi(final, sigmas=(1, 2, 3, 4), black_ridges=False))
    vn = v / (v[tissue].max() + 1e-9)
    seeds = colour & (vn >= vessel_min)
    lbl, _ = label(colour); keep = []
    for p in regionprops(lbl):
        if p.area < 8:
            continue
        if ((p.eccentricity >= 0.9) or (p.solidity <= 0.5)) and seeds[tuple(p.coords.T)].any():
            keep.append(p.label)
    shape = np.isin(lbl, keep)                                         # COLOUR + SHAPE
    sm = uniform_filter1d(stack, size=5, axis=0); below = sm < air
    anyb = below.any(0); lastb = (T - 1) - np.argmax(below[::-1], axis=0)
    onset = np.where(anyb, np.minimum(lastb + 1, T - 1), 0).astype(np.int32); onset[~shape] = -1
    counts = np.bincount(onset[shape], minlength=T); cum = np.cumsum(counts)
    frac = cum / max(int(tissue.sum()), 1)
    return dict(tissue=tissue, final=final, air=air, colour=colour, shape=shape,
                onset=onset, cum=cum, frac=frac, tissue_px=int(tissue.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("out")
    ap.add_argument("--bin", type=int, default=2)
    ap.add_argument("--air", type=float, default=None, help="manual air-colour threshold (else Otsu)")
    ap.add_argument("--tail-frac", type=float, default=0.12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.basename(os.path.normpath(a.folder))
    stack = load_frames(a.folder, a.bin)
    r = measure(stack, air=a.air, tail_frac=a.tail_frac)
    T = stack.shape[0]
    print("air=%.1f  colour=%.1f%%  colour+shape=%.1f%% of tissue"
          % (r["air"], 100 * r["colour"].sum() / max(r["tissue_px"], 1),
             100 * r["shape"].sum() / max(r["tissue_px"], 1)), flush=True)

    # proof PNG
    fig = plt.figure(figsize=(20, 5))
    for k, (m, title) in enumerate([(None, "raw final"), (r["colour"], "COLOUR only"),
                                    (r["shape"], "COLOUR + SHAPE")]):
        ax = fig.add_subplot(1, 4, k + 1); ax.imshow(r["final"], cmap="gray"); ax.axis("off"); ax.set_title(title)
        if m is not None:
            ov = np.zeros((*m.shape, 4)); ov[m] = [1, 0.15, 0.15, 0.55]; ax.imshow(ov)
    ax = fig.add_subplot(1, 4, 4); ax.plot(r["frac"] * 100, color="#c0392b", lw=2)
    ax.set_ylim(0, 100); ax.grid(alpha=.3); ax.set_xlabel("frame"); ax.set_ylabel("% tissue embolised")
    ax.set_title("cumulative embolised tissue")
    plt.tight_layout(); plt.savefig(os.path.join(a.out, f"{stem}_cs_proof.png"), dpi=95); plt.close()

    # curve CSV + interactive viewer
    import pandas as pd
    pd.DataFrame({"frame": np.arange(T), "embolised_area_px": r["cum"],
                  "frac_tissue_embolised": r["frac"]}).to_csv(
        os.path.join(a.out, f"{stem}_cs_curve.csv"), index=False)
    from types import SimpleNamespace
    res = SimpleNamespace(embolised=r["shape"], onset_frame=r["onset"], frac_embolised=r["frac"],
                          cum_area=r["cum"].astype(np.int64), cum_area_um2=r["cum"] * (2.5 ** 2),
                          tissue_px=r["tissue_px"], change_dn=r["final"])
    cout.save_embolism_viewer(stack, res, a.out, f"{stem}_cs")
    print("wrote:", os.path.join(a.out, f"{stem}_cs_proof.png"),
          "and", os.path.join(a.out, f"{stem}_cs_viewer.html"), flush=True)


if __name__ == "__main__":
    main()
