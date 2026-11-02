"""Direct cumulative-embolism measurement (the PRIMARY readout).

The strict event detector (`detect.py`) only accepts clean, abrupt, step-like brightenings,
which discards most real air-filling on the whole thallus (gradual fills, motion-smeared
fills). This module measures embolism the optical-vulnerability way instead: for every
*tissue* pixel, decide whether it undergoes a SUSTAINED net brightening from its wet
baseline to its dry end-state, and record the frame it half-filled. Accumulating those
onsets gives embolised-area-over-time — the vulnerability curve — without gating on event
shape.

Air-filling appears as brightening here; `direction` allows darkening / absolute change.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    from skimage.filters import threshold_otsu, frangi
except Exception:  # skimage optional; fall back to a percentile split / no tube gate
    threshold_otsu = None
    frangi = None
from scipy.ndimage import binary_erosion, binary_fill_holes


def _vesselness(img, sigmas):
    """Multi-scale tubeness of a bright-ridge image (0..~1). High on linear tubes, low on blobs."""
    if frangi is None:
        return np.ones_like(img, dtype=float)  # no skimage -> gate is a no-op
    v = frangi(img.astype(float), sigmas=sigmas, black_ridges=False)
    return np.nan_to_num(v)


@dataclass
class EmbolismResult:
    embolised: np.ndarray      # bool  [Y,X] tissue pixel air-filled by the end
    onset_frame: np.ndarray    # int   [Y,X] frame it half-filled (-1 where not embolised)
    change_dn: np.ndarray      # float [Y,X] sustained wet->dry change (grey levels)
    cum_area: np.ndarray       # int   [T]   cumulative embolised pixel count over time
    cum_area_um2: np.ndarray   # float [T]   cumulative embolised area in um^2
    frac_embolised: np.ndarray # float [T]   fraction of tissue embolised (0..1)
    tissue_px: int             # tissue pixel count (the denominator)


def tissue_mask(s: np.ndarray, base_frames: int = 10, erode: int = 4) -> np.ndarray:
    """Tissue vs background from the wet (early) mean: Otsu split, fill holes, erode edges.

    Eroding pulls the mask off the moving thallus rim, where shrinkage — not embolism —
    drives the brightness change.
    """
    early = s[:max(base_frames, 1)].mean(0)
    if threshold_otsu is not None:
        thr = float(threshold_otsu(early))
    else:
        thr = float(np.percentile(early, 40))
    m = early > thr
    m = binary_fill_holes(m)
    if erode > 0:
        m = binary_erosion(m, iterations=erode)
    return m


def measure_embolism(s: np.ndarray, sigma_dn: float, tissue_mask: np.ndarray, *,
                     base_frames: int = 10, tail_frac: float = 0.15,
                     change_min_dn: float = 10.0, k_sigma: float = 5.0,
                     direction: str = "bright",
                     tube_prior: bool = False, tube_grow: bool = True,
                     vessel_sigmas=(1, 2, 3, 4),
                     vessel_min: float = 0.04, onset_mode: str = "end",
                     um_per_px: float = 2.5, bin_xy: int = 1) -> EmbolismResult:
    """Measure embolism anchored on the DRY end-state.

    `tube_prior`: keep only air-filling that is part of a tube-like (linear) structure, via a
    multi-scale ridge filter on the wet->dry change — drops area-like surface/scale change.
    `onset_mode="end"`: date each pixel by the LAST frame it was still water before filling for
    good (robust to an early transient blip); "half" = first half-rise crossing.
    """
    T, Y, X = s.shape
    base = s[:max(base_frames, 1)].mean(0)                      # wet baseline [Y,X]
    tail = max(int(round(T * tail_frac)), 1)
    final = s[-tail:].mean(0)                                   # dry end-state [Y,X]

    if direction == "bright":
        level = s - base[None]                                  # signed, brightening positive
        sustained = final - base
    elif direction == "dark":
        level = base[None] - s
        sustained = base - final
    else:  # "abs"
        level = np.abs(s - base[None])
        sustained = np.abs(final - base)

    thr = max(change_min_dn, k_sigma * sigma_dn)
    embolised = (sustained > thr) & tissue_mask

    # tube-shape prior: the dry->wet increase must sit on a linear (tube) ridge, not a blob.
    # tube_grow: grow those ridge seeds out to the full width of the connected bright region
    # (keeps whole tubes; a round blob has no ridge seed so its component is dropped).
    if tube_prior:
        v = _vesselness(np.maximum(sustained, 0.0), vessel_sigmas)
        vmax = v[tissue_mask].max() if tissue_mask.any() else v.max()
        seeds = embolised & (v / (vmax + 1e-9) >= vessel_min)
        if tube_grow:
            # grow ridge seeds to the full connected bright region, but keep a component only
            # if it is TUBE-SHAPED (elongated or thin/curvy) — a round blob is dropped whole.
            from scipy.ndimage import label
            from skimage.measure import regionprops
            lbl, _ = label(embolised)
            keep = []
            for p in regionprops(lbl):
                if p.area < 5:
                    continue
                tube_like = (p.eccentricity >= 0.9) or (p.solidity <= 0.5)
                has_seed = bool(seeds[tuple(p.coords.T)].any())
                if tube_like and has_seed:
                    keep.append(p.label)
            embolised = np.isin(lbl, keep)
        else:
            embolised = seeds

    # onset timing
    target = 0.5 * np.maximum(sustained, 1e-6)
    if onset_mode == "end":
        below = level < target[None]                           # still water
        any_below = below.any(0)
        last_below = (T - 1) - np.argmax(below[::-1], axis=0)  # last water frame
        onset = np.where(any_below, np.minimum(last_below + 1, T - 1), 0).astype(np.int32)
    else:  # "half": first frame reaching half-rise
        crossed = level >= target[None]
        onset = np.argmax(crossed, axis=0).astype(np.int32)
    onset[~embolised] = -1

    # accumulate onsets into embolised-area-over-time
    counts = np.bincount(onset[embolised], minlength=T)[:T] if embolised.any() \
        else np.zeros(T, int)
    cum_area = np.cumsum(counts)
    scale_um2 = (um_per_px * bin_xy) ** 2
    tissue_px = int(tissue_mask.sum())
    frac = cum_area / tissue_px if tissue_px else np.zeros(T)

    return EmbolismResult(
        embolised=embolised, onset_frame=onset,
        change_dn=sustained.astype(np.float32),
        cum_area=cum_area.astype(np.int64),
        cum_area_um2=(cum_area * scale_um2).astype(np.float64),
        frac_embolised=frac.astype(np.float64), tissue_px=tissue_px)
