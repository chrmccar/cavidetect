"""Rhizoid tube map (§7.3): find the tubes and split them into segments.

A rhizoid is a tube; cavitation fills a whole tube segment at once. So detection must be
tube-centric. We build the tube map from a temporal high-percentile projection (a tube is
bright once it has cavitated at some point), enhance tubes with a Sato ridge filter,
skeletonise, break the skeleton at junctions into individual tube segments, and grow those
segment labels back out to the full tube width."""
from __future__ import annotations
import numpy as np
from skimage.filters import sato
from skimage.morphology import skeletonize
from skimage.measure import label
from scipy.ndimage import convolve, distance_transform_edt


def _drop_small(mask, min_px):
    lbl = label(mask)
    sizes = np.bincount(lbl.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lbl]


def tube_segments(s: np.ndarray, *, sigmas=(1, 1.5, 2, 2.5, 3), tube_percentile=80.0,
                  proj_percentile=98.0, min_tube_px=12, min_seg_px=4):
    """Return (seg_full [Y,X] int labels, proj [Y,X], tube_mask [Y,X] bool)."""
    proj = np.percentile(s, proj_percentile, axis=0)
    tub = sato(proj, sigmas=list(sigmas), black_ridges=False)
    tub = (tub - tub.min()) / (np.ptp(tub) + 1e-9)
    tube_mask = tub > np.percentile(tub, tube_percentile)
    tube_mask = _drop_small(tube_mask, min_tube_px)

    sk = skeletonize(tube_mask)
    nb = convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - sk.astype(int)
    seg_sk = label(sk & (nb <= 2))          # remove junction pixels -> separate tube branches
    seg_sk = seg_sk * _drop_small(seg_sk > 0, min_seg_px)

    if seg_sk.max() == 0:
        return np.zeros_like(seg_sk), proj, tube_mask
    _, inds = distance_transform_edt(seg_sk == 0, return_indices=True)
    seg_full = seg_sk[tuple(inds)]
    seg_full[~tube_mask] = 0
    return seg_full, proj, tube_mask
