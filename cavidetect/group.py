"""Group accepted pixels into events in (x,y,t) with event-time coherence (§7.2).

Two accepted pixels join the same event only if they are spatial 4-neighbours AND their
step frames differ by <= event_time_tol. This keeps two events on the same rhizoid at
different times from merging into one blob (an M0 failure mode)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from skimage.measure import regionprops


def _union_find_labels(accepted, step_frame, tol):
    Y, X = accepted.shape
    idx = -np.ones((Y, X), np.int64)
    ys, xs = np.nonzero(accepted)
    n = ys.size
    idx[ys, xs] = np.arange(n)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    tf = step_frame
    # right neighbours
    a = accepted[:, :-1] & accepted[:, 1:] & (np.abs(tf[:, :-1] - tf[:, 1:]) <= tol)
    for yy, xx in zip(*np.nonzero(a)):
        union(idx[yy, xx], idx[yy, xx + 1])
    # down neighbours
    b = accepted[:-1, :] & accepted[1:, :] & (np.abs(tf[:-1, :] - tf[1:, :]) <= tol)
    for yy, xx in zip(*np.nonzero(b)):
        union(idx[yy, xx], idx[yy + 1, xx])

    roots = np.array([find(i) for i in range(n)])
    _, comp = np.unique(roots, return_inverse=True)
    label = np.zeros((Y, X), np.int32)
    label[ys, xs] = comp + 1
    return label


def group_events(det, um_per_px: float, bin_xy: float, *, min_area_px: int = 4,
                 event_time_tol: int = 4) -> pd.DataFrame:
    label = _union_find_labels(det.accepted, det.step_frame, event_time_tol)
    scale = um_per_px * bin_xy
    rows = []
    for r in regionprops(label, intensity_image=det.step_dn):
        if r.area < min_area_px:
            continue
        ys, xs = r.coords[:, 0], r.coords[:, 1]
        frames = det.step_frame[ys, xs]
        major = r.axis_major_length; minor = max(r.axis_minor_length, 1e-6)
        rows.append(dict(
            event_id=int(r.label),
            frame_start=int(frames.min()),
            frame_peak=int(np.median(frames)),
            area_px=int(r.area),
            area_um2=float(r.area * scale * scale),
            length_um=float(major * scale),
            eccentricity=float(r.eccentricity),
            elongation=float(major / minor),
            orientation=float(r.orientation),
            step_height_dn=float(np.median(det.step_dn[ys, xs])),
            abrupt_ratio=float(np.nanmedian(det.abrupt_ratio[ys, xs])),
            overshoot_flag=bool(det.overshoot[ys, xs].any()),
            left_censored_flag=bool(det.left_censored[ys, xs].any()),
            centroid_r=float(r.centroid[0]), centroid_c=float(r.centroid[1]),
        ))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("frame_peak").reset_index(drop=True)
    return df, label
