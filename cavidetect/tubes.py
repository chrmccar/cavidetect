"""Aggregate per-pixel step detections onto rhizoid tube segments (§7.2 tube-centric).

Each tube segment that cavitated gets ONE cavitation frame; the whole segment is marked
embolised from that frame. This yields tubes that light up and spread (not dots), a robust
per-tube embolised-area curve, and per-rhizoid outcomes."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from skimage.measure import regionprops

BIG = np.iinfo(np.int32).max


@dataclass
class TubeDetections:
    accepted: np.ndarray    # bool [Y,X] — pixels belonging to a cavitated tube
    step_frame: np.ndarray  # int  [Y,X] — frame the tube cavitated (BIG where none)


def aggregate_tubes(det, seg_full: np.ndarray, um_per_px: float, bin_xy: float, *,
                    min_coverage: float = 0.15, min_det_px: int = 4, min_tube_px: int = 6):
    """Return (TubeDetections, per_tube DataFrame)."""
    scale = um_per_px * bin_xy
    seg_cav = np.full(seg_full.shape, BIG, np.int64)
    rows = []
    props = {r.label: r for r in regionprops(seg_full)}
    for lab, r in props.items():
        pix = seg_full == lab
        npx = int(pix.sum())
        if npx < min_tube_px:
            continue
        det_here = det.accepted & pix
        nd = int(det_here.sum())
        cavitated = (nd >= min_det_px) and (nd / npx > min_coverage)
        cav_frame = int(np.median(det.step_frame[det_here])) if cavitated else -1
        if cavitated:
            seg_cav[pix] = cav_frame
        major = r.axis_major_length
        rows.append(dict(
            tube_id=int(lab), cavitated=bool(cavitated), cav_frame=cav_frame,
            area_px=npx, area_um2=float(npx * scale * scale),
            length_um=float(major * scale), orientation=float(r.orientation),
            coverage=float(nd / npx), centroid_r=float(r.centroid[0]), centroid_c=float(r.centroid[1]),
        ))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["cavitated", "cav_frame"]).reset_index(drop=True)
    tdet = TubeDetections(accepted=(seg_cav < BIG), step_frame=seg_cav)
    return tdet, df
