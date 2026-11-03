"""Pure-Python BioVoxxel shape descriptors + a connected-component shape filter.

Reimplements the ImageJ/BioVoxxel *Shape Filter* measurements the expert uses
(no ImageJ dependency): thinness ratio, elongation, and the maximum inscribed
circle. ``filter_regions`` keeps the components whose descriptors fall inside a
set of ``[lo, hi]`` rules — the shape-filter detection method is built on it.

All functions are pure (numpy/scipy/skimage), so they unit-test without a GUI
or a TIFF stack. Sizes convert to micrometres via ``um_per_analysis_px`` =
``um_per_px * bin`` (one binned pixel spans ``bin`` original pixels).
"""
from __future__ import annotations
import math

import numpy as np


def thinness_ratio(area_px: float, perimeter_px: float) -> float:
    """BioVoxxel thinness ratio = 4*pi*area / perimeter**2, clipped to [0, 1].

    1.0 for a perfect disc, approaching 0 for a thin/elongated shape. Returns
    0.0 for a non-positive perimeter (degenerate region)."""
    if perimeter_px <= 0:
        return 0.0
    return float(min(1.0, max(0.0, 4.0 * math.pi * area_px / (perimeter_px ** 2))))


def elongation(minor_len: float, major_len: float) -> float:
    """Elongation = 1 - minor/major of the fitted ellipse.

    0 for a round shape, approaching 1 for a line. Returns 0.0 if the major
    axis is non-positive (degenerate region)."""
    if major_len <= 0:
        return 0.0
    return float(1.0 - minor_len / major_len)


def max_inscribed_diameter_px(region_mask: np.ndarray) -> float:
    """Diameter (px) of the largest circle that fits inside the region = twice
    the maximum of the Euclidean distance transform of the region mask."""
    from scipy.ndimage import distance_transform_edt
    if not region_mask.any():
        return 0.0
    # pad with a background border so the distance transform measures to the
    # region's OWN edge, not the array edge (a region that fills its bounding
    # box has no interior background otherwise).
    padded = np.pad(region_mask, 1)
    return float(2.0 * distance_transform_edt(padded).max())


def region_descriptors(prop, region_mask: np.ndarray,
                       um_per_analysis_px: float) -> dict:
    """Bundle the descriptors for one region.

    ``prop`` is a skimage ``RegionProperties``; ``region_mask`` is the region's
    own boolean mask (cropped or full-frame). Sizes are in micrometres where
    applicable."""
    insc_px = max_inscribed_diameter_px(region_mask)
    return {
        "inscribed_um": insc_px * um_per_analysis_px,
        "thinness": thinness_ratio(float(prop.area), float(prop.perimeter)),
        "elongation": elongation(float(prop.axis_minor_length),
                                 float(prop.axis_major_length)),
        "area_um2": float(prop.area) * (um_per_analysis_px ** 2),
        "area_px": float(prop.area),
    }


def _in_range(value: float, bounds) -> bool:
    """True if ``value`` is within ``(lo, hi)``; a None bound disables that side."""
    if bounds is None:
        return True
    lo, hi = bounds
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


def filter_regions(label_img: np.ndarray, rules: dict,
                   um_per_analysis_px: float) -> np.ndarray:
    """Boolean mask of the labelled components whose descriptors pass every rule.

    ``rules`` keys (all optional; omit or set None to disable a rule):
      ``inscribed_um=(lo,hi)``, ``thinness=(lo,hi)``, ``elongation=(lo,hi)``,
      ``min_area_px=int``.
    """
    from skimage.measure import regionprops
    out = np.zeros(label_img.shape, bool)
    min_area = rules.get("min_area_px")
    for p in regionprops(label_img):
        if min_area is not None and p.area < min_area:
            continue
        region_mask = label_img[p.slice] == p.label
        d = region_descriptors(p, region_mask, um_per_analysis_px)
        if (_in_range(d["inscribed_um"], rules.get("inscribed_um"))
                and _in_range(d["thinness"], rules.get("thinness"))
                and _in_range(d["elongation"], rules.get("elongation"))):
            out[p.slice][region_mask] = True
    return out
