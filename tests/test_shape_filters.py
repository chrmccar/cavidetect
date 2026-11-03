"""Tests for the pure-Python BioVoxxel shape descriptors + region filter."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import shape_filters as sf


def _disc(radius, pad=8):
    n = 2 * radius + 2 * pad
    yy, xx = np.mgrid[0:n, 0:n]
    c = n // 2
    return ((yy - c) ** 2 + (xx - c) ** 2 <= radius ** 2)


def _bar(length, width, pad=8):
    m = np.zeros((width + 2 * pad, length + 2 * pad), bool)
    m[pad:pad + width, pad:pad + length] = True
    return m


def _prop(mask):
    from skimage.measure import label, regionprops
    return regionprops(label(mask.astype(int)))[0]


def test_disc_descriptors():
    mask = _disc(15)
    p = _prop(mask)
    assert abs(sf.thinness_ratio(p.area, p.perimeter) - 1.0) < 0.15
    assert abs(sf.elongation(p.axis_minor_length, p.axis_major_length) - 0.0) < 0.1
    assert abs(sf.max_inscribed_diameter_px(mask) - 30) < 2.5


def test_bar_descriptors():
    mask = _bar(60, 6)
    p = _prop(mask)
    assert abs(sf.elongation(p.axis_minor_length, p.axis_major_length) - (1 - 6 / 60)) < 0.12
    assert abs(sf.max_inscribed_diameter_px(mask) - 6) < 2.5
    assert sf.thinness_ratio(p.area, p.perimeter) < 0.6


def test_inscribed_um_uses_bin():
    # um_per_analysis_px = um_per_px(2.5) * bin(2) = 5.0; disc r=10 -> diameter 20px -> 100 um
    mask = _disc(10)
    d = sf.region_descriptors(_prop(mask), mask, um_per_analysis_px=5.0)
    assert abs(d["inscribed_um"] - 100) < 12


def test_degenerate_region():
    assert sf.thinness_ratio(1.0, 0.0) == 0.0
    assert sf.elongation(5.0, 0.0) == 0.0
    single = np.zeros((5, 5), bool)
    single[2, 2] = True
    d = sf.region_descriptors(_prop(single), single, um_per_analysis_px=1.0)
    assert all(np.isfinite(v) for v in d.values())


def test_filter_regions_expert_defaults():
    # At um_per_analysis_px=1.0 and the expert's rules (inscribed 7-25um, thinness 0-0.6,
    # elongation 0.6-1): a 60x10 tube passes (inscribed ~10um, elongation ~0.83, thin);
    # a disc r=15 fails every rule (inscribed 30um, thinness ~1, elongation ~0).
    from skimage.measure import label
    canvas = np.zeros((140, 140), bool)
    disc = _disc(15, pad=2)
    canvas[5:5 + disc.shape[0], 5:5 + disc.shape[1]] = disc
    bar = _bar(60, 10, pad=2)
    canvas[100:100 + bar.shape[0], 20:20 + bar.shape[1]] = bar
    lbl = label(canvas.astype(int))
    rules = {"inscribed_um": (7, 25), "thinness": (0, 0.6),
             "elongation": (0.6, 1.0), "min_area_px": 8}
    keep = sf.filter_regions(lbl, rules, um_per_analysis_px=1.0)
    assert keep[100 + 2 + 5, 20 + 2 + 30]       # a point inside the bar kept
    assert not keep[5 + 2 + 15, 5 + 2 + 15]     # centre of the disc dropped
