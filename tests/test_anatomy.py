"""Increment 4 — anatomy segmentation: tube=lumen, rounded border sheet=scale, keep lumen-through-scale."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import anatomy as an


def _scene():
    """A tissue body with: a thin bright TUBE (lumen) and a big rounded bright SHEET at the
    border (scale). Background outside is dark."""
    final = np.full((80, 120), 20.0, np.float32)
    final[10:70, 10:110] = 110.0                       # tissue body
    tissue = np.zeros((80, 120), bool); tissue[10:70, 10:110] = True
    final[44:48, 20:100] = 230.0                       # thin elongated tube (high ridge)
    final[14:36, 12:40] = 225.0                        # big rounded sheet at the left margin
    tube = (slice(44, 48), slice(20, 100))
    sheet = (slice(14, 36), slice(12, 40))
    return final, tissue, tube, sheet


def test_tube_is_lumen_sheet_is_scale():
    final, tissue, tube, sheet = _scene()
    m = an.segment_anatomy(final, tissue, scale_area=200)
    assert m["lumen"][tube].mean() > 0.5                # the tube registers as lumen
    assert m["scale"][sheet].mean() > 0.3               # the rounded margin sheet is scale
    # the thin tube is mostly NOT classed as a scale sheet
    assert m["scale"][tube].mean() < 0.3


def test_masks_cover_tissue_without_gaps():
    final, tissue, tube, sheet = _scene()
    m = an.segment_anatomy(final, tissue, scale_area=200)
    # every tissue pixel is lumen, scale, or unknown (classes may overlap but must cover)
    covered = m["lumen"] | m["scale"] | m["unknown"]
    assert np.all(covered[tissue])
    # tape/background live strictly outside the thallus
    assert not (m["tape"] & tissue).any()
    assert not (m["background"] & tissue).any()


def test_refine_accepted_drops_tape_keeps_lumen_through_scale():
    final, tissue, tube, sheet = _scene()
    m = an.segment_anatomy(final, tissue, scale_area=200)
    # an accepted mask that includes: the tube (good), a patch of tape (bad), and a rhizoid
    # running under the scale sheet (good -> must be kept)
    shape = np.zeros_like(tissue)
    shape[tube] = True
    shape[2:6, 2:6] = True                              # outside tissue (tape/background)
    under_scale = (slice(26, 28), slice(14, 38))        # a lumen line beneath the scale
    final[under_scale] = 235.0
    m = an.segment_anatomy(final, tissue, scale_area=200)  # recompute with the under-scale line
    shape[under_scale] = True & m["lumen"][under_scale]
    keep = an.refine_accepted(shape, m)
    assert not keep[2:6, 2:6].any()                     # tape dropped
    assert keep[tube].mean() > 0.5                      # tube kept
