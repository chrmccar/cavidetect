"""Tests for the detection-method registry, parameter schema, and params resolution."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import methods as M


def test_list_methods_order():
    names = [n for n, _label in M.list_methods()]
    assert names[0] == "colour-frangi"
    assert set(names) >= {"colour-frangi", "shape-filter", "ridge-hysteresis"}


def test_resolve_fills_defaults():
    p = M.resolve_params("shape-filter", {})
    assert p["inscribed_hi_um"] == 25
    assert p["inscribed_lo_um"] == 7
    assert p["elongation_lo"] == 0.6


def test_resolve_coerces_and_clamps():
    p = M.resolve_params("shape-filter", {"bright_pct": "200"})
    assert p["bright_pct"] == 100       # str -> int, clamped to max 100


def test_resolve_drops_unknown():
    with pytest.warns(UserWarning):
        p = M.resolve_params("shape-filter", {"bogus": 1})
    assert "bogus" not in p


def test_get_method_unknown_lists_valid():
    with pytest.raises(KeyError) as e:
        M.get_method("nope")
    assert "colour-frangi" in str(e.value)


def _tube_final(Y=90, X=120):
    """A dark tissue blob with a bright elongated (air-filled) tube inside it.
    Bimodal so an Otsu cut isolates the tube from the body."""
    final = np.zeros((Y, X), np.float32)
    final[10:80, 10:110] = 40.0                 # dark tissue body
    final[40:45, 20:100] = 220.0                # a bright tube ~5px wide, 80 long
    tissue = final > 0
    return final, tissue


def test_shape_filter_detect_returns_masks():
    final, tissue = _tube_final()
    m = M.get_method("shape-filter")
    params = M.resolve_params("shape-filter", {"threshold": "otsu"})
    d = m.detect(final, tissue, **params)
    assert d["shape"].dtype == bool and d["shape"].shape == final.shape
    assert d["colour"].dtype == bool and d["colour"].shape == final.shape
    assert d["shape"].any()                     # the tube is detected
