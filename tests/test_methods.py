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


def test_shape_filter_defaults_reproduce_imagej_macro():
    # The shape-filter method is the faithful ImageJ flow: CLAHE on by default
    # with the expert's macro values, per-frame scope, and a near-zero size floor.
    p = M.resolve_params("shape-filter", {})
    assert p["contrast"] == "clahe"
    assert p["clahe_blocksize"] == 127
    assert p["clahe_bins"] == 256
    assert p["clahe_maxslope"] == 1.5
    assert p["clahe_scope"] == "per-frame"
    assert p["min_area_px"] == 1


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


def _rising_stack(T=8, Y=48, X=64):
    """A tube that fills bright over the last frames, on a textured body."""
    rng = np.random.default_rng(1)
    stack = np.full((T, Y, X), 30.0, np.float32)
    stack += rng.normal(0, 3, stack.shape).astype(np.float32)
    stack[:, 6:42, 6:58] += 60.0                 # tissue body
    for t in range(T):
        if t >= T - 2:
            stack[t, 16:21, 12:50] = 220.0        # tube fills near the end
    return stack


def test_measure_perframe_clahe_builds_enhanced_final():
    from cavidetect import colour_shape as cs
    from cavidetect import preprocess_final as pp
    stack = _rising_stack()
    tail_frac = 0.25
    plain = pp.final_from_stack(stack, tail_frac, contrast="none")

    r_final = cs.measure(stack, method="shape-filter", tail_frac=tail_frac,
                         params={"clahe_scope": "final"})
    assert np.array_equal(r_final["final"], plain)      # 'final' scope == plain mean

    r_pf = cs.measure(stack, method="shape-filter", tail_frac=tail_frac,
                      params={"clahe_scope": "per-frame"})
    expected = pp.final_from_stack(stack, tail_frac, contrast="clahe",
                                   clahe_blocksize=127, clahe_bins=256, clahe_maxslope=1.5)
    assert np.array_equal(r_pf["final"], expected)       # per-frame CLAHE'd then averaged
    assert not np.allclose(r_pf["final"], plain)


def test_shape_filter_detect_returns_masks():
    final, tissue = _tube_final()
    m = M.get_method("shape-filter")
    params = M.resolve_params("shape-filter", {"threshold": "otsu"})
    d = m.detect(final, tissue, **params)
    assert d["shape"].dtype == bool and d["shape"].shape == final.shape
    assert d["colour"].dtype == bool and d["colour"].shape == final.shape
    assert d["shape"].any()                     # the tube is detected
