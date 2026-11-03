"""Tests for the colour-shape onset timing (start-of-rise, not late global crossing)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import colour_shape as cs


def _step_stack(T=30, step=10, lo=20.0, hi=200.0):
    """A (T,1,2) stack: pixel 0 steps lo->hi at frame `step` and stays; pixel 1 stays dark."""
    stack = np.full((T, 1, 2), lo, np.float32)
    stack[step:, 0, 0] = hi
    return stack


def test_onset_dates_the_start_of_the_rise_not_the_end():
    stack = _step_stack(T=30, step=10)
    mask = np.array([[True, False]])
    onset = cs.onset_start_of_rise(stack, mask, rise_frac=0.4, persist=3, smooth=3)
    # the embolised pixel is dated near its step (frame 10), NOT near the end of the clip
    assert 8 <= onset[0, 0] <= 12, onset[0, 0]
    # the non-embolised pixel is marked -1
    assert onset[0, 1] == -1


def test_onset_is_earlier_than_the_old_last_crossing_rule():
    # a pixel that rises at frame 10 but flickers brighter again late would, under the old
    # "last frame below a high global threshold + 1" rule, be dated late. Start-of-rise stays early.
    stack = _step_stack(T=40, step=10)
    stack[30:, 0, 0] = 255.0            # a late extra-bright flicker
    mask = np.array([[True, False]])
    onset = cs.onset_start_of_rise(stack, mask)
    assert onset[0, 0] <= 14, onset[0, 0]


def test_onset_all_in_range_and_masked():
    stack = _step_stack()
    mask = np.array([[True, True]])
    onset = cs.onset_start_of_rise(stack, mask)
    assert onset.min() >= 0 and onset.max() <= stack.shape[0] - 1


def _emb_stack(T=40, Y=70, X=90, seed=0):
    """A stack whose dry final frame has several bright elongated (air-filled)
    tubes filling ~40% of a darker tissue body — enough for the default
    colour-frangi method to fire (the air fraction must clear the percentile cut)."""
    rng = np.random.default_rng(seed)
    stack = np.full((T, Y, X), 30.0, np.float32)
    body = np.zeros((Y, X), bool)
    body[10:60, 10:80] = True
    stack[:, body] = 45.0
    tubes = np.zeros((Y, X), bool)
    for r in range(14, 58, 6):
        tubes[r:r + 3, 16:74] = True            # thin tubes
    stack[T // 2:, tubes] = 210.0               # air-fill in the 2nd half
    stack = stack + rng.normal(0, 2, stack.shape).astype(np.float32)
    return np.clip(stack, 0, 255)


def test_colour_frangi_default_matches_detect_spatial():
    # Regression: the default method must reproduce the current detect_spatial mask.
    stack = _emb_stack()
    tail = max(int(round(stack.shape[0] * 0.12)), 1)
    final = stack[-tail:].mean(0)
    expected = cs.detect_spatial(final)["shape"]
    assert expected.any(), "fixture should produce a non-empty detection"
    got = cs.measure(stack)["shape"]
    assert np.array_equal(got, expected)


def test_empty_rules_give_zero_curve():
    stack = _emb_stack()
    r = cs.measure(stack, method="shape-filter",
                   params={"inscribed_lo_um": 1000, "threshold": "otsu"})
    assert r["frac"][-1] == 0.0


def test_run_writes_params_json(tmp_path):
    import json
    import tifffile
    folder = tmp_path / "cavi_test"
    folder.mkdir()
    for i, fr in enumerate(_emb_stack(T=12)):
        tifffile.imwrite(folder / f"t_{i:04d}.tif", fr.astype(np.uint8))
    out = tmp_path / "out"
    cs.run(str(folder), str(out), bin_xy=1)
    pj = out / "cavi_test_cs_params.json"
    assert pj.exists()
    data = json.loads(pj.read_text())
    assert data["method"] == "colour-frangi"
    assert "params" in data
