"""Increment 1 — diagnostic export: stage montage + per-frame/cumulative event TIFFs."""
import os
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import colour_shape as cs, diagnostics as dg, outputs as cout


def _tube_stack(T=24, step=10):
    """A dark tissue body with one bright tube that air-fills at frame `step`."""
    stack = np.full((T, 60, 90), 30.0, np.float32)
    stack[:, 10:50, 10:80] = 40.0                      # tissue body
    stack[step:, 25:30, 15:75] = 220.0                 # tube fills bright at `step`
    return stack


def test_detect_spatial_exposes_stages_without_changing_shape():
    final = _tube_stack()[-1]
    d = cs.detect_spatial(final)
    assert "stages" in d
    assert set(d["stages"]) >= {"brightness", "frangi", "seed", "accepted"}
    # the stages must not alter the accepted mask
    assert np.array_equal(d["stages"]["accepted"], d["shape"])


def test_stage_images_lists_panels_in_order():
    r = cs.measure(_tube_stack(), method="colour-frangi")
    panels = dg.stage_images(r)
    titles = [t for t, _img, _m in panels]
    assert titles[0] == "raw final" and titles[1] == "tissue"
    assert titles[-1].startswith("COLOUR + SHAPE")
    # every panel image matches the frame size
    for _t, img, _m in panels:
        assert img.shape == r["final"].shape


def test_save_stage_montage_writes_png(tmp_path):
    r = cs.measure(_tube_stack(), method="colour-frangi")
    p = dg.save_stage_montage(r, str(tmp_path), "demo")
    assert os.path.exists(p) and p.endswith("_cs_stages.png")
    assert os.path.getsize(p) > 0


def test_event_tiffs_are_consistent_and_binary(tmp_path):
    r = cs.measure(_tube_stack(), method="colour-frangi")
    T = _tube_stack().shape[0]
    paths = cout.save_event_tiffs(r["onset"], r["shape"], T, str(tmp_path), "demo")
    per = tifffile.imread(paths["per_frame"])
    cum = tifffile.imread(paths["cumulative"])
    assert per.shape == (T, *r["final"].shape)
    assert set(np.unique(per)).issubset({0, 255})
    # cumulative == running OR of per-frame, exactly
    expect = (np.cumsum(per.astype(bool), 0) > 0).astype(np.uint8) * 255
    assert np.array_equal(cum, expect)
    # every accepted pixel appears exactly once across the per-frame stack
    assert int((per > 0).sum()) == int(((r["onset"] >= 0) & r["shape"]).sum())
    # and the final cumulative frame equals the accepted mask
    assert np.array_equal(cum[-1] > 0, r["shape"] & (r["onset"] >= 0))
