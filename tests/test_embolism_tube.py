"""Tube-shape prior + end-anchored onset — the fixes for the early surface/scale confound.

Real air-filling is tube-like (linear); the early-frame surface change is blob-like. And an
early transient blip must not set the onset: we anchor on the END state (last-water frame)."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import embolism as cemb


def _tube_and_blob():
    T, Y, X = 40, 70, 70
    rng = np.random.default_rng(1)
    base = 100.0
    s = np.full((T, Y, X), base, np.float32) + rng.normal(0, 1.0, (T, Y, X)).astype(np.float32)
    tissue = np.ones((Y, X), bool)
    tube = np.zeros((Y, X), bool); tube[12:58, 20:25] = True            # 5px-wide tube
    yy, xx = np.ogrid[:Y, :X]
    blob = ((yy - 35) ** 2 + (xx - 52) ** 2) <= 64                      # radius-8 blob
    s[20:, tube] += 40
    s[20:, blob] += 40
    return s, tissue, tube, blob


def test_tube_grow_recovers_full_width():
    s, tissue, tube, blob = _tube_and_blob()
    r = cemb.measure_embolism(s, 1.0, tissue, base_frames=8, change_min_dn=15.0, k_sigma=5.0,
                              tube_prior=True, tube_grow=True, vessel_min=0.04)
    assert r.embolised[tube].mean() > 0.8, f"full tube width not recovered ({r.embolised[tube].mean():.2f})"


def test_tube_grow_still_drops_round_blob():
    s, tissue, tube, blob = _tube_and_blob()
    r = cemb.measure_embolism(s, 1.0, tissue, base_frames=8, change_min_dn=15.0, k_sigma=5.0,
                              tube_prior=True, tube_grow=True, vessel_min=0.04)
    assert r.embolised[blob].mean() < 0.3, f"round blob not dropped ({r.embolised[blob].mean():.2f})"


def test_without_tube_prior_blob_is_kept():
    s, tissue, tube, blob = _tube_and_blob()
    r = cemb.measure_embolism(s, 1.0, tissue, base_frames=8, change_min_dn=15.0, k_sigma=5.0,
                              tube_prior=False)
    assert r.embolised[blob].mean() > 0.8, "sanity: without the gate the blob should be kept"


def test_end_anchored_onset_ignores_early_transient():
    T, Y, X = 80, 12, 12
    base = 100.0
    s = np.full((T, Y, X), base, np.float32)
    tissue = np.zeros((Y, X), bool); tissue[2:10, 2:10] = True
    s[6:11, 5, 5] = base + 40     # early transient blip (surface-change analogue)
    s[55:, 5, 5] = base + 40      # the real, permanent air-fill
    r = cemb.measure_embolism(s, 1.0, tissue, base_frames=4, tail_frac=0.15,
                              change_min_dn=15.0, k_sigma=5.0, onset_mode="end")
    assert r.embolised[5, 5]
    assert int(r.onset_frame[5, 5]) >= 50, f"end-anchored onset wrongly early: {r.onset_frame[5,5]}"
    # the old half-rise rule would latch onto the early blip
    rh = cemb.measure_embolism(s, 1.0, tissue, base_frames=4, tail_frac=0.15,
                               change_min_dn=15.0, k_sigma=5.0, onset_mode="half")
    assert int(rh.onset_frame[5, 5]) < 15
