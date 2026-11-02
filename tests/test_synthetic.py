"""Behavioural tests on synthetic data: detector keeps events, rejects each confound,
and does not flood the frame (the M0 over-detection failure)."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import preprocess as cpp, characterise as cch, detect as cdet, group as cgrp
from tests.make_synthetic import make_synthetic


def _run():
    stack, gt = make_synthetic()
    stats = cch.characterise(stack, tail_frac=0.35)
    s, b = cpp.preprocess(stack, bin_xy=1, spatial_sigma=1.0, temporal_smooth=3)
    det = cdet.detect_pixels(s, stats.noise_sigma_dn, step_min_dn=10.0, step_k_sigma=5.0)
    df, label = cgrp.group_events(det, um_per_px=2.5, bin_xy=b, min_area_px=4, event_time_tol=4)
    return stack, gt, det, df


def test_recall_events():
    stack, gt, det, df = _run()
    assert len(df) > 0, "no events detected at all"
    # each ground-truth event centre should be near a detected centroid
    hits = 0
    for e in gt:
        if len(df):
            near = df[(df["frame_peak"] - e["frame"]).abs() <= 6]
            if len(near):
                d = np.hypot(near["centroid_r"] - e["cr"], near["centroid_c"] - e["cc"]).min()
                if d < 15:
                    hits += 1
    assert hits >= 4, f"recall too low: {hits}/5 injected events matched"


def test_not_flooded():
    stack, gt, det, df = _run()
    # over-detection guard: accepted pixels must be a small fraction, not ~half the frame (M0 bug)
    frac = det.accepted.mean()
    assert frac < 0.15, f"detector floods the frame: {frac:.2%} accepted"
    assert len(df) < 60, f"far too many events ({len(df)}) -> confounds leaking in"


def test_drift_not_detected():
    stack, gt, det, df = _run()
    # the drift region (x>55% of width) should hold almost no detections
    if len(df):
        in_drift = (df["centroid_c"] > stack.shape[2] * 0.6).mean()
        assert in_drift < 0.4, f"too many detections in the gradual-drift region: {in_drift:.2%}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
    print("ALL PASS")
