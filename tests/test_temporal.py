"""Increment 2 — temporal evidence: abrupt fill kept, gradual drying scored low."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import temporal as tp, colour_shape as cs, methods as M


def _scene(T=40, step=12):
    """Two bright endpoints reached differently inside one tissue body:
      - region A (abrupt): dark -> bright as a STEP at frame `step`  (a cavitation)
      - region B (gradual): dark -> bright as a slow RAMP over the whole clip (drying)
    Both finish equally bright, so a final-frame-only method cannot separate them."""
    stack = np.full((T, 40, 80), 20.0, np.float32)    # dark background
    stack[:, 5:35, 5:75] = 120.0                      # tissue body (clearly separable)
    tissue = np.zeros((40, 80), bool); tissue[5:35, 5:75] = True
    # A: step
    stack[step:, 12:17, 10:35] = 220.0
    # B: linear ramp to the same brightness
    ramp = np.linspace(120.0, 220.0, T)
    stack[:, 22:27, 45:70] = ramp[:, None, None]
    A = (slice(12, 17), slice(10, 35))
    B = (slice(22, 27), slice(45, 70))
    return stack, tissue, A, B


def test_abrupt_kept_gradual_scored_lower():
    stack, tissue, A, B = _scene()
    d = tp.temporal_evidence(stack, tissue, min_confidence=0.0, min_jump_frac=0.0)
    step, conf = d["step_score"], d["confidence"]
    # the abrupt region has a much higher step score than the gradual ramp
    assert step[A].mean() > 0.6
    assert step[B].mean() < 0.3
    assert conf[A].mean() > conf[B].mean() + 0.3


def test_candidate_gate_keeps_abrupt_rejects_gradual():
    stack, tissue, A, B = _scene()
    d = tp.temporal_evidence(stack, tissue)        # default permissive gate
    shape = d["shape"]
    assert shape[A].mean() > 0.8                    # abrupt fill is accepted
    assert shape[B].mean() < 0.2                    # slow drying is not


def test_onset_near_the_step_and_transition_within_two():
    stack, tissue, A, B = _scene(step=12)
    d = tp.temporal_evidence(stack, tissue)
    onset = d["onset"]
    on_A = onset[A][d["shape"][A]]
    assert on_A.size and abs(np.median(on_A) - 12) <= 3
    # transition width recorded as 1 or 2 frames for accepted pixels
    tr = d["transition"][d["shape"]]
    assert set(np.unique(tr)).issubset({1, 2})


def test_late_drift_after_fast_fill_is_not_rejected():
    """A fast fill that then dims/brightens slightly later must stay accepted (brief §2)."""
    T = 40
    stack = np.full((T, 10, 10), 40.0, np.float32)
    tissue = np.ones((10, 10), bool)
    stack[10:, 4:6, 4:6] = 210.0
    stack[25:, 4:6, 4:6] = 180.0        # later brightness drift downward
    stack[33:, 4:6, 4:6] = 230.0        # and back up
    d = tp.temporal_evidence(stack, tissue)
    assert d["shape"][4:6, 4:6].all()


def test_measure_uses_method_onset_for_temporal():
    stack, tissue, A, B = _scene()
    r = cs.measure(stack, method="temporal-evidence")
    assert r["confidence"] is not None
    assert r["shape"][A].mean() > 0.8
    # cumulative curve is monotonic non-decreasing
    assert np.all(np.diff(r["cum"]) >= 0)


def test_temporal_registered_and_needs_stack():
    names = [n for n, _ in M.list_methods()]
    assert "temporal-evidence" in names
    assert M.get_method("temporal-evidence").needs_stack is True
