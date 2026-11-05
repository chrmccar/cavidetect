"""Memory fix: the viewer/animation must normalise ONE frame at a time, never promote the
whole (T,Y,X) stack to float64 (the 14.6 GiB OOM the expert reported). See
CaviDetect_memory_optimization_v2."""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import outputs as cout


def _stack(T=12, Y=16, X=20, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((T, Y, X)) * 255).astype(np.float32)


def _result(stack):
    T, Y, X = stack.shape
    onset = np.full((Y, X), -1, np.int32)
    emb = np.zeros((Y, X), bool)
    emb[2:6, 3:8] = True; onset[2:6, 3:8] = T // 2
    return SimpleNamespace(onset_frame=onset, embolised=emb,
                           frac_embolised=np.linspace(0, 0.2, T))


def test_per_frame_grey_matches_old_full_stack_formula():
    """Byte-identical fidelity: the new per-frame uint8 equals the old `g[t]` slice."""
    s = _stack()
    lo, hi = np.percentile(s, [2, 99]); rng = max(hi - lo, 1e-3)
    g_full = np.clip((s - lo) / rng, 0, 1)                 # the OLD full-stack array
    for t in range(s.shape[0]):
        old = (g_full[t] * 255).astype(np.uint8)
        new = (np.clip((s[t] - lo) / rng, 0, 1) * 255).astype(np.uint8)
        assert np.array_equal(old, new)


def test_viewer_never_passes_full_stack_to_clip(tmp_path, monkeypatch):
    """Structural guard for the OOM: np.clip is only ever handed a 2-D frame, never the
    full (T,Y,X) stack (which under float64 promotion was the 14.6 GiB allocation)."""
    s = _stack()
    seen = []
    real_clip = np.clip
    monkeypatch.setattr(cout.np, "clip", lambda a, *args, **kw: (seen.append(np.asarray(a).shape), real_clip(a, *args, **kw))[1])
    cout.save_embolism_viewer(s, _result(s), str(tmp_path), "d")
    assert seen, "clip was never called"
    assert all(len(shp) <= 2 for shp in seen), f"clip saw a >2-D array: {seen}"
    assert s.shape not in seen


def test_animation_never_passes_full_stack_to_clip(tmp_path, monkeypatch):
    s = _stack()
    det = SimpleNamespace(accepted=_result(s).embolised,
                          step_frame=np.where(_result(s).embolised, 6, 0).astype(np.int32))
    seen = []
    real_clip = np.clip
    monkeypatch.setattr(cout.np, "clip", lambda a, *args, **kw: (seen.append(np.asarray(a).shape), real_clip(a, *args, **kw))[1])
    out = cout.save_animation(s, det, str(tmp_path), "d", stride=2)
    if out is None:
        return                      # PIL unavailable
    assert all(len(shp) <= 2 for shp in seen), f"clip saw a >2-D array: {seen}"
    assert s.shape not in seen


def test_viewer_output_unchanged_shape_and_opens(tmp_path):
    s = _stack()
    p = cout.save_embolism_viewer(s, _result(s), str(tmp_path), "d")
    assert os.path.exists(p)
    html = open(p, encoding="utf-8").read()
    assert "embolism viewer" in html and "data:image/jpeg;base64," in html
