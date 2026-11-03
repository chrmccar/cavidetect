"""Tests for final-frame preprocessing: contrast adjust + difference-of-Gaussians band-pass."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import preprocess_final as pp


def _grey_gradient(n=64):
    return np.tile(np.linspace(10, 240, n, dtype=np.float32), (n, 1))


def test_contrast_none_is_identity():
    img = _grey_gradient()
    out = pp.apply_contrast(img, "none")
    assert np.array_equal(out, img)


def test_gamma_changes_values_monotonically():
    img = _grey_gradient()
    out = pp.apply_contrast(img, "gamma", gamma=0.5)
    # gamma < 1 brightens mid-tones; monotonic mapping preserves ordering along the gradient
    row = out[0]
    assert np.all(np.diff(row) >= -1e-5)          # still monotonic increasing
    assert out.mean() > img.mean()                # overall brighter


def test_bandpass_removes_flat_background():
    flat = np.full((64, 64), 100.0, np.float32)
    out = pp.bandpass_dog(flat, small_um=2.5, large_um=40.0, um_per_analysis_px=2.5)
    assert np.allclose(out, 0.0, atol=1e-3)


def test_bandpass_passes_midscale_feature():
    img = np.zeros((80, 80), np.float32)
    # a bar whose width sits in the pass band (~ (2.5+40)/2 um at 2.5 um/px -> ~8 px)
    img[38:46, 10:70] = 200.0
    out = pp.bandpass_dog(img, small_um=2.5, large_um=40.0, um_per_analysis_px=2.5)
    assert out[42, 40] > 5.0                       # the feature survives with real contrast


def test_preprocess_default_is_identity():
    img = _grey_gradient()
    out = pp.preprocess_final(img)
    assert np.array_equal(out, img)
