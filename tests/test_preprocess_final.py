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


# --- Faithful ImageJ CLAHE (expert's macro: blocksize 127 / 256 bins / slope 1.5) ---

def test_imagej_clahe_clip_maps_maxslope_over_bins():
    # ImageJ 'Enhance Local Contrast' limits the transfer-function slope; the
    # per-bin clip as a fraction of the window is maxSlope / bins.
    assert pp.imagej_clahe_clip(1.5, 256) == 1.5 / 256
    assert pp.imagej_clahe_clip(3.0, 128) == 3.0 / 128


def _faint_local_feature(n=96):
    """A textured, uneven-illumination field with a low-contrast local patch —
    the kind of frame CLAHE is meant to pull a faint rhizoid out of."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    img = 60.0 + 0.6 * xx + 0.3 * yy                 # smooth illumination ramp
    img += rng.normal(0, 4, (n, n)).astype(np.float32)  # texture for the histogram
    img[40:56, 40:56] += 10.0                        # faint local feature
    return img.astype(np.float32)


def test_clahe_increases_local_contrast_of_faint_feature():
    img = _faint_local_feature()
    patch, bg = (slice(40, 56), slice(40, 56)), (slice(40, 56), slice(0, 16))
    base = float(img[patch].mean() - img[bg].mean())
    out = pp.apply_contrast(img, "clahe", clahe_blocksize=32,
                            clahe_bins=256, clahe_maxslope=3.0)
    # compare on a common scale: local contrast as a fraction of the frame's span
    base_n = base / (img.max() - img.min())
    enh_n = float(out[patch].mean() - out[bg].mean()) / (out.max() - out.min())
    assert enh_n > base_n                            # the patch stands out more


def test_clahe_maxslope_overrides_default_clip():
    # A higher maximum slope => larger clip => more aggressive equalisation,
    # so the two settings must not produce identical output.
    img = _faint_local_feature()
    mild = pp.apply_contrast(img, "clahe", clahe_blocksize=32, clahe_maxslope=1.1)
    strong = pp.apply_contrast(img, "clahe", clahe_blocksize=32, clahe_maxslope=4.0)
    assert not np.allclose(mild, strong)


# --- CLAHE scope: build the final frame from the stack (expert enhances every frame) ---

def _tiny_stack(T=6, n=48):
    rng = np.random.default_rng(0)
    base = np.full((T, n, n), 100.0, np.float32)
    base += rng.normal(0, 2, base.shape).astype(np.float32)
    base[:, 20:28, 20:28] += 10.0       # a faint persistent feature
    return base


def test_final_from_stack_default_is_plain_mean():
    stack = _tiny_stack()
    final = pp.final_from_stack(stack, tail_frac=0.5, contrast="none")
    tail = stack[-max(int(round(stack.shape[0] * 0.5)), 1):].mean(0)
    assert np.array_equal(final, tail)          # byte-identical to the old reduction


def test_final_from_stack_perframe_clahe_differs_from_plain_mean():
    stack = _tiny_stack()
    plain = pp.final_from_stack(stack, tail_frac=0.5, contrast="none")
    enhanced = pp.final_from_stack(stack, tail_frac=0.5, contrast="clahe",
                                   clahe_blocksize=16, clahe_maxslope=3.0)
    assert enhanced.shape == plain.shape
    assert not np.allclose(enhanced, plain)
