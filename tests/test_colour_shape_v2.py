"""Unit tests for the v2 ridge-gated detector and scale rejection.

Synthetic final frame: a thin bright TUBE (tubular -> high Frangi ridge) and a filled
bright SHEET at the tissue border (a 'scale' -> ~0 interior ridge). The detector must
KEEP the tube (recall) and DROP the sheet (scale rejection)."""
import numpy as np
import pytest

from cavidetect import colour_shape as cs


def _synthetic():
    H, W = 200, 300
    tissue = np.zeros((H, W), bool)
    tissue[20:180, 20:280] = True
    rng = np.random.default_rng(0)
    final = np.zeros((H, W), np.float32)
    final[tissue] = 60 + rng.normal(0, 4, tissue.sum()).astype(np.float32)  # dim tissue
    # thin bright tube (2 px tall) spanning most of the width
    tube = np.zeros((H, W), bool); tube[99:101, 40:260] = True
    final[tube] = 210
    # filled bright sheet near the bottom-right border = a scale
    scale = np.zeros((H, W), bool); scale[150:178, 232:272] = True
    final[scale] = 210
    return final, tissue, tube, scale


def test_v2_detects_thin_tube():
    final, tissue, tube, scale = _synthetic()
    d = cs.detect_spatial_v2(final, tissue=tissue)
    recall = d["shape"][tube].mean()
    assert recall > 0.7, f"tube recall too low: {recall:.2f}"


def test_v2_rejects_border_scale():
    final, tissue, tube, scale = _synthetic()
    d = cs.detect_spatial_v2(final, tissue=tissue)
    leaked = d["shape"][scale].mean()
    assert leaked < 0.3, f"scale not rejected: {leaked:.2f} of it marked"


def test_ridge_map_higher_on_tube_than_sheet_interior():
    final, tissue, tube, scale = _synthetic()
    vn = cs.ridge_map(final, tissue)
    interior = np.zeros_like(scale); interior[158:170, 240:264] = True  # sheet core
    assert vn[tube].mean() > 3 * vn[interior].mean() + 1e-6


def test_v2_returns_expected_keys():
    final, tissue, *_ = _synthetic()
    d = cs.detect_spatial_v2(final, tissue=tissue)
    assert set(d) >= {"tissue", "air", "colour", "shape", "ridge"}
    assert d["shape"].dtype == bool and d["shape"].shape == final.shape
