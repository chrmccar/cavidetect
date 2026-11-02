"""Preprocessing: optional binning, spatial smoothing, temporal smoothing.
Keeps results in GREY LEVELS (DN) so the absolute magnitude threshold is meaningful."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter1d
from skimage.measure import block_reduce


def preprocess(stack: np.ndarray, bin_xy: int = 1, spatial_sigma: float = 1.0,
               temporal_smooth: int = 3) -> tuple[np.ndarray, float]:
    """Return (smoothed stack in DN [T,Y,X], effective_bin). Reversible scale: bin multiplies um/px."""
    a = stack
    if bin_xy > 1:
        a = block_reduce(a, (1, bin_xy, bin_xy), np.mean)
    if spatial_sigma > 0:
        a = gaussian_filter(a, (0, spatial_sigma, spatial_sigma))
    if temporal_smooth > 1:
        a = uniform_filter1d(a, size=temporal_smooth, axis=0)
    return a.astype(np.float32, copy=False), float(bin_xy)


def anscombe(x: np.ndarray) -> np.ndarray:
    """Variance-stabilising transform for Poisson-like noise (used for optional VST-domain work)."""
    return 2.0 * np.sqrt(np.maximum(x, 0) + 3.0 / 8.0)
