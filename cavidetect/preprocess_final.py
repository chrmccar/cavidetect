"""Final-frame preprocessing shared by the detection methods.

Two optional, default-off steps applied to the dry final frame before a
method's threshold:

- ``contrast`` — gamma or CLAHE, to pull faint rhizoids out of the background.
- ``bandpass`` — a difference-of-Gaussians band-pass that keeps structures in a
  size band (default 2.5-40 um, mirroring the band-pass the expert tried in
  ImageJ) and removes slow background and pixel noise. Dependency-free (two
  Gaussian blurs), so no FFT/ImageJ needed.

Sizes convert to pixels via ``um_per_analysis_px`` = ``um_per_px * bin``.
All functions return float32 arrays of the input shape.
"""
from __future__ import annotations

import numpy as np


def _to01(img):
    """Scale to [0,1] by the image's own min/max; return (scaled, lo, span)."""
    lo = float(img.min())
    span = float(img.max()) - lo
    if span <= 0:
        return np.zeros_like(img, np.float32), lo, 0.0
    return ((img - lo) / span).astype(np.float32), lo, span


def apply_contrast(img: np.ndarray, mode: str = "none",
                   gamma: float = 1.0, clahe_clip: float = 0.01) -> np.ndarray:
    """Contrast adjust the frame. ``mode`` in {"none","gamma","clahe"}.

    ``none`` returns the input unchanged. ``gamma`` applies ``out = in**gamma``
    on the [0,1]-scaled image (gamma<1 brightens). ``clahe`` applies contrast-
    limited adaptive histogram equalisation."""
    img = img.astype(np.float32)
    if mode == "none":
        return img
    from skimage import exposure
    scaled, lo, span = _to01(img)
    if span == 0:
        return img
    if mode == "gamma":
        out01 = exposure.adjust_gamma(scaled, gamma)
    elif mode == "clahe":
        out01 = exposure.equalize_adapthist(scaled, clip_limit=clahe_clip)
    else:
        raise ValueError(f"unknown contrast mode: {mode!r}")
    return (out01 * span + lo).astype(np.float32)


def bandpass_dog(img: np.ndarray, small_um: float, large_um: float,
                 um_per_analysis_px: float) -> np.ndarray:
    """Difference-of-Gaussians band-pass: blur at the small scale minus blur at
    the large scale. Passes structures between ``small_um`` and ``large_um``,
    zeroing flat background. Sigmas = scale_um / um_per_analysis_px."""
    from scipy.ndimage import gaussian_filter
    img = img.astype(np.float32)
    s_small = max(small_um / um_per_analysis_px, 1e-3)
    s_large = max(large_um / um_per_analysis_px, s_small + 1e-3)
    return (gaussian_filter(img, s_small) - gaussian_filter(img, s_large)).astype(np.float32)


def preprocess_final(final: np.ndarray, *, contrast: str = "none",
                     gamma: float = 1.0, clahe_clip: float = 0.01,
                     bandpass: bool = False, bp_small_um: float = 2.5,
                     bp_large_um: float = 40.0,
                     um_per_analysis_px: float = 2.5) -> np.ndarray:
    """Apply contrast then (optionally) the band-pass. All defaults = identity."""
    out = apply_contrast(final, contrast, gamma=gamma, clahe_clip=clahe_clip)
    if bandpass:
        out = bandpass_dog(out, bp_small_um, bp_large_um, um_per_analysis_px)
    return out.astype(np.float32)
