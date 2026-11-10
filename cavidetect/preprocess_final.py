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


def imagej_clahe_clip(maxslope: float, bins: int) -> float:
    """skimage ``clip_limit`` equivalent to ImageJ's 'maximum slope'.

    ImageJ's 'Enhance Local Contrast (CLAHE)' limits the slope of the per-tile
    transfer function to ``maxSlope``; skimage's ``equalize_adapthist`` instead
    takes ``clip_limit`` as the per-bin clip expressed as a fraction of the tile
    pixel count. For a uniform histogram each of ``bins`` bins would hold
    ``tile_pixels / bins`` counts (slope 1), so a slope cap of ``maxSlope``
    corresponds to a per-bin fraction of ``maxSlope / bins``.
    """
    return float(maxslope) / float(bins)


def apply_contrast(img: np.ndarray, mode: str = "none",
                   gamma: float = 1.0, clahe_clip: float | None = None,
                   clahe_blocksize: int = 127, clahe_bins: int = 256,
                   clahe_maxslope: float = 1.5) -> np.ndarray:
    """Contrast adjust the frame. ``mode`` in {"none","gamma","clahe"}.

    ``none`` returns the input unchanged. ``gamma`` applies ``out = in**gamma``
    on the [0,1]-scaled image (gamma<1 brightens). ``clahe`` applies contrast-
    limited adaptive histogram equalisation.

    CLAHE mirrors the expert's ImageJ macro ('Enhance Local Contrast', the
    ``fast_(less_accurate)`` tiled+interpolated variant that ``equalize_adapthist``
    implements): ``clahe_blocksize`` is the tile size (skimage ``kernel_size``),
    ``clahe_bins`` the histogram bins (``nbins``), and ``clahe_maxslope`` the
    maximum transfer-function slope, mapped to ``clip_limit`` via
    ``imagej_clahe_clip``. Defaults = the macro (127 / 256 / 1.5). An explicit
    ``clahe_clip`` overrides the slope-derived value."""
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
        clip = clahe_clip if clahe_clip is not None else imagej_clahe_clip(clahe_maxslope, clahe_bins)
        # the tile can't exceed the image; clamp per axis so small frames still run
        ksize = [min(int(clahe_blocksize), int(s)) for s in scaled.shape]
        out01 = exposure.equalize_adapthist(scaled, kernel_size=ksize,
                                            clip_limit=clip, nbins=int(clahe_bins))
    else:
        raise ValueError(f"unknown contrast mode: {mode!r}")
    return (out01 * span + lo).astype(np.float32)


def final_from_stack(stack: np.ndarray, tail_frac: float = 0.12,
                     contrast: str = "none", **clahe_kw) -> np.ndarray:
    """Build the dry 'final' frame from the tail of the stack.

    ``contrast="none"`` (default) is the plain mean of the tail window —
    byte-identical to ``stack[-tail:].mean(0)``, so every existing method is
    unaffected. ``contrast="clahe"`` enhances *each* tail frame before averaging,
    reproducing the expert's macro (which runs CLAHE over every frame). Only the
    spatial 'final' frame is enhanced; the raw stack handed to temporal detectors
    and to onset timing is never touched. ``clahe_kw`` are forwarded to
    ``apply_contrast`` (``clahe_blocksize``/``clahe_bins``/``clahe_maxslope``/``clahe_clip``)."""
    T = stack.shape[0]
    tail = stack[-max(int(round(T * tail_frac)), 1):]
    if contrast == "none":
        return tail.mean(0).astype(np.float32)
    acc = np.zeros(stack.shape[1:], np.float32)
    for frame in tail:
        acc += apply_contrast(frame, contrast, **clahe_kw)
    return (acc / len(tail)).astype(np.float32)


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
                     gamma: float = 1.0, clahe_clip: float | None = None,
                     clahe_blocksize: int = 127, clahe_bins: int = 256,
                     clahe_maxslope: float = 1.5,
                     bandpass: bool = False, bp_small_um: float = 2.5,
                     bp_large_um: float = 40.0,
                     um_per_analysis_px: float = 2.5) -> np.ndarray:
    """Apply contrast then (optionally) the band-pass. All defaults = identity."""
    out = apply_contrast(final, contrast, gamma=gamma, clahe_clip=clahe_clip,
                         clahe_blocksize=clahe_blocksize, clahe_bins=clahe_bins,
                         clahe_maxslope=clahe_maxslope)
    if bandpass:
        out = bandpass_dog(out, bp_small_um, bp_large_um, um_per_analysis_px)
    return out.astype(np.float32)
