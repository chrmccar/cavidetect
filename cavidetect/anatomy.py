"""Anatomy / background segmentation (v0.2.0 brief §4).

Separately label the structures that an embolism measurement must treat differently:

* **thallus**  — the plant body (the analysis region).
* **lumen**    — tubular rhizoid interiors (where real cavitation lives). High ridge.
* **scale**    — thin translucent sheets, usually rounded and at the tissue margin, that
                 whiten *gradually*. Low ridge, rounded, border-touching, large.
* **tape**     — non-biological background behind/around the thallus, often textured.
* **unknown**  — tissue that is none of the above.

Layers may overlap: a rhizoid may run *beneath* a scale or *over* tape, and those events
stay valid. The classes are evidence, not hard gates — :func:`refine_accepted` uses them to
drop tape-only / clearly-non-lumen pixels from an accepted mask while **keeping** lumen
pixels even where a scale overlies them.

This is a first, rough, single-final-frame segmentation (ridge + brightness + shape +
position). It is deliberately conservative; richer layered labels come from the annotation
tool (§5). Nothing here runs by default inside a measurement — call it explicitly.
"""
from __future__ import annotations
import numpy as np


def _local_std(img, size=5):
    from scipy.ndimage import uniform_filter
    img = img.astype(np.float32)
    mean = uniform_filter(img, size)
    sq = uniform_filter(img * img, size)
    return np.sqrt(np.maximum(sq - mean * mean, 0))


def segment_anatomy(final, tissue, ridge=None, lumen_ridge=0.04, scale_ridge=0.02,
                    scale_area=400, scale_ecc=0.85, border_px=10, bright_pct=60,
                    tape_texture_pct=70):
    """Return a dict of boolean masks: thallus, lumen, scale, tape, background, unknown.

    ``ridge`` is a normalised Frangi tubeness map (computed if omitted). Parameters mirror
    the ridge-hysteresis scale logic so behaviour is familiar.
    """
    from scipy.ndimage import label, distance_transform_edt
    from skimage.measure import regionprops
    final = np.asarray(final, np.float32)
    tissue = np.asarray(tissue, dtype=bool)
    if ridge is None:
        from .colour_shape import ridge_map
        ridge = ridge_map(final, tissue)

    thallus = tissue
    lumen = tissue & (ridge >= lumen_ridge)

    # Bright structures, separated from the tissue body by Otsu (a flat percentile swallows a
    # uniform body); fall back to a percentile if Otsu degenerates on near-uniform tissue.
    if tissue.any():
        from skimage.filters import threshold_otsu
        vals = final[tissue]
        try:
            air = float(threshold_otsu(vals))
        except Exception:
            air = float(np.percentile(vals, bright_pct))
    else:
        air = 0.0
    bright = tissue & (final >= air)
    # A scale is a bright region distinguished from a rhizoid lumen by SHAPE, not ridge:
    # LARGE, ROUNDED (low eccentricity) and touching the tissue margin. A rhizoid is thin and
    # elongated (high eccentricity) so it fails the roundedness gate and is kept out. (A bright
    # SHEET gives a strong Frangi response too, so ridge cannot separate the two.)
    edge_dist = distance_transform_edt(tissue)
    scale = np.zeros_like(tissue)
    lbl, _ = label(bright)
    for p in regionprops(lbl):
        if p.area < scale_area:
            continue
        coords = tuple(p.coords.T)
        touches_border = float(edge_dist[coords].min()) <= border_px
        if p.eccentricity <= scale_ecc and touches_border:   # rounded sheet at the margin
            scale[coords] = True

    # tape/background: everything outside the thallus. Textured => tape, flat => background.
    outside = ~tissue
    tex = _local_std(final)
    if outside.any():
        tthr = float(np.percentile(tex[outside], tape_texture_pct))
    else:
        tthr = np.inf
    tape = outside & (tex >= tthr)
    background = outside & ~tape

    unknown = tissue & ~lumen & ~scale
    return dict(thallus=thallus, lumen=lumen, scale=scale, tape=tape,
                background=background, unknown=unknown, ridge=ridge)


def refine_accepted(shape, masks):
    """Drop pixels that anatomy says are not cavitation, while keeping lumen-through-scale.

    Removes tape/background pixels from ``shape`` (events never live outside the thallus),
    and removes accepted pixels that fall on a scale sheet UNLESS they are also lumen
    (a rhizoid running beneath the scale). Inter-rhizoid drying handling (Class B) is left
    to the detector's temporal evidence; this only applies the spatial anatomy classes.
    """
    shape = np.asarray(shape, dtype=bool)
    keep = shape & masks["thallus"] & ~masks["tape"] & ~masks["background"]
    on_scale_not_lumen = masks["scale"] & ~masks["lumen"]
    keep = keep & ~on_scale_not_lumen
    return keep
