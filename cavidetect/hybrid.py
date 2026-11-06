"""Hybrid ridge-hysteresis + temporal-evidence detector (CaviDetect v0.3.1 expert brief).

The two shipped detectors are complementary and each fails in a way the other does not:

* **ridge-hysteresis** (``colour_shape.detect_spatial_v2``) finds bright, elongated, tubular
  structure and grows whole rhizoids from strong seeds, but it reads only the final (dry)
  composite — it cannot tell an abrupt cavitation from gradual drying, and its scale/size
  gates can discard genuine rhizoids. When it *does* fire it captures most of a rhizoid.
* **temporal-evidence** (``temporal.temporal_evidence``) finds rapid positive brightness
  steps over 1-2 frames anywhere in the stack — recovering dense bundles the ridge misses —
  but has no anatomical requirement, so it lights up scales/inter-rhizoid drying and its
  true positives are splotchy fragments of rhizoids, not whole tubes.

This module **fuses the continuous evidence** (not the two binary masks) so that:

* nothing is accepted without a credible, noise-aware *optical transition* (a strong ridge
  that never rapidly brightens is NOT cavitation);
* a strong temporal event with weak ridge still qualifies when it sits in plausible tissue
  (bundle recovery);
* ridge connectivity is used to **grow a temporally-confirmed seed along its whole tube**,
  but only into pixels that themselves show a transition (so drying scales are not filled);
* onset is taken from the **start of the strongest abrupt jump**, correcting the
  premature-onset behaviour of the standalone temporal method;
* a gradual scale / tape / background exclusion map is produced, while genuine rhizoid
  lumens that run *beneath* a scale remain eligible.

A single interpretable ``confidence`` map (0..1, explicitly *not* a calibrated probability)
is returned so the user can re-threshold after processing via the confidence viewer.

**Memory:** the smoothed stack is built once as float32 and freed before the ridge/fusion
stage; no operation promotes the full stack to float64 (the 16 GB-laptop constraint).
``temporal.py`` and ``colour_shape.detect_spatial_v2`` are left untouched for benchmarking;
this module re-uses ``colour_shape.ridge_evidence`` and mirrors the temporal scoring (adding
the noise-aware magnitude and start-of-jump onset the brief asks for).
"""
from __future__ import annotations
import numpy as np


def _temporal_maps(stack, tissue, smooth=3, max_transition=2, baseline_pct=20):
    """Per-pixel temporal evidence from the frame stack, as continuous maps.

    Mirrors ``temporal.temporal_evidence`` but (a) adds a per-pixel temporal **noise**
    estimate and an absolute signal-to-noise ratio, and (b) records the **start** of the
    strongest jump (not its end) for onset. Returns float32 maps; the smoothed stack is
    freed before returning so the caller's peak stays near one stack copy.
    """
    from scipy.ndimage import uniform_filter1d
    stack = np.asarray(stack, np.float32)
    T = stack.shape[0]
    sm = stack if smooth <= 1 else uniform_filter1d(stack, size=int(smooth), axis=0).astype(np.float32)

    lo = np.percentile(sm, baseline_pct, axis=0).astype(np.float32)   # each pixel's own dark baseline
    hi = sm.max(0)                                                     # its own bright level
    rng = np.maximum(hi - lo, np.float32(1e-6))

    # per-pixel temporal noise: robust sigma of 1-frame increments (MAD -> sigma).
    d1 = np.diff(sm, axis=0)
    noise = (1.4826 * np.median(np.abs(d1), axis=0)).astype(np.float32)
    noise = np.maximum(noise, np.float32(1e-3))
    del d1

    # largest positive jump over a window of up to `max_transition` frames, and WHERE it is.
    best_jump = np.zeros(stack.shape[1:], np.float32)
    best_end = np.zeros(stack.shape[1:], np.int32)      # frame the jump completes on
    best_w = np.zeros(stack.shape[1:], np.int32)        # width (frames) of that jump
    for w in range(1, int(max_transition) + 1):
        diff = sm[w:] - sm[:-w]
        jmax = diff.max(0).astype(np.float32)
        jarg = (diff.argmax(0) + w).astype(np.int32)
        upd = jmax > best_jump
        best_jump = np.where(upd, jmax, best_jump).astype(np.float32)
        best_end = np.where(upd, jarg, best_end).astype(np.int32)
        best_w = np.where(upd, w, best_w).astype(np.int32)
        del diff

    jump_frac = (best_jump / rng).astype(np.float32)    # step size vs the pixel's own full rise
    step_score = np.clip(jump_frac, 0, 1).astype(np.float32)
    snr = (best_jump / noise).astype(np.float32)        # NOISE-AWARE magnitude (absolute)

    has_rise = (hi - lo) > 0

    # persistence: generous median post-onset level, normalised; tolerant of later drift.
    onset_end = np.clip(best_end, 0, T - 1)
    idx = np.arange(T)[:, None, None]
    post = idx >= onset_end[None]
    post_mean = (np.where(post, sm, 0).sum(0) / np.maximum(post.sum(0), 1)).astype(np.float32)
    persistence = np.clip((post_mean - lo) / rng, 0, 1).astype(np.float32)
    del sm, post, idx

    onset_start = np.clip(best_end - best_w, 0, T - 1).astype(np.int32)   # START of the jump
    return dict(step_score=step_score, snr=snr, jump_frac=jump_frac,
                best_jump=best_jump, rise=(hi - lo).astype(np.float32),
                persistence=persistence, onset_start=onset_start,
                transition=best_w.astype(np.int32), has_rise=has_rise, noise=noise)


def hybrid_detect(final, tissue, stack=None,
                  # --- temporal ---
                  smooth=3, max_transition=2, baseline_pct=20,
                  min_jump_frac=0.20, snr_min=4.0, weak_snr_min=2.0, abs_jump_frac=0.15,
                  # --- ridge (anatomy) ---
                  bright_hi_pct=70, bright_lo_pct=45, seed_ridge=0.08, grow_ridge=0.02,
                  scale_area=600, scale_ecc=0.85, border_px=10,
                  # --- fusion / acceptance ---
                  w_temporal=0.5, w_ridge=0.3, w_persist=0.2,
                  min_confidence=0.15, use_anatomy=True,
                  um_per_px=2.5, **_ignored):
    """Fuse temporal + ridge evidence into an accepted mask, onset and confidence.

    ``stack`` (T,Y,X) is required; ``final`` is the dry composite and ``tissue`` the analysis
    region (both supplied by ``colour_shape.measure``). Returns the standard detector dict
    (``tissue``, ``colour``, ``shape``, ``ridge``, ``onset``, ``confidence``, ``transition``,
    ``stages``) plus an ``evidence`` dict of maps to export (temporal / ridge / combined /
    artifact). See the module docstring for the fusion rationale.
    """
    from scipy.ndimage import label
    from skimage.measure import regionprops
    from . import colour_shape as cs
    if stack is None:
        raise ValueError("hybrid-ridge-temporal needs the frame stack")

    # -- Step 2: temporal evidence (freed internally) --
    tm = _temporal_maps(stack, tissue, smooth=smooth, max_transition=max_transition,
                        baseline_pct=baseline_pct)
    step_score, snr, persistence = tm["step_score"], tm["snr"], tm["persistence"]
    has_rise, onset_start, transition = tm["has_rise"], tm["onset_start"], tm["transition"]
    best_jump, rise = tm["best_jump"], tm["rise"]

    # Absolute-magnitude floor (the brief's noise-aware criterion): a relative jump fraction
    # is ~1 for noise on an already-bright pixel whose own range is tiny, so acceptance must
    # also require a brightness change that is a real fraction of the scene's dynamic range.
    # This is what rejects a static bright ridge (small own-range) and low-amplitude flicker.
    if tissue.any():
        glob = float(np.percentile(final[tissue], 99) - np.percentile(final[tissue], 5))
    else:
        glob = 0.0
    abs_floor = float(abs_jump_frac) * max(glob, 1e-6)
    big_jump = best_jump >= abs_floor          # the transition itself is large in absolute DN
    big_rise = rise >= abs_floor               # the pixel genuinely goes dark -> bright

    # -- Step 3: ridge (anatomical) evidence, continuous scores retained --
    rev = cs.ridge_evidence(final, tissue, bright_hi_pct=bright_hi_pct,
                            bright_lo_pct=bright_lo_pct, seed_ridge=seed_ridge,
                            grow_ridge=grow_ridge)
    vn, tubes, edge_dist = rev["ridge"], rev["tubes"], rev["edge_dist"]
    ridge_term = np.clip(vn / max(seed_ridge, 1e-6), 0, 1).astype(np.float32)

    # -- Step 4: fuse --
    # credible optical transition: a noise-aware SNR, an absolute DN magnitude, AND a step
    # that is a real fraction of the pixel's own rise. This is the floor for EVERY accepted
    # pixel (a strong ridge that never rapidly brightens cannot pass).
    transition_ok = (tissue & has_rise & big_jump & big_rise
                     & (snr >= snr_min) & (step_score >= min_jump_frac))
    # a weaker transition, used only to validate ridge-guided growth (never on its own). Still
    # requires the absolute floor so growth cannot leak into static-bright / drying tissue.
    weak_transition = (tissue & has_rise & big_rise & (best_jump >= 0.5 * abs_floor)
                       & (snr >= weak_snr_min) & (step_score >= 0.5 * min_jump_frac))

    # ridge-guided growth: a tube component that CONTAINS a confirmed transition pixel is
    # grown to its full extent, but only across pixels that themselves show >= a weak
    # transition (never fill a gradually-drying scale just because it is connected).
    lblt, _ = label(tubes)
    touched = set(int(x) for x in np.unique(lblt[transition_ok])) - {0}
    grown_tube = np.isin(lblt, list(touched)) if touched else np.zeros_like(tissue)
    grown = grown_tube & weak_transition

    candidate = transition_ok | grown            # strong-temporal pixels always qualify

    # -- Step 7: interpretable confidence (NOT a calibrated probability) --
    wsum = max(w_temporal + w_ridge + w_persist, 1e-6)
    snr_term = np.clip(snr / max(snr_min, 1e-6), 0, 1).astype(np.float32)
    temporal_term = (0.5 * step_score + 0.5 * snr_term).astype(np.float32)
    confidence = ((w_temporal * temporal_term + w_ridge * ridge_term
                   + w_persist * persistence) / wsum).astype(np.float32)
    confidence = np.where(tissue & has_rise, confidence, 0.0).astype(np.float32)

    accepted = candidate & (confidence >= min_confidence)

    # -- Step 5: artifact exclusion (gradual scale / tape / background) --
    artifact = np.zeros_like(tissue)
    if use_anatomy:
        from . import anatomy
        masks = anatomy.segment_anatomy(final, tissue, ridge=vn, scale_area=scale_area,
                                        scale_ecc=scale_ecc, border_px=border_px)
        refined = anatomy.refine_accepted(accepted, masks)
        artifact = accepted & ~refined           # what anatomy removed (scale-not-lumen / tape)
        accepted = refined

    # -- Step 6: onset = start of the strongest abrupt jump, for accepted pixels --
    onset = np.where(accepted, onset_start, -1).astype(np.int32)
    transition = np.where(accepted, transition, 0).astype(np.int32)

    colour = tissue & has_rise                   # "any rise at all" (the broad COLOUR panel)
    stages = {"brightness": colour, "frangi": vn, "seed": rev["seed"],
              "scale": artifact, "accepted": accepted}
    evidence = {"ridge_response": vn, "temporal_confidence": temporal_term * (tissue & has_rise),
                "combined_confidence": confidence, "artifact_excluded": artifact}
    return dict(tissue=tissue, air=rev["air_hi"], colour=colour, shape=accepted,
                ridge=vn, onset=onset, confidence=confidence, transition=transition,
                step_score=step_score, stages=stages, evidence=evidence)


def intersection_detect(final, tissue, stack=None, smooth=3, max_transition=2,
                        baseline_pct=20, min_jump_frac=0.25, min_confidence=0.15,
                        rise_frac=0.5, bright_hi_pct=70, bright_lo_pct=45,
                        seed_ridge=0.08, grow_ridge=0.02, min_area=8, scale_area=600,
                        scale_ecc=0.85, border_px=10, um_per_px=2.5, **_ignored):
    """Diagnostic: the simple BINARY INTERSECTION of the two legacy final masks.

    This is Phase-1 / the validation baseline from the brief ("Simple binary intersection
    as a diagnostic"): accept only pixels that *both* the standalone temporal method and the
    standalone ridge-hysteresis method accept. High precision, poor recall — kept selectable
    so the hybrid can be benchmarked against the naive combination it is meant to beat. It
    deliberately calls the unchanged legacy functions so it tracks them exactly.
    """
    from . import colour_shape as cs
    from .temporal import temporal_evidence
    if stack is None:
        raise ValueError("ridge-temporal-intersection needs the frame stack")
    te = temporal_evidence(stack, tissue, smooth=smooth, max_transition=max_transition,
                           rise_frac=rise_frac, min_jump_frac=min_jump_frac,
                           min_confidence=min_confidence, baseline_pct=baseline_pct)
    rh = cs.detect_spatial_v2(final, tissue=tissue, bright_hi_pct=bright_hi_pct,
                              bright_lo_pct=bright_lo_pct, seed_ridge=seed_ridge,
                              grow_ridge=grow_ridge, min_area=min_area, scale_area=scale_area,
                              scale_ecc=scale_ecc, border_px=border_px)
    shape = te["shape"] & rh["shape"]
    onset = np.where(shape, te["onset"], -1).astype(np.int32)
    colour = te["colour"] | rh["colour"]
    confidence = np.where(shape, te["confidence"], 0.0).astype(np.float32)
    stages = {"brightness": colour, "frangi": rh["ridge"], "accepted": shape}
    return dict(tissue=tissue, air=rh["air"], colour=colour, shape=shape, ridge=rh["ridge"],
                onset=onset, confidence=confidence, transition=te.get("transition"),
                stages=stages)
