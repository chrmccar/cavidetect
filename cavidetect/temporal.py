"""Temporal + spatial evidence layer (v0.2.0 brief §2).

The shipped ``colour-frangi`` method reads only the averaged final (dry) frame, so it
cannot tell an abrupt lumen cavitation from slow inter-rhizoid drying or scale whitening —
both end bright. This module re-engages the **frame stack** and scores each pixel on *how*
it got bright:

* **positive 1- and 2-frame differences** — cavitation is intrinsically fast (ms); in
  15-s sampling it appears as a step across one, occasionally two, consecutive frames.
* **step-vs-gradual score** — the largest 1-2 frame jump as a fraction of the pixel's
  total rise. ~1 for a clean step, small for a slow ramp (drying/scale).
* **persistence** — the pixel stays bright after onset. A fast transition may vary a
  little or the rhizoid may drift afterwards; persistence is measured generously (median
  of the post-onset level), so later movement/brightness drift does **not** reject it.
* **confidence** — jump magnitude x step-score x persistence, in [0, 1].

Key rule from the brief: permit a **maximum 2-frame optical transition**, not a maximum
2-frame bright *state*. The initial frame is expected to be event-free. Candidates are
kept at high recall for review; the confidence score (not a hard gate) ranks them.
"""
from __future__ import annotations
import numpy as np


def _smooth_t(stack, k):
    if k <= 1:
        return stack
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(stack, size=int(k), axis=0)


def temporal_evidence(stack, tissue, smooth=3, max_transition=2, rise_frac=0.5,
                      min_jump_frac=0.25, min_confidence=0.15, baseline_pct=20,
                      **_ignored):
    """Per-pixel temporal evidence for an abrupt, persistent brightening.

    Returns a dict with boolean ``shape`` (candidate embolised mask) and ``colour`` (any
    rise at all), plus float maps ``onset`` (frame, -1 if none), ``confidence``,
    ``step_score`` and int ``transition`` (1 or 2 frames; 0 where none). ``tissue`` gates
    everything. ``max_transition`` caps the jump window; ``min_jump_frac`` and
    ``min_confidence`` set the (deliberately permissive) candidate gate.
    """
    stack = np.asarray(stack, np.float32)
    T = stack.shape[0]
    sm = _smooth_t(stack, smooth)

    lo = np.percentile(sm, baseline_pct, axis=0)          # each pixel's own dark baseline
    hi = sm.max(0)                                         # its own bright level
    rng = np.maximum(hi - lo, 1e-6)

    # largest positive jump over a window of up to `max_transition` frames, and where it is
    best_jump = np.zeros(stack.shape[1:], np.float32)
    best_at = np.zeros(stack.shape[1:], np.int32)
    best_w = np.zeros(stack.shape[1:], np.int32)
    for w in range(1, int(max_transition) + 1):
        diff = sm[w:] - sm[:-w]                            # rise across w frames
        jmax = diff.max(0)
        jarg = diff.argmax(0) + w                          # onset = END of the transition
        upd = jmax > best_jump
        best_jump = np.where(upd, jmax, best_jump)
        best_at = np.where(upd, jarg, best_at).astype(np.int32)
        best_w = np.where(upd, w, best_w).astype(np.int32)

    jump_frac = best_jump / rng                            # step size relative to full rise
    step_score = np.clip(jump_frac, 0, 1)

    # onset = first frame the pixel passes rise_frac of its own rise AND is part of the jump
    thr = lo + rise_frac * rng
    above = sm >= thr[None]
    first = np.argmax(above, axis=0)
    has_rise = above.any(0)

    # persistence: generous — median post-onset level, normalised. Tolerates later drift.
    onset_i = np.clip(best_at, 0, T - 1)
    persistence = np.zeros(stack.shape[1:], np.float32)
    # vectorised-ish: for each pixel take mean of frames >= its onset
    idx = np.arange(T)[:, None, None]
    post = idx >= onset_i[None]
    post_sum = np.where(post, sm, 0).sum(0)
    post_cnt = np.maximum(post.sum(0), 1)
    post_mean = post_sum / post_cnt
    persistence = np.clip((post_mean - lo) / rng, 0, 1)

    confidence = np.clip(step_score * persistence, 0, 1)
    confidence = np.where(tissue & has_rise, confidence, 0.0)

    candidate = (tissue & has_rise & (jump_frac >= min_jump_frac)
                 & (confidence >= min_confidence))

    onset = np.where(candidate, np.minimum(first, onset_i), -1).astype(np.int32)
    transition = np.where(candidate, best_w, 0).astype(np.int32)

    colour = tissue & has_rise
    return dict(tissue=tissue, air=None, colour=colour, shape=candidate,
                ridge=None, onset=onset, confidence=confidence.astype(np.float32),
                step_score=step_score.astype(np.float32), transition=transition,
                stages={"brightness": colour, "step_score": step_score,
                        "confidence": confidence, "accepted": candidate})
