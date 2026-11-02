"""Per-stack characterisation: noise sigma from the quiet tail, active-at-start, drift score."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class StackStats:
    noise_sigma_dn: float     # per-pixel per-frame noise (grey levels), robust, from quiet tail
    active_at_start: bool     # True if the stack is already changing at frame 0 (no quiet head)
    drift_px: float           # estimated rigid drift magnitude (px) early->late
    tail_start: int           # first frame index of the quiet tail


def _mad_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def characterise(stack: np.ndarray, tail_frac: float = 0.35) -> StackStats:
    T = stack.shape[0]
    tail_start = int(T * (1.0 - tail_frac))
    tail = stack[tail_start:]
    # noise: MAD of frame-to-frame differences in the quiet tail, /sqrt(2) (diff of two frames)
    d = np.diff(tail, axis=0)
    sigma = _mad_sigma(d) / np.sqrt(2.0)

    # active-at-start: compare early activity to tail activity
    head = stack[: max(4, T // 20)]
    head_act = _mad_sigma(np.diff(head, axis=0))
    tail_act = _mad_sigma(d)
    active_at_start = head_act > 3.0 * tail_act

    # drift: phase correlation between an early and a late frame (rigid translation magnitude)
    drift = _phase_drift(stack[min(2, T - 1)], stack[tail_start])
    return StackStats(noise_sigma_dn=float(max(sigma, 1e-3)),
                      active_at_start=bool(active_at_start),
                      drift_px=float(drift), tail_start=tail_start)


def _phase_drift(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from skimage.registration import phase_cross_correlation
        shift, _, _ = phase_cross_correlation(a, b, upsample_factor=10)
        return float(np.hypot(*shift))
    except Exception:
        return 0.0
