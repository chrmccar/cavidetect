"""Core per-pixel detection (§7.1): positive + abrupt + persistent steps.

Fixes from the M0 trial:
  * ABSOLUTE magnitude floor in grey levels (slow drying never clears it) + a sigma floor.
  * step-vs-ramp MODEL COMPARISON replaces the noisy two-window abruptness ratio:
    an event fits a single step far better than a straight line; slow drying does not.
All heavy work is vectorised (cumsum + advanced indexing); no per-frame Python loops.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class PixelDetections:
    accepted: np.ndarray    # bool [Y,X]
    step_frame: np.ndarray  # int  [Y,X] frame of the step (valid where accepted)
    step_dn: np.ndarray     # float[Y,X] step height in grey levels
    abrupt_ratio: np.ndarray# float[Y,X] RSS_step/RSS_linear (lower = more step-like)
    overshoot: np.ndarray   # bool [Y,X]
    left_censored: np.ndarray # bool [Y,X] step at/near frame 0 (onset before record)


def _windowed_step(s: np.ndarray, w: int) -> np.ndarray:
    """step(t,y,x) = mean(s[t:t+w]) - mean(s[t-w:t]); NaN where invalid. Vectorised via cumsum."""
    T = s.shape[0]
    c = np.concatenate([np.zeros((1,) + s.shape[1:], s.dtype), np.cumsum(s, axis=0)], axis=0)
    step = np.full_like(s, np.nan)
    t = np.arange(w, T - w)
    step[t] = (c[t + w] - c[t]) / w - (c[t] - c[t - w]) / w
    return step


def detect_pixels(s: np.ndarray, sigma_dn: float, *, w_long: int = 7,
                  step_min_dn: float = 10.0, step_k_sigma: float = 5.0,
                  rampfit_half: int = 10, abrupt_ratio_max: float = 0.5,
                  persist_frac: float = 0.5, persist_frames: int = 8) -> PixelDetections:
    T, Y, X = s.shape
    step = _windowed_step(s, w_long)
    sl = np.nan_to_num(step, nan=-1e9)
    best_t = np.argmax(sl, axis=0)
    best = np.take_along_axis(sl, best_t[None], 0)[0]           # max positive step per pixel (DN)

    # dual threshold: absolute grey-level floor AND k*sigma
    thr = max(step_min_dn, step_k_sigma * sigma_dn)
    cand = best > thr
    ys, xs = np.nonzero(cand)
    n = ys.size
    accepted = np.zeros((Y, X), bool)
    abrupt_ratio = np.full((Y, X), np.nan, np.float32)
    overshoot = np.zeros((Y, X), bool)
    left_censored = np.zeros((Y, X), bool)
    if n == 0:
        return PixelDetections(accepted, best_t, best.astype(np.float32),
                               abrupt_ratio, overshoot, left_censored)

    h = rampfit_half
    tc = best_t[ys, xs]
    # window indices clipped into [0,T)
    offs = np.arange(-h, h + 1)
    tidx = np.clip(tc[:, None] + offs[None, :], 0, T - 1)       # (n, L)
    win = s[tidx, ys[:, None], xs[:, None]]                      # (n, L) traces around step
    L = win.shape[1]
    centre = h                                                  # local index of the step

    # step model: two constant levels split at the step
    pre = win[:, :centre]; post = win[:, centre:]
    lvl_before = pre.mean(1, keepdims=True); lvl_after = post.mean(1, keepdims=True)
    rss_step = (((pre - lvl_before) ** 2).sum(1) + ((post - lvl_after) ** 2).sum(1))

    # linear model over the whole window (closed-form LS)
    xpos = offs.astype(np.float32)
    xm = xpos.mean(); xc = xpos - xm; denom = (xc ** 2).sum()
    ym = win.mean(1, keepdims=True)
    slope = ((win - ym) * xc[None, :]).sum(1) / denom
    inter = win.mean(1) - slope * xm
    pred = slope[:, None] * xpos[None, :] + inter[:, None]
    rss_lin = ((win - pred) ** 2).sum(1)

    ratio = rss_step / (rss_lin + 1e-6)
    stepheight = (lvl_after - lvl_before).ravel()
    persist_ok = stepheight >= persist_frac * best[ys, xs]      # post level stays elevated
    is_abrupt = (ratio < abrupt_ratio_max) & persist_ok
    over = post.max(1) > (lvl_after.ravel() + 0.5 * best[ys, xs])  # transient peak above plateau
    censored = tc <= (w_long + 1)

    accepted[ys, xs] = is_abrupt
    abrupt_ratio[ys, xs] = ratio.astype(np.float32)
    overshoot[ys, xs] = over & is_abrupt
    left_censored[ys, xs] = censored & is_abrupt
    return PixelDetections(accepted, best_t, best.astype(np.float32),
                           abrupt_ratio, overshoot, left_censored)
