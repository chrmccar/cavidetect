"""Synthetic stack generator for tests: injects elongated positive STEP events (the signal)
plus the confounds (slow drift, moving bright/dark pairs, round early blobs) and
signal-dependent noise. Returns (stack uint8-like float, ground_truth events)."""
from __future__ import annotations
import numpy as np


def make_synthetic(T=120, Y=120, X=120, seed=0):
    rng = np.random.default_rng(seed)
    base = np.full((Y, X), 60.0, np.float32)
    stack = np.repeat(base[None], T, axis=0)

    # 1) slow drift (gradual drying) over a broad region -> must NOT be detected
    yy, xx = np.mgrid[0:Y, 0:X]
    drift_region = (xx > X * 0.55).astype(np.float32)
    ramp = np.linspace(0, 18, T)[:, None, None]  # +18 DN over the whole run, gradual
    stack = stack + ramp * drift_region[None]

    # 2) elongated positive STEP events (the signal) -> must be detected
    gt = []
    events = [(25, 20, 30, 90, +28), (40, 70, 10, 55, +22), (35, 45, 80, 95, +34),
              (50, 95, 30, 70, +20), (30, 60, 60, 15, +26)]  # (frame, r, c, length, amp)
    for (f, r, c, length, amp) in events:
        ang = rng.uniform(0, np.pi)
        drawn = []
        for k in range(length):
            ry = int(r + k * np.sin(ang)); rx = int(c + k * np.cos(ang))
            if 1 <= ry < Y - 1 and 1 <= rx < X - 1:
                stack[f:, ry - 1:ry + 2, rx - 1:rx + 2] += amp  # persistent step, tube-width
                drawn.append((ry, rx))
        d = np.array(drawn)
        gt.append(dict(frame=f, r=r, c=c, cr=float(d[:, 0].mean()), cc=float(d[:, 1].mean())))

    # 3) moving bright/dark pairs (texture motion) -> must NOT be detected as events
    for _ in range(8):
        r = rng.integers(10, Y - 10); c = rng.integers(10, X - 30); f = rng.integers(45, 75)
        stack[f:f + 6, r - 1:r + 2, c - 1:c + 2] += 25     # bright
        stack[f:f + 6, r - 1:r + 2, c + 2:c + 5] -= 25     # dark neighbour (the motion signature)

    # 4) round early blobs (both signs, transient) -> must NOT be detected
    for _ in range(6):
        r = rng.integers(10, Y - 10); c = rng.integers(10, X - 10)
        rr = (yy - r) ** 2 + (xx - c) ** 2 < 25
        amp = rng.choice([+20, -20]); f = rng.integers(2, 14)
        stack[f:f + 7] += (amp * rr[None]).astype(np.float32)  # transient, comes and goes
        stack[f + 7:] += 0  # fades (no persistence)

    # 5) signal-dependent (Poisson-like) noise
    stack = stack + rng.normal(0, 1, stack.shape).astype(np.float32) * np.sqrt(np.maximum(stack, 1) / 60.0)
    stack = np.clip(stack, 0, 255)
    return stack.astype(np.float32), gt
