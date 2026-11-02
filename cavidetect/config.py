"""Run configuration. Every parameter is here, documented, and written into each output folder."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import yaml

@dataclass
class Config:
    # --- preprocess ---
    bin_xy: int = 1               # spatial binning factor (1 = full res on Mac; 2 used on the RAM-tight VM)
    spatial_sigma: float = 1.0    # Gaussian spatial smoothing (px, post-bin)
    temporal_smooth: int = 3      # moving-average window over time (frames)
    # --- detection (§7.1) ---
    w_long: int = 7               # before/after window for the step statistic (frames)
    step_min_dn: float = 10.0     # ABSOLUTE magnitude floor in grey levels (fixes M0 over-detection)
    step_k_sigma: float = 5.0     # also require step >= k * per-stack noise sigma
    rampfit_half: int = 10        # half-window (frames) for the step-vs-ramp model comparison
    abrupt_ratio_max: float = 0.5 # accept if RSS_step / RSS_linear < this (step clearly beats a ramp)
    persist_frac: float = 0.5     # post-step level must stay >= persist_frac * step above baseline
    persist_frames: int = 8       # ... for at least this many frames
    # --- grouping (§7.2) ---
    min_area_px: int = 4          # drop events smaller than this (post-bin)
    event_time_tol: int = 4       # pixels merge into one event only if step frames within +/- this
    # --- rhizoid tube map (§7.3) ---
    ridge_sigmas: tuple = (1, 1.5, 2, 2.5, 3)  # Sato ridge scales (px) for tube enhancement
    tube_percentile: float = 80.0  # keep ridges above this percentile of tubeness
    min_tube_px: int = 12          # drop tube blobs smaller than this
    min_seg_px: int = 4            # drop skeleton segments shorter than this
    tube_min_coverage: float = 0.15 # a tube counts as cavitated if >this frac of its pixels fired
    tube_min_det_px: int = 4       # ... and at least this many detected pixels
    # --- animation ---
    make_animation: bool = False   # write a GIF of embolism accumulating (slower)
    anim_stride: int = 3           # use every Nth frame in the GIF
    # --- primary embolism measurement (§ direct cumulative air-filling) ---
    emb_base_frames: int = 12      # wet-baseline window (first N frames)
    emb_tail_frac: float = 0.12    # dry end-state window (last fraction of frames)
    emb_change_min_dn: float = 10.0 # absolute grey-level floor for a sustained air-fill
    emb_k_sigma: float = 5.0       # also require change >= k * per-stack noise sigma
    emb_direction: str = "bright"  # air-filling appears as brightening ("bright"|"dark"|"abs")
    emb_tube_prior: bool = True    # keep only tube-like (linear) air-filling; drops surface/scale change
    emb_tube_grow: bool = True     # grow ridge seeds to full tube width (drop round blobs whole)
    emb_vessel_sigmas: tuple = (1, 2, 3, 4)  # ridge scales (px) for the tube-shape filter
    emb_vessel_min: float = 0.04   # min normalised tubeness to keep a detection
    emb_onset_mode: str = "end"    # "end" = last-water frame (ignores early transients) | "half"
    tissue_erode: int = 5          # erode tissue mask off the moving thallus rim (px)
    skip_events: bool = False      # True = skip the slow tube/event detection (fast test runs)
    make_viewer: bool = True       # write the interactive HTML scrubber (raw|overlay|curve)
    make_annotator: bool = True    # write the ground-truth annotation tool on the final frame
    # --- characterise ---
    tail_frac: float = 0.35       # fraction of the stack (from the end) treated as the quiet tail
    drift_px_thresh: float = 1.0  # register only if measured drift exceeds this (px)
    # --- io ---
    default_um_per_px: float = 2.5  # fallback only; real value read from metadata

    @classmethod
    def load(cls, path: str | None):
        if not path:
            return cls()
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        known = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
