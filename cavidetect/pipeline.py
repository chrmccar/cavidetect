"""Orchestrate one stack end-to-end (tube-centric).
io -> characterise -> preprocess -> per-pixel detect -> rhizoid tube map -> aggregate to tubes -> outputs."""
from __future__ import annotations
import os
import pandas as pd
from . import io as cio, characterise as cch, preprocess as cpp, detect as cdet
from . import rhizoid as crh, tubes as ctub, outputs as cout, embolism as cemb
from .config import Config

__version__ = "0.8.0"


def run_stack(path: str, out_dir: str, cfg: Config | None = None,
              make_overlay: bool = True, make_animation: bool | None = None) -> dict:
    cfg = cfg or Config()
    if make_animation is None:
        make_animation = cfg.make_animation
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(os.path.normpath(path)))[0]

    stack, meta = cio.load(path, cfg.default_um_per_px)
    stats = cch.characterise(stack, tail_frac=cfg.tail_frac)
    s, bin_eff = cpp.preprocess(stack, bin_xy=cfg.bin_xy, spatial_sigma=cfg.spatial_sigma,
                                temporal_smooth=cfg.temporal_smooth)

    # --- PRIMARY: direct cumulative embolised-tissue measurement (optical-vulnerability style) ---
    tissue = cemb.tissue_mask(s, base_frames=cfg.emb_base_frames, erode=cfg.tissue_erode)
    emb = cemb.measure_embolism(
        s, stats.noise_sigma_dn, tissue, base_frames=cfg.emb_base_frames,
        tail_frac=cfg.emb_tail_frac, change_min_dn=cfg.emb_change_min_dn,
        k_sigma=cfg.emb_k_sigma, direction=cfg.emb_direction,
        tube_prior=cfg.emb_tube_prior, tube_grow=cfg.emb_tube_grow,
        vessel_sigmas=cfg.emb_vessel_sigmas,
        vessel_min=cfg.emb_vessel_min, onset_mode=cfg.emb_onset_mode,
        um_per_px=meta.um_per_px, bin_xy=bin_eff)
    cout.write_embolism_curve(emb, meta.n_frames, out_dir, stem)
    if make_overlay:
        cout.save_embolism_view(s, emb, out_dir, stem)
        if cfg.make_viewer:
            cout.save_embolism_viewer(s, emb, out_dir, stem)
        if cfg.make_annotator:
            # the RAW final frame itself — no mask, no processing layer
            tail_n = max(int(round(meta.n_frames * cfg.emb_tail_frac)), 1)
            raw_final = stack[-tail_n:].mean(0)
            cout.save_final_png(raw_final, out_dir, stem)
            cout.save_annotator(raw_final, out_dir, stem)

    if cfg.skip_events:
        manifest = dict(
            stem=stem, version=__version__, source=path, shape=meta.shape,
            um_per_px=meta.um_per_px, scale_from_metadata=meta.scale_from_metadata,
            noise_sigma_dn=stats.noise_sigma_dn, active_at_start=stats.active_at_start,
            drift_px=stats.drift_px, registered=False, bin_xy=cfg.bin_xy,
            ingest_notes=list(meta.notes), tissue_px=emb.tissue_px,
            frac_embolised_final=float(emb.frac_embolised[-1]),
            n_tubes=0, n_cavitated=0, n_events=0, skip_events=True,
            config={k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        )
        cout.write_manifest(out_dir, stem, manifest)
        return dict(meta=meta, stats=stats, embolism=emb, manifest=manifest)

    # per-pixel "when did this pixel brighten" signal
    det = cdet.detect_pixels(
        s, stats.noise_sigma_dn, w_long=cfg.w_long, step_min_dn=cfg.step_min_dn,
        step_k_sigma=cfg.step_k_sigma, rampfit_half=cfg.rampfit_half,
        abrupt_ratio_max=cfg.abrupt_ratio_max, persist_frac=cfg.persist_frac,
        persist_frames=cfg.persist_frames)
    # tubes: find rhizoid segments, assign each a single cavitation frame, mark whole tube
    seg_full, proj, tube_mask = crh.tube_segments(
        s, sigmas=cfg.ridge_sigmas, tube_percentile=cfg.tube_percentile,
        min_tube_px=cfg.min_tube_px, min_seg_px=cfg.min_seg_px)
    tdet, tube_df = ctub.aggregate_tubes(
        det, seg_full, meta.um_per_px, bin_eff,
        min_coverage=cfg.tube_min_coverage, min_det_px=cfg.tube_min_det_px, min_tube_px=cfg.min_tube_px)

    scale_um2 = (meta.um_per_px * bin_eff) ** 2
    events = tube_df[tube_df["cavitated"]].copy() if len(tube_df) else tube_df
    events = events.rename(columns={"cav_frame": "frame_peak"}) if len(events) else events

    cout.write_event_table(events, out_dir, stem)
    cout.write_tube_table(tube_df, out_dir, stem)
    path_curve, cum = cout.write_area_curve(tdet, events, meta.n_frames, scale_um2, out_dir, stem)
    if make_overlay:
        cout.save_temporal_view(s, tdet, cum, out_dir, stem)
        cout.save_tube_map(proj, seg_full, out_dir, stem)
    if make_animation:
        cout.save_animation(s, tdet, out_dir, stem, stride=cfg.anim_stride)

    n_cav = int(tube_df["cavitated"].sum()) if len(tube_df) else 0
    manifest = dict(
        stem=stem, version=__version__, source=path, shape=meta.shape,
        um_per_px=meta.um_per_px, scale_from_metadata=meta.scale_from_metadata,
        noise_sigma_dn=stats.noise_sigma_dn, active_at_start=stats.active_at_start,
        drift_px=stats.drift_px, registered=False, bin_xy=cfg.bin_xy,
        ingest_notes=list(meta.notes), tissue_px=emb.tissue_px,
        frac_embolised_final=float(emb.frac_embolised[-1]),
        n_tubes=int(len(tube_df)), n_cavitated=n_cav, n_events=int(len(events)),
        config={k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
    )
    cout.write_manifest(out_dir, stem, manifest)
    return dict(meta=meta, stats=stats, tubes=tube_df, events=events, seg=seg_full,
                tdet=tdet, embolism=emb, manifest=manifest)
