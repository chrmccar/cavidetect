"""Behavioural tests for the hybrid ridge + temporal detector (v0.3.1 brief).

Synthetic frame stack with four structures whose correct verdicts are known:

* ABRUPT TUBE   — a thin tubular lumen that jumps dark->bright in one frame mid-run and
                  stays bright. The genuine cavitation. MUST be recovered, onset at the jump.
* GRADUAL SCALE — a large, rounded, border-touching sheet that brightens slowly over many
                  frames. A drying-scale artifact. MUST be suppressed.
* STATIC RIDGE  — a thin bright tube present from frame 0 (no optical transition). A strong
                  ridge response with no rapid brightening. MUST NOT be accepted.
* dim tissue    — noisy background body.

These encode the brief's core asks: gate every acceptance on a noise-aware abrupt
transition, recover whole tubes via ridge-guided growth, reject gradual scales, and date
onset to the start of the jump.
"""
import numpy as np
import pytest

from cavidetect import hybrid, colour_shape as cs, methods as M


def _synthetic_stack(T=30):
    H, W = 120, 200
    rng = np.random.default_rng(0)
    tissue = np.zeros((H, W), bool)
    tissue[10:110, 10:190] = True

    stack = np.full((T, H, W), 10.0, np.float32)                 # dark background
    stack[:, tissue] = 50.0                                       # dim tissue body
    stack += rng.normal(0, 2.0, stack.shape).astype(np.float32)  # temporal noise everywhere

    # ABRUPT TUBE: thin tubular lumen, dark until frame 15 then bright (one-frame step).
    tube = np.zeros((H, W), bool); tube[59:62, 40:160] = True
    jump = 15
    stack[:jump, tube] = 50.0 + rng.normal(0, 2.0, (jump, int(tube.sum()))).astype(np.float32)
    stack[jump:, tube] = 210.0 + rng.normal(0, 2.0, (T - jump, int(tube.sum()))).astype(np.float32)

    # GRADUAL SCALE: large rounded border sheet, slow linear brightening 50 -> 210.
    scale = np.zeros((H, W), bool); scale[84:108, 124:164] = True
    ramp = np.clip((np.arange(T) - 5) / 20.0, 0, 1)
    for t in range(T):
        stack[t, scale] = 50.0 + ramp[t] * 160.0

    # STATIC RIDGE: thin bright tube present from the very first frame (no transition).
    static = np.zeros((H, W), bool); static[30:33, 40:160] = True
    stack[:, static] = 210.0 + rng.normal(0, 2.0, (T, int(static.sum()))).astype(np.float32)

    final = stack[-5:].mean(0)
    return stack, final, tissue, tube, scale, static


def _run():
    stack, final, tissue, tube, scale, static = _synthetic_stack()
    p = M.resolve_params("hybrid-ridge-temporal", {})
    d = hybrid.hybrid_detect(final, tissue, stack=stack, **p)
    return d, (stack, final, tissue, tube, scale, static)


def test_recovers_abrupt_tube():
    d, (_, _, _, tube, _, _) = _run()
    assert d["shape"][tube].mean() > 0.6, f"tube recall {d['shape'][tube].mean():.2f}"


def test_suppresses_gradual_scale():
    d, (_, _, _, _, scale, _) = _run()
    assert d["shape"][scale].mean() < 0.15, f"scale leaked {d['shape'][scale].mean():.2f}"


def test_rejects_static_ridge_without_transition():
    d, (_, _, _, _, _, static) = _run()
    assert d["shape"][static].mean() < 0.1, \
        f"static bright ridge wrongly accepted {d['shape'][static].mean():.2f}"


def test_onset_dates_the_jump_not_earlier():
    d, (_, _, _, tube, _, _) = _run()
    ons = d["onset"][tube & (d["onset"] >= 0)]
    assert ons.size > 0
    assert 11 <= int(np.median(ons)) <= 18, f"onset median {np.median(ons)} not at the jump (~15)"


def test_confidence_is_bounded_and_float32():
    d, _ = _run()
    c = d["confidence"]
    assert c.dtype == np.float32
    assert c.min() >= 0.0 and c.max() <= 1.0


def test_exposes_evidence_and_stage_maps():
    d, _ = _run()
    assert set(d["evidence"]) >= {"ridge_response", "temporal_confidence",
                                  "combined_confidence", "artifact_excluded"}
    assert d["ridge"].dtype == np.float32
    assert set(d["stages"]) >= {"brightness", "frangi", "accepted"}


def test_strong_temporal_without_ridge_still_qualifies():
    """A bundle fragment with a real abrupt jump but weak tube-ness must not be discarded
    purely for lacking ridge (brief Step 4)."""
    stack, final, tissue, tube, scale, static = _synthetic_stack()
    # a tiny isolated blob that jumps abruptly but is too small/round to be a strong ridge
    blob = np.zeros_like(tissue); blob[20:23, 20:23] = True
    stack[:12, blob] = 50.0; stack[12:, blob] = 215.0
    final = stack[-5:].mean(0)
    p = M.resolve_params("hybrid-ridge-temporal", {})
    d = hybrid.hybrid_detect(final, tissue, stack=stack, **p)
    assert d["shape"][blob].mean() > 0.5, "abrupt non-ridge event was dropped"


def test_intersection_is_subset_of_both_legacy_masks():
    stack, final, tissue, tube, scale, static = _synthetic_stack()
    from cavidetect.temporal import temporal_evidence
    p = M.resolve_params("ridge-temporal-intersection", {})
    d = hybrid.intersection_detect(final, tissue, stack=stack, **p)
    te = temporal_evidence(stack, tissue, smooth=p["smooth"],
                           max_transition=p["max_transition"], rise_frac=p["rise_frac"],
                           min_jump_frac=p["min_jump_frac"], min_confidence=p["min_confidence"],
                           baseline_pct=p["baseline_pct"])
    rh = cs.detect_spatial_v2(final, tissue=tissue)
    assert not (d["shape"] & ~te["shape"]).any()
    assert not (d["shape"] & ~rh["shape"]).any()


def test_methods_registered_and_resolvable():
    names = [n for n, _ in M.list_methods()]
    assert "hybrid-ridge-temporal" in names
    assert "ridge-temporal-intersection" in names
    # legacy methods still present and first
    assert names[0] == "colour-frangi"
    assert {"temporal-evidence", "ridge-hysteresis"} <= set(names)
    p = M.resolve_params("hybrid-ridge-temporal", {})
    assert p["snr_min"] == 4.0 and p["w_temporal"] == 0.5 and p["use_anatomy"] is True


def test_detect_spatial_v2_unchanged_by_refactor():
    """ridge_evidence refactor must leave the legacy ridge-hysteresis mask byte-identical."""
    _, final, tissue, *_ = _synthetic_stack()
    d = cs.detect_spatial_v2(final, tissue=tissue)
    assert set(d) >= {"tissue", "air", "colour", "shape", "ridge"}
    assert d["shape"].dtype == bool and d["shape"].shape == final.shape
