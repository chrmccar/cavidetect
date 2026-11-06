# CaviDetect v0.4.0 — Hybrid ridge + temporal detector (design)

Date: 2026-11-06. Implements the expert's **CaviDetect v0.3.1 "Hybrid ridge-hysteresis and
temporal-evidence detector"** brief. Additive: both legacy methods are left byte-identical
and remain selectable for benchmarking.

## Objective

A selectable `hybrid-ridge-temporal` method that fuses the two complementary detectors so
genuine cavitation in pegged-rhizoid lumens is recovered (ridge-hysteresis's strength) while
scale / inter-rhizoid drying / tape false positives are suppressed (temporal-evidence's
weakness). Plus a diagnostic `ridge-temporal-intersection` baseline (the naive AND).

## Why fuse evidence, not the two binary masks

- Ridge-hysteresis reads only the dry composite → cannot tell abrupt cavitation from gradual
  drying; scale/size gates discard real rhizoids; no temporal confidence.
- Temporal-evidence has no anatomical requirement → lights up scales/drying; its true
  positives are splotchy rhizoid *fragments*.
- Intersecting the final masks keeps only what both already agree on → high precision, poor
  recall (kept as the diagnostic baseline). The hybrid must beat it.

## Pipeline (maps to the brief's Steps 1–7)

1. **Shared preprocessing** — `colour_shape.measure` already builds the dry `final`, the
   `tissue` mask and the stack once; the hybrid is called with all three. The smoothed stack
   is built once as float32 and freed before the ridge/fusion stage (16 GB constraint).
2. **Temporal evidence** (`hybrid._temporal_maps`) — mirrors `temporal.temporal_evidence`
   (left untouched) and adds the two corrections the brief asks for:
   - **noise-aware magnitude**: a per-pixel temporal noise (MAD→σ of 1-frame increments) and
     an **absolute** jump floor tied to the scene brightness span, so a relative jump
     fraction cannot be ~1 from noise on an already-bright pixel whose own range is tiny.
   - **start-of-jump onset**: onset = the frame the strongest 1–2-frame jump *begins*, not
     an earlier gradual crossing (corrects the premature-onset behaviour).
3. **Ridge evidence** (`colour_shape.ridge_evidence`, split out of `detect_spatial_v2`) —
   continuous Frangi ridge score + seed / grow / hysteresis-tube / edge-distance maps, none
   discarded by rigid thresholds before fusion.
4. **Fuse** — every accepted pixel must clear a credible optical transition (noise-aware SNR
   **and** absolute DN magnitude **and** a real step fraction of its own rise). On top of
   that core: ridge-guided growth completes a temporally-confirmed tube to full extent, but
   only across pixels that themselves show ≥ a weak transition (never fills a drying scale);
   and a strong abrupt event with weak ridge still qualifies (dense-bundle recovery). A
   strong ridge with no transition never qualifies.
5. **Artifact exclusion** (`anatomy.segment_anatomy` + `refine_accepted`) — drop gradual
   scale / tape / background, but keep a rhizoid lumen that runs *beneath* a scale. The
   removed pixels are exported as the artifact map; nothing is silently dropped.
6. **Onset** — start-of-jump per accepted pixel; each pixel contributes to cumulative area
   once (bincount on onset, shared with every method).
7. **Confidence** — one interpretable 0–1 score = weighted temporal(step×SNR) + ridge +
   persistence. Explicitly **not** a calibrated probability. The existing §6 confidence
   viewer lets the user re-threshold after processing without recomputing the stack.

## Outputs (brief "Outputs" section — all already wired through `run()`)

per-frame + cumulative binary TIFF, event-time LUT, cumulative-area CSV, onset table,
run metadata, **ridge-response / temporal / combined-confidence / artifact maps**
(`outputs.save_evidence_maps`), confidence viewer, diagnostic stage montage. All keep
image/frame correspondence.

## Preserved / unchanged

`temporal.temporal_evidence`, `colour_shape.detect_spatial`, `detect_spatial_v2` (now calls
`ridge_evidence` but returns the same mask — regression-pinned), `colour-frangi`,
`shape-filter`. CLI/GUI pick the new methods up automatically from the registry + schema.

## Validation

`tests/test_hybrid.py` — synthetic stack with a known abrupt tube (must recover, onset at
the jump), a gradual border scale (must suppress), a static-bright ridge with no transition
(must reject), an abrupt non-ridge blob (must keep — bundle recovery), intersection ⊆ both
legacy masks, bounded float32 confidence, registry + param resolution, and a guard that the
`ridge_evidence` refactor left `detect_spatial_v2` intact. Real held-out-plant benchmarking
(ridge vs temporal vs intersection vs hybrid: recall / FP area / precision / onset accuracy /
RAM) is a **Mac** task on the `annotated/` expert masks — the VM cannot hold a full-res stack.

## Deferred (brief's Phase 3)

ML scale/tape discrimination beyond the current shape-based anatomy — only "if beneficial";
the hybrid prototype ships first, as the brief directs.
