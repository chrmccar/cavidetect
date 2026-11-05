# CaviDetect v0.2.0 — staged roadmap

**Date:** 2026-11-05
**Status:** approved decomposition — each increment below gets its own spec → plan → build cycle
**Source brief:** `../../../CaviDetect_v0.2.0_compact_development_spec.md` (expert, 2026-11-05)
**Baseline:** tag `v0.2.0` (selectable detection methods: `colour-frangi` [default], `shape-filter`, `ridge-hysteresis`; 63 tests)

## Purpose

The expert returned a 7-item development brief plus an optional thallus-shrinkage
companion. It is too large for a single spec. This document decomposes it into **8
independently-shippable increments**, sequenced in the expert's stated priority order
(1→7, shrinkage last), with the dependency tensions made explicit. It is the parent
plan; it is **not** itself an implementation spec. Each increment is brainstormed into
its own `docs/superpowers/specs/` design, then a `docs/superpowers/plans/` plan, then
built on its own branch with its own tests and regression pins.

## Audit finding (2026-11-05)

A grep of the repo for `torch|sklearn|train|checkpoint|classifier|random_forest|logistic`
returns **nothing**. There is no ML, no training code, and no trained model used at
inference. The existing `annotated/` masks are single-frame hand paints produced by the
`save_annotator` HTML tool; they have never trained anything. This confirms the expert's
own suspicion in brief item 3 ("an annotation interface alone is not evidence of
training"). **Increment 3 therefore starts from zero**, not from an existing model.

The current method of record (`colour-shape` / `colour-frangi`) measures embolised tissue
on the **averaged final (dry) frame only**, end-anchored. Several increments below require
re-engaging the **full frame stack**; this is the single largest architectural shift in
the roadmap and is concentrated in increment 2.

## Design principles (apply to every increment)

1. **Additive, never a pivot.** `colour-frangi` and all current methods stay, each pinned
   by a byte-identical regression test. Every new capability is a new method, layer, or
   flag. A baseline-only mode is always reachable. (Expert: *"do not replace ridge
   hysteresis… keep baseline-only mode."*)
2. **Diagnostic/optional first.** A new feature ships as an inspectable diagnostic before
   it is ever allowed to change an accepted-event measurement.
3. **Split by plant / dry-down, never by crop or frame** for any evaluation or ML. There
   are only **3 biological plants** (49, 53, 60); the 5–6 files per folder are crops of
   one dry-down and are **not** independent. Leave-one-plant-out is the eval, n=3.
4. **Do not hard-exclude whole components** by area, width, or global orientation.
   Rhizoids may be isolated, crossed, curved, perpendicular, or in packed bundles.
5. **Benchmark against v0.2.0** on the independent manually-reviewed plants at each
   increment: event recall/precision, timing error, false positives on scales/tape,
   cumulative area, runtime, memory.

## The 8 increments

### 1 — Diagnostic export *(additive, no detector change)*
Per run, dump every intermediate stage as inspectable images: final composite, tissue
mask, brightness-only mask, Frangi response, seed mask, grown mask, pre/post shape mask,
pre/post scale mask, and the final accepted mask. Build a diagnostic montage comparing
**colour-only vs colour+shape**, focused on the dense left/central bundles where recall is
worst. Test thresholds **separately** rather than tightening everything at once, and run
at **full resolution** (not just binned previews). Also emit the **per-frame binary TIFF**
for the current method here, as a diagnostic (formalised in increment 7).
*Delivers:* visibility into where false negatives come from. *Blocks:* nothing. *Risk:* low.

### 2 — Temporal + spatial evidence layer *(the backbone)*
Re-engage the frame stack. Introduce a shared **event-candidate representation**: onset
frame, one- and two-frame positive image-difference features, a step-vs-gradual-drying
score, local noise, persistence after onset, and an event-confidence score. Enforce a
**maximum 2-frame optical transition** (not a maximum 2-frame bright state) — a fast
transition may stay bright and then vary slightly; do **not** reject on later
movement/brightness drift. Keep ridge-hysteresis candidate generation. Preserve
high-recall candidates for user review. Output one-frame and two-frame onset candidates
with explicit ambiguity. The initial frame is expected to be event-free.
*Delivers:* the candidate/onset/confidence data structure that 3, 6, and 7 consume.
*Depends on:* increment 1 diagnostics for validation. *Risk:* high (core re-architecture).
*Carries flag 2:* ship a **rough** tissue+tape+scale mask here to down-weight scale/tape,
then formalise the layered anatomy model at increment 4.

### 3 — Lightweight trainable model *(starts from zero — see audit)*
With only 3 labelled plants, use a **lightweight feature classifier** (e.g. logistic
regression / random forest over raw frame(s), 1-/2-frame differences, and optional ridge
maps) to distinguish lumen cavitation from inter-rhizoid drying, scale changes, and tape
artifacts. **Combine** ML probability with ridge/temporal evidence; keep a baseline-only
(no-ML) mode. Leave-one-plant-out evaluation.
*Depends on:* increment 2 features. *Carries flag 1:* bootstrap on the **existing
single-frame masks only**; the rich multi-frame labelling arrives in increment 5, which
makes the model *retrainable*. So increment 3 ships a *first* model, increment 5 makes it
improve over time. *Risk:* medium.

### 4 — Anatomy / background segmentation
Separately label, with permitted overlap/layers: thallus tissue, translucent scales,
rhizoid lumens, tape/background, and unknown/occluded. Exclude inter-rhizoid drying
(Class B) and tape from **accepted** cavitation measurements, but retain them as
diagnostics. Keep real lumen events **through** scales and **over** tape — never discard a
true rhizoid event just because the background is tape or a scale lies above it. Tissue and
tape masks must also support shrinkage measurement (increment 8).
*Formalises:* the rough masks from increment 2. *Serves:* increments 7 (denominators) and
8 (shrinkage). *Risk:* medium.

### 5 — Annotation UX rebuild
From single-frame paint to **multi-frame** annotation: frame slider, zoom, brush/erase,
label categories (lumen event / inter-rhizoid drying / scale / tape / unknown), and
per-action undo/redo. The "Annotate this dataset" view overlays **(a)** threshold/colour-
only, **(b)** colour+shape, **(c)** their union/difference, plus raw frames and frame
differences. Save editable masks with **frame ID, label, original-resolution coordinates,
and source version**. Existing one-frame annotations are **not** automatically treated as
frame-specific onset labels. Reviewed corrections join a **versioned training queue**;
retrain only on explicit request, validate first, allow model rollback.
*Enables:* retraining the increment-3 model on richer labels. *Risk:* medium (UI-heavy;
Tk/HTML tooling is Mac-only to exercise live).

### 6 — Confidence UI
A score/probability **threshold slider** with immediate overlay and area-curve updates
driven from **cached** predictions. Show accepted, rejected, and uncertain candidates
separately. Persist threshold and model version. Do **not** label an uncalibrated score a
statistical confidence interval.
*Depends on:* increments 2/3 (scores to threshold). *Risk:* low–medium.

### 7 — Outputs
Original-compatible CSV / area denominators, an **event-onset table**, a **per-frame binary
TIFF** of newly-detected pixels, an optional **cumulative binary TIFF**, a final
**event-time coloured overlay/LUT**, and metadata (calibration, 15-s frame interval or TIFF
timestamps, settings, model/checkpoint, thresholds). Prefer onset-frame marking of complete
event regions where supportable while preserving pixelwise timing and documenting
ambiguities. **No tape or inter-rhizoid drying in accepted event area.** No default image
registration (reconsider only if validated necessary).
*Depends on:* increment 2 onset data, increment 4 anatomy for denominators. *Formalises:*
the diagnostic TIFF from increment 1. *Risk:* low (well-defined, additive).

### 8 — Thallus-shrinkage companion *(optional; does not block cavitation release)*
A **separate module**, built only after the detector and shared segmentation (increment 4)
are stable. Preserve the existing `thallus_shrinkage_estimator.py` Lucas–Kanade
point-tracking baseline as an independent reference. Add optional projected thallus
area/perimeter + boundary displacement, plus robust tracks of stable **internal** thallus
features (including between rhizoids) for local in-plane strain. **Do not** track transient
cavitation highlights as landmarks. Detect/flag rolling, folds, and loss of visible tissue;
**projected area loss is not necessarily true tissue shrinkage** in a 2D image — flag
unusable frames rather than fabricating a corrected value. Export separate projected-area,
local-distance/strain, and rolling-quality curves with validity flags. Align cavitation and
shrinkage curves by original frame/time. **Recalibrate/validate** the
shrinkage→water-potential relation before substituting any metric; do **not** assume the
old calibration transfers.
*Depends on:* increment 4. *Lives in:* optional companion repo
`github.com/chrmccar/thallus_shrinkage`. *Risk:* medium; explicitly non-blocking.

## Dependency summary

```
1 (diagnostics) ──► 2 (temporal backbone) ──► 3 (ML, bootstrap)
                         │                      ▲
                         ├─ rough anatomy       │ retrainable
                         ▼                      │
                    4 (anatomy) ───────────────►5 (annotation UX)
                         │                      │
                         ├──────────► 7 (outputs)
                         │            ▲
                    2/3 ─┴──► 6 (confidence UI)
                         │
                         └──► 8 (shrinkage, optional)
```

Three places where strict 1→7 ordering is bent, each resolved by shipping a thin slice
early and formalising later (agreed 2026-11-05):

1. **ML before its labels/anatomy** — increment 3 bootstraps on existing single-frame
   masks; increment 5 makes it retrainable on richer labels.
2. **Anatomy needed before its slot** — a rough tissue+tape+scale mask ships inside
   increment 2; increment 4 formalises the layered model.
3. **Outputs depend on onset data** — the per-frame binary TIFF ships as a diagnostic in
   increment 1; the full output suite is formalised at increment 7.

## Definition of done (whole roadmap)

Benchmark against v0.2.0 on independent manually-reviewed plants: event recall/precision,
timing errors, false positives on scales/tape, cumulative area, runtime, memory. Verify
thin/hidden rhizoids, one-/two-frame transitions, persistent post-event brightness,
movement, rhizoids over tape, scale-end whitening, onset annotations, exact
per-frame/cumulative TIFF consistency, original CSV compatibility, and shrinkage rolling
flags. Ridge-hysteresis is never replaced; no compulsory heavyweight ML or shrinkage
component. Document decisions and limitations per increment.

## Next step

Brainstorm **increment 1 (diagnostic export)** into its own spec at
`docs/superpowers/specs/2026-11-05-cavidetect-increment-1-diagnostics-design.md`.
