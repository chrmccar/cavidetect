# Design spec — automated detection of cavitation events in liverwort rhizoid time-lapses

**Date:** 2026-10-30
**Status:** draft for review
**Supersedes framing in:** `HANDOFF_for_claude_code.md` (still the authoritative description of the biology; this spec refines the engineering after inspecting all 31 stacks and running an M0 feasibility trial).

---

## 1. Context — why this, and what changed after looking at the data

Chris images liverwort **rhizoids** under a microscope while the plant **dries down**. As a rhizoid cavitates, its water column breaks, air enters, and the rhizoid **suddenly brightens and stays bright**. We want a tool that finds these events automatically — **where** (pixels/region), **when** (frame), **how big** (area, length) — and separates them from the several non-cavitation changes that also brighten the image.

The handoff laid out the biology and a sensible staged plan. Before committing to a build, we (a) inspected the metadata and pixel dynamics of **all 31 stacks**, and (b) ran a throwaway **M0 trial** of the core idea on one crop from each plant. Both changed how the pipeline should be built. The key empirical conclusions drive this spec and are recorded in §3 and Appendix A.

The outcome we want: a **publication-ready, deterministic, rerunnable Python tool** that other labs can install and point at new stacks, producing event tables, embolised-area curves, per-rhizoid outcomes and QC overlays — with every parameter documented and defensible in a methods section.

## 2. Goals and non-goals

**Goals**
- Per-stack detection of cavitation events, output as four readouts (§5): cumulative embolised-area curve (per class), discrete event list, per-rhizoid outcome, and spatial maps/QC overlays.
- Classify each event into **two classes** (§4): air-in-lumen vs inter-rhizoid drainage.
- Transparent, physics-based, deterministic pipeline with pinned dependencies, a versioned config, and a clear rerun path. Batch over many stacks (more data is coming).
- Honest evaluation against hand labels, respecting the real independence structure (3 plants).

**Non-goals (now)**
- **No deep learning.** True independent n = 3 plants (§3); it cannot be trained or validated honestly. A reference tool (Mandelli et al. 2026) needed 65 leaves of one species and still reached only IoU 0.38 at pixel level.
- **No water-potential / mass linking yet.** Chris records both and will join them later. The event table must be designed so a frame→timestamp→Ψ join is trivial, but we do not build it now.
- No real-time/online processing. Detection is offline; post-event frames are used to confirm events.

## 3. Data reality (evidence-based — this is what makes the problem specific)

31 stacks, ~1.2 GB, all **8-bit uint8**, stored by ImageJ as z-slices (treat **axis 0 = time**). Three plants: **49, 53, 60**. **The multiple files inside each plant's folder are different crops of ONE physical drydown** (confirmed by Chris) — so they are **not independent**.

| | Plant 49 | Plant 53 | Plant 60 |
|---|---|---|---|
| Crops | 6 | 5 | full field + positions `60_2/`, `60_4/`, `60_5/` |
| Frames | 375–376 | 377–387 | 400 |
| Pixel size (µm/px) | 2.5 | 2.5–3.1 | 3.5 (and 60_4/60_5 differ from 60.tif) |
| Noise σ (quiet tail) | 1.5 | 3.0 | 3.0 |
| Event timing | window ~25–86, median f35 | mostly early + sparse late tail, median f20 | peaks at **f0–6** |
| Frame-mean over run | rises 62→74 | ~flat | ~flat |
| Dominant character | discrete abrupt events + overshoot | whole-network gradual brightening | starts mid-action, heavy movement |

**Consequences that shape the design:**

1. **The three plants behave qualitatively differently.** No fixed threshold/window/model works across them. Everything (noise σ, baseline, thresholds, activity window) must be **estimated per stack** from its own quiet tail.
2. **Discrimination must rest on persistence + sign + shape, NOT timing.** In plant 60 the confounds (early blobs, movement) coincide *in time* with events, so frame-pair/timing methods are weakest exactly where the data is hardest.
3. **No guaranteed quiet "before" baseline.** The quiet *tail* is reliable everywhere (good noise reference), but plant 60 is already active at frame 0 → first events are **left-censored**; a clean pre-event window cannot be assumed.
4. **Global frame statistics are useless.** Frame-mean is flat in 53/60 because brightening is local and redistributive (bundles brighten while background darkens). All detection is local.
5. **n = 3.** Train/test splits and any tuning must be **leave-one-plant-out**; crops from one plant may never straddle the split; overlapping crops within a plant risk double-counting in per-plant aggregates.
6. **Scale varies; depth is 8-bit; saturation is negligible.** Always read pixel size from metadata. No 12/16-bit originals appear to exist (the large top-level files are the same 8-bit full field, byte-identical to a position file). A handful of frames touch 255 (<0.001% pixels) — note but not a problem.
7. **Registration is sample-dependent.** Drift <0.3 px in 49 (skip); substantial movement in 60 (needed). Must be a per-stack *conditional* step gated on a measured score.

## 4. Event definition — two classes, assigned by location

Per Chris: detect **both** and **label them separately**, decide later which matters.

- **Class A — lumen fill (true embolism):** air entering a single rhizoid tube. Bright, persistent change **inside** a rhizoid.
- **Class B — inter-rhizoid drainage:** water receding from capillary spaces **between** rhizoids. Also bright and persistent.

Both brighten and both persist, so they cannot be separated temporally. The **only** reliable discriminator is **location relative to a rhizoid map** (§7.3): inside a tube → A; in interstitial space → B. This makes the rhizoid map **load-bearing** (three of four outputs depend on it). The pipeline must still emit a useful **unclassified** event table when the rhizoid map is weak, with a per-event `class_confidence`.

## 5. Outputs (all four, per stack, plus per-plant aggregates)

1. **Event table** (CSV/Parquet): `event_id, class {A,B,unknown}, class_confidence, frame_start, frame_peak, area_px, area_um2, length_um, centroid_rc, orientation, step_height_sigma, step_height_dn, persistence_frames, abruptness, overshoot_flag, left_censored_flag, crop_id, plant_id`.
2. **Cumulative embolised-area curve** per class, in **frame units** (x = frame), with columns ready to left-join a `frame,timestamp,Ψ` table later. Also normalised (fraction of detectable rhizoid area).
3. **Per-rhizoid outcome table**: `rhizoid_id, embolised {0,1}, frame_embolised, class, crop_id, plant_id` (depends on rhizoid instance map).
4. **Spatial maps & QC**: event-time colour map, class-overlay image, and a QC overlay movie (raw + detections) for fast human spot-checking. Plus a **run manifest** (config snapshot + package version/git hash + per-stack decisions: σ, registration on/off, active-at-start flag).

## 6. Architecture — modular pipeline

A Python package (working name `cavidetect`) + CLI:
`cavidetect run STACK.tif --config params.yaml --out results/` and `cavidetect batch FOLDER/ ...`.
Each module is independently testable with a single clear responsibility.

1. **`io`** — read TIFF; resolve ImageJ z-as-t explicitly; read pixel size from metadata (never hardcode); emit a `StackMeta` record. Raw is opened read-only; nothing is ever written back to source.
2. **`characterise`** — estimate noise σ from the quiet tail (MAD of temporal diffs); flag **active-at-start**; compute a drift/movement score (phase correlation between early frames).
3. **`register`** *(conditional)* — `pystackreg` rigid registration **only if** the movement score exceeds a threshold; logged per stack.
4. **`preprocess`** — optional spatial binning (default off on Mac at full res; the VM trial used 2× because noise is correlated at 1 px); light spatial Gaussian; **variance-stabilising transform** (Anscombe) so one threshold works across brightness (noise is signal-dependent in every stack).
5. **`detect`** — per-pixel **change-point step-vs-ramp** detection (the core; §7.1).
6. **`group`** — assemble per-pixel detections into events in (x, y, t) with tight event-time coherence and bundle-splitting (§7.2).
7. **`rhizoid_map`** — ridge filter → tube skeleton + width, optional instance segmentation (§7.3).
8. **`classify`** — confound rejection + two-class assignment + interpretable features (§7.4).
9. **`outputs`** — the four readouts (§5) + run manifest.
10. **`evaluate`** *(separate entry point)* — label ingestion, matching, metrics, OSOV baseline (§8).

Seam for later ML: the per-event feature table from module 8 is exactly the input a scikit-learn high-recall filter would consume (Stage 3). Not built now; the column contract is fixed so it can be added without refactor.

## 7. Core algorithms (with the M0 lessons baked in)

M0 proved the signal is separable (clean spike vs ramp at pixel level) and that the rhizoid structure extracts cleanly, but the naive detector over-detected (~50% of pixels) for two diagnosed reasons: a σ-threshold made meaningless by over-smoothing, and a too-noisy two-window "abruptness" ratio. The fixes below are the whole point of M1.

### 7.1 Per-pixel detection (`detect`)
For each pixel's VST time series:
- **Change-point model** (`ruptures`, PELT with an L2 cost, penalty set from the per-stack noise σ) to find segment boundaries. An event = a single changepoint separating two near-constant segments with a **positive** jump; slow drying = many small changepoints or a poor piecewise-constant fit. This replaces the noisy abruptness ratio.
- **Dual threshold (fixes M0 over-detection):** accept a step only if it clears **both** an absolute magnitude floor (≈10–12 DN in raw units, matching the handoff proof-of-concept) **and** k·σ in VST units. The magnitude floor is what slow drying fails to clear.
- **Persistence:** the post-step segment must stay elevated ≥ N frames (handles overshoot via a step-plus-decay tolerance — the decay must not drop back to baseline).
- **Sign:** positive only.
- Output per pixel: `step_frame, step_dn, step_sigma, persistence, abruptness, overshoot_flag`.
- Efficiency: windowed means via cumsum (no per-frame Python loops — this mattered: naive loops were ~10× slower and the VM swapped). `ruptures` runs per active-pixel only (pre-masked by the dual threshold on a fast windowed-step pass) to keep it tractable.

### 7.2 Grouping into events (`group`)
- Build a spatiotemporal mask from accepted pixels; **3D connected components / hysteresis** in (x, y, t) so a multi-frame event (2–4 frames) stays one object.
- **Split merged bundles** (M0 fused whole bundles into 5000-px blobs): watershed seeded on local step maxima, and require tight event-time coherence (pixels within ±k frames) so two events on the same tube at different times don't merge.
- Drop objects below a min area; flag `left_censored` when `step_frame ≤ w` (onset before the record began — important for plant 60).

### 7.3 Rhizoid map (`rhizoid_map`)
- **Frangi/Sato ridge filter** (scikit-image, CPU, deterministic) on a temporal max/mean projection → tubeness map → skeleton + local width. Defines inside-tube vs interstitial masks.
- Optional **instance** segmentation of individual rhizoids (Cellpose, can use the Mac GPU/MPS) to enable per-rhizoid outputs; gated behind a flag, with the ridge map as the no-dependency fallback.

### 7.4 Reject + classify (`classify`)
- **Reject movement:** discard candidates with an adjacent strong **negative** step in the same time window (bright/dark pair = texture motion). This is the primary confound rejector for plant 60, where timing can't help.
- **Reject early blobs:** round, low-eccentricity, two-signed candidates.
- **Classify A vs B by location** vs the rhizoid map; `class_confidence` from distance-to-ridge and shape agreement.
- **Features per event:** area (px, µm²), length (µm), eccentricity, major/minor ratio, orientation, step σ and DN, persistence, abruptness, distance-to-ridge, negative-companion strength.

## 8. Evaluation & baseline (`evaluate`)
- **Labels:** napari (Points/Shapes) → versioned CSV/GeoJSON (`plant, crop, frame_start, frame_end, class, polygon/point`). Target 2–3 crops spanning the plants; ideally one crop double-labelled to measure the inter-observer ceiling.
- **Matching:** detected ↔ labelled within ±k frames and spatial overlap (IoU or centroid distance); report **precision/recall/F1 per class**.
- **Curve agreement:** automatic vs manual cumulative-area curve.
- **Splits:** **leave-one-plant-out** only; never split crops of one plant across train/test.
- **Baseline to beat:** a Python re-implementation of the **OSOV** frame-difference + threshold workflow (so reviewers see the familiar number the new method improves on).

## 9. Reproducibility & engineering
- Packaging: `pyproject.toml` + lockfile; core deps `tifffile, numpy, scipy, scikit-image, pandas`; optional `ruptures, scikit-learn, napari, pystackreg, cellpose`.
- Every run writes its **resolved config** + package version/git hash into the output folder.
- **Raw data is never written.** Read-only opens; all derived data to the output dir.
- **Determinism:** fixed seeds; no nondeterministic parallel reductions in the scored path.
- **Tests:** a synthetic stack generator (injected elongated positive steps + moving-texture bright/dark pairs + round early blobs + slow drift + signal-dependent noise) → unit tests assert the detector keeps events and rejects each confound, independent of real data.
- **Methods text** maintained alongside the code, one paragraph per module/parameter.
- **Performance:** full resolution on the Mac (M3/24 GB); binning optional. The VM swaps on full-res stacks and is for small reviews only.

## 10. Risk register
| # | Risk | Mitigation |
|---|---|---|
| R1 | Class A vs B genuinely ambiguous | Location-based assignment + `class_confidence`; emit unclassified table regardless |
| R2 | No quiet baseline (plant 60 active at f0); onset censored | Robust early-quantile baseline; `active_at_start` + `left_censored` flags; don't assume a pre-window |
| R3 | n≈3 → overfit/underpowered eval | Physics core; leave-one-plant-out; classifier (if ever) only as high-recall filter |
| R4 | Slow drying brightens everything (M0 over-detection) | Dual threshold (absolute DN floor + σ); change-point step-vs-ramp |
| R5 | Overshoot breaks single-step model | Step-plus-decay tolerance; never count decay as a negative event |
| R6 | Movement coincident with events (plant 60) | Bright/dark-pair rejection + shape + persistence; conditional registration |
| R7 | Per-stack noise & scale heterogeneity | Per-stack σ; Anscombe VST; pixel size from metadata |
| R8 | Merged-bundle blobs (M0) | Watershed split + event-time coherence |
| R9 | Overlapping crops double-count per-plant area | Track crop overlap; dedupe in per-plant aggregates (needs crop geometry — open Q) |
| R10 | Raw-data safety on shared mount | Read-only opens; copy to working dir; never write source |

## 11. Build milestones
- **M1 — core pipeline (Mac, full res):** modules 1–6 + event table + cumulative-area curve + QC overlays. Change-point detector with the dual threshold; validate over-detection is fixed on all 3 plants. *Exit:* event counts and overlays look right by eye on 49/53/60.
- **M2 — classes & per-rhizoid:** module 7 (ridge map, optional Cellpose) + module 8 (reject/classify) + per-rhizoid outputs + class-split curves.
- **M3 — evaluation & baseline:** napari labelling workflow + OSOV baseline + `evaluate` (LOPO, per-class P/R/F1, curve agreement) + synthetic-stack unit tests.
- **M4 — packaging & docs:** CLI/batch, config, methods text, README, pinned env, Mac packaging; optional classifier seam.

Each milestone gets its own implementation plan (via the writing-plans skill) before coding.

## 12. Open questions (non-blocking; fold in as answered)
1. **Frame interval** — constant across stacks and known? (Needed only for cross-stack rate comparisons; outputs stay in frame units meanwhile.)
2. **Crop geometry** — do crops within a plant overlap, and can we get their offsets? (For R9 double-count handling.)
3. **Labelling capacity** — still TBD; sets whether M3 includes a trainable filter. Physics core stands alone regardless.
4. **12/16-bit originals** — metadata says 8-bit only; confirm none exist.
5. **Class priority** — if forced to tune for one class's recall, which? (A, lumen fill, assumed primary unless told otherwise.)

## Appendix A — M0 trial findings (throwaway code, 2026-10-30)
Ran a per-pixel abrupt/gradual detector on 2×-binned crops of plants 49, 53, 60.
- **Validated:** event vs gradual pixels are clearly separable at the trace level (sharp spike incl. overshoot vs slow ramp) in both abrupt (49) and gradual-dominated (53) plants; the max-step map cleanly traces the rhizoid network.
- **Failed as expected:** naive detector kept ~50% of pixels. Causes: (a) σ-threshold meaningless after smoothing — fixed by an absolute DN floor; (b) two-window abruptness ratio too noisy — replaced by change-point step-vs-ramp; (c) grouping merged whole bundles — fixed by watershed + event-time coherence.
- **Engineering:** VM swap-thrashes on full-res stacks (a bare sqrt took 42 s); binning + intermediate frees made it finish. Confirms development/iteration belongs on the Mac.

## Appendix B — Selectable detection methods (2026-11-03)
The spatial-detection stage is now pluggable (`cavidetect/methods.py`). A `Method(name, label,
detect, schema)` is registered in a REGISTRY; `detect(final, tissue, **params) -> dict` returns at
least `shape` and `colour` masks. The `schema` (a list of `ParamSpec`) is the single source of
truth that drives both the GUI settings panel and the CLI `--params` validation (`resolve_params`
coerces/clamps/defaults and drops unknown keys). See the design doc at
`docs/superpowers/specs/2026-11-03-selectable-detection-methods-design.md`.

**To add an experimental method:** write a `detect(final, tissue, **params)` function (reuse
`shape_filters.filter_regions` / `preprocess_final` as needed), define its `ParamSpec` schema
(group each param General/Preprocess/Threshold/"Shape rules"), and `register(Method(...))`. No GUI
or CLI edits are needed — the dropdown and the settings panel pick it up from `list_methods()`.

Shipped methods: `colour-frangi` (default, = the calibrated current method), `shape-filter`
(BioVoxxel inscribed-circle/thinness/elongation on a thresholded final frame), `ridge-hysteresis`
(exposes `detect_spatial_v2`). Shape rules are entered in µm; `um_per_px` (default 2.5, a fallback —
set to the real optical calibration) times `bin` gives µm per analysis pixel.
