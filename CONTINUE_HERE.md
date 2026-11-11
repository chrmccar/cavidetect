# CONTINUE HERE — cavitation detection

Pick-up note for the next session. Read this first, then `DESIGN_SPEC.md`.

## 2026-11-11 — v0.6.0: INSPECTOR OVERLAY (Phase 1, view-only) — read this box first

New **spatial + temporal review overlay** Chris asked for (the OSOV / cavicam look — Vimeo
215769356, stills `11nov-1/2/3.png` at project root). Design:
`docs/superpowers/specs/2026-11-11-inspector-design.md`. Fully additive; all 4 legacy
detectors + `colour-frangi` byte-identical.

**What shipped (Phase 1):** `cavidetect/inspector.py` + `<stem>_cs_inspect.html`, written by
`colour_shape.run()` beside the viewer, opened from a new **🔍 Inspect** button in the GUI
(`gui.py`: `artefact_paths` gained `"inspect"`, button + `_open_inspector`, enable logic).
- Three **vertically-stacked, synced** panels (shared scrubber + ▶ play + wheel-zoom/drag-pan):
  **raw** / **difference** (change between shown frames, 128=no change) / **conduits** on black.
- Conduits colour toggles **by time** (onset frame, blue→red ramp) and **by tube** (each
  rhizoid a distinct hue via `inspector.label_tubes` = skeleton→split-at-junctions→grow). A
  **static** toggle shows the all-time map; otherwise conduits reveal at their onset frame.
- Build reuses the memory-safe `_grey_u8` per-frame pattern for raw/diff JPEGs; the onset map
  and tube-id map are embedded as **lossless RGBA PNGs** (value=R+256·G, alpha=valid) and
  coloured **live in JS**, so mode/static/animation are instant and nothing is baked server-side.

**Tests:** `tests/test_inspector.py` (+10) — label_tubes (line→1, crossing→split, empty→zeros,
tiny→component fallback, 2 tubes→2 labels), PNG u16 round-trip, prepare_frames counts,
save_inspector content + empty-mask survival. **123 pass on the VM** (was 113). **Rendered
headless in Firefox** (geckodriver) — both colour modes + difference + controls verified working,
no JS errors. (That browser check normally can't run on the VM; it worked this time.)

**Mac follow-ups / revision inputs:** eyeball Inspect on real `data/cavi*` full-res; set real
µm/px + frame interval. **Question list for the collaborator:**
`docs/INSPECTOR_QUESTIONS_FOR_COLLABORATOR.md` (difference-panel definition, tube =
strand-vs-segment, LUT, calibration, what corrections they do most) — feeds the next revision.

**Phase 2 (next):** editing toolkit on the conduits canvas (add-trace→snap-to-ridge→onset from
stack; erase; split; merge; relabel) → **Commit** exports edits JSON → `cavidetect inspect-apply`
recomputes onset/area-curve/per-tube tables → regenerates + reopens, area curve updating live.
Auto run preserved; corrected run saved alongside. ("view by default, commit on demand" +
"full editing toolkit", Chris's choices 2026-11-11.)

**SHIPPED: committed to main + bumped to 0.6.0 + tagged `v0.6.0` + pushed** — CI builds the
macOS/Windows apps and publishes the Release (verify
github.com/chrmccar/cavidetect/releases/tag/v0.6.0).

---

## 2026-11-10 — EXPERT ImageJ WORKFLOW FOLDED IN (CLAHE-faithful `shape-filter`)

The expert described how they actually use the tool: (1) an ImageJ macro that runs
**CLAHE** ("Enhance Local Contrast") along the rhizoids — block size 127, 256 bins, max
slope 1.5, the `fast_(less_accurate)` variant — before detection; (2) at the end they
filter out detected objects smaller than ~0.5 px (i.e. keep almost everything). Their macro
is saved at the project root: `../CLAHE.ijm.ijm`.

**What changed (all additive; `colour-frangi` stays byte-identical, regression-pinned):**
- **`preprocess_final.py`:** CLAHE now takes ImageJ-semantic knobs — `clahe_blocksize`
  (→ skimage `kernel_size`), `clahe_bins` (→ `nbins`), `clahe_maxslope` (→ `clip_limit` via
  the new `imagej_clahe_clip(maxslope, bins) = maxslope/bins`), with `clahe_clip` as an
  optional explicit override. The expert runs ImageJ's `fast_(less_accurate)` CLAHE, which is
  exactly the tiled+interpolated algorithm `skimage.exposure.equalize_adapthist` implements,
  so this is a faithful analogue (not pixel-identical at tile seams). New
  `final_from_stack(stack, tail_frac, contrast, **clahe_kw)` builds the dry final frame —
  plain mean (default, byte-identical) **or** per-frame-CLAHE-then-average (the macro enhances
  every frame).
- **`shape-filter` method = the faithful ImageJ flow.** Defaults now reproduce the macro:
  `contrast=clahe`, blocksize 127 / bins 256 / slope 1.5, `clahe_scope=per-frame`, and
  `min_area_px=1` (≈ "keep ≥0.5 px"). New params auto-appear in the CLI `--params` schema and
  the GUI settings panel (registry-driven). Collaborator picks `--method shape-filter`.
- **`colour_shape.measure`** owns the stack→final reduction: when the method asks for
  per-frame CLAHE it CLAHEs each tail frame then averages and suppresses the method's own
  contrast step (CLAHE applied exactly once). The **raw stack** (onset timing +
  temporal/hybrid detectors) is never CLAHE'd — per-frame CLAHE is non-linear and would
  corrupt the absolute DN jumps those detectors key on. `clahe_scope=final` enhances only the
  averaged final frame.

**Tests:** +7 (TDD) in `tests/test_preprocess_final.py` (clip mapping, local-contrast gain,
slope override, `final_from_stack` plain-mean vs per-frame) and `tests/test_methods.py`
(macro defaults, measure per-frame-scope plumbing). **113 pass on the VM** (was 106).
End-to-end `run()` smoke with `shape-filter --params clahe_scope=per-frame` detects the tube
and writes the full output set.

**SHIPPED: committed + merged to main + tagged `v0.5.0` + pushed** (commit e826039; CI builds
the Release on the tag push — verify at github.com/chrmccar/cavidetect/releases/tag/v0.5.0).
**Mac-only follow-ups:** run `shape-filter` on real full-res
`data/cavi*` and eyeball vs the expert `annotated/` masks; confirm per-frame CLAHE recall on
the dense bundles; set the real µm/px. Decision log: user chose faithful-ImageJ reproduction,
selectable CLAHE scope (default per-frame), lower min-size floor, and **keep colour-frangi
untouched**.

---

## 2026-11-06 — v0.4.0: HYBRID RIDGE + TEMPORAL DETECTOR (read this box first)

Implements the expert's **CaviDetect v0.3.1 "Hybrid ridge-hysteresis and temporal-evidence
detector"** brief. Fully additive — both legacy methods are byte-identical and still
selectable. Design: `docs/superpowers/specs/2026-11-06-hybrid-ridge-temporal-design.md`.

**Two new selectable methods** (registry → appear automatically in CLI `--method` and the GUI
dropdown; no CLI/GUI code change needed):
- **`hybrid-ridge-temporal`** (`cavidetect/hybrid.py:hybrid_detect`) — fuses the *continuous*
  temporal + ridge evidence (not the two binary masks). Every accepted pixel must clear a
  credible optical transition (noise-aware **SNR** + **absolute** DN magnitude tied to the
  scene span + a real step fraction of its own rise), so a static bright ridge with no rapid
  brightening is rejected and noise on already-bright pixels can't score. Ridge-guided growth
  completes a temporally-confirmed tube to full extent but only across pixels that also show
  a transition (never fills a drying scale); a strong abrupt event with weak ridge still
  qualifies (dense-bundle recovery). Gradual scale/tape excluded via `anatomy` but
  lumen-under-scale kept; the removed pixels are exported as an artifact map. Onset = **start
  of the strongest jump** (fixes the premature-onset behaviour). One interpretable 0–1
  `confidence` (NOT calibrated) → the §6 confidence viewer re-thresholds after the run.
- **`ridge-temporal-intersection`** (`hybrid.py:intersection_detect`) — diagnostic baseline:
  the naive AND of the two *legacy* final masks (the thing the hybrid must beat).

**Refactor (byte-identical):** `colour_shape.ridge_evidence` was split out of
`detect_spatial_v2` so the hybrid can reuse the ridge seed/grow/tube maps; v2's mask is
unchanged and pinned by a test.

**New outputs:** `outputs.save_evidence_maps` writes `<stem>_cs_ridge_response.png`,
`_temporal_confidence.png`, `_combined_confidence.png`, `_artifact_excluded.png` whenever the
method yields an `evidence` dict (hybrid only). Everything else (binary TIFFs, event LUT,
onset table, run metadata, confidence viewer, stage montage) is reused as-is.

**Tests:** `tests/test_hybrid.py` (10) — recovers the abrupt tube, suppresses the gradual
scale, rejects the static non-transition ridge, keeps an abrupt non-ridge blob, onset at the
jump, intersection ⊆ both legacy masks, bounded float32 confidence, registry/params, and a
guard that the `ridge_evidence` refactor left v2 intact. **106 tests pass on the VM** (was 96).

**STILL TO DO on the Mac (VM can't — no full-res stack, no browser):**
1. Run all four detectors (`ridge-hysteresis`, `temporal-evidence`, `ridge-temporal-intersection`,
   `hybrid-ridge-temporal`) on the real `data/cavi*` full-res and compare against the expert
   `annotated/` masks — **recall / FP area / precision / onset accuracy / dense-bundle /
   under-scale / RAM**, using **held-out plants** (n=3, LOPO) not crops of one plant. This is
   the brief's acceptance test ("demonstrated against manually reviewed data, not cleaner-looking
   masks"). The cached eval harness from v0.1.3 (scratchpad) is the starting point.
2. Eyeball the hybrid evidence maps + confidence viewer in a browser; tune `snr_min`,
   `abs_jump_frac`, `min_confidence`, fusion weights on the real bundles the expert flagged.
3. Set the collaborator's REAL µm/px (still the 2.5 fallback).
4. Smoke-tested end-to-end on the VM at `--bin 4` on `cavi490001` (full `run()` path: evidence
   maps + confidence viewer + TIFFs) — see the v0.4.0 commit. Full-res timing/RAM is a Mac check.

Run it: `cavidetect colour-shape ../data/cavi490001 --out results_whole/hybrid --bin 1 --method hybrid-ridge-temporal`

---

## ⭐ NEXT SESSION / TOMORROW (2026-11-06) — start here

State at end of 2026-11-05: shipped **v0.3.0** (expert's 7-item v0.2.0 brief, 8 increments)
and **v0.3.1** (viewer 14.6 GiB OOM fix). `main` == `origin/main`, working tree clean,
tags v0.3.0 + v0.3.1 on origin, CI green, both Releases published w/ macOS+Windows zips.
96 tests pass on the VM. Everything below is **Mac-only** (VM has no Tk/browser display and
can't hold a full-res stack — only 7.4 GB, and it was load-avg ~9 all session so tests ran slow).

Do these in order:
1. **Confirm the v0.3.1 OOM fix on REAL full-res data.** On the Mac run the viewer export on a
   400-frame full-res stack (`cavidetect colour-shape ../data/cavi490001 --out /tmp/x --bin 1`),
   watch peak RAM. Expect it to complete now (was 14.6 GiB hard-fail). If still tight, the peak is
   the DETECTION stage holding the full float32 stack (~7.3 GiB) + onset/temporal bool arrays — see
   the v0.3.1 box below. Next step for ≤5 GB = pixel-chunked streaming of `onset_start_of_rise` /
   `temporal_evidence` (fidelity-sensitive → assert byte-identical onset/mask vs current before
   shipping). I did NOT do this blind; needs the real data.
2. **Eyeball the new UIs in a browser** (can't on VM): the diagnostic stage montage
   (`*_cs_stages.png`), the confidence viewer (`*_cs_confidence.html`, temporal method only) and the
   multi-frame annotator (`*_cs_multiframe_annotate.html`). Check the method dropdown in the GUI now
   lists **temporal-evidence**.
3. **Run each method on real `data/cavi*` full-res** and compare recall vs v0.2.0, esp. the dense
   left/central bundles the expert flagged. `temporal-evidence` is the recall play (uses the frame
   stack, not just the final frame).
4. **Set the collaborator's REAL µm/px** (still the 2.5 fallback) — shape rules are only
   meaningful once it's right.
5. **Train the §3 ML model LOPO** on the `annotated/` expert masks (sklearn is in `extra`; VM has it
   installed via --break-system-packages, Mac needs `pip install -e .[extra]`). Compare to baseline.
   NB there was NO ML before v0.3.0 — this is from scratch.
6. **§8 shrinkage companion** — deferred; separate repo `thallus_shrinkage`. Only after the detector
   + shared segmentation are stable.
7. Housekeeping (optional): local merged branches `docs/v0.2.0-roadmap`, `feature/v0.2.0-build`,
   `fix/viewer-memory` can be deleted (all merged to main); nothing on them is missing from origin.

Roadmap doc: `docs/superpowers/specs/2026-11-05-cavidetect-v0.2.0-roadmap.md`.


## 2026-11-05 — v0.3.1: VIEWER MEMORY FIX (read first)

Expert hit `numpy ArrayMemoryError: 14.6 GiB (400,1920,2560) float64` in
`save_embolism_viewer` on a 16 GB Windows laptop (`CaviDetect_memory_optimization_v2.md`).
**Root cause:** `g = np.clip((s - lo)/rng, 0, 1)` — `s` is float32 but `lo` (from
`np.percentile`) is float64, so the subtraction promoted the WHOLE stack to float64 (14.6 GiB),
and it built all 400 frames though only ~90 timepoints are rendered.
**Fix (byte-identical output, pins added):**
- `outputs.save_embolism_viewer` / `save_animation`: normalise ONE frame at a time to uint8
  (`_grey_u8(t)`), never the full stack. Per-frame temp ~tens of MB vs 14.6 GiB.
- `colour_shape.load_frames_binned`: preallocate + fill in place instead of
  `np.stack([list of all frames])` (was ~2x peak during load).
- `colour_shape.onset_start_of_rise`: `del sm` (the full smoothed copy) before allocating the
  bool arrays.
Tests: `tests/test_viewer_memory.py` (per-frame grey == old `g[t]`; structural guard that
`np.clip` never receives the 3-D stack). All pass.
**Remaining memory floor (NOT a bug):** the current algorithm holds the full float32 stack
(~7.3 GiB at full res) and `onset`/`temporal` add bool/smoothed arrays on top — the peak is now
the detection stage, not the viewer. Getting to ≤5 GB needs pixel-chunked streaming of
onset/temporal (and/or holding the stack as uint8 with per-op float casts); that is
fidelity-sensitive and must be validated on real full-res data on the Mac — recommended next,
not done blind. Could not reproduce full-res on the VM (7.4 GB total).

## 2026-11-05 — v0.3.0: EXPERT v0.2.0 BRIEF IMPLEMENTED (read this box first)

The expert returned a 7-item development brief (`../CaviDetect_v0.2.0_compact_development_spec.md`).
I decomposed it into 8 increments (roadmap: `docs/superpowers/specs/2026-11-05-cavidetect-v0.2.0-roadmap.md`)
and built them on branch `feature/v0.2.0-build`, tagged **`v0.3.0`**. All headless/algorithmic
work is unit-tested on the VM; the HTML UIs need a desktop browser to validate (VM has no display).

**Audit finding:** there was NO ML/training code in the repo (grep clean). The `annotated/`
masks were single-frame hand paints, never used at inference. So the ML increment starts from zero.

**What shipped (all additive; `colour-frangi` stays byte-identical, regression-pinned):**
- **§1 Diagnostics** (`diagnostics.py`): `save_stage_montage` dumps every pipeline stage
  (tissue / brightness / Frangi / seed / accepted) as one PNG; per-run so recall loss is visible.
- **§2 Temporal evidence** (`temporal.py`, new method `temporal-evidence`): re-engages the
  FRAME STACK (the old method read only the averaged final frame). Scores each pixel on 1-/2-frame
  positive differences, step-vs-gradual, persistence, confidence; **max 2-frame transition**;
  tolerant of later drift. This is the recall fix. `Method.needs_stack` added so `measure` passes
  the stack; methods may return their own `onset`/`confidence`.
- **§3 Lightweight ML** (`ml.py`, sklearn optional → degrades to baseline): deterministic features
  (brightness, jump1/2, step_score, ridge, local_std, persistence) + logistic/forest; `combine()`
  blends ML prob with evidence; `leave_one_plant_out` enforces split-by-plant (n=3). Needs real
  training on the Mac (sklearn in `extra`; joblib added).
- **§4 Anatomy** (`anatomy.py`): `segment_anatomy` → thallus/lumen/scale/tape/background/unknown
  layered masks; `refine_accepted` drops tape, keeps lumen-through-scale.
- **§5 Multi-frame annotator** (`outputs.save_multiframe_annotator`): frame slider, add/erase,
  category labels, per-stroke undo, exports JSON with frame id + label + **original-res coords** +
  source version (the training-queue payload). Helper `prepare_annotator_frames` unit-tested.
- **§6 Confidence UI** (`outputs.save_confidence_viewer`): threshold slider over the cached
  confidence map; accepted/uncertain/rejected; live %; "not a statistical CI". Written by `run()`
  whenever the method yields confidence (i.e. temporal-evidence).
- **§7 Outputs**: `save_event_tiffs` (per-frame + cumulative binary TIFF, **streamed** so full-res
  400-frame runs don't allocate GBs), `build/write_onset_table` (per-region onset/area/centroid/
  confidence/transition/ambiguous), `save_event_lut` (onset-time coloured overlay), `write_run_metadata`
  (calibration + 15-s interval + settings + version). `run()` writes them all; CLI flags
  `--frame-interval`, `--no-tiffs`, `--no-diagnostics`.
- **§8 Shrinkage companion: DEFERRED** (explicitly optional + non-blocking in the brief; lives in a
  separate repo `thallus_shrinkage` not present here). Next session.

**STILL TO DO on the Mac (VM can't):** (1) run each method on real `data/cavi*` at full res and
eyeball the diagnostic montage + confidence viewer + multi-frame annotator in a browser; (2) set the
collaborator's REAL µm/px; (3) train the §3 model LOPO on the expert masks and check recall vs v0.2.0;
(4) build §8 shrinkage. **VM was under load avg ~9 this session → tests ran 2–7 min each; that is
environment, not the code.**

---

## 2026-11-03 — WHERE THINGS STAND RIGHT NOW (read this box first)

- **Selectable detection methods SHIPPED.** Merged to `main` and tagged **`v0.2.0`** (pushed).
  CI `build-apps` then builds the macOS + Windows apps and **publishes them to the public
  Release**: https://github.com/chrmccar/cavidetect/releases/tag/v0.2.0
- **Repo is PUBLIC** (verified via API). The collaborator downloads the new app from that
  Releases page exactly as before — no change to how they get it.
- **Everything committed + pushed; working tree clean; on `main`.** The feature branch
  `feature/selectable-detection-methods` is also on origin (a PR was requested but opening it
  needs a browser/PAT — not blocking; the merge already landed on main).

### Still TO DO (next session / on the Mac)
1. **Confirm the `v0.2.0` CI build went green** (GitHub → Actions) and that the Release has both
   zips attached. If the build FAILED, read the log and fix (the frozen app is never launched in
   CI — past Windows-only crashes slipped through: relative-import and cp1252/encoding bugs).
2. **Mac GUI manual checks** (VM has no Tk display): `pip install -e . && cavidetect-gui` →
   (a) the **Method** dropdown switches the Settings panel; (b) run each method on
   `data/cavi600005`; (c) `<stem>_cs_params.json` is written and names the method;
   (d) the results header names the method.
3. **Set `µm per pixel` to the collaborator's REAL ImageJ calibration** — the shape rules are in
   µm and only physically meaningful once it's right; **2.5 is a fallback guess**.
4. Two deferred minors (non-urgent): `colour-shape --list-methods` still requires the `folder`
   positional + `--out` (argparse quirk); the GUI lays schema groups out in columns, which may be
   wide for `shape-filter` (4 groups) — check spacing on the Mac.

Design + plan for this work: `docs/superpowers/specs/2026-11-03-selectable-detection-methods-design.md`
and `docs/superpowers/plans/2026-11-03-selectable-detection-methods.md`.

## 2026-11-03 SESSION 3 — SELECTABLE DETECTION METHODS + CONFIGURABLE SETTINGS — read this first

Built in response to the expert's request for **shape-based thresholding** (BioVoxxel Shape
Filter: max inscribed circle, thinness ratio, elongation) and Chris's ask to **choose the
method** and **expose more settings** before a run. Fully self-contained — **no ImageJ**.

- **Spec + plan:** `docs/superpowers/specs/2026-11-03-selectable-detection-methods-design.md`
  and `docs/superpowers/plans/2026-11-03-selectable-detection-methods.md`.
- **Branch:** `feature/selectable-detection-methods` (not yet merged/pushed at time of writing).
- **New method system (`cavidetect/methods.py`):** a registry maps a method name → a detector
  + a **parameter schema** (`ParamSpec` list). The schema drives BOTH the GUI panel and the CLI
  `--params` validation — add a future experimental method = one `detect` fn + its schema.
- **Three methods:**
  1. `colour-frangi` — **the current method, DEFAULT, byte-identical by default** (guarded by a
     regression test). Optional `apply_shape_filter` toggle adds the BioVoxxel rules on top.
  2. `shape-filter` — the expert's flow: contrast/band-pass → threshold (otsu|percentile) →
     connected components → keep by **inscribed circle (µm) / thinness / elongation**. Defaults
     seeded to the expert's numbers (7–25 µm, 0–0.6, 0.6–1).
  3. `ridge-hysteresis` — exposes the old `detect_spatial_v2` (experimental).
- **Shared pure-Python pieces (TDD'd on the VM):** `shape_filters.py` (BioVoxxel descriptors),
  `preprocess_final.py` (gamma/CLAHE contrast + difference-of-Gaussians band-pass, default off).
- **Pixel scale:** `um_per_px` is a setting (default **2.5** — a FALLBACK guess). The shape rules
  are only physically meaningful once it's set to the collaborator's **real ImageJ calibration**.
  Binning is folded in automatically (`um_per_analysis_px = um_per_px * bin`).
- **Reproducibility:** every run writes `<stem>_cs_params.json` (method + resolved params + bin +
  tail_frac + um_per_px + version); method is stamped on the proof image, results header, bundle.
- **CLI:** `cavidetect colour-shape FOLDER --out X [--method NAME] [--params p.json] [--list-methods]`.
  No `--method` → `colour-frangi` (unchanged). `--air` still works (now an optional param).
- **GUI (Mac):** a **Method** dropdown; selecting a method rebuilds a grouped **Settings** panel
  (General / Preprocess / Threshold / Shape rules) from the schema. Everything else (Run/Cancel,
  viewer, annotate, bundle) unchanged.
- **Tests:** 63 pass on the VM (`pytest -q`). Only the live Tk window needs the Mac.

**MAC MANUAL CHECKS still to do (VM has no Tk display):** `pip install -e .` then `cavidetect-gui`
→ (1) the Method dropdown switches the Settings panel; (2) run each method on `data/cavi600005`;
(3) `<stem>_cs_params.json` is written and names the method; (4) the results header names the
method. If a method errors, read the Progress log — the pipeline runs as a subprocess.

## 2026-11-02 SESSION 2 — LIVE STATE (GitHub + distributable apps) — read this first

- GUI v2 (click-to-enlarge proof + auto-open viewer), **onset-timing fix** (start-of-rise),
  **✎ Annotate** + **📦 Bundle** buttons, and **standalone-app packaging** are all DONE and pushed.
- **GitHub repo (PUBLIC):** https://github.com/chrmccar/cavidetect — code only (data/results are
  gitignored). Pushed from the VM over the SSH key in `~/.ssh` (authenticates as `chrmccar`).
- **CI build `v0.1.0` SUCCEEDED** on macOS + Windows (Actions run 41377249712).
- **Built apps downloaded to the PROJECT ROOT** (one level ABOVE this repo):
  `../cavidetect-macos.zip` (~112 MB) and `../cavidetect-windows.zip` (~138 MB) — ready to send.
- **ACTIONS FOR CHRIS:** (1) **revoke the personal-access-token** at
  https://github.com/settings/tokens; (2) email the right zip to the collaborator (unsigned →
  Mac: right-click→Open; Windows: More info→Run anyway, first launch only); (3) relaunch
  `cavidetect-gui` on the Mac and run `cavi600005` to eyeball the new onset timing in the viewer.
- **To rebuild later:** push a new tag `git tag vX.Y.Z && git push origin vX.Y.Z`, OR GitHub →
  Actions → build-apps → Run workflow. Download from the run's **Artifacts** (browser) or via the
  API with a fresh token. Full handoff + recipient steps: `DISTRIBUTION.md`.
- **GitHub ops from the VM:** SSH push works (keys auth as `chrmccar`); there is **no `gh` CLI and
  no saved API token**, so creating a repo and downloading artifacts need a user-supplied PAT.

## 2026-11-02 STATE — read this first (current method of record)

**Objective (settled):** measure **total embolised tissue over time** (optical-vulnerability /
OSOV style — opensourceov.org, Brodribb & Skelton), NOT per-event onset. Chris's cavicam is
this method. The OLD tube/event detector (everything below "## Where we are") is superseded.

**Current method = `cavidetect colour-shape` (in `cavidetect/colour_shape.py`).**
Embolised tissue = **air-COLOUR in the dry final frame, within the thallus, AND tube-SHAPED**;
timing = the frame each pixel **last crossed** into the air-colour (read backwards from the end).
Pipeline, in order:
1. **tissue = `tissue_from_final(final)`** — Otsu on the DRY final frame, close+fill, **largest
   connected component only** (= the thallus), erode 2. (Was built from WET frames, which
   excluded the bottom rhizoids since they're dark until they air-fill — fixed 2026-11-02:
   bottom-third coverage 7% → 86%.)
2. **colour** = tissue & final ≥ `percentile(final[tissue], 70)` (the "air-colour" cut).
3. **shape** = frangi tube-ness (sigmas 1–4) seeds; grow each colour component to full width;
   keep a component only if tube-like (eccentricity ≥ 0.9 OR solidity ≤ 0.5) AND has a seed →
   drops round blobs / surface patches.
4. **timing** = per-pixel **start-of-rise** (`onset_start_of_rise`, 2026-11-02): each embolised
   pixel is dated to the first frame it rises past `rise_frac`(=0.4) of the way from its own dark
   baseline to its own bright level AND stays there (`persist`=3). Replaced the old "last frame
   below the global air-colour +1", which dated fill to the mid/end of the transition and looked
   delayed. Only timing changed; the mask and final % are unchanged.
5. → cumulative embolised-area curve.
Calibrated to Chris's hand annotation (`embolised_mask.png` in project root): **F1 = 0.68**,
total area matches (17.8% vs 17.9%). Params baked: bright_pct=70, vessel_min=0.04, grow=True,
tissue erode=2, tail_frac=0.12, rise_frac=0.4 (onset timing).

**Run (Mac only — bare `python3` lacks numpy; the `cavidetect` console script has the env):**
```
cd .../cavidetect
cavidetect colour-shape ../data/cavi490001 --out results_whole/cavi490001 --bin 2
open results_whole/cavi490001/cavi490001_cs_viewer.html
```
Outputs: `*_cs_viewer.html` (scrub raw|overlay|curve), `*_cs_curve.csv`, `*_cs_proof.png`
(colour-only vs colour+shape).

**GUI (added 2026-11-02):** `cavidetect/gui.py` — a simple Tkinter desktop window
over `colour-shape` (choose input folder → choose save location → set bin / air
(auto|manual) / tail-frac → Run/Cancel → live progress log → results summary with
headline % embolised, the proof PNG thumbnail, and buttons to open the viewer /
results folder). Runs the pipeline as a SUBPROCESS so Cancel is safe. **Mac only**
(VM has no tkinter): `pip install -e .` then `cavidetect-gui`, or `python3 -m
cavidetect.gui`, or double-click `run_gui.command`. UI-free helpers are unit-tested
in `tests/test_gui_helpers.py` (run on the VM). First real test target = `data/cavi600005`.
GUI now also has: bigger **click-to-enlarge proof**, auto-open **frame viewer** on finish,
**✎ Annotate this dataset** (opens `<stem>_cs_annotate.html`; colour-shape now emits it), and
**📦 Bundle to send…** (zips results + attached annotation PNG + `info.txt`).
**Packaging to a self-contained double-click app** (2026-11-02, for handing to another user on
Mac/Windows): one dual-entry exe via `cavidetect/app.py` (GUI by default, pipeline worker when
arg[0] is a CLI subcommand) so the GUI's subprocess/Cancel still works when frozen; PyInstaller
`cavidetect.spec`; GitHub Actions `.github/workflows/build.yml` builds macOS+Windows on tag push;
full instructions in `DISTRIBUTION.md`. Pillow added to deps. **Not buildable/testable on the VM**
(no macOS/Windows, no Tk) — CI + launching on each OS is the real test. 32 tests pass on the VM.

**Also built this session (all in the package, 18/18 unit tests pass):**
- `io.load_frames` + `io.load` dispatcher — ingest a FOLDER of per-frame TIFFs (mixed grey/RGB;
  drops the stray `t_0001 - Copy.tif`). Scale: no real metadata → fallback 2.5 µm/px.
- `embolism.py measure_embolism` — the earlier change-based primary measure (tube_prior + grow +
  end-anchored). Still wired into `cavidetect run --quick`. The `colour-shape` path is the newer,
  better one; `run --quick` is the older change-based one.
- `outputs.save_embolism_viewer` (interactive scrubber), `save_annotator` (**paint on the RAW
  final frame**, Export PNG), `save_final_png`.
- CLI: `run [--quick]`, `batch`, `colour-shape`.

**Ground-truth / key artefacts in the project:**
- `embolised_mask.png` (root) — Chris's hand annotation of the whole-image final (900×1200, ~18%).
- `cavidetect/results_whole/cavi490001/cavi490001_final.png` — raw dry final frame (1920×2560).
- `Screenshot ... cavi490001_cs — embolism viewer.png` (root) — a result screenshot.

**REMAINING IMPROVEMENTS (priority order for next session):**
1. **Background creep INSIDE the thallus** — if dark/texture inside the outline still gets picked,
   add an absolute brightness floor on top of the pct70 cut. (Corner background already dropped
   via largest-component.)
2. **Faint thin tubes under-caught** (lower half) — the F1=0.68 gap; the pct70 global cut misses
   dim tubes. Consider a LOCAL/adaptive threshold or lean harder on frangi for dim ridges.
3. **Dense central tuft** — shape filter may drop it as a blob; is it embolism? Chris's call.
4. **Motion** — thallus shrinks + loose bottom rhizoids move during dry-down; the averaged final
   blurs them. Registration (`pystackreg`) would sharpen and de-confound.
5. **`colour_shape.py` has NO unit tests yet** — add them (tissue_from_final, measure).
6. Decide whether `colour-shape` becomes the default `run` path (retire the change-based one).
7. Acacia reference video (Vimeo 215769356) — blocked from the VM; Chris to share another way to
   match the exact cavicam visual.

Everything below "## Where we are" is the OLD tube-centric design, now historical/secondary.

## Where we are
Built `cavidetect`, a tube-centric cavitation-event detector for the rhizoid time-lapses.
**M1 (core detection) and the M2 tube rewrite are done and working on all 3 plants / 31 stacks.**

### The key correction made tonight
A rhizoid is a **tube**; cavitation fills a **whole tube segment at once** and spreads along it.
The first detector marked scattered *pixels* (looked like dots / a before-after) — wrong granularity.
Detection is now **tube-centric**: find rhizoid tubes → give each tube one cavitation frame →
mark the whole tube embolised from that frame. Output shows tubes lighting up and spreading, and
the cumulative embolised-area curve is a **staircase of discrete tube cavitation events**.

### What to look at
- `results_report_ALL.html` — all 31 stacks (open in a browser).
- `example_outputs/49_1_1_temporal.png` and `_animation.gif` — the clearest single result.
- `example_outputs/49_1_1_tubemap.png` — the detected tube segments.

## How to run
```
cd .../project-liverwort-cavitation-event/cavidetect
pip install -e .                              # once
cavidetect batch "../starting substack" --out results   # full-res, all stacks
python tools/build_report.py results results_report.html # combined HTML
```
Per stack you get: `*_temporal.png`, `*_tubemap.png`, `*_tubes.csv` (**per-rhizoid outcome**),
`*_events.csv`, `*_area_curve.csv`, `*_manifest.json`, and `*_animation.gif` if `make_animation: true`.
Full res on the Mac uses `bin_xy: 1`; the VM previews used `bin_xy: 2` (RAM-limited).

## Code map (`cavidetect/cavidetect/`)
- `io.py` — load TIFF, z-as-time, pixel size from metadata.
- `characterise.py` — per-stack noise σ, active-at-start flag, drift score (**drift is buggy, see below**).
- `preprocess.py` — binning + Gaussian + Anscombe VST.
- `detect.py` — per-pixel positive/abrupt/persistent step (dual DN+σ threshold, step-vs-ramp test).
- `rhizoid.py` — tube map: Sato ridge → skeleton → branch-split segments. **← tune here**
- `tubes.py` — assign each tube a cavitation frame; per-tube table. **← tune here**
- `group.py` — old per-pixel grouping (kept; no longer on the main path).
- `outputs.py`, `pipeline.py`, `cli.py`, `config.py`.
- `tests/test_synthetic.py` — 3 behavioural tests (pass). `configs/default.yaml` — all params.

## NEXT STEPS (priority order)
1. **Review tube results with Chris across all 3 plants** (open `results_report_ALL.html`).
   Plant 60 (dense, moving tissue) is where tube segmentation is weakest — check it first.
2. **Two-class split (the main open M2 feature):** classify each event as **air-in-lumen**
   (brightening sits ON a tube ridge) vs **inter-rhizoid drainage** (brightening BETWEEN tubes).
   The tube map already gives the "on a ridge" mask; interstitial = accepted pixels NOT near any ridge.
3. **Fix the drift estimate** in `characterise.py` (`_phase_drift` compares pre/post-event frames →
   garbage, e.g. 142 px). Use consecutive-frame shifts within the quiet tail, or a bounded search.
   Then wire conditional `pystackreg` registration (needed for plant 60, skip for 49).
4. **Tube segmentation quality:** dense crossing bundles merge; segments fragment. Try tuning
   `ridge_sigmas` / `tube_percentile`, or a proper junction-aware tracing. Consider Cellpose (Mac GPU)
   as an alternative rhizoid instance map (optional dependency already noted).
5. **Then M3:** napari labelling workflow, OSOV baseline, evaluation (leave-one-plant-out,
   per-class precision/recall/F1), synthetic tests for the confound rejection.

## Known issues / caveats
- Results are **recall-tuned**; some off-tube red remains. Two-class split + ridge masking tightens it.
- `n_tubes` / `cavitated` counts depend on segmentation granularity — treat **embolised area over time**
  as the robust readout, not raw counts.
- Drift value in manifests is wrong (unused; see step 3).
- VM swaps on full-res stacks; develop previews with `bin_xy: 2`, do real runs on the Mac.

## Open questions for Chris (non-blocking)
1. Frame interval constant/known across stacks? (outputs are in frames for now; needed for the Ψ join.)
2. Do crops within a plant overlap, and can we get pixel offsets? (to avoid double-counting per-plant area.)
3. If forced to prioritise one class's recall, which — lumen fill (assumed) or drainage?
4. Confirm no 12/16-bit originals exist (metadata says 8-bit only).

Design rationale and the full data analysis: `DESIGN_SPEC.md`. Original brief: `../HANDOFF_for_claude_code.md`.
