# Design: `rhizoid-events` — per-rhizoid cavitation detection

Date: 2026-11-12
Status: approved (brainstorm), implementing on `feature/rhizoid-events`

## Intent

The expert wants a detector that works at the level of **individual pegged
rhizoids**, not total embolised area. Each rhizoid is a tube; cavitation is a
**whole-strand brightness jump in one frame (rarely two)**. A *partial*
brightening along a strand is **drying**, not cavitation, and must not be
counted. The deliverable is **how many rhizoids cavitated and when** (a count
over time + a per-rhizoid timing table), plus two pictures the expert named: a
single traced-rhizoid map and a frame-by-frame series highlighting each rhizoid
as it cavitates.

Source: `../12_11_feedback.docx` (expert, 2026-11-12).

### Decisions settled in brainstorming
- **Fit:** a new *selectable method* alongside the existing detectors; the
  area-% `colour-shape`/Inspector path stays untouched and selectable.
- **Primary readout:** total count of cavitated rhizoids + per-rhizoid timing
  (not a normalised fraction).
- **Correction:** automatic tracing will err; the existing Inspector editing
  toolkit (add/erase/split/merge → recompute) is the safety net, extended to
  recompute the per-rhizoid count/timing (not only area).
- **Core algorithm:** Approach A — late-frame ridge skeleton traced and read
  backwards. ML/Cellpose instance segmentation is explicitly out of scope,
  left as a future pluggable backend. No full-stack registration.

## Non-goals (YAGNI)
- No Cellpose/ML instance backend in this build.
- No `pystackreg`/global registration (a dilated sampling corridor absorbs the
  modest within-window drift instead; the existing drift estimate is known buggy
  and is not used).
- No change to `colour-frangi`, `shape-filter`, `ridge-hysteresis`,
  `temporal-evidence`, `hybrid-ridge-temporal` — all stay byte-identical,
  regression-pinned.
- Real µm/px calibration remains a Mac follow-up (2.5 fallback).

## Architecture

### Registry integration (no CLI/GUI code change)
A new `Method("rhizoid-events", "Per-rhizoid events", detect, schema,
needs_stack=True)` registered in `methods.py`. Because the registry drives both
the CLI `--method` list and the GUI settings panel, the method and its params
appear automatically. All new logic lives in a **new module
`cavidetect/rhizoid_events.py`**; `methods.py` only imports and registers it.

### Return contract
`detect(final, tissue, stack=…, **params) -> dict` returns:
- `shape`, `colour` — boolean masks (union of accepted *cavitated* rhizoid
  pixels) so every existing output (viewer, event TIFFs, LUT, onset table) keeps
  working unchanged.
- `onset` — int32 per-pixel onset map where **every pixel of a cavitated strand
  inherits that strand's single cavitation frame** (whole tube lights at once).
- `tubes` — int32 instance-label map of traced rhizoids (0 = background).
- `events` — list of per-rhizoid records (the new payload):
  `{id, cavitation_frame | None, cls ∈ {"cavitation","drying","none"},
    length_px, bright_frac_peak, confidence}`.

`colour_shape.run()` gains one branch: `if r.get("events"): <new writers>`.
`measure()` already passes `stack=` to `needs_stack` methods and already accepts
a method-supplied `onset`; it needs a small addition to forward `tubes`/`events`
through to `run()` (added to the returned dict alongside `evidence`).

## Pipeline (`rhizoid_events.detect`)

1. **CLAHE** every loaded frame first (reuse `preprocess_final` ImageJ-semantic
   knobs: blocksize 127, bins 256, slope 1.5). The CLAHE'd stack is used for
   tracing and for the brightness profiles.
2. **Reference frame** = per-pixel max over the last `ref_tail_n` loaded frames
   (strands are brightest after cavitation → most separable). Fallback to mean
   if max is too noisy on synthetic/degenerate input (guard empty tissue).
3. **Instance tracing:**
   - `colour_shape.ridge_evidence(reference, tissue)` → ridge response + seed.
   - `inspector.label_tubes(ridge_mask)` → segments split at junctions.
   - **Divergence-merge** (new): merge adjacent segments into one strand when
     they are collinear (orientation continuity within a tolerance) and their
     endpoints are close, EXCEPT where a segment forks into two segments that
     diverge in orientation — those stay separate strands that share the upper
     trunk. Deterministic, graph-over-segments; no ML.
4. **Per-strand history (read backwards):** for each strand, build a sampling
   mask = its skeleton dilated by `corridor_dilate_px`; per frame compute
   `bright_frac` = fraction of the strand's length whose sampled brightness is
   in air-colour (strand-local threshold from its own dark/bright levels, like
   `onset_start_of_rise`'s per-pixel logic but aggregated along the strand).
5. **Event detection + classification** (next section) → per strand: a
   cavitation frame or a `drying`/`none` label.
6. Build `onset` (strand frame broadcast to its pixels), `shape`/`colour`
   (cavitated strands' pixels), `tubes`, `events`.

### Whole-strand vs. drying discriminator
Per strand, over frames, inspect the `bright_frac(t)` trace:
- **cavitation** ⇔ `bright_frac` rises from below `bright_frac_thresh`
  (default 0.7) to at/above it within a window of `step_window` frames
  (default 2) — a sharp, near-complete jump. Onset = the first frame of that
  jump (start of the rise).
- **drying** ⇔ `bright_frac` increases meaningfully but either plateaus below
  `bright_frac_thresh` (partial strand) or climbs gradually — the largest
  single-frame rise within the window is below `gradual_reject`. Recorded,
  **not counted**.
- **none** ⇔ no meaningful rise within the window.
`confidence` = normalised jump sharpness × fraction covered (0–1, interpretable,
not calibrated).

## Outputs (new writers in `outputs.py`, called from `run()`)
- `save_rhizoid_map(tubes, final, out_dir, stem)` → `<stem>_cs_rhizoids.png`:
  one image, each traced rhizoid a distinct hue over the (dimmed) final frame.
- `save_cavitation_series(stack, tubes, events, out_dir, stem)` →
  `<stem>_cs_cavitation/` PNG-per-frame **+** `<stem>_cs_cavitation.gif`: each
  cavitating strand lit at its event frame on the CLAHE'd raw (strands already
  cavitated stay lit, matching the "spreading staircase" read).
- `write_rhizoid_table(events, out_dir, stem)` → `<stem>_cs_rhizoids.csv`:
  `id, cavitation_frame, cls, length_px, bright_frac_peak, confidence`.
- **Count-over-time staircase:** `build/write_rhizoid_count` →
  `<stem>_cs_rhizoid_count.csv` (`frame, n_cavitated_cumulative`) + headline
  "N rhizoids cavitated" in the results summary. The existing per-pixel area
  `cum` curve is still written for continuity.

## Correction loop (reuse + extend Inspector)
- `tubes` is passed to `save_inspector(… tube_ids=tubes …)` — the overlay
  already colours by tube identity, so the expert reviews strand identity
  directly and uses the existing **add / erase / split / merge** tools.
- `inspector.recompute_curve` / `apply_corrections` currently recompute
  per-pixel area only. Add a **per-rhizoid recompute path**: after edits, re-run
  the strand history + classifier over the edited tube map to regenerate
  `events`, the timing table, and the count staircase (not just area). Auto run
  preserved; corrected run saved alongside, edits seeded so rounds accumulate.

## Memory / windowing
New param `max_frames` (default 100). Only the first `max_frames` frames are
loaded/analysed (cavitation is in the first ~70). This bounds RAM and makes the
`ref_tail_n` reference post-cavitation. A strand not cavitated within the window
= `none`. Reuses the existing memory-safe per-frame load path.

## Params (schema; auto-surface in GUI/CLI)
| key | default | group | meaning |
|-----|---------|-------|---------|
| `max_frames` | 100 | General | frames loaded/analysed from the start |
| `ref_tail_n` | 8 | General | frames maxed for the reference |
| `bright_frac_thresh` | 0.7 | Event | strand fraction bright ⇒ cavitation |
| `step_window` | 2 | Event | max frames for the jump |
| `gradual_reject` | 0.3 | Event | min single-frame rise in `bright_frac` during the jump; below ⇒ gradual ⇒ drying |
| `corridor_dilate_px` | 3 | Trace | strand sampling corridor |
| `divergence_merge` | true | Trace | merge collinear segments into strands |
| CLAHE knobs | ImageJ defaults | Preprocess | inherited |

## Error handling
- Empty tissue / no ridges ⇒ return empty masks, `events=[]`, no crash (guard
  the `np.percentile`-on-empty fragility noted in CONTINUE_HERE).
- `stack` shorter than `max_frames` ⇒ use what exists.
- A strand with zero length after tracing ⇒ dropped before classification.
- GIF/PNG writers tolerate zero cavitation events (emit the traced map + an
  empty series rather than failing).

## Testing (TDD, VM-runnable on synthetic stacks)
1. Whole-strand 1-frame jump ⇒ exactly one `cavitation` event at the right frame.
2. Partial strand brightening (only part lights) ⇒ `drying`, not counted.
3. Gradual whole-strand rise over many frames ⇒ `drying`, not counted.
4. Two strands sharing an upper trunk but diverging lower ⇒ **2** ids.
5. Collinear fragments of one strand ⇒ merged to **1** id.
6. `max_frames` windowing stops the load/analysis at N.
7. Per-rhizoid recompute after a split edit changes the count; after a merge,
   likewise.
8. Registry: method + params present; `resolve_params` round-trips.
9. **Regression guard:** the other five detectors' masks byte-identical.
10. End-to-end `run()` smoke at `--bin 4` writes rhizoid map + series + table +
    count CSV + inspector.

Full-res timing, browser eyeballing of the series/inspector, and tuning against
the expert `annotated/` masks stay Mac-only follow-ups.

## Files touched
- **new** `cavidetect/rhizoid_events.py` — detect, tracing, divergence-merge,
  strand history, classifier.
- `cavidetect/methods.py` — register the method + schema.
- `cavidetect/colour_shape.py` — `measure()` forwards `tubes`/`events`;
  `run()` branches to the new writers.
- `cavidetect/outputs.py` — `save_rhizoid_map`, `save_cavitation_series`.
- `cavidetect/build.py` (or `outputs.py`) — `write_rhizoid_table`,
  `write_rhizoid_count`.
- `cavidetect/inspector.py` — per-rhizoid recompute path in
  `recompute_curve`/`apply_corrections`.
- **new** `tests/test_rhizoid_events.py` — the ten tests above.
