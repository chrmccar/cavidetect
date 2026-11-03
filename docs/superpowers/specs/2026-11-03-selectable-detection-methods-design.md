# Selectable detection methods + user-configurable settings — design

**Date:** 2026-11-03
**Status:** approved (brainstorming) — ready for implementation plan
**Repo:** `cavidetect` (feature branch `feature/selectable-detection-methods`)

## Intent

The expert reviewing cavitation results asked for **shape-based thresholding**: keep
only embolism events that are tube-shaped, filtered by the BioVoxxel *Shape Filter*
descriptors they use in ImageJ —

- **maximum inscribed circle 7–25 µm** (tube width)
- **thinness ratio 0–0.6** (not round / elongated)
- **elongation 0.6–1** (line-like)

Chris wants this added **without disturbing the current method**: the GUI should let the
user **choose which detection method to run** (preserve the current one, add experimental
ones) and expose **more settings before a run** — the shape rules above, contrast, pixel
scale, etc. — so the tool is customisable rather than one fixed pipeline.

Extra context from the expert (informational): they also tried an ImageJ **band-pass FFT
filter** (pass 2.5–40 µm structures) but found the current cavidetect model works better.
This tells us the useful structure sits in a ~2.5–40 µm size band; we offer a band-pass as
an *optional, default-off* preprocessing step, not a core change.

**Success:** on the Mac, the collaborator can pick a method from a dropdown, adjust its
rules (in µm), run, and compare results; the current calibrated method (F1≈0.68) is
reachable and unchanged by default; every result records the method + exact settings that
produced it.

**Fully self-contained — no ImageJ.** Every descriptor the expert uses in ImageJ
(BioVoxxel Shape Filter: inscribed circle, thinness ratio, elongation) and the band-pass
are **reimplemented in pure Python** (skimage/scipy) inside the `cavidetect` package. The
words "ImageJ flow"/"ImageJ-style" below describe *which measurements we replicate*, not a
dependency: the pipeline never calls out to, requires, or assumes ImageJ at any step.

### Non-goals (YAGNI)

- No new detection algorithms beyond the three below.
- No batch-sweep over methods, no saved user presets/profiles beyond baked defaults.
- No change to the viewer/annotator/bundle formats except stamping the method + params in.

## Decisions (from brainstorming)

- **Pixel scale:** expose an editable **µm/px** field (default 2.5); shape rules are entered
  in µm and converted internally, accounting for `bin`. The collaborator sets their real
  ImageJ calibration.
- **Relationship to current method:** **Both** — ship a *separate* ImageJ-style method AND
  let the BioVoxxel descriptors *optionally augment* the current colour+frangi method.
- **Expert default ranges** are the seed defaults: inscribed circle 7–25 µm, thinness
  0–0.6, elongation 0.6–1.

## Architecture

### Method registry + parameter schema (the one new abstraction)

A registry maps a method name → a `Method` record:

```
Method:
  name: str                      # e.g. "colour-frangi"
  label: str                     # human label for the GUI dropdown
  detect: Callable               # (final, tissue, **params) -> dict; MUST return `shape` (bool mask)
                                 #   and `colour` (bool mask) for the shared proof panel;
                                 #   `ridge` optional (run() tolerates its absence)
  schema: list[ParamSpec]        # drives GUI widgets AND CLI validation

ParamSpec:
  key: str                       # param name passed to detect()
  label: str
  type: "float" | "int" | "bool" | "choice"
  default: Any
  min / max: optional            # numeric range (validated + clamped)
  choices: optional              # for type=choice
  unit: optional                 # "µm", "%", "px" — shown in GUI, used for conversion
  group: "General" | "Threshold" | "Shape rules" | "Preprocess"
  help: str
```

The schema is the **single source of truth**: the GUI builds its settings panel from it, and
the CLI validates/coerces `--params` against it. A future experimental method is one new
`detect` function + its schema — no GUI or CLI edits.

Lives in a new `cavidetect/methods.py` (registry + `Method`/`ParamSpec` + `get_method`,
`list_methods`, `resolve_params(method, raw) -> dict` with coercion/range-clamp/defaults).

### Shared descriptor module — `cavidetect/shape_filters.py` (pure, testable)

BioVoxxel descriptors computed per labelled region; no I/O, no tkinter.

- `thinness_ratio(area_px, perimeter_px) -> float`  = `4π·area / perimeter²`, clipped to 1
  (1 = disc, →0 = line).
- `elongation(minor_len, major_len) -> float`       = `1 − minor/major` (0 = round, →1 = line).
- `max_inscribed_diameter_px(region_mask) -> float` = `2 · max(distance_transform_edt(mask))`.
- `region_descriptors(prop, region_mask, um_per_analysis_px) -> dict` bundling the above in
  **µm** where applicable (inscribed circle in µm, area in µm²).
- `filter_regions(label_img, rules, um_per_analysis_px) -> mask` — keep components whose
  descriptors fall inside every `[min,max]` rule. `rules` omit/`None` = that bound disabled.

`um_per_analysis_px = um_per_px * bin` (one pixel in the binned image spans `bin` original
pixels). Centralised so every method and the GUI agree.

### Methods at launch

1. **`colour-frangi` — DEFAULT, current behaviour preserved exactly.**
   Wraps today's `detect_spatial` (air-colour percentile → frangi tubeness + eccentricity /
   solidity gate). With **no `--method` and no overridden params, output is byte-identical to
   the current pipeline** (regression-tested). Schema exposes the existing knobs
   (`bright_pct`, `vessel_min`, `ecc_min`, `sol_max`, `min_area`) **plus** an optional
   `apply_shape_filter` bool; when on, BioVoxxel descriptor rules are applied as an extra
   gate (the "augment current" half of *Both*).

2. **`shape-filter` — NEW, the expert's ImageJ flow.**
   `preprocess(contrast/bandpass) → threshold within tissue (Otsu | percentile) → connected
   components → filter_regions(inscribed µm, thinness, elongation, min area)`. Schema seeds
   the expert's defaults. No frangi.

3. **`ridge-hysteresis` — experimental.** Exposes the already-written `detect_spatial_v2`
   (seed/grow hysteresis + scale rejection) with its params in the schema, so it is runnable
   and comparable instead of dead code.

### Shared preprocessing — `cavidetect/preprocess_final.py`

Applied to the dry final frame before a method's threshold; feeds all methods via the
`Preprocess` param group:

- `contrast`: `none | gamma | clahe` with a strength param (default `none` → no change).
- `bandpass`: optional default-off FFT/DoG band-pass passing `bp_small_um`–`bp_large_um`
  (default 2.5–40, mirroring the expert's attempt). Implemented as difference-of-Gaussians
  at the two µm scales (cheaper and dependency-free vs FFT; equivalent band-pass behaviour).

### CLI

`colour-shape` gains:

- `--method NAME` (default `colour-frangi`)
- `--params PATH.json` (method params as a flat JSON object; validated against the schema)
- `--list-methods` (print names + labels and exit)

Existing `--bin`, `--air`, `--tail-frac` stay (general run controls). `--params` values
override schema defaults; absent keys fall back to defaults. Unknown/out-of-range keys are
reported and clamped/dropped with a warning line. `um_per_px` is a param (default 2.5).

### Reproducibility

`run()` writes `<stem>_cs_params.json` into the output folder: `{method, params (resolved),
bin, tail_frac, um_per_px, cavidetect version}`. The proof-image title, GUI results header,
and bundle `info.txt` all name the method + key rules used.

### GUI

- A **Method** dropdown (`ttk.Combobox`) at the top, populated from `list_methods()`.
- Selecting a method **rebuilds a schema-driven settings area**, widgets grouped by
  `group`: **General** (µm/px, bin, contrast, bandpass), **Threshold**, **Shape rules** (the
  three expert rules, units shown as µm). Numeric → `Spinbox`/`Entry`, bool → `Checkbutton`,
  choice → `Combobox`. No per-method widget code — it's generated from the schema.
- On Run: collect field values → coerce via the schema → write a temp `params.json` → build
  `colour-shape --method … --params …`. Everything else (subprocess/Cancel, live log, proof
  thumbnail, viewer/annotate/folder/bundle) is unchanged.
- Results header shows the method that ran.
- Pure helpers (schema→params dict, params→argv, temp-file writing, method list formatting)
  live outside the Tk class so they're unit-tested on the VM, matching the existing pattern.

## Data flow

```
GUI fields ──coerce(schema)──▶ params.json ─┐
CLI --method/--params ──────────────────────┼─▶ resolve_params ─▶ detect(final,tissue,**p)
                                             │                         │
load frames (bin) ─▶ final (tail mean) ─▶ tissue_from_final ─▶ preprocess ─▶ threshold/method
                                             │                         ▼
                                             └────────────▶ shape gate (method-specific) ─▶ mask
                                                                       ▼
                                   onset_start_of_rise (unchanged) ─▶ curve / viewer / proof / params.json
```

Only the **spatial detection** stage is swapped by method choice; tissue definition,
onset-timing, cumulative curve, and all outputs are shared and unchanged.

## Error handling

- Unknown `--method` → list valid names, exit non-zero.
- `--params` not valid JSON / not an object → clear error, exit non-zero.
- Out-of-range or wrong-type param → warn, clamp to range (numeric) or fall back to default;
  never crash mid-run.
- Empty result mask (no region passes the rules) → run completes, curve is all-zero, proof
  shows "0.0% — no regions passed the shape rules"; not an error (over-strict rules are a
  normal thing to discover).
- Missing optional deps for a preprocessing choice → warn and skip that step.

## Testing (TDD on the VM; Tk window on the Mac)

New/changed unit tests (`tests/`):

- `test_shape_filters.py` — descriptors on synthetic shapes: disc radius r → thinness≈1,
  elongation≈0, inscribed diameter≈2r; thin bar L×w → elongation≈1−w/L, inscribed≈w,
  thinness low; `filter_regions` keeps the bar, drops the disc under expert defaults; µm
  conversion with `bin` correct.
- `test_methods.py` — registry lookups, `resolve_params` coercion/clamp/defaults, unknown
  key handling.
- `test_colour_shape.py` (extend) — **regression:** `colour-frangi` with default params
  reproduces the current `detect_spatial` mask on a synthetic final frame (guards "preserve
  the current").
- `test_gui_helpers.py` (extend) — schema→params→argv round-trip, temp params.json written
  and parsed back, method-list formatting.

Manual (Mac): dropdown switches panels; run each method on `data/cavi600005`; `params.json`
written; results header names the method.

## Files

- **New:** `cavidetect/methods.py`, `cavidetect/shape_filters.py`, `cavidetect/preprocess_final.py`,
  `tests/test_shape_filters.py`, `tests/test_methods.py`.
- **Changed:** `cavidetect/colour_shape.py` (methods call into it; `run`/`measure` take
  `method`+`params`, write `params.json`; `detect_spatial` gains optional descriptor augment),
  `cavidetect/cli.py` (`--method`/`--params`/`--list-methods`), `cavidetect/gui.py` (method
  dropdown + schema-driven panel + temp params), `tests/test_colour_shape.py`,
  `tests/test_gui_helpers.py`.
- **Docs:** update `CONTINUE_HERE.md` / `DESIGN_SPEC.md` after implementation.
