# Selectable Detection Methods + Configurable Settings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the cavidetect GUI/CLI choose among multiple embolism-detection methods (preserving the current colour+frangi one as default) and expose each method's settings — including the expert's BioVoxxel shape rules and pixel scale — before a run.

**Architecture:** A method *registry* maps a name to a `(detect, schema)` pair; the schema (a list of `ParamSpec`) is the single source of truth that drives both the GUI's generated settings panel and the CLI's `--params` validation. The BioVoxxel descriptors and a band-pass are reimplemented in pure Python (skimage/scipy) in shared modules — no ImageJ anywhere. Only the spatial-detection stage is swapped by method; tissue, onset-timing and outputs stay shared.

**Tech Stack:** Python 3, numpy, scipy.ndimage, skimage (measure/filters/exposure), tifffile, matplotlib; Tkinter for the GUI (helpers kept pure for VM testing); pytest.

**Spec:** `docs/superpowers/specs/2026-11-03-selectable-detection-methods-design.md`

## Global Constraints

- **No ImageJ dependency** — every descriptor and the band-pass are pure-Python; the pipeline never calls out to ImageJ.
- **Preserve the current method exactly** — `colour-frangi` with default params must reproduce today's `detect_spatial` output; `cavidetect colour-shape FOLDER --out X` with no new flags behaves as before.
- **Scale rule:** `um_per_analysis_px = um_per_px * bin`; shape rules are entered in µm and converted with this everywhere. Default `um_per_px = 2.5`.
- **Expert default shape rules:** inscribed circle 7–25 µm, thinness ratio 0–0.6, elongation 0.6–1.
- **GUI testability:** all non-Tk logic lives in module-level pure helpers (matches existing `gui.py` pattern); the Tk window imports tkinter lazily inside `main()`.
- Follow existing file/style conventions; parameters documented where defined.

## Review Focus

- **Out-of-range / wrong-type `--params` value** → coerced/clamped to the schema, never a crash. (Task 3)
- **Over-strict rules → empty result mask** → run completes with a 0 % curve and a "no regions passed" proof, not an error. (Task 4)
- **`um_per_px` with `bin>1`** → inscribed-circle µm conversion stays correct (uses `um_per_px*bin`). (Task 1)
- **Unknown `--method` / malformed `--params` JSON** → clear message, non-zero exit, no traceback. (Task 5)
- **Degenerate region (perimeter 0, single pixel, minor axis 0)** → thinness/elongation return finite values, no divide-by-zero. (Task 1)

---

### Task 1: Shape descriptors + region filter (`shape_filters.py`)

**Files:**
- Create: `cavidetect/shape_filters.py`
- Test: `tests/test_shape_filters.py`

**Interfaces:**
- Consumes: nothing (pure numpy/scipy/skimage).
- Produces:
  - `thinness_ratio(area_px: float, perimeter_px: float) -> float` — `4*pi*area/perimeter**2`, clipped to `[0,1]`; returns `0.0` if `perimeter_px <= 0`.
  - `elongation(minor_len: float, major_len: float) -> float` — `1 - minor_len/major_len`; returns `0.0` if `major_len <= 0`.
  - `max_inscribed_diameter_px(region_mask: np.ndarray) -> float` — `2*max(distance_transform_edt(region_mask))`.
  - `region_descriptors(prop, region_mask: np.ndarray, um_per_analysis_px: float) -> dict` — keys `inscribed_um`, `thinness`, `elongation`, `area_um2`, `area_px` (`prop` is a skimage `RegionProperties`).
  - `filter_regions(label_img: np.ndarray, rules: dict, um_per_analysis_px: float) -> np.ndarray` — bool mask of kept components. `rules` keys (each optional / `None` disables): `inscribed_um=(lo,hi)`, `thinness=(lo,hi)`, `elongation=(lo,hi)`, `min_area_px:int`.

- [ ] **Step 1: Write failing tests** in `tests/test_shape_filters.py`:
  - `test_disc_descriptors`: a filled disc radius 15 px → `thinness` ≈ 1 (±0.1), `elongation` ≈ 0 (±0.1), `max_inscribed_diameter_px` ≈ 30 (±2).
  - `test_bar_descriptors`: a 60×6 px bar → `elongation` ≈ 1−6/60 (±0.1), `max_inscribed_diameter_px` ≈ 6 (±2), `thinness` < 0.6.
  - `test_inscribed_um_uses_bin`: disc radius 10 px with `um_per_analysis_px = 2.5*2 = 5.0` → `inscribed_um` ≈ 2*10*5.0 = 100 (±10).
  - `test_degenerate_region`: `thinness_ratio(1,0)==0.0`, `elongation(5,0)==0.0`, single-pixel region descriptors are finite.
  - `test_filter_regions_expert_defaults`: label image with one disc (r=15px) + one bar (60×6px), `um_per_analysis_px=1.0`, rules `inscribed_um=(7,25)`, `thinness=(0,0.6)`, `elongation=(0.6,1)` → kept mask covers the bar, not the disc.
- [ ] **Step 2: Run** `pytest tests/test_shape_filters.py -v` — expect FAIL (module missing).
- [ ] **Step 3: Implement** the functions in `cavidetect/shape_filters.py`. Use `scipy.ndimage.distance_transform_edt` for inscribed circle, `skimage.measure.regionprops`/`label` in `filter_regions`, `prop.minor_axis_length`/`major_axis_length` for elongation, `prop.area`/`prop.perimeter` for thinness. Guard all divisions.
- [ ] **Step 4: Run** `pytest tests/test_shape_filters.py -v` — expect PASS.
- [ ] **Step 5: Commit** `feat: pure-Python BioVoxxel shape descriptors + region filter`.

---

### Task 2: Final-frame preprocessing — contrast + band-pass (`preprocess_final.py`)

**Files:**
- Create: `cavidetect/preprocess_final.py`
- Test: `tests/test_preprocess_final.py`

**Interfaces:**
- Produces:
  - `apply_contrast(img: np.ndarray, mode: str = "none", gamma: float = 1.0, clahe_clip: float = 0.01) -> np.ndarray` — `mode in {"none","gamma","clahe"}`; `none` returns input unchanged (identity). Uses `skimage.exposure.adjust_gamma` / `equalize_adapthist`.
  - `bandpass_dog(img: np.ndarray, small_um: float, large_um: float, um_per_analysis_px: float) -> np.ndarray` — difference-of-Gaussians band-pass; sigmas = `small_um/um_per_analysis_px` and `large_um/um_per_analysis_px`.
  - `preprocess_final(final: np.ndarray, *, contrast: str = "none", gamma: float = 1.0, clahe_clip: float = 0.01, bandpass: bool = False, bp_small_um: float = 2.5, bp_large_um: float = 40.0, um_per_analysis_px: float = 2.5) -> np.ndarray` — contrast then (optional) band-pass; returns float32 same shape.

- [ ] **Step 1: Write failing tests** in `tests/test_preprocess_final.py`:
  - `test_contrast_none_is_identity`: `apply_contrast(img, "none")` returns array equal to input.
  - `test_gamma_changes_values_monotonically`: gamma 0.5 brightens mid-tones (output ≥ input where input in (0,1)).
  - `test_bandpass_removes_flat_background`: constant image → band-pass output ≈ 0 everywhere (±1e-3).
  - `test_bandpass_passes_midscale_feature`: a bar of width ≈ (small+large)/2 µm survives with non-trivial contrast.
  - `test_preprocess_default_is_identity`: `preprocess_final(img)` (all defaults) returns array equal to input.
- [ ] **Step 2: Run** `pytest tests/test_preprocess_final.py -v` — expect FAIL.
- [ ] **Step 3: Implement** in `cavidetect/preprocess_final.py`. Normalise to `[0,1]` internally for gamma/clahe, restore scale on return; DoG via `scipy.ndimage.gaussian_filter`.
- [ ] **Step 4: Run** `pytest tests/test_preprocess_final.py -v` — expect PASS.
- [ ] **Step 5: Commit** `feat: contrast + difference-of-Gaussians band-pass preprocessing`.

---

### Task 3: Method registry + schema + params resolution (`methods.py`)

**Files:**
- Create: `cavidetect/methods.py`
- Test: `tests/test_methods.py`
- Modify: `cavidetect/colour_shape.py` (add `detect_spatial` optional descriptor augment — see Step 3)

**Interfaces:**
- Consumes: `shape_filters.filter_regions`, `preprocess_final.preprocess_final`, `colour_shape.detect_spatial`, `colour_shape.detect_spatial_v2`.
- Produces:
  - `@dataclass ParamSpec(key:str, label:str, type:str, default, min=None, max=None, choices=None, unit=None, group:str="General", help:str="")` (`type in {"float","int","bool","choice"}`).
  - `@dataclass Method(name:str, label:str, detect:Callable, schema:list[ParamSpec])`.
  - `REGISTRY: dict[str, Method]`; `register(m)`, `get_method(name)->Method` (raises `KeyError` with valid names in message), `list_methods()->list[tuple[str,str]]` (ordered `[(name,label),...]`, `colour-frangi` first).
  - `resolve_params(method_name:str, raw:dict) -> dict` — start from schema defaults, overlay `raw`, coerce to each spec's `type`, clamp numerics to `[min,max]`, drop unknown keys with a `warnings.warn`, return the full param dict.
  - Three registered methods with `detect(final, tissue, **params) -> dict` returning at least `shape` and `colour` bool masks:
    - `colour-frangi` — calls `detect_spatial(...)`; schema: `um_per_px`(General), `bright_pct`,`vessel_min`,`ecc_min`,`sol_max`,`min_area`(Threshold/Shape) + `apply_shape_filter`(bool) and the three µm rules used only when that is on.
    - `shape-filter` — `preprocess_final(...) → threshold within tissue (Otsu|percentile) → label → filter_regions(...)`; schema: `um_per_px`, `contrast`,`gamma`,`clahe_clip`,`bandpass`,`bp_small_um`,`bp_large_um`(Preprocess), `threshold`(choice otsu|percentile),`bright_pct`(Threshold), `inscribed_lo_um`=7,`inscribed_hi_um`=25,`thinness_lo`=0,`thinness_hi`=0.6,`elongation_lo`=0.6,`elongation_hi`=1,`min_area_px`=8(Shape rules).
    - `ridge-hysteresis` — calls `detect_spatial_v2(...)`; schema exposes its existing params.

- [ ] **Step 1: Write failing tests** in `tests/test_methods.py`:
  - `test_list_methods_order`: `list_methods()[0][0] == "colour-frangi"`; all three names present.
  - `test_resolve_fills_defaults`: `resolve_params("shape-filter", {})["inscribed_hi_um"] == 25`.
  - `test_resolve_coerces_and_clamps`: `resolve_params("shape-filter", {"bright_pct":"200"})["bright_pct"] == 100` (string→int, clamped to max 100).
  - `test_resolve_drops_unknown`: `resolve_params("shape-filter", {"bogus":1})` has no `bogus` key (and warns).
  - `test_get_method_unknown_lists_valid`: `get_method("nope")` raises `KeyError` whose message contains `colour-frangi`.
  - `test_shape_filter_detect_returns_masks`: on a synthetic final+tissue, `get_method("shape-filter").detect(final, tissue, **resolve_params("shape-filter",{}))` returns dict with bool `shape` and `colour` of `final.shape`.
- [ ] **Step 2: Run** `pytest tests/test_methods.py -v` — expect FAIL.
- [ ] **Step 3: Implement** `methods.py` and add an optional `apply_shape_filter`/rules path to `detect_spatial` in `colour_shape.py` (when off, behaviour is unchanged). Thresholding helper for `shape-filter`: Otsu via `skimage.filters.threshold_otsu(final[tissue])`, percentile via `np.percentile(final[tissue], bright_pct)`.
- [ ] **Step 4: Run** `pytest tests/test_methods.py -v` — expect PASS.
- [ ] **Step 5: Commit** `feat: method registry, param schema, and three registered methods`.

---

### Task 4: Wire `measure`/`run` to method+params; write `params.json` (`colour_shape.py`)

**Files:**
- Modify: `cavidetect/colour_shape.py` (`measure`, `run`)
- Test: `tests/test_colour_shape.py` (extend)

**Interfaces:**
- Consumes: `methods.get_method`, `methods.resolve_params`.
- Produces:
  - `measure(stack, method:str="colour-frangi", params:dict|None=None, tail_frac:float=0.12, rise_frac:float=0.4) -> dict` — resolves params, builds `final`, calls the method's `detect`, then the unchanged `onset_start_of_rise`/cumulative-curve code; dict gains `method` and `params` keys; `ridge` defaults to zeros if the method omits it.
  - `run(folder, out_dir, bin_xy:int=2, method:str="colour-frangi", params:dict|None=None, tail_frac:float=0.12) -> dict` — unchanged outputs plus `<stem>_cs_params.json` = `{method, params, bin, tail_frac, um_per_px, version}`; proof title and summary line name the method.

- [ ] **Step 1: Write failing tests** (extend `tests/test_colour_shape.py`):
  - `test_colour_frangi_default_matches_detect_spatial`: on a synthetic final frame, `measure(stack)["shape"]` equals `detect_spatial(final)["shape"]` (regression — preserves current method).
  - `test_empty_rules_give_zero_curve`: `shape-filter` with impossible rules (`inscribed_lo_um=1000`) → `frac[-1] == 0.0`, no exception.
  - `test_run_writes_params_json`: after `run(...)`, `<stem>_cs_params.json` exists and parses, with `method` and `params` keys. (Use `bin_xy` large / tiny fixture to keep it fast.)
- [ ] **Step 2: Run** the three tests — expect FAIL.
- [ ] **Step 3: Implement** the `method`/`params` plumbing in `measure`/`run`; keep `onset_start_of_rise`, curve, viewer, annotator calls as-is; write the JSON with `json.dump`.
- [ ] **Step 4: Run** `pytest tests/test_colour_shape.py -v` — expect PASS.
- [ ] **Step 5: Commit** `feat: run/measure take method+params and record them in params.json`.

---

### Task 5: CLI `--method` / `--params` / `--list-methods` (`cli.py`)

**Files:**
- Modify: `cavidetect/cli.py` (`colour-shape` subparser + dispatch)
- Test: `tests/test_cli_methods.py` (create)

**Interfaces:**
- Consumes: `methods.list_methods`, `colour_shape.run`.
- Produces: `colour-shape` accepts `--method` (default `colour-frangi`), `--params PATH` (JSON object), `--list-methods` (print `name — label` lines, exit 0). Existing `--bin`,`--air`,`--tail-frac` kept; `--air` (if given) folds into `params["air"]`. Invalid method → stderr message listing valid names, exit 2; unreadable/non-object `--params` → stderr message, exit 2.

- [ ] **Step 1: Write failing tests** in `tests/test_cli_methods.py` (invoke `cli.main([...])`, capture via `capsys`/`SystemExit`):
  - `test_list_methods_prints_all`: `--list-methods` prints `colour-frangi` and exits 0.
  - `test_unknown_method_exits_2`: `colour-shape f --out o --method nope` raises `SystemExit(2)`, stderr names valid methods. (monkeypatch `colour_shape.run` so no real work runs.)
  - `test_bad_params_json_exits_2`: `--params` pointing at non-JSON → `SystemExit(2)`.
  - `test_params_and_method_forwarded`: valid `--method shape-filter --params good.json` calls a monkeypatched `run` with matching `method=` and a dict `params=`.
- [ ] **Step 2: Run** `pytest tests/test_cli_methods.py -v` — expect FAIL.
- [ ] **Step 3: Implement** argparse additions + validation + dispatch in `cli.py`.
- [ ] **Step 4: Run** `pytest tests/test_cli_methods.py -v` — expect PASS.
- [ ] **Step 5: Commit** `feat: colour-shape CLI gains --method/--params/--list-methods`.

---

### Task 6: GUI method dropdown + schema-driven settings panel (`gui.py`)

**Files:**
- Modify: `cavidetect/gui.py` (pure helpers + Tk `App`)
- Test: `tests/test_gui_helpers.py` (extend)

**Interfaces:**
- Consumes: `methods.list_methods`, `methods.get_method`, `methods.resolve_params`.
- Produces (pure helpers, no tkinter):
  - `params_from_fields(method_name:str, values:dict) -> dict` — coerce raw string field values through `resolve_params`.
  - `write_params_tempfile(params:dict) -> str` — write JSON to a temp file, return its path.
  - `build_command(folder, outdir, *, method:str="colour-frangi", params_path:str|None=None, bin_xy:int=2, tail_frac:float=0.12, python_exe=None, frozen=None) -> list[str]` — extend existing to add `--method` and (if given) `--params`.
- Tk `App`: a **Method** `ttk.Combobox` (from `list_methods()`); `_rebuild_params_panel()` clears and regenerates widgets from `get_method(name).schema`, grouped by `ParamSpec.group` (float/int→`Spinbox`/`Entry`, bool→`Checkbutton`, choice→`Combobox`, units shown in the label); `_run()` builds params via `params_from_fields` → `write_params_tempfile` → `build_command(... method=, params_path=)`; results header shows the method name.

- [ ] **Step 1: Write failing tests** (extend `tests/test_gui_helpers.py`):
  - `test_build_command_includes_method_and_params`: `build_command("f","o",method="shape-filter",params_path="p.json")` contains `["--method","shape-filter","--params","p.json"]`.
  - `test_build_command_default_method_backcompat`: with defaults, argv still contains `colour-shape`,`f`,`--out`,`o` (unchanged prefix).
  - `test_params_from_fields_coerces`: `params_from_fields("shape-filter", {"bright_pct":"70"})["bright_pct"] == 70`.
  - `test_write_params_tempfile_roundtrips`: file exists and JSON-loads to the same dict.
- [ ] **Step 2: Run** `pytest tests/test_gui_helpers.py -v` — expect FAIL for the new tests.
- [ ] **Step 3: Implement** the pure helpers, then the Tk dropdown + `_rebuild_params_panel()` (tkinter stays lazily imported inside `main()`).
- [ ] **Step 4: Run** `pytest tests/test_gui_helpers.py -v` — expect PASS. (Tk window itself verified on the Mac — see Task 7.)
- [ ] **Step 5: Commit** `feat: GUI method dropdown + schema-generated settings panel`.

---

### Task 7: Docs + full suite + Mac verification note

**Files:**
- Modify: `CONTINUE_HERE.md`, `DESIGN_SPEC.md` (note the new method system + how to add a method), `README`/help text if present.

- [ ] **Step 1:** Run the whole suite: `pytest -q` — expect all green.
- [ ] **Step 2:** Update `CONTINUE_HERE.md` with: the method system, how to pick a method in the GUI/CLI, how `params.json` records a run, the `um_per_px` caveat (default 2.5, set to the real ImageJ calibration), and the Mac-only manual checks (dropdown switches panels; run each method on `data/cavi600005`; `params.json` written; header names the method).
- [ ] **Step 3: Commit** `docs: document the selectable-method system and Mac verification steps`.

---

## Notes for the executor

- The VM has numpy/scipy/skimage/pytest — Tasks 1–6 test fully here. Only the live Tk window (Task 7 manual checks) needs the Mac; do not try to open a Tk window on the VM.
- Keep `colour-frangi` byte-identical by default — the regression test in Task 4 is the guard; if it fails, the method wrapper changed behaviour and must be fixed, not the test.
- Synthetic fixtures: reuse/extend `tests/make_synthetic.py` where one already exists for final-frame shapes.
