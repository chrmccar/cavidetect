# cavidetect

Automated, transparent detection of **cavitation events** in liverwort rhizoid time-lapse microscopy.
Finds where / when / how big each event is, separating true events from the confounds that also
brighten the image (slow drying, movement, early blobs). Physics-based, deterministic, no deep learning.

See `DESIGN_SPEC.md` for the full rationale and the data analysis behind the design.

## Status: M1 (core detector)
Implemented: IO + per-stack characterisation + preprocess + per-pixel abrupt/persistent step
detection + spatiotemporal grouping + outputs (event table, cumulative-area curve, overlays, manifest).
Tuned for **recall**; confound rejection and A/B class assignment arrive in M2.

## Install
```bash
cd cavidetect
pip install -e .          # core deps only
pip install -e ".[dev]"   # + pytest
```

## Run
```bash
# one stack (full resolution — recommended on a machine with >=16 GB RAM)
cavidetect run path/to/49_1_1.tif --out results/49_1_1

# a whole folder (recursive), e.g. all plants
cavidetect batch "starting substack/" --out results/

# custom parameters
cavidetect run stack.tif --out out/ --config configs/default.yaml
```

Outputs per stack (in `--out`): `*_events.csv`, `*_area_curve.csv`, `*_overlay.png`, `*_manifest.json`.

### Performance note
On a RAM-constrained machine, set `bin_xy: 2` in the config (2x spatial binning; the sensor noise
is correlated at 1 px so this loses little). On an M3/24 GB use `bin_xy: 1` (full resolution).

## Test
```bash
pytest -q          # behavioural tests on a synthetic stack (events kept, confounds rejected)
```

## Key parameters (`configs/default.yaml`)
- `step_min_dn` (10 DN) — absolute grey-level floor; what slow drying never clears.
- `step_k_sigma` (5) — also require step >= k x per-stack noise sigma.
- `abrupt_ratio_max` (0.5) — accept only if a single step fits the trace far better than a straight
  line (the step-vs-ramp test that separates abrupt events from gradual drying).
- `w_long`, `rampfit_half`, `persist_frames` — temporal windows (frames).

## Known limitations (M1)
- Drift estimate in the manifest is unreliable (compares pre/post-event frames); registration is not
  yet applied. Fixed in M2.
- No confound rejection (bright/dark movement pairs, round early blobs) or A/B classification yet (M2).
- Event counts are recall-tuned and include false positives by design.
