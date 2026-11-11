"""Tests for the inspector Phase 2 editing/correction recompute core."""
import os
import numpy as np
import pytest

from cavidetect import inspector as insp


# ---- stroke rasterisation --------------------------------------------------------

def test_rasterize_single_point_disc():
    m = insp.rasterize_stroke([[10, 10]], radius=3, hw=(20, 20))
    assert m[10, 10]
    assert m[10, 13] and not m[10, 15]          # within / beyond radius


def test_rasterize_polyline_connected():
    m = insp.rasterize_stroke([[5, 2], [5, 18]], radius=1, hw=(10, 20))
    assert m[5, 2] and m[5, 10] and m[5, 18]    # the whole segment filled


# ---- curve recompute -------------------------------------------------------------

def test_recompute_curve_monotone_and_total():
    onset = np.full((5, 5), -1, np.int32)
    onset[0, 0] = 1; onset[0, 1] = 3; onset[0, 2] = 3
    shape = onset >= 0
    cum, frac = insp.recompute_curve(onset, shape, tissue_px=25, T=6)
    assert list(cum) == [0, 1, 1, 3, 3, 3]       # 1 at f1, +2 at f3
    assert cum[-1] == shape.sum()
    assert np.all(np.diff(cum) >= 0)


# ---- erase / add -----------------------------------------------------------------

def _state():
    Y, X = 40, 60
    shape = np.zeros((Y, X), bool); shape[20, 5:55] = True
    onset = np.full((Y, X), -1, np.int32); onset[20, 5:55] = 10
    tissue = np.ones((Y, X), bool)
    ridge = np.zeros((Y, X), np.float32); ridge[18:23, :] = 0.5
    return onset, shape, tissue, ridge


def test_apply_edits_erase_removes_pixels():
    onset, shape, tissue, ridge = _state()
    edits = [{"op": "erase", "points": [[20, 30]], "radius": 4}]
    r = insp.apply_edits(onset, shape, tissue, ridge, edits)
    assert r["shape"].sum() < shape.sum()
    assert not r["shape"][20, 30]
    assert r["onset"][20, 30] == -1


def _brightening_stack(Y=40, X=60, T=20, row=30):
    rng = np.random.default_rng(0)
    s = rng.normal(30, 2, size=(T, Y, X)).astype(np.float32)
    s[:, 15:25, :] += 60                          # a tissue band
    for t in range(T):
        if t >= 12:
            s[t, row, 5:55] += 150                 # a NEW tube fills late
    return s


def test_apply_edits_add_inserts_with_onset():
    onset, shape, tissue, ridge = _state()
    ridge[30, :] = 0.5                             # mark the added tube's ridge
    s = _brightening_stack(row=30)
    edits = [{"op": "add", "points": [[30, 6], [30, 54]], "radius": 1}]
    r = insp.apply_edits(onset, shape, tissue, ridge, edits, stack=s)
    assert r["shape"][30, 30]                      # new tube present
    assert r["shape"].sum() > shape.sum()
    assert r["onset"][30, 30] >= 10                # dated near its (late) brightening


def test_apply_edits_add_without_stack_raises():
    onset, shape, tissue, ridge = _state()
    with pytest.raises(ValueError):
        insp.apply_edits(onset, shape, tissue, ridge,
                         [{"op": "add", "points": [[30, 6]], "radius": 2}], stack=None)


# ---- split / merge (identity) ----------------------------------------------------

def test_apply_edits_split_divides_one_tube():
    onset, shape, tissue, ridge = _state()
    before = insp.label_tubes(shape).max()
    edits = [{"op": "split", "point": [20, 30]}]
    r = insp.apply_edits(onset, shape, tissue, ridge, edits)
    assert r["tube_ids"].max() > before           # one tube -> two ids
    assert (r["tube_ids"][shape] > 0).all()


def test_apply_edits_merge_joins_two_tubes():
    Y, X = 40, 60
    shape = np.zeros((Y, X), bool); shape[12, 5:55] = True; shape[28, 5:55] = True
    onset = np.where(shape, 10, -1).astype(np.int32)
    tissue = np.ones((Y, X), bool); ridge = np.zeros((Y, X), np.float32)
    edits = [{"op": "merge", "points": [[12, 30], [28, 30]]}]
    r = insp.apply_edits(onset, shape, tissue, ridge, edits)
    assert len(set(r["tube_ids"][shape].tolist())) == 1   # merged to one identity


# ---- edits JSON coordinate conversion --------------------------------------------

def test_edits_from_json_scales_display_to_analysis():
    doc = {"disp_size": [100, 50], "orig_size": [100, 200],
           "edits": [{"op": "erase", "points": [[50, 25]], "radius": 10}]}
    out = insp.edits_from_json(doc)
    # x=50 of W=100 -> col = 50 * 200/100 = 100 ; y=25 of H=50 -> row = 25 * 100/50 = 50
    assert out[0]["points"][0] == [50.0, 100.0]
    assert out[0]["op"] == "erase"


# ---- state round-trip + full apply_corrections -----------------------------------

def test_state_roundtrip(tmp_path):
    onset, shape, tissue, ridge = _state()
    p = insp.save_inspector_state(str(tmp_path), "demo_cs", onset, shape, tissue, ridge,
                                  {"n_frames": 20, "bin": 1, "um_per_analysis_px": 2.5})
    st = insp.load_inspector_state(p)
    assert (st["shape"] == shape).all()
    assert (st["onset"] == onset).all()
    assert st["meta"]["n_frames"] == 20


def test_apply_corrections_end_to_end(tmp_path):
    import json, tifffile
    onset, shape, tissue, ridge = _state()
    # a source folder whose frames match the state resolution
    src = tmp_path / "src"; src.mkdir()
    s = _brightening_stack(row=30)
    for t in range(s.shape[0]):
        tifffile.imwrite(src / f"t_{t:04d}.tif", np.clip(s[t], 0, 255).astype(np.uint8))
    run = tmp_path / "run"; run.mkdir()
    ridge[30, :] = 0.5
    insp.save_inspector_state(str(run), "demo_cs", onset, shape, tissue, ridge,
                              {"n_frames": s.shape[0], "bin": 1, "um_per_analysis_px": 2.5})
    doc = {"disp_size": [shape.shape[1], shape.shape[0]],
           "orig_size": [shape.shape[0], shape.shape[1]],
           "edits": [{"op": "add", "points": [[6, 30], [54, 30]], "radius": 1},
                     {"op": "erase", "points": [[30, 20]], "radius": 3}]}
    ep = tmp_path / "edits.json"; ep.write_text(json.dumps(doc))
    out = insp.apply_corrections(str(run), "demo_cs", str(ep), str(src), bin_xy=1)
    assert os.path.exists(out["inspector"])
    assert out["inspector"].endswith("demo_cs_inspect_corrected.html")
    assert os.path.exists(out["curve"])
    html = open(out["inspector"], encoding="utf-8").read()
    assert "data:image/png" in html
