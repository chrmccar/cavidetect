"""Behavioural tests for the per-rhizoid cavitation detector (``rhizoid-events``).

The design: trace individual rhizoids in a late-tail reference, read each
strand's brightness backwards, and call cavitation only when the WHOLE strand
jumps in 1-2 frames (partial/gradual = drying, not counted). These synthetic
stacks encode the behaviours the expert asked for.
"""
import numpy as np
import pytest

from cavidetect import rhizoid_events as re, methods as M, colour_shape as cs


# --------------------------------------------------------------------------- #
# synthetic builders
# --------------------------------------------------------------------------- #

def _base(T=24, H=120, W=200, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    tissue = np.zeros((H, W), bool)
    tissue[10:110, 10:190] = True
    stack = np.full((T, H, W), 10.0, np.float32)     # dark background
    stack[:, tissue] = 55.0                           # dim tissue body
    stack += rng.normal(0, noise, stack.shape).astype(np.float32)
    return stack, tissue


def _add_tube(stack, rows, cols, base=70.0, bright=215.0, jump=None, ramp=None):
    """A tube pixel block. ``base`` keeps it faintly tube-visible before filling.
    ``jump`` = one-frame air-fill at that frame; ``ramp`` = (start,end) gradual."""
    T = stack.shape[0]
    region = (slice(*rows), slice(*cols))
    stack[:, region[0], region[1]] = base
    if jump is not None:
        stack[jump:, region[0], region[1]] = bright
    elif ramp is not None:
        s, e = ramp
        for t in range(T):
            f = 0.0 if t < s else min(1.0, (t - s) / max(1, e - s))
            stack[t, region[0], region[1]] = base + f * (bright - base)
    return stack


# --------------------------------------------------------------------------- #
# 1. whole-strand one-frame jump -> one cavitation at the right frame
# --------------------------------------------------------------------------- #

def test_whole_strand_jump_is_one_cavitation():
    stack, tissue = _base()
    _add_tube(stack, (59, 62), (40, 160), jump=14)
    r = re.detect(stack[-1], tissue, stack=stack)
    cav = [e for e in r["events"] if e["cls"] == "cavitation"]
    assert len(cav) == 1, r["events"]
    assert abs(cav[0]["cavitation_frame"] - 14) <= 1
    assert re.count_curve(r["events"], stack.shape[0])[-1] == 1


# --------------------------------------------------------------------------- #
# 2. partial-strand brightening -> drying, not counted
# --------------------------------------------------------------------------- #

def test_partial_strand_is_drying_not_counted():
    stack, tissue = _base()
    # full tube faintly visible; only the LEFT half air-fills
    _add_tube(stack, (59, 62), (40, 160), base=70.0)
    stack[14:, 59:62, 40:98] = 215.0                 # left half jumps, right half stays 70
    r = re.detect(stack[-1], tissue, stack=stack)
    assert re.count_curve(r["events"], stack.shape[0])[-1] == 0
    assert any(e["cls"] == "drying" for e in r["events"]), r["events"]


# --------------------------------------------------------------------------- #
# 3. gradual whole-strand rise -> drying, not counted
# --------------------------------------------------------------------------- #

def test_gradual_whole_strand_is_drying():
    stack, tissue = _base()
    _add_tube(stack, (59, 62), (40, 160), base=70.0, ramp=(4, 20))  # slow fill over 16 frames
    r = re.detect(stack[-1], tissue, stack=stack)
    assert re.count_curve(r["events"], stack.shape[0])[-1] == 0
    assert any(e["cls"] == "drying" for e in r["events"]), r["events"]


# --------------------------------------------------------------------------- #
# 4/5. divergence-merge geometry (unit-tested on label maps)
# --------------------------------------------------------------------------- #

def test_merge_joins_collinear_fragments():
    # one straight horizontal strand broken into two collinear fragments with a gap
    seg = np.zeros((40, 80), np.int32)
    seg[20, 10:35] = 1
    seg[20, 40:70] = 2
    merged = re._merge_collinear(seg, gap_max=10.0)
    assert merged.max() == 1


def test_merge_keeps_diverging_branches_separate():
    # a Y: a trunk that forks into two branches diverging at a wide angle.
    seg = np.zeros((80, 80), np.int32)
    for i in range(30):                              # trunk, horizontal, label 1
        seg[40, 5 + i] = 1
    for i in range(30):                              # branch up-right, label 2
        seg[40 - i, 35 + i] = 2
    for i in range(30):                              # branch down-right, label 3
        seg[40 + i, 35 + i] = 3
    merged = re._merge_collinear(seg, gap_max=6.0, angle_tol_deg=35.0)
    # trunk fuses with at most one branch; the two diverging branches never fuse
    # into a single strand -> at least two distinct rhizoids remain.
    assert merged.max() == 2


# --------------------------------------------------------------------------- #
# 6. max_frames windowing bounds the analysis
# --------------------------------------------------------------------------- #

def test_max_frames_windows_the_stack():
    stack, tissue = _base(T=60)
    _add_tube(stack, (59, 62), (40, 160), jump=12)
    # a second, later event beyond the window should be invisible
    _add_tube(stack, (80, 83), (40, 160), jump=48)
    r = re.detect(stack[-1], tissue, stack=stack, max_frames=20)
    # only the within-window event is counted; onset frames are < 20
    frames = [e["cavitation_frame"] for e in r["events"] if e["cls"] == "cavitation"]
    assert frames and all(f < 20 for f in frames)


# --------------------------------------------------------------------------- #
# 7. per-rhizoid recompute after an edit changes the count
# --------------------------------------------------------------------------- #

def test_recompute_events_after_split_changes_count():
    stack, tissue = _base()
    _add_tube(stack, (59, 62), (40, 160), jump=14)
    r = re.detect(stack[-1], tissue, stack=stack)
    tubes = r["tubes"].copy()
    # emulate a user SPLIT: relabel the right half of the (single) strand as a new id
    tid = int(tubes.max())
    right = tubes.copy()
    k = tid + 1
    right[:, 100:][tubes[:, 100:] == tid] = k
    ev2, onset2, shape2 = re.recompute_events(right, stack)
    cav2 = [e for e in ev2 if e["cls"] == "cavitation"]
    assert len(cav2) == 2                              # the one strand became two cavitations


# --------------------------------------------------------------------------- #
# 8. registry wiring
# --------------------------------------------------------------------------- #

def test_method_registered_with_schema():
    assert "rhizoid-events" in dict(M.list_methods())
    m = M.get_method("rhizoid-events")
    assert m.needs_stack
    p = M.resolve_params("rhizoid-events", {"bright_frac_thresh": 0.6, "max_frames": 50})
    assert p["bright_frac_thresh"] == pytest.approx(0.6)
    assert p["max_frames"] == 50
    assert p["divergence_merge"] is True


# --------------------------------------------------------------------------- #
# 9. regression: other detectors untouched by import side-effects
# --------------------------------------------------------------------------- #

def test_colour_frangi_still_registered_and_default():
    names = [n for n, _ in M.list_methods()]
    assert names[0] == "colour-frangi"           # default stays first
    assert not M.get_method("colour-frangi").needs_stack


# --------------------------------------------------------------------------- #
# 11. one over-segmented tube is counted once (coincident-cavitation merge)
# --------------------------------------------------------------------------- #

def test_coincident_cavitations_merge_to_one_event():
    tubes = np.zeros((40, 80), np.int32)
    tubes[20, 10:40] = 1               # two touching fragments of ONE tube...
    tubes[20, 41:70] = 2               # ...both cavitating on the same frame
    tubes[5, 10:70] = 3                # a separate tube, different frame
    events = [
        {"id": 1, "cls": "cavitation", "cavitation_frame": 12, "length_px": 30,
         "bright_frac_peak": 0.9, "confidence": 0.8},
        {"id": 2, "cls": "cavitation", "cavitation_frame": 12, "length_px": 29,
         "bright_frac_peak": 0.9, "confidence": 0.8},
        {"id": 3, "cls": "cavitation", "cavitation_frame": 25, "length_px": 60,
         "bright_frac_peak": 0.9, "confidence": 0.8},
    ]
    _, merged = re.merge_coincident_cavitations(tubes, events, step_window=2, dilate=3)
    cav = [e for e in merged if e["cls"] == "cavitation"]
    assert len(cav) == 2                # the touching same-frame pair fused; the far one stayed
    assert re.count_curve(merged, 40)[-1] == 2


# --------------------------------------------------------------------------- #
# 12. full correction loop: run -> split edit -> corrected per-rhizoid count
# --------------------------------------------------------------------------- #

def test_correction_loop_recomputes_rhizoid_count(tmp_path):
    import os, json, tifffile, pandas as pd
    from cavidetect import inspector as insp
    src = tmp_path / "cavi_syn"; src.mkdir()
    out = tmp_path / "out"; out.mkdir()
    T, H, W = 20, 100, 200
    rng = np.random.default_rng(3)
    tissue = np.zeros((H, W), bool); tissue[10:90, 10:190] = True
    stack = np.full((T, H, W), 12.0, np.float32); stack[:, tissue] = 120.0
    stack += rng.normal(0, 2.0, stack.shape).astype(np.float32)
    stack[:, 48:51, 30:170] = 132.0; stack[8:, 48:51, 30:170] = 225.0   # one whole tube, cav @8
    for t in range(T):
        tifffile.imwrite(str(src / f"t_{t:04d}.tif"), np.clip(stack[t], 0, 255).astype(np.uint8))

    cs.run(str(src), str(out), bin_xy=1, method="rhizoid-events")
    auto = pd.read_csv(out / "cavi_syn_cs_rhizoid_count.csv")["n_cavitated_cumulative"].iloc[-1]
    assert int(auto) == 1

    # split the single strand at its midpoint (display coords == analysis coords)
    doc = {"disp_size": [W, H], "orig_size": [H, W],
           "edits": [{"op": "split", "point": [100, 49]}]}
    edits_path = tmp_path / "edits.json"
    edits_path.write_text(json.dumps(doc))
    res = insp.apply_corrections(str(out), "cavi_syn_cs", str(edits_path), str(src), bin_xy=1)
    corr = pd.read_csv(out / "cavi_syn_cs_corrected_rhizoid_count.csv")
    assert int(corr["n_cavitated_cumulative"].iloc[-1]) == 2       # split -> two counted events
    assert res["n_cavitated"] == 2


# --------------------------------------------------------------------------- #
# 10. empty / degenerate input does not crash
# --------------------------------------------------------------------------- #

def test_empty_tissue_returns_empty_without_crash():
    stack = np.zeros((10, 60, 60), np.float32)       # no tissue at all
    r = re.detect(stack[-1], np.zeros((60, 60), bool), stack=stack)
    assert r["events"] == []
    assert r["shape"].sum() == 0
