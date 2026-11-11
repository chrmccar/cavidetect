"""Tests for the inspector review overlay (Phase 1, view-only).

Covers the headless/array helpers; the live HTML canvas is a Mac-browser check (the VM has
no display). See docs/superpowers/specs/2026-11-11-inspector-design.md.
"""
import numpy as np
import pytest

from cavidetect import inspector as insp


# ---- label_tubes: per-rhizoid identity labelling ---------------------------------

def test_label_tubes_single_line_one_label():
    m = np.zeros((20, 40), bool)
    m[10, 5:35] = True              # a single straight tube
    seg = insp.label_tubes(m)
    assert seg.shape == m.shape
    assert (seg[~m] == 0).all()     # background stays 0
    assert (seg[m] > 0).all()       # every tube pixel labelled
    assert len(set(seg[m].tolist())) == 1   # one rhizoid


def test_label_tubes_crossing_splits_into_branches():
    # a '+' crossing: removing the junction must give more than one segment
    m = np.zeros((40, 40), bool)
    m[20, 2:38] = True
    m[2:38, 20] = True
    seg = insp.label_tubes(m)
    assert (seg[m] > 0).all()
    assert seg.max() >= 2           # the junction split it into branches


def test_label_tubes_empty_mask():
    m = np.zeros((12, 12), bool)
    seg = insp.label_tubes(m)
    assert seg.shape == m.shape
    assert seg.max() == 0


def test_label_tubes_tiny_blob_connected_component_fallback():
    # too small to skeletonise into branches -> every pixel still gets a label
    m = np.zeros((12, 12), bool)
    m[5:7, 5:7] = True
    seg = insp.label_tubes(m)
    assert (seg[m] > 0).all()
    assert (seg[~m] == 0).all()


def test_label_tubes_two_separate_tubes_distinct_labels():
    m = np.zeros((30, 40), bool)
    m[8, 2:38] = True
    m[22, 2:38] = True
    seg = insp.label_tubes(m)
    assert len(set(seg[m].tolist())) == 2


# ---- lossless u16 PNG map encode/decode round-trip -------------------------------

def test_encode_decode_u16_png_roundtrip():
    rng = np.random.default_rng(0)
    vals = rng.integers(0, 4000, size=(16, 24)).astype(np.int32)
    valid = rng.random((16, 24)) > 0.3
    b64 = insp._encode_u16_png(vals, valid)
    assert b64.startswith("data:image/png;base64,")
    got_vals, got_valid = insp._decode_u16_png(b64)
    assert (got_valid == valid).all()
    # values only need to match where valid (invalid pixels are don't-care)
    assert (got_vals[valid] == vals[valid]).all()


def test_encode_u16_png_handles_large_frame_numbers():
    vals = np.full((4, 4), 405, np.int32)      # >256 needs the high byte
    valid = np.ones((4, 4), bool)
    got, _ = insp._decode_u16_png(insp._encode_u16_png(vals, valid))
    assert (got == 405).all()


# ---- prepare_inspector_frames ----------------------------------------------------

def _synthetic_stack(T=10, Y=18, X=30):
    rng = np.random.default_rng(1)
    s = rng.normal(40, 3, size=(T, Y, X)).astype(np.float32)
    # one tube brightens partway through
    for t in range(T):
        if t >= 5:
            s[t, 9, 5:25] += 120
    return s


def test_prepare_inspector_frames_counts_and_size():
    s = _synthetic_stack()
    tps, raw, diff, newsize = insp.prepare_inspector_frames(s, max_px=16, n_timepoints=6)
    assert len(tps) == len(raw) == len(diff)
    assert all(x.startswith("data:image/jpeg") for x in raw + diff)
    assert newsize[0] <= 16 and newsize[1] <= 16


# ---- save_inspector end to end (file content) ------------------------------------

def _result_for(s):
    from cavidetect import colour_shape as cs
    shape = np.zeros(s.shape[1:], bool)
    shape[9, 5:25] = True
    onset = np.full(s.shape[1:], -1, np.int32)
    onset[9, 5:25] = 5
    return dict(final=s[-1], shape=shape, onset=onset, tissue=np.ones(s.shape[1:], bool))


def test_save_inspector_writes_html_with_both_modes(tmp_path):
    s = _synthetic_stack()
    path = insp.save_inspector(s, _result_for(s), str(tmp_path), "demo_cs",
                               max_px=24, n_timepoints=6)
    assert path.endswith("demo_cs_inspect.html")
    html = open(path, encoding="utf-8").read()
    assert "by time" in html and "by tube" in html
    assert "data:image/jpeg" in html          # raw/diff frames embedded
    assert "data:image/png" in html           # onset + tube-id maps embedded
    assert "raw" in html and "difference" in html and "conduits" in html


def test_save_inspector_survives_empty_mask(tmp_path):
    s = _synthetic_stack()
    r = _result_for(s)
    r["shape"] = np.zeros(s.shape[1:], bool)
    r["onset"] = np.full(s.shape[1:], -1, np.int32)
    path = insp.save_inspector(s, r, str(tmp_path), "empty_cs", max_px=24, n_timepoints=5)
    assert open(path, encoding="utf-8").read().count("data:image/png") >= 2
