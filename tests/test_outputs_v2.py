"""Increment 7 — outputs: onset table, event-time LUT, run metadata, TIFF suite."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import outputs as cout


def _two_events():
    """A 20x30 field with two separate accepted regions filling at frames 5 and 12."""
    onset = np.full((20, 30), -1, np.int32)
    shape = np.zeros((20, 30), bool)
    conf = np.zeros((20, 30), np.float32)
    trans = np.zeros((20, 30), np.int32)
    shape[3:7, 4:10] = True; onset[3:7, 4:10] = 5; conf[3:7, 4:10] = 0.8; trans[3:7, 4:10] = 1
    shape[12:15, 18:26] = True; onset[12:15, 18:26] = 12; conf[12:15, 18:26] = 0.4; trans[12:15, 18:26] = 2
    return onset, shape, conf, trans


def test_onset_table_one_row_per_region_sorted():
    onset, shape, conf, trans = _two_events()
    df = cout.build_onset_table(onset, shape, conf, trans, scale_um2=4.0, frame_interval_s=15.0)
    assert len(df) == 2
    assert list(df["onset_frame"]) == [5, 12]                   # sorted by onset
    r0 = df.iloc[0]
    assert r0["area_px"] == 24 and r0["area_um2"] == 96.0       # 24 px * 4 µm²
    assert r0["onset_time_s"] == 75.0                            # 5 * 15 s
    assert abs(r0["mean_confidence"] - 0.8) < 1e-6
    assert set(df["transition_frames"]) == {1, 2}


def test_onset_table_empty_when_nothing_accepted():
    onset = np.full((10, 10), -1, np.int32)
    shape = np.zeros((10, 10), bool)
    df = cout.build_onset_table(onset, shape)
    assert len(df) == 0


def test_write_onset_table_and_lut_and_metadata(tmp_path):
    onset, shape, conf, trans = _two_events()
    final = np.random.default_rng(0).random((20, 30)).astype(np.float32) * 200
    p1 = cout.write_onset_table(onset, shape, str(tmp_path), "d", confidence=conf, transition=trans)
    p2 = cout.save_event_lut(onset, shape, final, str(tmp_path), "d", n_frames=20)
    p3 = cout.write_run_metadata(str(tmp_path), "d",
                                 {"method": "temporal-evidence", "um_per_px": 2.5,
                                  "frame_interval_s": 15, "version": "x"})
    for p in (p1, p2, p3):
        assert os.path.exists(p) and os.path.getsize(p) > 0
    assert p1.endswith("_onset_table.csv")
    assert p2.endswith("_event_lut.png")
    assert p3.endswith("_metadata.json")


def test_ambiguous_flag_when_onset_spread_exceeds_transition():
    onset = np.full((10, 10), -1, np.int32)
    shape = np.zeros((10, 10), bool)
    shape[2:8, 2:8] = True
    # same region, pixels filling across frames 4..9 (spread 5) but transition recorded as 1
    rng = np.arange(2, 8)
    for i, y in enumerate(rng):
        onset[y, 2:8] = 4 + i
    trans = np.where(shape, 1, 0).astype(np.int32)
    df = cout.build_onset_table(onset, shape, transition=trans)
    assert bool(df.iloc[0]["ambiguous"]) is True
