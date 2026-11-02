"""Direct cumulative-embolism measurement: the PRIMARY readout.

Unlike the strict event detector, this must count tissue that air-fills *gradually*,
not only clean abrupt steps — that was the cause of the whole-image under-counting.
It records, per tissue pixel, whether it air-filled and the frame it did so (half-rise),
and accumulates that into embolised-area-over-time."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import embolism as cemb


def _synthetic():
    """20x20 tissue on a quiet background; one ABRUPT and one GRADUAL air-filling pixel."""
    T, Y, X = 60, 20, 20
    rng = np.random.default_rng(0)
    base = 100.0
    s = np.full((T, Y, X), base, np.float32) + rng.normal(0, 1.0, (T, Y, X)).astype(np.float32)
    tissue = np.zeros((Y, X), bool); tissue[2:18, 2:18] = True
    # abrupt step at frame 20 (+40)
    s[20:, 5, 5] += 40
    # gradual ramp over frames 10..40 (+40 total) — the strict step test would reject this
    for i, t in enumerate(range(10, 40)):
        s[t, 10, 10] = base + 40 * (i + 1) / 30
    s[40:, 10, 10] = base + 40
    return s, tissue


def test_detects_both_gradual_and_abrupt_airfilling():
    s, tissue = _synthetic()
    r = cemb.measure_embolism(s, sigma_dn=1.0, tissue_mask=tissue,
                              base_frames=8, change_min_dn=15.0, k_sigma=5.0)
    assert r.embolised[5, 5], "abrupt air-filling missed"
    assert r.embolised[10, 10], "GRADUAL air-filling missed (the core bug)"
    assert not r.embolised[3, 3], "quiet tissue pixel wrongly flagged"


def test_background_outside_mask_never_embolised():
    s, tissue = _synthetic()
    s[20:, 0, 0] += 80  # a big change OUTSIDE the tissue mask
    r = cemb.measure_embolism(s, sigma_dn=1.0, tissue_mask=tissue,
                              base_frames=8, change_min_dn=15.0, k_sigma=5.0)
    assert not r.embolised[0, 0], "change outside tissue mask must not count"


def test_onset_frames_are_reasonable():
    s, tissue = _synthetic()
    r = cemb.measure_embolism(s, sigma_dn=1.0, tissue_mask=tissue,
                              base_frames=8, change_min_dn=15.0, k_sigma=5.0)
    assert abs(int(r.onset_frame[5, 5]) - 20) <= 2, "abrupt onset should be ~frame 20"
    assert 20 <= int(r.onset_frame[10, 10]) <= 30, "gradual half-rise should be mid-ramp"


def test_cumulative_area_monotonic_and_counts():
    s, tissue = _synthetic()
    r = cemb.measure_embolism(s, sigma_dn=1.0, tissue_mask=tissue,
                              base_frames=8, change_min_dn=15.0, k_sigma=5.0)
    assert r.cum_area.shape[0] == s.shape[0]
    assert np.all(np.diff(r.cum_area) >= 0), "cumulative embolised area must never decrease"
    assert int(r.cum_area[-1]) == 2, "exactly two tissue pixels air-fill"
    assert abs(r.frac_embolised[-1] - 2 / tissue.sum()) < 1e-9
