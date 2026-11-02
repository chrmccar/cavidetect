"""The interactive embolism viewer: a self-contained HTML scrubber (raw | overlay | curve)."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import outputs as cout, embolism as cemb


def _tiny_result():
    T, Y, X = 12, 16, 16
    s = np.full((T, Y, X), 100.0, np.float32)
    tissue = np.zeros((Y, X), bool); tissue[2:14, 2:14] = True
    s[6:, 8, 8] += 50  # one pixel air-fills at frame 6
    r = cemb.measure_embolism(s, sigma_dn=1.0, tissue_mask=tissue,
                              base_frames=3, change_min_dn=15.0, k_sigma=5.0)
    return s, r


def test_viewer_writes_self_contained_html(tmp_path):
    s, r = _tiny_result()
    path = cout.save_embolism_viewer(s, r, str(tmp_path), "demo")
    assert os.path.exists(path) and path.endswith(".html")
    html = open(path).read()
    # self-contained: embedded frame data + the three panels + a slider
    assert "RAW=" in html and "OV=" in html and "CUM=" in html
    assert 'id="sld"' in html
    assert "data:image/jpeg;base64," in html
