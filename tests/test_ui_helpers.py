"""Increments 5 & 6 — UI data-prep helpers (live HTML is validated on a desktop browser)."""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import outputs as cout


def test_confidence_to_bytes_quantises_and_thresholds():
    conf = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 2.0]], np.float32)  # 2.0 clips to 1
    b = cout.confidence_to_bytes(conf)
    assert b.dtype == np.uint8
    assert b[0, 0] == 0 and b[0, 2] == 255 and b[1, 2] == 255
    assert abs(int(b[0, 1]) - 128) <= 1
    # thresholding the bytes reproduces thresholding the floats
    assert int((b >= 128).sum()) == int((conf >= 0.5).sum())


def test_prepare_annotator_frames_count_scale_and_datauri():
    stack = (np.random.default_rng(0).random((30, 200, 400)) * 255).astype(np.uint8)
    idx, imgs, scale = cout.prepare_annotator_frames(stack, n_frames=8, max_px=100)
    assert len(idx) == len(imgs) == 8
    assert idx[0] == 0 and idx[-1] == 29                   # spans the full stack
    assert all(u.startswith("data:image/jpeg;base64,") for u in imgs)
    assert scale == 100 / 400                              # limited by the larger dim


def test_save_confidence_viewer_writes_html(tmp_path):
    final = (np.random.default_rng(1).random((40, 50)) * 200).astype(np.float32)
    conf = np.clip(np.random.default_rng(2).random((40, 50)), 0, 1).astype(np.float32)
    p = cout.save_confidence_viewer(final, conf, str(tmp_path), "d")
    assert os.path.exists(p) and p.endswith("_confidence.html")
    assert "confidence threshold" in open(p, encoding="utf-8").read()


def test_save_multiframe_annotator_embeds_categories_and_version(tmp_path):
    stack = (np.random.default_rng(3).random((12, 60, 80)) * 255).astype(np.uint8)
    p = cout.save_multiframe_annotator(stack, str(tmp_path), "d",
                                       categories=("lumen", "scale"), version="9.9")
    assert os.path.exists(p) and p.endswith("_multiframe_annotate.html")
    html = open(p, encoding="utf-8").read()
    assert '"lumen"' in html and '"scale"' in html       # categories embedded as JSON
    assert "9.9" in html                                  # source version embedded
