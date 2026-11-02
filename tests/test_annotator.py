"""Ground-truth annotator: self-contained HTML (threshold + brush) over the final frame."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import outputs as cout


def test_annotator_writes_self_contained_html(tmp_path):
    final = (np.arange(16 * 16).reshape(16, 16) % 256).astype(np.float32)
    path = cout.save_annotator(final, str(tmp_path), "demo")
    assert os.path.exists(path) and path.endswith(".html")
    html = open(path).read()
    assert 'id="exp"' in html and 'id="mAdd"' in html          # export + brush
    assert "FIN=dec(" in html and "MSK=dec(" not in html        # raw image only, no tissue layer


def test_final_png_written(tmp_path):
    final = (np.arange(16 * 16).reshape(16, 16) % 256).astype(np.float32)
    path = cout.save_final_png(final, str(tmp_path), "demo")
    assert os.path.exists(path) and path.endswith("_final.png")
