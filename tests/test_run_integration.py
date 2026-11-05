"""End-to-end: colour_shape.run writes the full v0.2.0 output set for each method."""
import os
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import colour_shape as cs


def _write_stack(folder, T=16, step=6):
    os.makedirs(folder, exist_ok=True)
    base = np.full((48, 64), 20.0, np.float32)
    base[6:42, 6:58] = 120.0                      # tissue body
    for t in range(T):
        f = base.copy()
        if t >= step:
            f[14:19, 12:40] = 220.0               # a tube that fills bright at `step`
        tifffile.imwrite(os.path.join(folder, f"t_{t:04d}.tif"),
                         f.astype(np.uint8))


def _outputs_present(out_dir, stem):
    want = [f"{stem}_cs_params.json", f"{stem}_cs_metadata.json", f"{stem}_cs_curve.csv",
            f"{stem}_cs_proof.png", f"{stem}_cs_viewer.html", f"{stem}_cs_annotate.html",
            f"{stem}_cs_event_lut.png", f"{stem}_cs_onset_table.csv",
            f"{stem}_cs_events_per_frame.tif", f"{stem}_cs_events_cumulative.tif"]
    return {w: os.path.exists(os.path.join(out_dir, w)) for w in want}


def test_run_colour_frangi_full_outputs(tmp_path):
    folder = str(tmp_path / "caviXX")
    _write_stack(folder)
    out = str(tmp_path / "out")
    cs.run(folder, out, bin_xy=1, method="colour-frangi")
    present = _outputs_present(out, "caviXX")
    assert all(present.values()), present
    # TIFF pages == frame count; last cumulative page == accepted mask footprint
    per = tifffile.imread(os.path.join(out, "caviXX_cs_events_per_frame.tif"))
    assert per.shape[0] == 16


def test_run_temporal_writes_onset_table_with_confidence(tmp_path):
    import pandas as pd
    folder = str(tmp_path / "caviYY")
    _write_stack(folder)
    out = str(tmp_path / "out")
    cs.run(folder, out, bin_xy=1, method="temporal-evidence")
    present = _outputs_present(out, "caviYY")
    assert all(present.values()), present
    df = pd.read_csv(os.path.join(out, "caviYY_cs_onset_table.csv"))
    assert len(df) >= 1
    assert "mean_confidence" in df.columns and "onset_time_s" in df.columns
    # the tube fills around frame 6; onset should be near it
    assert df["onset_frame"].min() <= 8


def test_run_can_skip_tiffs_and_diagnostics(tmp_path):
    folder = str(tmp_path / "caviZZ")
    _write_stack(folder)
    out = str(tmp_path / "out")
    cs.run(folder, out, bin_xy=1, method="colour-frangi",
           write_tiffs=False, write_diagnostics=False)
    assert not os.path.exists(os.path.join(out, "caviZZ_cs_events_per_frame.tif"))
    assert not os.path.exists(os.path.join(out, "caviZZ_cs_stages.png"))
    # core outputs still present
    assert os.path.exists(os.path.join(out, "caviZZ_cs_curve.csv"))
