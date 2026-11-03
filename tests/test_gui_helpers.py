"""Unit tests for the tkinter-free helpers in cavidetect.gui.

These exercise the command-building, summary parsing, curve reading and artefact
naming that the GUI relies on, without needing a Tk display — so they run on the
VM as well as the Mac."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import gui


def test_build_command_auto_air_omits_flag():
    cmd = gui.build_command("/data/cavi600005", "/out/cavi600005",
                            bin_xy=2, air=None, tail_frac=0.12,
                            python_exe="python3")
    assert cmd[:4] == ["python3", "-m", "cavidetect.cli", "colour-shape"]
    assert "/data/cavi600005" in cmd
    assert cmd[cmd.index("--out") + 1] == "/out/cavi600005"
    assert cmd[cmd.index("--bin") + 1] == "2"
    assert cmd[cmd.index("--tail-frac") + 1] == "0.12"
    assert "--air" not in cmd


def test_build_command_manual_air_adds_flag():
    cmd = gui.build_command("/d", "/o", bin_xy=1, air=142.0, tail_frac=0.1,
                            python_exe="py")
    assert cmd[cmd.index("--air") + 1] == "142.0"
    assert cmd[cmd.index("--bin") + 1] == "1"


def test_build_command_dev_uses_module():
    cmd = gui.build_command("/d", "/o", python_exe="py", frozen=False)
    assert cmd[:4] == ["py", "-m", "cavidetect.cli", "colour-shape"]


def test_build_command_frozen_omits_module():
    # a PyInstaller build invokes the one bundled exe with the subcommand directly
    cmd = gui.build_command("/d", "/o", python_exe="/app/cavidetect", frozen=True)
    assert cmd[:2] == ["/app/cavidetect", "colour-shape"]
    assert "-m" not in cmd


def test_build_command_includes_method_and_params():
    cmd = gui.build_command("/d", "/o", method="shape-filter",
                            params_path="/tmp/p.json", python_exe="py")
    assert cmd[cmd.index("--method") + 1] == "shape-filter"
    assert cmd[cmd.index("--params") + 1] == "/tmp/p.json"


def test_build_command_default_method_backcompat():
    cmd = gui.build_command("/d", "/o", python_exe="py")
    assert cmd[:4] == ["py", "-m", "cavidetect.cli", "colour-shape"]
    assert "/d" in cmd and cmd[cmd.index("--out") + 1] == "/o"
    assert cmd[cmd.index("--method") + 1] == "colour-frangi"
    assert "--params" not in cmd            # omitted when no params file


def test_params_from_fields_coerces():
    p = gui.params_from_fields("shape-filter", {"bright_pct": "70"})
    assert p["bright_pct"] == 70


def test_write_params_tempfile_roundtrips():
    import json
    path = gui.write_params_tempfile({"bright_pct": 70, "contrast": "gamma"})
    try:
        with open(path, encoding="utf-8") as fh:
            assert json.load(fh) == {"bright_pct": 70, "contrast": "gamma"}
    finally:
        os.remove(path)


def test_build_bundle_zips_results_and_extras(tmp_path):
    out = tmp_path / "cavi600005"
    out.mkdir()
    (out / "cavi600005_cs_curve.csv").write_text("frame,frac_tissue_embolised\n0,0.1\n")
    ann = tmp_path / "my_annotation.png"
    ann.write_bytes(b"\x89PNG\r\n\x1a\n")  # a stand-in PNG
    zip_path = tmp_path / "cavi600005_report.zip"
    gui.build_bundle(str(out), str(zip_path), extra_files=[str(ann)],
                     info_text="dataset: cavi600005\n")
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    assert "cavi600005/cavi600005_cs_curve.csv" in names
    assert "cavi600005/annotations/my_annotation.png" in names
    assert "cavi600005/info.txt" in names


def test_parse_summary_line():
    line = "air=142.0  colour=21.3%  colour+shape=18.0% of tissue"
    s = gui.parse_summary_line(line)
    assert s == {"air": 142.0, "colour_pct": 21.3, "shape_pct": 18.0}


def test_parse_summary_line_ignores_other_lines():
    assert gui.parse_summary_line("loaded 404 frames (960, 1280) in 38s") is None
    assert gui.parse_summary_line("") is None


def test_final_fraction_reads_last_row(tmp_path):
    csv_path = tmp_path / "cavi600005_cs_curve.csv"
    csv_path.write_text(
        "frame,embolised_area_px,frac_tissue_embolised\n"
        "0,0,0.0\n"
        "1,50,0.05\n"
        "2,180,0.179\n")
    assert gui.final_fraction(str(csv_path)) == 0.179


def test_final_fraction_missing_file():
    assert gui.final_fraction("/no/such/curve.csv") is None


def test_stem_and_output_dir():
    assert gui.stem_of("/data/cavi600005/") == "cavi600005"
    assert gui.output_dir("/repo/results_whole", "/data/cavi600005") == \
        os.path.join("/repo/results_whole", "cavi600005")


def test_artefact_paths():
    p = gui.artefact_paths("/out/cavi600005", "cavi600005")
    assert p["viewer"].endswith("cavi600005_cs_viewer.html")
    assert p["curve"].endswith("cavi600005_cs_curve.csv")
    assert p["proof"].endswith("cavi600005_cs_proof.png")
    assert os.path.dirname(p["viewer"]) == "/out/cavi600005"
