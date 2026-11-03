"""Tests for the colour-shape CLI method/params options."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cavidetect import cli


def test_list_methods_prints_all(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["colour-shape", "x", "--out", "o", "--list-methods"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "colour-frangi" in out and "shape-filter" in out


def test_unknown_method_exits_2(capsys, monkeypatch):
    monkeypatch.setattr("cavidetect.colour_shape.run", lambda *a, **k: None)
    with pytest.raises(SystemExit) as e:
        cli.main(["colour-shape", "folder", "--out", "o", "--method", "nope"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "colour-frangi" in err


def test_bad_params_json_exits_2(tmp_path, monkeypatch):
    monkeypatch.setattr("cavidetect.colour_shape.run", lambda *a, **k: None)
    bad = tmp_path / "bad.json"
    bad.write_text("not json {{{")
    with pytest.raises(SystemExit) as e:
        cli.main(["colour-shape", "folder", "--out", "o", "--params", str(bad)])
    assert e.value.code == 2


def test_params_and_method_forwarded(tmp_path, monkeypatch):
    captured = {}

    def fake_run(folder, out, **kw):
        captured["folder"] = folder
        captured["out"] = out
        captured.update(kw)

    monkeypatch.setattr("cavidetect.colour_shape.run", fake_run)
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"bright_pct": 70}))
    cli.main(["colour-shape", "folder", "--out", "o",
              "--method", "shape-filter", "--params", str(good)])
    assert captured["method"] == "shape-filter"
    assert captured["params"] == {"bright_pct": 70}
