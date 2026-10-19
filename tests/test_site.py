import json
import runpy
import sys
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parent.parent / "site" / "build.py"


def test_site_builds_with_real_data(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build.py", "--out", str(tmp_path)])
    runpy.run_path(str(BUILD), run_name="__main__")
    data = json.loads((tmp_path / "data.json").read_text())
    assert "ag_news" in data
    items = data["ag_news"]["items"]
    assert len(items) == 1000 and items[0][0] >= items[-1][0]  # sorted by confidence
    assert all("#39;" not in it[3] for it in items)
    assert data["ag_news"]["summary"]["held_out"]["0.1"]["feasible"] == 10
    assert "Synthetic data" in (tmp_path / "synthetic-cascade.html").read_text()
    assert (tmp_path / "index.html").exists()


def test_clean_text():
    clean = runpy.run_path(str(BUILD))["_clean"]
    assert clean("CBS #39;s quot;hit quot; a\\\\b") == "CBS's \"hit \" a b"
    assert clean("word " * 100).endswith("…")
