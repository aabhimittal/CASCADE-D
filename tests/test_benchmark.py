import json

import numpy as np
import pytest

from cascaded.backends.base import DecisionBackend
from cascaded.benchmark import collect, load, main, render, run, synthetic_rows
from cascaded.schema import Answer, DecisionRequest, DecisionResponse

from cascaded.benchmark import SYNTHETIC_COSTS as COSTS, SYNTHETIC_TIERS as NAMES


class Fake(DecisionBackend):
    def __init__(self, name, fail=False):
        self.name, self.fail = name, fail

    def _decide(self, request):
        if self.fail:
            raise ValueError("no option letter among the top logprobs")
        return DecisionResponse(tier=self.name, answers={"q": Answer(probabilities={"true": 0.8, "false": 0.2})})


def test_collect_caches_answers_latency_and_failures(tmp_path):
    path = tmp_path / "cache.jsonl"
    q = {"q": {"type": "noul", "instructions": "?"}}
    collect([Fake("d1"), Fake("small", fail=True)], [(DecisionRequest(state="s", questions=q), {"q": "true"})] * 2, str(path))
    rows = load(str(path))
    assert len(rows) == 2 and rows[0]["truth"] == {"q": "true"}
    assert rows[0]["tiers"]["d1"]["answers"]["q"] == {"true": 0.8, "false": 0.2}
    assert rows[0]["tiers"]["d1"]["ms"] >= 0
    assert "no option letter" in rows[0]["tiers"]["small"]["error"]


@pytest.fixture(scope="module")
def result():
    return run(synthetic_rows(), NAMES, COSTS, epsilons=(0.03, 0.05, 0.1), splits=3)


def test_cascade_meets_target_and_beats_llm_only(result):
    s = result["summary"]
    for eps, p in s["cascade"].items():
        assert p["met_eps"] == 3, f"ε={eps}"
        assert p["error"]["mean"] <= eps
    for eps in set(s["cascade"]) & set(s["llm_only"]):
        assert s["cascade"][eps]["cost_1k"]["mean"] < s["llm_only"][eps]["cost_1k"]["mean"]
    assert s["always"]["large"]["cost_1k"]["mean"] == pytest.approx(10.0)


def test_calibration_and_latency_are_reported(result):
    t = result["summary"]["tiers"]
    assert t["small"]["temperature"]["mean"] == pytest.approx(2.0, rel=0.2)  # overconfident tier cooled
    assert t["small"]["ece"]["mean"] < t["small"]["ece_raw"]["mean"]
    always = result["summary"]["always"]
    assert always["d1"]["p50_ms"]["mean"] < always["small"]["p50_ms"]["mean"] < always["large"]["p50_ms"]["mean"]
    # a request that reached the large tier paid every tier's latency on the way
    p = result["summary"]["cascade"][0.05]
    assert p["p99_ms"]["mean"] > always["large"]["p50_ms"]["mean"]


def test_failed_tier_counts_as_unsure_and_wrong():
    rows = synthetic_rows(400, seed=3)
    for r in rows:
        r["tiers"]["d1"] = {"error": "boom", "ms": 1.0}
    res = run(rows, NAMES, COSTS, epsilons=(0.1,), splits=2)
    assert res["summary"]["tiers"]["d1"]["failures"]["mean"] == 1.0
    assert res["summary"]["cascade"][0.1]["coverage"][0] == 0.0  # never trusted
    assert res["summary"]["always"]["d1"]["error"]["mean"] == 1.0


def test_report_html(result):
    page = render(result, "Bench <x>", note="Synthetic data")
    assert "<svg" in page and "<script" not in page
    assert "Bench &lt;x&gt;" in page and "Synthetic data" in page
    assert "always-large" in page and "LLM-only cascade" in page


def test_cli_writes_report(tmp_path):
    cache = tmp_path / "c.jsonl"
    cache.write_text("\n".join(json.dumps(r) for r in synthetic_rows(600, seed=4)))
    out = tmp_path / "r.html"
    main([str(cache), "--tiers", ",".join(NAMES), "--costs", ",".join(map(str, COSTS)),
          "--out", str(out), "--splits", "2"])
    assert out.read_text().startswith("<!doctype html>")


def test_tier_summary_held_out():
    from cascaded.benchmark import tier_summary
    s = tier_summary(synthetic_rows(1000, seed=6), "large", epsilons=(0.05,), splits=3)
    h = s["held_out"][0.05]
    assert h["feasible"] == 3 and h["worst_error"] <= 0.05
    assert h["coverage"]["mean"] > 0.8 and 0.9 < s["accuracy"] < 1.0


def test_learned_router_baseline_uses_request_text():
    rows = synthetic_rows(2000, seed=5)
    for r in rows:  # make the small LLM's correctness predictable from the text
        ok = next(iter(r["tiers"]["small"]["answers"]["q"].items()))[0] == r["truth"]["q"]
        r["state"] = "easy routine request" if ok else "tricky unusual request"
    res = run(rows, NAMES, COSTS, epsilons=(0.1,), splits=2)
    p = res["summary"]["router"][0.1]
    assert p["met_eps"] == 2 and p["error"]["mean"] <= 0.1
    small_share = p["coverage"][1]
    assert small_share > 0.5  # the router learned to send "easy" text to the cheap model
    assert "Learned router" in render(res, "t")


def test_router_skipped_without_request_text(result):
    assert result["summary"]["router"] == {}
