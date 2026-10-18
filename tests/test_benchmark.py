import json

import numpy as np
import pytest

from cascaded.backends.base import DecisionBackend
from cascaded.benchmark import collect, load, main, render, run
from cascaded.schema import Answer, DecisionRequest, DecisionResponse

NAMES, COSTS = ["d1", "small", "large"], [0.0001, 0.001, 0.01]


def synthetic_rows(n=3000, seed=0):
    """Cache rows for 3 tiers on one yes/no question; the small LLM is overconfident (T≈2)."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(-3, 3, n)
    rows = []
    for i in range(n):
        truth = "true" if rng.uniform() < 0.5 else "false"
        row = {"truth": {"q": truth}, "tiers": {}}
        for name, skill, sharpen, ms in zip(NAMES, (1.0, 2.0, 5.0), (1.0, 2.0, 1.0), (10, 200, 900)):
            p_right = 1 / (1 + np.exp(-(skill - d[i] + rng.normal(0, 0.5))))
            right = rng.uniform() < p_right
            p_label = 1 / (1 + np.exp(-sharpen * np.log(p_right / (1 - p_right))))  # reported confidence
            other = "false" if truth == "true" else "true"
            said, alt = (truth, other) if right else (other, truth)
            row["tiers"][name] = {"answers": {"q": {said: float(max(p_label, 1 - p_label)),
                                                    alt: float(min(p_label, 1 - p_label))}},
                                  "ms": float(ms * rng.uniform(0.8, 1.2))}
        rows.append(row)
    return rows


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


def test_tier_summary_coverage_at_risk():
    from cascaded.benchmark import tier_summary
    rows = [{"truth": {"q": "true"}, "tiers": {"t": {"answers": {"q": {"true": c, "false": 1 - c}}, "ms": 1.0}}}
            for c in (0.99, 0.95, 0.9, 0.85)]
    rows.append({"truth": {"q": "false"}, "tiers": {"t": {"answers": {"q": {"true": 0.6, "false": 0.4}}, "ms": 1.0}}})
    s = tier_summary(rows, "t", risks=(0.0, 0.2))
    assert s["accuracy"] == 0.8
    assert s["coverage_at_risk"] == {0.0: 0.8, 0.2: 1.0}  # the one miss is the least confident
