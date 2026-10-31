import json

import pytest
from fastapi.testclient import TestClient

from cascaded.backends.base import DecisionBackend
from cascaded.dashboard import Ledger, render
from cascaded.optimizer import SKIP, Cascade
from cascaded.proxy import make_app
from cascaded.schema import Answer, DecisionResponse

COSTS = [0.01, 1.0, 10.0]


class Fake(DecisionBackend):
    """Picks the first option; confident only if the state mentions one of `sure_of`."""

    def __init__(self, name, sure_of):
        self.name, self.sure_of = name, sure_of

    def _decide(self, request):
        c = 0.95 if any(w in str(request.state) for w in self.sure_of) else 0.55
        answers = {}
        for k, q in request.questions.items():
            first, *rest = q.options
            answers[k] = Answer(probabilities={first: c, **{o: (1 - c) / len(rest) for o in rest}})
        return DecisionResponse(tier=self.name, answers=answers, usage={"input_tokens": 7, "output_tokens": 0})


def client(thresholds=(0.9, 0.9)):
    tiers = [Fake("d1", ["easy"]), Fake("small", ["easy", "medium"]), Fake("large", [])]
    return TestClient(make_app(Cascade(tiers, thresholds), COSTS))


DECIDE = {"questions": {"team": {"type": "choice", "instructions": "Which team?",
                                 "criteria": {"billing": "money", "technical": "bugs"}}}}


def test_decide_routes_and_charges_each_tier_passed():
    c = client()
    for state, tier in [("easy one", "d1"), ("medium one", "small"), ("hard one", "large")]:
        r = c.post("/v1/decide", json={**DECIDE, "state": state})
        assert r.status_code == 200 and r.json()["tier"] == tier
    s = c.get("/stats").json()
    assert s["requests"] == 3
    assert [t["requests"] for t in s["tiers"]] == [1, 1, 1]
    assert s["mean_cost"] == pytest.approx((0.01 + 1.01 + 11.01) / 3)
    assert s["savings"] == pytest.approx(1 - s["mean_cost"] / 10.0)


def test_skipped_tier_is_not_charged():
    c = client(thresholds=(SKIP, 0.9))
    assert c.post("/v1/decide", json={**DECIDE, "state": "easy"}).json()["tier"] == "small"
    assert c.get("/stats").json()["mean_cost"] == pytest.approx(1.0)


def test_decide_rejects_invalid_request():
    bad = {"state": "x", "questions": {"team": {"type": "choice", "instructions": "?", "criteria": {"one": "1"}}}}
    assert client().post("/v1/decide", json=bad).status_code == 422


def openai_body(props, text="easy: I was charged twice"):
    return {"model": "anything", "messages": [{"role": "system", "content": "Route tickets."},
                                              {"role": "user", "content": [{"type": "text", "text": text}]}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "t", "schema": {
                "type": "object", "properties": props}}}}


def test_openai_compatible_typed_task():
    props = {"team": {"type": "string", "enum": ["billing", "technical"], "description": "Which team?"},
             "refund": {"type": "boolean", "description": "Asking for a refund?"}}
    r = client().post("/v1/chat/completions", json=openai_body(props))
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion" and body["model"] == "d1"
    assert json.loads(body["choices"][0]["message"]["content"]) == {"team": "billing", "refund": True}
    assert body["cascade"]["confidence"]["team"] == pytest.approx(0.95)


@pytest.mark.parametrize("body", [
    {"model": "m", "messages": [{"role": "user", "content": "write a poem"}]},             # free-form
    openai_body({"summary": {"type": "string"}}),                                         # not an enum
    openai_body({"team": {"type": "string", "enum": ["only"]}}),                          # 1 option
])
def test_openai_endpoint_rejects_untyped_requests(body):
    assert client().post("/v1/chat/completions", json=body).status_code == 400


def test_duplicate_tier_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        make_app(Cascade([Fake("x", []), Fake("x", [])], (0.9,)), [1.0, 2.0])


def test_dashboard_html():
    c = client()
    assert "No requests yet" in c.get("/dashboard").text
    c.post("/v1/decide", json={**DECIDE, "state": "easy"})
    page = c.get("/dashboard")
    assert page.headers["content-type"].startswith("text/html")
    assert "Saved vs always-large" in page.text and "99.9%" in page.text and "<script" not in page.text


def test_dashboard_escapes_tier_names_and_embeds_calibration():
    from cascaded.calibration import reliability
    ledger = Ledger(["<b>evil</b>"], [1.0])
    ledger.add(0, 1.0, 5.0)
    rel = reliability([Answer(probabilities={"a": 0.9, "b": 0.1})], ["a"])
    page = render(ledger, [{"tier": "d1", "temperature": 1.5, "ece_before": 0.12, "ece_after": 0.02, "reliability": rel}])
    assert "<b>evil</b>" not in page and "&lt;b&gt;evil&lt;/b&gt;" in page
    assert "<svg" in page and "0.120" in page


def test_per_task_costs_in_ledger():
    tiers = [Fake("d1", ["easy"]), Fake("small", ["easy", "medium"]), Fake("large", [])]
    c = TestClient(make_app(Cascade(tiers, {"routing": (0.9, 0.9), "emotion": (SKIP, 0.9)}), COSTS))
    c.post("/v1/decide", json={**DECIDE, "state": "easy", "task": "routing"})   # d1 answers: 0.01
    c.post("/v1/decide", json={**DECIDE, "state": "easy", "task": "emotion"})   # d1 skipped, small: 1.0
    c.post("/v1/decide", json={**DECIDE, "state": "easy", "task": "unknown"})   # straight to large: 10.0
    assert c.get("/stats").json()["mean_cost"] == pytest.approx((0.01 + 1.0 + 10.0) / 3)


def test_openai_metadata_task_is_used():
    tiers = [Fake("d1", ["easy"]), Fake("small", ["easy"]), Fake("large", [])]
    c = TestClient(make_app(Cascade(tiers, {"routing": (0.9, 0.9)}), COSTS))
    body = openai_body({"team": {"type": "string", "enum": ["billing", "technical"]}})
    assert c.post("/v1/chat/completions", json={**body, "metadata": {"task": "routing"}}).json()["model"] == "d1"
    assert c.post("/v1/chat/completions", json=body).json()["model"] == "large"  # no task: not validated


def test_drift_alarm_when_live_coverage_departs_from_validation():
    tiers = [Fake("d1", ["easy"]), Fake("small", ["easy", "medium"]), Fake("large", [])]
    app = make_app(Cascade(tiers, (0.9, 0.9)), COSTS, expected_coverage={None: [0.8, 0.1, 0.1]})
    c = TestClient(app)
    for _ in range(60):  # live traffic is all "hard": d1 answers nothing, unlike validation's 80%
        c.post("/v1/decide", json={**DECIDE, "state": "hard"})
    drift = {d["tier"]: d for d in c.get("/stats").json()["drift"]}
    assert drift["d1"]["alarm"] and drift["d1"]["observed"] == 0.0
    assert "recalibrate" in c.get("/dashboard").text


def test_ledger_persists_across_restarts(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    tiers = [Fake("d1", ["easy"]), Fake("small", []), Fake("large", [])]
    TestClient(make_app(Cascade(tiers, (0.9, 0.9)), COSTS, ledger_path=path)).post(
        "/v1/decide", json={**DECIDE, "state": "easy"})
    again = TestClient(make_app(Cascade(tiers, (0.9, 0.9)), COSTS, ledger_path=path))
    assert again.get("/stats").json()["requests"] == 1


def test_api_key_and_rate_limit():
    tiers = [Fake("d1", ["easy"]), Fake("small", []), Fake("large", [])]
    c = TestClient(make_app(Cascade(tiers, (0.9, 0.9)), COSTS, api_key="s3cret", rate_limit_per_minute=2))
    req = {**DECIDE, "state": "easy"}
    assert c.post("/v1/decide", json=req).status_code == 401
    assert c.get("/dashboard").status_code == 401
    ok = {"Authorization": "Bearer s3cret"}
    assert [c.post("/v1/decide", json=req, headers=ok).status_code for _ in range(3)] == [200, 200, 429]


def test_rate_limit_cannot_be_bypassed_with_fresh_headers():
    tiers = [Fake("d1", ["easy"]), Fake("small", []), Fake("large", [])]
    c = TestClient(make_app(Cascade(tiers, (0.9, 0.9)), COSTS, rate_limit_per_minute=2))
    codes = [c.post("/v1/decide", json={**DECIDE, "state": "easy"}, headers={"Authorization": f"Bearer x{i}"}).status_code
             for i in range(3)]
    assert codes == [200, 200, 429]


def test_oversized_task_rejected_and_ledger_file_bounded(tmp_path):
    tiers = [Fake("d1", ["easy"]), Fake("small", []), Fake("large", [])]
    assert TestClient(make_app(Cascade(tiers, (0.9, 0.9)), COSTS)).post(
        "/v1/decide", json={**DECIDE, "state": "easy", "task": "x" * 65}).status_code == 422
    path = tmp_path / "l.jsonl"
    ledger = Ledger(["a"], [1.0], maxlen=10, path=str(path))
    for _ in range(55):
        ledger.add(0, 1.0, 1.0)
    assert len(path.read_text().splitlines()) < 20 and len(Ledger(["a"], [1.0], maxlen=10, path=str(path)).records) == 10


def test_concurrent_requests_are_safe(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path / "l.jsonl"
    ledger = Ledger(["a"], [1.0], maxlen=20, path=str(path))
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda _: ledger.add(0, 1.0, 1.0), range(400)))
    lines = path.read_text().splitlines()
    assert all(json.loads(line)["tier"] == 0 for line in lines) and len(lines) < 40
    path.write_text(path.read_text() + '{"tier": 0, "cost": 1')  # torn write from a crash
    assert len(Ledger(["a"], [1.0], maxlen=20, path=str(path)).records) == 20

    tiers = [Fake("d1", ["easy"]), Fake("small", []), Fake("large", [])]
    c = TestClient(make_app(Cascade(tiers, (0.9, 0.9)), COSTS, rate_limit_per_minute=1000))
    with ThreadPoolExecutor(16) as pool:
        codes = list(pool.map(lambda _: c.post("/v1/decide", json={**DECIDE, "state": "easy"}).status_code, range(200)))
    assert codes.count(200) == 200
