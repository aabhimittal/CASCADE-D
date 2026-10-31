"""M3: HTTP proxy in front of a Cascade, plus the live cost dashboard.

Endpoints
    POST /v1/decide            native: DecisionRequest in, DecisionResponse out
    POST /v1/chat/completions  OpenAI-compatible, for typed tasks only: the request must carry a
                               json_schema response_format whose properties are string enums
                               (-> choice) or booleans (-> noul). Free-form generation is a
                               non-goal (SPEC §4) and gets a 400.
    GET  /dashboard            static HTML cost dashboard
    GET  /stats                the same numbers as JSON

Run: uvicorn --factory "yourmodule:build_app"  where build_app returns make_app(cascade, costs).
"""

import json
import os
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import ValidationError

from cascaded.dashboard import Ledger, render
from cascaded.optimizer import SKIP, Cascade
from cascaded.schema import DecisionRequest, DecisionResponse


def make_app(cascade: Cascade, costs: list[float], calibration: list[dict] | None = None,
             ledger_path: str | None = None, expected_coverage: dict | None = None,
             api_key: str | None = None, rate_limit_per_minute: int | None = None) -> FastAPI:
    """expected_coverage: {task: Result.coverage} from validation, enables the drift table.
    api_key (or env CASCADED_API_KEY): if set, every endpoint needs "Authorization: Bearer <key>"."""
    if len(costs) != len(cascade.tiers):
        raise ValueError("one cost per tier")
    names = [t.name for t in cascade.tiers]
    if len(set(names)) != len(names):
        raise ValueError(f"tier names must be unique (they identify who answered): {names}")
    ledger = Ledger(names, costs, path=ledger_path, expected=expected_coverage)
    api_key = api_key or os.environ.get("CASCADED_API_KEY")
    recent: dict[str, deque] = defaultdict(deque)
    recent_lock = threading.Lock()  # sync endpoints run on FastAPI's thread pool

    def guard(req: Request) -> None:
        if api_key and not secrets.compare_digest(req.headers.get("authorization", ""), f"Bearer {api_key}"):
            raise HTTPException(401, "missing or wrong API key")
        if rate_limit_per_minute:
            # shortcut: per-process memory; use a shared store (e.g. Redis) when running several workers
            # key on the client address, never on a header the client chooses (a fresh header per request
            # would get a fresh bucket); drop idle buckets so memory stays bounded by active clients
            who, now = req.client.host if req.client else "?", time.monotonic()
            with recent_lock:
                for k in [k for k, v in recent.items() if not v or now - v[-1] > 60]:
                    recent.pop(k, None)
                q = recent[who]
                while q and now - q[0] > 60:
                    q.popleft()
                limited = len(q) >= rate_limit_per_minute
                if not limited:
                    q.append(now)
            if limited:
                raise HTTPException(429, "rate limit exceeded, retry in a minute")

    app = FastAPI(title="CASCADE-D", dependencies=[Depends(guard)])
    app.state.ledger = ledger

    def paid(task: str | None, k: int) -> float:
        """Cost of a request answered by tier k: every tier it passed through (skipped tiers are free)."""
        taus = cascade.thresholds_for(task)
        return sum(c for c, tau in zip(costs[: k + 1], taus) if tau != SKIP)

    def run(request: DecisionRequest) -> DecisionResponse:
        start = time.perf_counter()
        response = cascade.decide(request)
        k = names.index(response.tier)
        ledger.add(k, paid(request.task, k), (time.perf_counter() - start) * 1000, request.task)
        return response

    @app.post("/v1/decide")
    def decide(request: DecisionRequest) -> DecisionResponse:
        return run(request)

    @app.post("/v1/chat/completions")
    def chat_completions(body: dict) -> dict:
        request, booleans = _to_decision_request(body)
        r = run(request)
        content = {k: (a.label == "true") if k in booleans else a.label for k, a in r.answers.items()}
        return {
            "id": f"cascade-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": r.tier,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(content)}}],
            "usage": {"prompt_tokens": r.usage.get("input_tokens", 0),
                      "completion_tokens": r.usage.get("output_tokens", 0)},
            "cascade": {"tier": r.tier, "confidence": {k: a.confidence for k, a in r.answers.items()}},
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        return render(ledger, calibration)

    @app.get("/stats")
    def stats() -> dict:
        return ledger.summary()

    return app


def _to_decision_request(body: dict) -> tuple[DecisionRequest, set[str]]:
    """OpenAI chat body with a json_schema response_format -> DecisionRequest (+ names of boolean fields)."""
    fmt = body.get("response_format") or {}
    schema = (fmt.get("json_schema") or {}).get("schema") or {}
    props = schema.get("properties")
    if fmt.get("type") != "json_schema" or not props:
        raise HTTPException(400, "typed tasks only: send response_format json_schema with enum/boolean properties")
    questions, booleans = {}, set()
    for name, p in props.items():
        instructions = p.get("description") or name
        if p.get("type") == "boolean":
            questions[name] = {"type": "noul", "instructions": instructions}
            booleans.add(name)
        elif isinstance(p.get("enum"), list) and all(isinstance(v, str) for v in p["enum"]):
            questions[name] = {"type": "choice", "instructions": instructions, "criteria": {v: v for v in p["enum"]}}
        else:
            raise HTTPException(400, f"property {name!r}: only string enums and booleans are supported")
    state = "\n\n".join(f"{m.get('role', 'user')}: {_text(m.get('content'))}" for m in body.get("messages", []))
    try:
        task = (body.get("metadata") or {}).get("task")  # OpenAI's free-form metadata field
        return DecisionRequest(state=state, questions=questions, task=task), booleans
    except ValidationError as e:
        raise HTTPException(400, str(e)) from e


def _text(content) -> str:
    if isinstance(content, list):  # multimodal parts: keep the text ones
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return content or ""
