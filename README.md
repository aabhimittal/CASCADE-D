# CASCADE-D

Decision-model-first cascade: d1-3B answers typed questions when confident, escalating to a small then a large LLM. See [SPEC.md](SPEC.md).

**Status: M3**
- M0: typed answer schema (`cascaded/schema.py`), `DecisionBackend` interface, a d1-3B backend and an OpenAI-compatible logprob backend for the LLM tiers.
- M1: `cascaded/calibration.py` — per-tier temperature scaling, ECE, reliability diagrams (SVG). `calibrate(tiers, fit, test)` fits on one split and reports ECE before/after on the other.
- M2: `cascaded/optimizer.py` — replays the cascade over offline tier outputs, grid-searches the cheapest thresholds whose error (plus a Wilson margin) stays ≤ ε, AURC per tier, and a runtime `Cascade` router.
- M3: `cascaded/proxy.py` — FastAPI proxy (`/v1/decide`, OpenAI-compatible `/v1/chat/completions` for enum/boolean structured outputs) and `cascaded/dashboard.py` — static HTML cost dashboard at `/dashboard`.

```bash
pip install -e ".[dev]"        # add ",d1" to run d1-3B locally, ",proxy" to serve
pytest
```

Serve a cascade (thresholds from `optimizer.optimize`):

```python
# app.py
from cascaded.backends import D1Backend, LLMChoiceBackend
from cascaded.optimizer import Cascade
from cascaded.proxy import make_app

tiers = [D1Backend(), LLMChoiceBackend("small-model"), LLMChoiceBackend("large-model")]
# in practice: the CalibratedBackend wrappers returned by calibration.calibrate()
costs = [0.0001, 0.001, 0.01]  # your per-request cost per tier
build_app = lambda: make_app(Cascade(tiers, thresholds=(0.92, 0.85)), costs)
```

```bash
uvicorn --factory app:build_app   # then open http://localhost:8000/dashboard
```
