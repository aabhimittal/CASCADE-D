# CASCADE-D

Decision-model-first cascade: d1-3B answers typed questions when confident, escalating to a small then a large LLM. See [SPEC.md](SPEC.md).

**Status: M2**
- M0: typed answer schema (`cascaded/schema.py`), `DecisionBackend` interface, a d1-3B backend and an OpenAI-compatible logprob backend for the LLM tiers.
- M1: `cascaded/calibration.py` — per-tier temperature scaling, ECE, reliability diagrams (SVG). `calibrate(tiers, fit, test)` fits on one split and reports ECE before/after on the other.
- M2: `cascaded/optimizer.py` — replays the cascade over offline tier outputs, grid-searches the cheapest thresholds whose error (plus a Wilson margin) stays ≤ ε, AURC per tier, and a runtime `Cascade` router.

```bash
pip install -e ".[dev]"        # add ",d1" to run d1-3B locally
pytest
```
