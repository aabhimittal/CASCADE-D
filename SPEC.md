# CASCADE-D

## 1. Thesis

Most routers ask "which LLM should answer?". CASCADE-D asks a cheaper question first: "does this need generation at all?". A decision model answers classify/route/score requests directly when it is confident; only the remainder escalates to a small LLM, then a large one. Thresholds come from risk-coverage curves, so you choose a target error rate and get the cheapest cascade that meets it.

Analogy: a hospital triage desk. A nurse resolves the clear cases in seconds, sends uncertain ones to a GP, and only the hard ones reach a specialist. The system works because the nurse knows when they're unsure.

## 2. Why now

- Decision models return calibrated typed answers with zero output tokens (d1-3B open weights; Jev hosted). TypeSafe frames them as "smart if-statements" inside workflows.
- Large price gaps between model tiers make cascades worth engineering.
- LangChain published "How to Build a Model Router in the Harness" (Oct 2026): routing is now a recognized harness component.

## 3. Prior art & differentiation

RouteLLM, NotDiamond, Martian and harness-level routers pick among generative models using a learned router. CASCADE-D differs in two ways: the bottom tier is a non-generative decision model that answers directly, and escalation uses calibrated confidence with an explicit risk-coverage target rather than a router's preference score.

## 4. Scope

- MVP: three tiers (d1-3B → small LLM → large LLM) for typed decision tasks (classify, route, score, extract-choice).
- v1: threshold optimizer, cost dashboard, OpenAI-compatible proxy mode for typed tasks.
- Non-goals: free-form generation tasks (they skip tier 0 entirely).

## 5. Architecture

```
request (state + typed questions)
   ▼
Tier 0: decision model ── conf ≥ τ0 ──▶ answer
   │ else
Tier 1: small LLM via structured-decision adapter ── conf ≥ τ1 ──▶ answer
   │ else
Tier 2: large LLM ──▶ answer (+ optional human flag)
```

Every tier returns the same typed answer schema with a probability, so tiers are interchangeable and comparable.

## 6. Core method

For tier k with confidence c_k(x), selective risk at threshold τ:
R_k(τ) = P(wrong | c_k ≥ τ), coverage Cov_k(τ) = P(c_k ≥ τ).

Expected cost per request for thresholds (τ0, τ1):
C = c0 + (1 − Cov0)·c1 + (1 − Cov0)(1 − Cov1|rest)·c2

Optimize C subject to overall error ≤ ε on a validation set (grid search over τ is sufficient; tiers ≤ 3). Report AURC (area under risk-coverage curve) per tier.

Important: LLM self-reported confidence is usually poorly calibrated. For tiers 1–2 use token logprobs over constrained answer options where the API exposes them, or TypeSafe's open-source System One adapter approach; fit temperature scaling on validation data.

## 7. Evaluation

- Tasks: public labeled classification/routing datasets with human labels; optionally TypeSafe's published workflow evals (check license).
- Metrics: cost per 1k requests, end-to-end error, p50/p99 latency, coverage per tier.
- Baselines: always-large, always-small, RouteLLM-style router, LLM-self-confidence cascade.
- Headline: cost vs error Pareto curve, with each baseline as a point.

## 8. Milestones

- **M0** Typed answer schema + backend interface + three backends. Acceptance: same request runs through each tier and returns schema-valid output.
- **M1** Calibration module (temperature scaling, reliability diagrams). Acceptance: ECE reported per tier on validation.
- **M2** Threshold optimizer. Acceptance: chosen thresholds meet ε on held-out test across 10 splits.
- **M3** Proxy mode + cost dashboard (static HTML).
- **M4** Benchmark report.

## 9. Repo layout & stack

`cascaded/ {schema.py, backends/, calibration.py, optimizer.py, proxy.py}`, Python, pydantic, numpy, FastAPI.

## 10. Risks & honest weaknesses

- If tier 0 coverage is low on your tasks, the cascade adds latency for little saving. Report coverage honestly per task family.
- Validation-set thresholds drift as traffic changes; needs periodic recalibration (DECIDE-DISTILL's drift monitor can be shared).
- Crowded "LLM router" space; the decision-model tier is the only real differentiator.

## 11. Verify before building

Current pricing and logprob availability for each provider tier; d1-3B question schema; TypeSafe adapter license.

### Verification log (2026-10-18)

- **d1-3B** is `LiquidAI/d1-3B` on Hugging Face (built on LFM2.5-VL-3B, LFM Open License 1.0, not Apache). Needs `transformers>=5.14` and `trust_remote_code=True`. API: `model.system_one(state, questions, images=None)` → `{"answers": {name: answer}, "usage": {...}}`. Question schema (Decision Index): `type` ∈ `noul | choice | score`, `instructions`, `criteria`. Answers: `noul` → `{"noul": P(yes)}`; `choice` → `choice, confidence, probabilities`; `score` → `score, confidence, probabilities, legend`. The model card does not show the exact shape of `score.probabilities`; the adapter accepts a list (lowest level first) or a dict.
- **Jev** (TypeSafe) is hosted-only; weights not released as of Sept 2026. Not used in M0.
- **LLM tiers** use the OpenAI-compatible `/chat/completions` endpoint with `logprobs: true, top_logprobs: 20, max_tokens: 1`. This works with OpenAI and self-hosted vLLM/llama.cpp servers. The Anthropic Messages API does not expose token logprobs, so Claude models cannot serve tiers 1–2 under this method without a different confidence source.
- Still open: per-provider pricing (M2 needs it), TypeSafe adapter license.

## 12. Claude Code kickoff

"Read SPEC.md. Implement M0: the typed answer schema and a DecisionBackend interface with a d1-3B backend and one LLM backend using constrained choices. Verify current APIs first. Tests for schema validation and backend parity."
