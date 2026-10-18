"""M2: pick the cheapest cascade thresholds that keep end-to-end error under a target.

Offline, every tier answers every validation request once. Per request that gives a confidence
(the lowest over its questions: one unsure answer escalates the whole request) and whether all
labelled questions were right. Replaying the cascade over those arrays is then cheap, so we grid
search thresholds exactly as SPEC §6 describes:

    C = c0 + (1 - Cov0)·c1 + (1 - Cov0)(1 - Cov1|rest)·c2

A threshold of SKIP drops a tier entirely (never called, never paid for), so the grid also
contains every single-tier baseline (always-small, always-large); the result is never worse
than the best of them on validation.
"""

import math
from dataclasses import dataclass, field
from itertools import product
from statistics import NormalDist

import numpy as np

from cascaded.backends.base import DecisionBackend
from cascaded.calibration import Example
from cascaded.schema import DecisionRequest, DecisionResponse

SKIP = math.inf


def outcomes(backend: DecisionBackend, examples: list[Example]) -> tuple[np.ndarray, np.ndarray]:
    """Per-request (confidence, correct) for one tier. Pass calibrated tiers (M1)."""
    conf, correct = [], []
    for request, truth in examples:
        r = backend.decide(request)
        conf.append(min(a.confidence for a in r.answers.values()))
        correct.append(all(r.answers[k].label == v for k, v in truth.items()))
    return np.array(conf), np.array(correct)


@dataclass
class Result:
    thresholds: tuple[float, ...]  # one per tier except the last, which always answers
    cost: float                    # mean cost per request
    error: float                   # end-to-end error rate
    coverage: tuple[float, ...]    # share of requests each tier answered
    per_request: np.ndarray = field(default=None, repr=False)  # each request's summed cost


def simulate(conf: np.ndarray, correct: np.ndarray, costs, thresholds) -> Result:
    """Replay a cascade over (tiers, n) arrays of confidence and correctness.

    costs[k] is a scalar or a per-request array; pass measured latencies to get per-request latency.
    """
    n = conf.shape[1]
    taus = (*thresholds, 0.0)  # the last tier accepts everything that reaches it
    answered_by = np.full(n, -1)
    cost = np.zeros(n)
    for k, tau in enumerate(taus):
        if tau == SKIP:
            continue
        pending = answered_by < 0
        cost[pending] += np.broadcast_to(costs[k], n)[pending]
        answered_by[pending & (conf[k] >= tau)] = k
    wrong = ~correct[answered_by, np.arange(n)]
    coverage = tuple(float(np.mean(answered_by == k)) for k in range(len(taus)))
    return Result(tuple(thresholds), float(cost.mean()), float(wrong.mean()), coverage, cost)


def error_upper_bound(error: float, n: int, delta: float) -> float:
    """One-sided Wilson upper bound on the true error rate at confidence 1 - delta."""
    z = NormalDist().inv_cdf(1 - delta)
    centre = error + z * z / (2 * n)
    half = z * math.sqrt(error * (1 - error) / n + z * z / (4 * n * n))
    return (centre + half) / (1 + z * z / n)


def optimize(conf: np.ndarray, correct: np.ndarray, costs, epsilon: float,
             delta: float = 0.01, grid: int = 50) -> Result:
    """Cheapest thresholds whose validation error, plus a Wilson margin, stays <= epsilon.

    The margin makes the target hold on unseen traffic, not just on the validation set.
    """
    # shortcut: the margin is per-config and ignores the grid search's own selection bias
    # (no multiple-testing correction); switch to Learn-then-Test if held-out misses show up.
    n = conf.shape[1]
    candidates = [[0.0, *np.unique(np.quantile(c, np.linspace(0, 1, grid))), SKIP] for c in conf[:-1]]
    best = None
    for taus in product(*candidates):
        r = simulate(conf, correct, costs, taus)
        if error_upper_bound(r.error, n, delta) <= epsilon and (best is None or r.cost < best.cost):
            best = r
    if best is None:
        floor = simulate(conf, correct, costs, (SKIP,) * (len(conf) - 1))
        raise ValueError(f"no thresholds reach error <= {epsilon}; the last tier alone errs {floor.error:.4f}")
    return best


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Selective risk at each coverage level (most confident first) and its area, AURC."""
    order = np.argsort(-conf, kind="stable")
    wrong = ~correct[order]
    k = np.arange(1, len(conf) + 1)
    risk = np.cumsum(wrong) / k
    return k / len(conf), risk, float(risk.mean())


class Cascade(DecisionBackend):
    """Runtime router: ask each tier in turn until one is confident enough."""

    def __init__(self, tiers: list[DecisionBackend], thresholds, name: str = "cascade"):
        assert len(thresholds) == len(tiers) - 1, "one threshold per tier except the last"
        self.tiers, self.thresholds, self.name = tiers, (*thresholds, 0.0), name

    def _decide(self, request: DecisionRequest) -> DecisionResponse:
        for tier, tau in zip(self.tiers, self.thresholds):
            if tau == SKIP:
                continue
            r = tier.decide(request)
            if min(a.confidence for a in r.answers.values()) >= tau:
                return r  # r.tier says who answered
        raise AssertionError("unreachable: the last tier's threshold is 0")
