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
from functools import lru_cache

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
    certified: bool = True  # False: too few labels to certify any setting, so this is the always-last-tier fallback


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


def binom_pvalue(errors: int, n: int, epsilon: float) -> float:
    """P(at most `errors` mistakes in n | true error = epsilon): small means the error is really below epsilon."""
    i = np.arange(errors + 1)
    lg = np.vectorize(math.lgamma)
    logpmf = lg(n + 1) - lg(i + 1) - lg(n - i + 1) + i * math.log(epsilon) + (n - i) * math.log1p(-epsilon)
    return float(min(1.0, np.exp(logpmf).sum()))


def optimize(conf: np.ndarray, correct: np.ndarray, costs, epsilon: float,
             delta: float = 0.05, grid: int = 50, method: str = "bonferroni", seed: int = 0) -> Result:
    """Cheapest thresholds whose true error is <= epsilon with probability >= 1 - delta.

    Learn-then-Test (Angelopoulos et al., 2021): every threshold setting is a hypothesis test of
    "error >= epsilon" with an exact binomial p-value, so the search itself is accounted for.
      method="bonferroni"  test every setting on all the data at delta / (number of settings)
      method="pareto"      Pareto Testing (Laufer-Goldshmidt et al., 2023): half the data finds the
                           cost/error frontier and orders it safest first; the other half tests it in
                           that fixed order at the full delta, stopping at the first failure
    """
    n = conf.shape[1]
    settings = _settings(conf, grid)
    if method == "pareto":
        chosen = _pareto_test(conf, correct, costs, epsilon, delta, settings, seed)
    elif method == "bonferroni":
        pvalue = lru_cache(maxsize=None)(lambda k: binom_pvalue(k, n, epsilon))  # many settings share an error count
        chosen = None
        for taus in settings:
            r = simulate(conf, correct, costs, taus)
            if (chosen is None or r.cost < chosen.cost) and pvalue(round(r.error * n)) <= delta / len(settings):
                chosen = r
        chosen = chosen and chosen.thresholds
    else:
        raise ValueError(f"unknown method {method!r}")
    if chosen is None:
        floor = simulate(conf, correct, costs, (SKIP,) * (len(conf) - 1))
        raise ValueError(f"no thresholds reach error <= {epsilon} at delta {delta} with n={n}; "
                         f"the last tier alone errs {floor.error:.4f}")
    return simulate(conf, correct, costs, chosen)


def _settings(conf, grid):
    """Every threshold combination searched: quantiles of each tier's confidence, plus 0 and SKIP."""
    candidates = [[0.0, *np.unique(np.quantile(c, np.linspace(0, 1, grid))), SKIP] for c in conf[:-1]]
    return list(product(*candidates))


def _pareto_test(conf, correct, costs, epsilon, delta, settings, seed):
    idx = np.random.default_rng(seed).permutation(conf.shape[1])
    a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    on_a = [(simulate(conf[:, a], correct[:, a], costs, t), t) for t in settings]
    frontier, best_err = [], np.inf
    for r, t in sorted(on_a, key=lambda x: (x[0].cost, x[0].error)):  # cheapest first; keep strict error improvements
        if r.error < best_err:
            frontier.append((r.error, r.cost, t))
            best_err = r.error
    chosen = None
    for _, _, t in sorted(frontier):  # safest first: fixed-sequence testing needs no multiplicity correction
        r = simulate(conf[:, b], correct[:, b], costs, t)
        if binom_pvalue(round(r.error * len(b)), len(b), epsilon) > delta:
            break
        chosen = t  # each later setting on the frontier is cheaper
    return chosen


def optimize_per_task(conf, correct, costs, epsilon, tasks, **kw) -> dict[str, Result]:
    """Separate thresholds per task family. Each task meets epsilon on its own, so the mix does too;
    a task where tier 0 can't be trusted simply gets SKIP for it. A task with too few labels to certify
    anything falls back to the last tier alone (certified=False) instead of failing every task."""
    tasks, out = np.asarray(tasks), {}
    for t in np.unique(tasks):
        c, ok = conf[:, tasks == t], correct[:, tasks == t]
        try:
            out[t] = optimize(c, ok, costs, epsilon, **kw)
        except ValueError:
            r = simulate(c, ok, costs, (SKIP,) * (len(conf) - 1))
            r.certified = False
            out[t] = r
    return out


def label_plan(conf, correct, costs, epsilon, tasks, delta: float = 0.05, grid: int = 50,
               method: str = "bonferroni", margins=(0.6, 0.7, 0.8, 0.9)) -> dict[str, dict]:
    """Per task: what the labels in hand certify now, and what more labels would buy.

    `now` is optimize_per_task's answer (cost, and whether anything was certified). `options` lists, for
    each margin m, the cheapest setting whose observed error is <= m * epsilon, its cost, and the labels
    needed to certify it (assuming its observed error is its true error, so an estimate). Tighter margins
    are cheaper to certify but cost more to run: the menu is labels vs running cost.
    """
    tasks, plan = np.asarray(tasks), {}
    now = optimize_per_task(conf, correct, costs, epsilon, tasks, delta=delta, grid=grid, method=method)
    for t in np.unique(tasks):
        c, ok = conf[:, tasks == t], correct[:, tasks == t]
        settings = _settings(c, grid)
        runs = [simulate(c, ok, costs, s) for s in settings]
        options = []
        for m in margins:
            good = [r for r in runs if r.error <= m * epsilon]
            if not good:
                continue
            best = min(good, key=lambda r: r.cost)
            need = (2 * required_n(epsilon, best.error, delta) if method == "pareto"  # half the data tests
                    else required_n(epsilon, best.error, delta, settings=len(settings)))
            options.append({"margin": m, "cost": best.cost, "error": best.error, "thresholds": best.thresholds,
                            "need": need, "shortfall": max(0, need - c.shape[1])})
        plan[t] = {"have": c.shape[1], "certified": now[t].certified, "cost_now": now[t].cost, "options": options}
    return plan


def required_n(epsilon: float, true_error: float, delta: float = 0.05, settings: int = 1) -> int:
    """Smallest validation size at which a setting with this true error passes the Learn-then-Test check.

    Planning aid: e.g. required_n(0.05, 0.03, settings=2704) says how many labels a 3-tier grid needs.
    """
    if not 0 <= true_error < epsilon:
        raise ValueError("true_error must be below epsilon")
    lo, hi = 1, 1
    while binom_pvalue(round(true_error * hi), hi, epsilon) > delta / settings:
        lo, hi = hi, hi * 2
    while lo < hi:  # p-value is close enough to monotone in n for planning purposes
        mid = (lo + hi) // 2
        lo, hi = (mid + 1, hi) if binom_pvalue(round(true_error * mid), mid, epsilon) > delta / settings else (lo, mid)
    return lo


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Selective risk at each coverage level (most confident first) and its area, AURC."""
    order = np.argsort(-conf, kind="stable")
    wrong = ~correct[order]
    k = np.arange(1, len(conf) + 1)
    risk = np.cumsum(wrong) / k
    return k / len(conf), risk, float(risk.mean())


class Cascade(DecisionBackend):
    """Runtime router: ask each tier in turn until one is confident enough.

    `thresholds` is one tuple for every request, or {task: tuple} from optimize_per_task, keyed by
    DecisionRequest.task. A task with no thresholds goes straight to the last tier: never trust
    tier 0 on traffic it was not validated on.
    """

    def __init__(self, tiers: list[DecisionBackend], thresholds, name: str = "cascade"):
        self.tiers, self.name = tiers, name
        per_task = thresholds if isinstance(thresholds, dict) else {None: thresholds}
        for t in per_task.values():
            assert len(t) == len(tiers) - 1, "one threshold per tier except the last"
        self.per_task = {k: (*t, 0.0) for k, t in per_task.items()}
        self.thresholds = self.per_task.get(None)  # single-tuple form, kept for callers that read it

    def thresholds_for(self, task: str | None) -> tuple:
        if task in self.per_task:
            return self.per_task[task]
        if None in self.per_task:
            return self.per_task[None]
        return (SKIP,) * (len(self.tiers) - 1) + (0.0,)

    def _decide(self, request: DecisionRequest) -> DecisionResponse:
        for tier, tau in zip(self.tiers, self.thresholds_for(request.task)):
            if tau == SKIP:
                continue
            r = tier.decide(request)
            if min(a.confidence for a in r.answers.values()) >= tau:
                return r  # r.tier says who answered
        raise AssertionError("unreachable: the last tier's threshold is 0")
