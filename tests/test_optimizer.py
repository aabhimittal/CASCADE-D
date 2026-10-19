import numpy as np
import pytest

from cascaded.backends.base import DecisionBackend
from cascaded.optimizer import SKIP, Cascade, binom_pvalue, optimize, outcomes, risk_coverage, simulate
from cascaded.schema import Answer, DecisionRequest, DecisionResponse

COSTS = [0.01, 1.0, 10.0]  # d1 ~free, small LLM, large LLM


def synthetic(n, seed):
    """Three calibrated tiers sharing a per-request difficulty; bigger tiers are more accurate."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(-3, 3, n)
    skill = np.array([[2.5], [2.0], [5.0]])  # tier 0 about as accurate as the real d1-3B on AG News
    conf = 1 / (1 + np.exp(-(skill - d + rng.normal(0, 0.5, (3, n)))))
    conf = np.maximum(conf, 1 - conf)  # binary decisions: confidence is never below 0.5
    correct = rng.uniform(size=(3, n)) < conf
    return conf, correct


def test_simulate_hand_case():
    conf = np.array([[0.9, 0.6, 0.6, 0.95],
                     [0.5, 0.9, 0.7, 0.5],
                     [0.5, 0.5, 0.5, 0.5]])
    correct = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 1, 1]], dtype=bool)
    r = simulate(conf, correct, COSTS, (0.8, 0.8))
    # tier0 takes requests 0, 3; tier1 takes 1; tier2 takes 2
    assert r.coverage == (0.5, 0.25, 0.25)
    assert r.error == 0.25  # request 3: tier0 confident and wrong
    assert r.cost == pytest.approx((0.01 * 4 + 1.0 * 2 + 10.0 * 1) / 4)


def test_skip_removes_tier_and_its_cost():
    conf, correct = synthetic(1000, 0)
    always_large = simulate(conf, correct, COSTS, (SKIP, SKIP))
    always_small = simulate(conf, correct, COSTS, (SKIP, 0.0))
    assert always_large.cost == 10.0 and always_large.coverage == (0.0, 0.0, 1.0)
    assert always_small.cost == 1.0 and always_small.coverage == (0.0, 1.0, 0.0)


def test_binom_pvalue():
    assert binom_pvalue(50, 1000, 0.05) == pytest.approx(0.5, abs=0.04)   # observed == epsilon: no evidence
    assert binom_pvalue(20, 1000, 0.05) < 1e-5                            # far below epsilon: strong evidence
    assert binom_pvalue(2, 100, 0.05) > binom_pvalue(20, 1000, 0.05)      # same rate, more data: smaller p


def test_meets_epsilon_on_held_out_across_10_splits():
    """M2 acceptance: thresholds chosen on validation meet epsilon on held-out test, 10 splits."""
    eps = 0.05
    conf, correct = synthetic(8000, seed=42)
    always_large = simulate(conf, correct, COSTS, (SKIP, SKIP))
    for split in range(10):
        idx = np.random.default_rng(split).permutation(conf.shape[1])
        val, test = idx[:4000], idx[4000:]
        chosen = optimize(conf[:, val], correct[:, val], COSTS, eps)
        held_out = simulate(conf[:, test], correct[:, test], COSTS, chosen.thresholds)
        assert held_out.error <= eps, f"split {split}: {held_out.error:.4f} > {eps}"
        assert held_out.cost < 0.7 * always_large.cost  # the cascade has to be worth it
        assert held_out.coverage[0] > 0.05              # and tier 0 actually answers traffic


def test_loose_target_lets_tier0_answer_everything():
    conf, correct = synthetic(4000, 1)
    r = optimize(conf, correct, COSTS, epsilon=0.5)
    assert r.coverage[0] == 1.0 and r.cost == pytest.approx(0.01)


def test_unreachable_target_raises():
    conf, correct = synthetic(2000, 2)
    with pytest.raises(ValueError, match="no thresholds"):
        optimize(conf, correct, COSTS, epsilon=0.0001)


def test_risk_coverage_and_aurc():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    good = np.array([True, True, True, False])  # the one mistake is the least confident
    bad = good[::-1].copy()                     # the one mistake is the most confident
    cov, risk, aurc_good = risk_coverage(conf, good)
    assert cov.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert risk.tolist() == [0, 0, 0, 0.25]
    assert aurc_good == pytest.approx(0.0625)
    assert risk_coverage(conf, bad)[2] > aurc_good


class TableTier(DecisionBackend):
    def __init__(self, name, conf, correct):
        self.name, self.conf, self.correct = name, conf, correct

    def _decide(self, request):
        c, ok = self.conf[request.state], self.correct[request.state]
        hit, miss = ("true", "false") if ok else ("false", "true")
        return DecisionResponse(tier=self.name, answers={"q": Answer(probabilities={hit: c, miss: 1 - c})})


def test_runtime_cascade_matches_simulation():
    conf, correct = synthetic(300, 3)
    tiers = [TableTier(f"t{k}", conf[k], correct[k]) for k in range(3)]
    q = {"q": {"type": "noul", "instructions": "?"}}
    examples = [(DecisionRequest(state=i, questions=q), {"q": "true"}) for i in range(300)]

    got = np.array([outcomes(t, examples)[0] for t in tiers])
    assert np.allclose(got, conf)

    taus = (0.9, 0.8)
    expected = simulate(conf, correct, COSTS, taus)
    cascade = Cascade(tiers, taus)
    answered = [cascade.decide(r).tier for r, _ in examples]
    assert tuple(answered.count(f"t{k}") / 300 for k in range(3)) == pytest.approx(expected.coverage)
