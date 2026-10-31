import numpy as np
import pytest

from cascaded.backends.base import DecisionBackend
from cascaded.calibration import CalibratedBackend, calibrate, fit_temperature, reliability, scale
from cascaded.schema import Answer, DecisionRequest, DecisionResponse

OPTS = ["billing", "technical", "fraud"]


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def synthetic(n, sharpen, seed=0):
    """True labels drawn from softmax(z); the tier reports softmax(sharpen * z), i.e. true T = sharpen."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, (n, len(OPTS)))
    labels = [OPTS[rng.choice(len(OPTS), p=p)] for p in softmax(z)]
    answers = [Answer(probabilities=dict(zip(OPTS, p.tolist()))) for p in softmax(sharpen * z)]
    return answers, labels


class TableTier(DecisionBackend):
    """Fake tier: the request state is a row index into precomputed answers."""

    def __init__(self, name, answers):
        self.name, self.answers = name, answers

    def _decide(self, request):
        return DecisionResponse(tier=self.name, answers={"team": self.answers[request.state]})


def examples(labels):
    q = {"team": {"type": "choice", "instructions": "Which team?", "criteria": {o: o for o in OPTS}}}
    return [(DecisionRequest(state=i, questions=q), {"team": y}) for i, y in enumerate(labels)]


def test_fit_recovers_known_temperature():
    answers, labels = synthetic(4000, sharpen=3.0)
    assert fit_temperature(answers, labels) == pytest.approx(3.0, rel=0.1)


def test_fit_leaves_calibrated_tier_alone():
    answers, labels = synthetic(4000, sharpen=1.0)
    assert fit_temperature(answers, labels) == pytest.approx(1.0, rel=0.1)


def test_fit_survives_zero_probability_on_true_label():
    answers = [Answer(probabilities={"a": 1.0, "b": 0.0})] * 9 + [Answer(probabilities={"a": 0.0, "b": 1.0})]
    t = fit_temperature(answers, ["a"] * 9 + ["a"])
    assert np.isfinite(t) and t > 1  # one confident miss -> cool the tier down


def test_fit_handles_mixed_option_counts():
    answers = [Answer(probabilities={"true": 0.9, "false": 0.1}),
               Answer(probabilities={"x": 0.7, "y": 0.2, "z": 0.1})] * 50
    assert np.isfinite(fit_temperature(answers, ["true", "x"] * 50))


def test_scale_keeps_argmax_and_validity():
    a = Answer(probabilities={"a": 0.6, "b": 0.3, "c": 0.1})
    assert scale(a, 1.0).probabilities == pytest.approx(a.probabilities)
    hot, cold = scale(a, 0.5), scale(a, 4.0)
    assert hot.label == cold.label == "a"
    assert hot.confidence > 0.6 > cold.confidence


def test_ece_hand_computed():
    # conf 0.6 x2, one right -> |0.5 - 0.6| = 0.1, weight 2/4; conf 1.0 x2, both right -> 0, weight 2/4
    answers = [Answer(probabilities={"a": 0.6, "b": 0.4})] * 2 + [Answer(probabilities={"a": 1.0, "b": 0.0})] * 2
    r = reliability(answers, ["a", "b", "a", "a"])
    assert r.ece == pytest.approx(0.05)
    assert r.count.sum() == 4 and r.count[-1] == 2


def test_reliability_svg_has_a_bar_per_nonempty_bin():
    answers, labels = synthetic(500, sharpen=2.0)
    r = reliability(answers, labels, bins=10)
    assert r.svg().count("<rect ") == 1 + int((r.count > 0).sum())  # frame + bars


def test_calibrate_reports_ece_per_tier_on_held_out_data():
    """M1 acceptance: ECE reported per tier, fitted on one split and measured on another."""
    answers, labels = synthetic(6000, sharpen=3.0, seed=1)
    ok_answers, _ = synthetic(6000, sharpen=1.0, seed=1)  # same seed -> same labels, calibrated tier
    tiers = [TableTier("overconfident", answers), TableTier("calibrated", ok_answers)]
    ex = examples(labels)
    calibrated, rows = calibrate(tiers, fit=ex[:3000], test=ex[3000:])

    over, ok = rows
    assert over["tier"] == "overconfident" and over["n"] == 3000
    assert over["ece_before"] > 0.1 and over["ece_after"] < 0.03
    assert ok["ece_before"] < 0.03 and ok["temperature"] == pytest.approx(1.0, rel=0.15)
    assert over["accuracy"] == ok["accuracy"]  # temperature never changes the argmax

    r = calibrated[0].decide(ex[3000][0])
    assert isinstance(calibrated[0], CalibratedBackend) and r.tier == "overconfident"
    assert r.answers["team"].confidence < answers[3000].confidence


def test_per_type_temperatures_fix_types_miscalibrated_differently():
    """Yes/no answers overconfident (T=3), choice answers calibrated: one shared T can't fix both."""
    rng = np.random.default_rng(4)
    z = rng.normal(0, 1.5, 3000)
    p_true = 1 / (1 + np.exp(-z))
    yes_lab = ["true" if rng.uniform() < p else "false" for p in p_true]
    shown = 1 / (1 + np.exp(-3.0 * z))  # overconfident yes/no
    yes_ans = [Answer(probabilities={"true": float(p), "false": float(1 - p)}) for p in shown]
    ch_ans, ch_lab = synthetic(3000, sharpen=1.0, seed=3)

    noul = {"type": "noul", "instructions": "?"}
    choice = {"type": "choice", "instructions": "?", "criteria": {o: o for o in OPTS}}
    rows = [x for i in range(3000) for x in (("noul", yes_ans[i], yes_lab[i]), ("choice", ch_ans[i], ch_lab[i]))]

    class Mixed(DecisionBackend):
        name = "mixed"

        def _decide(self, request):
            return DecisionResponse(tier=self.name, answers={"q": rows[request.state][1]})

    examples = [(DecisionRequest(state=i, questions={"q": noul if t == "noul" else choice}), {"q": y})
                for i, (t, _, y) in enumerate(rows)]
    _, shared = calibrate([Mixed()], examples[:3000], examples[3000:])
    calibrated, per = calibrate([Mixed()], examples[:3000], examples[3000:], per_type=True)
    t = per[0]["temperature"]
    assert t["noul"] == pytest.approx(3.0, rel=0.2) and t["choice"] == pytest.approx(1.0, rel=0.2)
    assert per[0]["ece_after"] < shared[0]["ece_after"]
    assert calibrated[0].decide(examples[0][0]).answers["q"].confidence < rows[0][1].confidence
