"""M1: temperature scaling, expected calibration error and reliability diagrams.

Every tier answers with a distribution p over its options. Temperature scaling sharpens or
softens it as p_i^(1/T) / sum_j p_j^(1/T), which is softmax(logits / T) up to a constant, so
it works on probabilities alone and keeps the argmax. T > 1 cools an overconfident tier.

A labelled example is a (DecisionRequest, {question_name: true_option}) pair.
"""

from dataclasses import dataclass

import numpy as np

from cascaded.backends.base import DecisionBackend
from cascaded.schema import Answer, DecisionRequest, DecisionResponse

EPS = 1e-6  # floor for log(p): logprob tiers can give an option exactly 0
LOG_T_GRID = np.linspace(np.log(0.05), np.log(20.0), 801)

Example = tuple[DecisionRequest, dict[str, str]]


def _log_matrix(answers: list[Answer], labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Ragged answers -> (n, k_max) log-probs padded with -inf, plus each true label's column."""
    k = max(len(a.probabilities) for a in answers)
    logp = np.full((len(answers), k), -np.inf)
    y = np.empty(len(answers), dtype=int)
    for i, (a, label) in enumerate(zip(answers, labels)):
        opts = list(a.probabilities)
        logp[i, : len(opts)] = np.log(np.clip(list(a.probabilities.values()), EPS, 1.0))
        y[i] = opts.index(label)  # ValueError if the label is not an option: bad validation data
    return logp, y


def _nll(logp: np.ndarray, y: np.ndarray, temperature: float) -> float:
    z = logp / temperature
    m = z.max(axis=1)
    lse = m + np.log(np.exp(z - m[:, None]).sum(axis=1))
    return float(-(z[np.arange(len(y)), y] - lse).mean())


def fit_temperature(answers: list[Answer], labels: list[str]) -> float:
    """Temperature minimising NLL on a validation set (grid over log T, 0.05..20)."""
    logp, y = _log_matrix(answers, labels)
    temps = np.exp(LOG_T_GRID)
    return float(temps[np.argmin([_nll(logp, y, t) for t in temps])])


def scale(answer: Answer, temperature: float) -> Answer:
    p = np.clip(np.array(list(answer.probabilities.values())), EPS, 1.0) ** (1.0 / temperature)
    return Answer(probabilities=dict(zip(answer.probabilities, (p / p.sum()).tolist())))


@dataclass
class Reliability:
    """Per-bin mean confidence, accuracy and count; empty bins have count 0 and NaN means."""
    edges: np.ndarray
    confidence: np.ndarray
    accuracy: np.ndarray
    count: np.ndarray

    @property
    def ece(self) -> float:
        """Expected calibration error: count-weighted |accuracy - confidence| over bins."""
        hit = self.count > 0
        gap = np.abs(self.accuracy[hit] - self.confidence[hit])
        return float((gap * self.count[hit]).sum() / self.count.sum())

    def svg(self, size: int = 240) -> str:
        """Reliability diagram as a standalone SVG (bars = accuracy, diagonal = perfect calibration)."""
        w = size / len(self.count)
        bars = "".join(
            f'<rect x="{lo * size:.1f}" y="{size - a * size:.1f}" width="{w:.1f}" height="{a * size:.1f}" '
            f'fill="#4c78a8" stroke="#fff"><title>conf {c:.2f}, acc {a:.2f}, n={n}</title></rect>'
            for lo, c, a, n in zip(self.edges, self.confidence, self.accuracy, self.count) if n
        )
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}">'
                f'<rect width="{size}" height="{size}" fill="none" stroke="#888"/>{bars}'
                f'<line x1="0" y1="{size}" x2="{size}" y2="0" stroke="#e45756" stroke-dasharray="4"/></svg>')


def reliability(answers: list[Answer], labels: list[str], bins: int = 15) -> Reliability:
    conf = np.array([a.confidence for a in answers])
    correct = np.array([a.label == y for a, y in zip(answers, labels)], dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)  # bin i = (edges[i], edges[i+1]]
    count = np.bincount(idx, minlength=bins)
    with np.errstate(invalid="ignore"):
        mean_conf = np.bincount(idx, conf, bins) / count
        acc = np.bincount(idx, correct, bins) / count
    return Reliability(edges[:-1], mean_conf, acc, count)


class CalibratedBackend(DecisionBackend):
    """Wraps a tier and temperature-scales every answer it returns.

    `temperature` is one float, or {question type: float} ("noul", "choice", "score"); types missing
    from the dict are left unscaled.
    """

    def __init__(self, backend: DecisionBackend, temperature: float | dict[str, float]):
        self.backend, self.temperature, self.name = backend, temperature, backend.name

    def _t(self, qtype: str) -> float:
        return self.temperature.get(qtype, 1.0) if isinstance(self.temperature, dict) else self.temperature

    def _decide(self, request: DecisionRequest) -> DecisionResponse:
        r = self.backend.decide(request)
        answers = {k: scale(a, self._t(request.questions[k].type)) for k, a in r.answers.items()}
        return r.model_copy(update={"answers": answers})


def collect(backend: DecisionBackend, examples: list[Example]) -> tuple[list[Answer], list[str]]:
    """Run a tier over labelled examples; flatten to one (answer, label) pair per question."""
    answers, labels = [], []
    for request, truth in examples:
        response = backend.decide(request)
        for name, label in truth.items():
            answers.append(response.answers[name])
            labels.append(label)
    return answers, labels


def _types(examples: list[Example]) -> list[str]:
    """Question type per (answer, label) pair, in collect()'s order."""
    return [request.questions[name].type for request, truth in examples for name in truth]


def calibrate(backends: list[DecisionBackend], fit: list[Example], test: list[Example],
              bins: int = 15, per_type: bool = False,
              min_per_type: int = 30) -> tuple[list[CalibratedBackend], list[dict]]:
    """Fit temperatures per tier on `fit`, report ECE before and after on held-out `test`.

    per_type=True fits one temperature per question type (yes/no answers are often calibrated
    differently from 6-way choices); a type with fewer than `min_per_type` fit examples falls back
    to the tier-wide temperature. Reporting on the fit set would understate ECE, so the two sets
    must not overlap.
    """
    calibrated, rows = [], []
    fit_types, test_types = _types(fit), _types(test)
    for b in backends:
        fit_answers, fit_labels = collect(b, fit)
        t_all = fit_temperature(fit_answers, fit_labels)
        t = t_all
        if per_type:
            t = {}
            for qtype in sorted(set(fit_types) | set(test_types)):
                idx = [i for i, q in enumerate(fit_types) if q == qtype]
                t[qtype] = (fit_temperature([fit_answers[i] for i in idx], [fit_labels[i] for i in idx])
                            if len(idx) >= min_per_type else t_all)
        answers, labels = collect(b, test)
        scaled = [scale(a, t[q] if per_type else t) for a, q in zip(answers, test_types)]
        before = reliability(answers, labels, bins)
        after = reliability(scaled, labels, bins)
        calibrated.append(CalibratedBackend(b, t))
        rows.append({"tier": b.name, "temperature": t, "n": len(labels),
                     "accuracy": float(np.mean([a.label == y for a, y in zip(answers, labels)])),
                     "ece_before": before.ece, "ece_after": after.ece, "reliability": after})
    return calibrated, rows
