"""M4: benchmark report — cost vs error of CASCADE-D against baselines (SPEC §7).

Two steps, so the expensive part runs once:

1. collect(): run every tier on every labelled example and cache raw answers + latency (JSONL).
2. run() + render(): replay offline over 10 random fit/test splits. On each split, fit
   temperatures and thresholds on `fit`, measure on `test`. Report means ± std across splits.

Compared on held-out data:
  CASCADE-D           all tiers, thresholds from optimizer.optimize at each target ε
  LLM-only cascade    same optimizer without tier 0: isolates what the decision model adds
  always-<tier>       each tier alone
A learned RouteLLM-style router is not implemented (it needs a trained router per dataset).

    python -m cascaded.benchmark CACHE.jsonl --tiers d1-3B,small,large --costs 0.0001,0.001,0.01 --out report.html
"""

import argparse
import html
import json
import time

import numpy as np

from cascaded.backends.base import DecisionBackend
from cascaded.calibration import Example, fit_temperature, reliability, scale
from cascaded.optimizer import SKIP, optimize, risk_coverage, simulate
from cascaded.schema import Answer

EPSILONS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2)


# --- 1. collect --------------------------------------------------------------

def collect(tiers: list[DecisionBackend], examples: list[Example], path: str) -> None:
    """Append one JSON line per example: truth plus each tier's raw answers and latency.

    A tier that raises (e.g. an LLM ignoring the answer format) is recorded as an error and
    later counts as confidence 0 and wrong, so it escalates or costs accuracy honestly.
    """
    with open(path, "a") as f:
        for request, truth in examples:
            row = {"truth": truth, "tiers": {}}
            for t in tiers:
                start = time.perf_counter()
                try:
                    r = t.decide(request)
                    out = {"answers": {k: a.probabilities for k, a in r.answers.items()}}
                except Exception as e:  # recorded, not swallowed: shows up as a failure in the report
                    out = {"error": f"{type(e).__name__}: {e}"}
                out["ms"] = (time.perf_counter() - start) * 1000
                row["tiers"][t.name] = out
            f.write(json.dumps(row) + "\n")


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --- 2. replay ---------------------------------------------------------------

def _answers(rows, name, temperature=1.0):
    """Per row: {question: Answer} (temperature-scaled), or None if the tier failed."""
    out = []
    for row in rows:
        t = row["tiers"][name]
        out.append(None if "error" in t else
                   {k: scale(Answer(probabilities=p), temperature) for k, p in t["answers"].items()})
    return out


def _arrays(rows, names, temps):
    conf, correct, ms = [], [], []
    for name, temp in zip(names, temps):
        c, ok = [], []
        for row, ans in zip(rows, _answers(rows, name, temp)):
            c.append(0.0 if ans is None else min(a.confidence for a in ans.values()))
            ok.append(ans is not None and all(ans[k].label == v for k, v in row["truth"].items()))
        conf.append(c), correct.append(ok), ms.append([row["tiers"][name]["ms"] for row in rows])
    return np.array(conf), np.array(correct), np.array(ms)


def _flat(rows, name, temperature=1.0):
    """Flatten to (answers, labels) over every labelled question the tier answered."""
    answers, labels = [], []
    for row, ans in zip(rows, _answers(rows, name, temperature)):
        if ans is not None:
            for k, v in row["truth"].items():
                answers.append(ans[k])
                labels.append(v)
    return answers, labels


def _fit(rows, name) -> float:
    answers, labels = _flat(rows, name)
    return fit_temperature(answers, labels) if answers else 1.0  # tier failed everywhere: nothing to fit


def _ece(rows, name, temperature=1.0) -> float:
    answers, labels = _flat(rows, name, temperature)
    return reliability(answers, labels).ece if answers else float("nan")


def _point(conf, correct, ms, costs, taus) -> dict:
    r = simulate(conf, correct, costs, taus)
    lat = simulate(conf, correct, ms, taus).per_request
    return {"cost_1k": r.cost * 1000, "error": r.error, "coverage": r.coverage,
            "p50_ms": float(np.percentile(lat, 50)), "p99_ms": float(np.percentile(lat, 99))}


def run(rows: list[dict], names: list[str], costs: list[float], epsilons=EPSILONS,
        splits: int = 10, fit_frac: float = 0.5, delta: float = 0.01, seed: int = 0) -> dict:
    n, k = len(rows), len(names)
    per_split = []
    for s in range(splits):
        idx = np.random.default_rng(seed + s).permutation(n)
        fit = [rows[i] for i in idx[: int(n * fit_frac)]]
        test = [rows[i] for i in idx[int(n * fit_frac):]]
        temps = [_fit(fit, name) for name in names]
        fc, fok, _ = _arrays(fit, names, temps)
        tc, tok, tms = _arrays(test, names, temps)

        out = {"temperature": temps, "cascade": {}, "llm_only": {}, "always": {}, "tiers": {}}
        for eps in epsilons:
            for key, sl in (("cascade", slice(0, k)), ("llm_only", slice(1, k))):
                try:
                    chosen = optimize(fc[sl], fok[sl], costs[sl], eps, delta)
                except ValueError:  # unreachable target on this split
                    continue
                taus = chosen.thresholds if key == "cascade" else (SKIP, *chosen.thresholds)
                out[key][eps] = {**_point(tc, tok, tms, costs, taus), "thresholds": taus}
        for j, name in enumerate(names):
            only = tuple(SKIP if i != j else 0.0 for i in range(k - 1)) if j < k - 1 else (SKIP,) * (k - 1)
            out["always"][name] = _point(tc, tok, tms, costs, only)
            out["tiers"][name] = {
                "ece_raw": _ece(test, name),
                "ece": _ece(test, name, temps[j]),
                "aurc": risk_coverage(tc[j], tok[j])[2],
                "failures": sum("error" in r["tiers"][name] for r in test) / len(test),
            }
        per_split.append(out)
    return {"names": names, "costs": costs, "n": n, "splits": splits, "per_split": per_split,
            "summary": _summarise(per_split, names, epsilons)}


def _ms(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def _summarise(per_split, names, epsilons):
    metrics = ("cost_1k", "error", "p50_ms", "p99_ms")
    agg = {"cascade": {}, "llm_only": {}, "always": {}, "tiers": {}}
    for key in ("cascade", "llm_only"):
        for eps in epsilons:
            pts = [s[key][eps] for s in per_split if eps in s[key]]
            if len(pts) == len(per_split):  # only targets reachable on every split
                agg[key][eps] = {m: _ms([p[m] for p in pts]) for m in metrics}
                agg[key][eps]["coverage"] = np.mean([p["coverage"] for p in pts], axis=0).tolist()
                agg[key][eps]["met_eps"] = sum(p["error"] <= eps for p in pts)
    for name in names:
        pts = [s["always"][name] for s in per_split]
        agg["always"][name] = {m: _ms([p[m] for p in pts]) for m in metrics}
        agg["tiers"][name] = {m: _ms([s["tiers"][name][m] for s in per_split])
                              for m in ("ece_raw", "ece", "aurc", "failures")}
        agg["tiers"][name]["temperature"] = _ms([s["temperature"][names.index(name)] for s in per_split])
    return agg


def tier_summary(rows: list[dict], name: str, risks=(0.01, 0.02, 0.05)) -> dict:
    """One tier alone: accuracy, ECE, AURC, and how much traffic it could answer at each error rate.

    Coverage at risk r = largest share of requests, most confident first, whose error stays <= r.
    In-sample and uncalibrated, so treat it as an optimistic first look, not a threshold.
    """
    conf, correct, _ = _arrays(rows, [name], [1.0])
    cov, risk, aurc = risk_coverage(conf[0], correct[0])
    return {"n": len(rows), "accuracy": float(correct[0].mean()), "ece": _ece(rows, name), "aurc": aurc,
            "coverage_at_risk": {r: float(cov[risk <= r].max()) if np.any(risk <= r) else 0.0 for r in risks}}


# --- 3. report ---------------------------------------------------------------

CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#e3e3e3;--s1:#2a78d6;--s2:#eb6834;--pt:#555}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--fg:#f0efec;--muted:#a3a29a;--line:#333;--s1:#3987e5;--s2:#d95926;--pt:#c3c2b7}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 16px}
.note{border-left:4px solid var(--s2);padding:8px 12px;background:color-mix(in srgb,var(--s2) 10%,transparent)}
table{border-collapse:collapse;width:100%;margin:1rem 0 2rem}th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}th,caption{color:var(--muted);font-weight:500}caption{text-align:left}
td{font-variant-numeric:tabular-nums}svg{width:100%;height:auto}svg text{fill:var(--muted);font-size:12px}
.legend{display:flex;gap:16px;color:var(--muted);font-size:13px}.legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:6px}
"""


def _pm(m, fmt="{:.3f}"):
    return f"{fmt.format(m['mean'])} ± {fmt.format(m['std'])}"


def _chart(summary, names) -> str:
    """Pareto chart: x = error, y = cost per 1k requests (log). Lines = cascades over ε, dots = baselines."""
    w, h, pad = 720, 380, 56
    series = [("CASCADE-D", "var(--s1)", summary["cascade"]), ("LLM-only cascade", "var(--s2)", summary["llm_only"])]
    pts = [(p["error"]["mean"], p["cost_1k"]["mean"]) for _, _, d in series for p in d.values()]
    pts += [(p["error"]["mean"], p["cost_1k"]["mean"]) for p in summary["always"].values()]
    xmax = max(x for x, _ in pts) * 1.1 or 1
    step = next(t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5) if xmax / t <= 6)  # round-number ticks
    ys = [y for _, y in pts if y > 0]
    lo, hi = np.log10(min(ys)) - 0.2, np.log10(max(ys)) + 0.2

    def X(x): return pad + x / xmax * (w - 2 * pad)
    def Y(y): return h - pad - (np.log10(max(y, 10 ** lo)) - lo) / (hi - lo) * (h - 2 * pad)

    grid = "".join(f'<line x1="{pad}" x2="{w - pad}" y1="{Y(10 ** e):.1f}" y2="{Y(10 ** e):.1f}" stroke="var(--line)"/>'
                   f'<text x="{pad - 6}" y="{Y(10 ** e) + 4:.1f}" text-anchor="end">{10 ** e:g}</text>'
                   for e in range(int(np.ceil(lo)), int(np.floor(hi)) + 1))
    xt = "".join(f'<text x="{X(v):.1f}" y="{h - pad + 18}" text-anchor="middle">{v:.0%}</text>'
                 for v in np.arange(0, xmax, step))
    marks = ""
    for label, color, data in series:
        coords = [(X(p["error"]["mean"]), Y(p["cost_1k"]["mean"]), eps, p) for eps, p in sorted(data.items())]
        if coords:
            marks += f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y, *_ in coords)}"/>'
        for x, y, eps, p in coords:
            marks += (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="var(--bg)" stroke-width="2">'
                      f'<title>{label}, target ε={eps:g}: error {p["error"]["mean"]:.3f}, cost/1k {p["cost_1k"]["mean"]:.4g}</title></circle>')
    for name, p in summary["always"].items():
        x, y = X(p["error"]["mean"]), Y(p["cost_1k"]["mean"])
        marks += (f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" fill="var(--pt)" stroke="var(--bg)" stroke-width="2">'
                  f'<title>always-{html.escape(name)}: error {p["error"]["mean"]:.3f}, cost/1k {p["cost_1k"]["mean"]:.4g}</title></rect>'
                  f'<text x="{x + 9:.1f}" y="{y + 4:.1f}">always-{html.escape(name)}</text>')
    axes = (f'<text x="{w / 2}" y="{h - 12}" text-anchor="middle">end-to-end error (held-out)</text>'
            f'<text transform="translate(14 {h / 2}) rotate(-90)" text-anchor="middle">cost per 1k requests (log)</text>')
    legend = ('<div class=legend><span><i style="background:var(--s1)"></i>CASCADE-D</span>'
              '<span><i style="background:var(--s2)"></i>LLM-only cascade</span><span>■ single-tier baselines</span></div>')
    return f'{legend}<svg viewBox="0 0 {w} {h}" role="img" aria-label="Cost versus error">{grid}{xt}{axes}{marks}</svg>'


def render(result: dict, title: str, note: str = "") -> str:
    s, names, esc = result["summary"], result["names"], html.escape
    head = f"<h1>{esc(title)}</h1>" + (f"<p class=note>{esc(note)}</p>" if note else "")
    head += (f"<p>{result['n']:,} labelled requests · {result['splits']} random 50/50 fit/test splits · "
             f"tier costs {', '.join(f'{esc(n)} {c:g}' for n, c in zip(names, result['costs']))} · mean ± std over splits.</p>")

    def cascade_rows(data):
        return "".join(
            f"<tr><td>{eps:g}</td><td>{_pm(p['error'])}</td><td>{p['met_eps']}/{result['splits']}</td>"
            f"<td>{_pm(p['cost_1k'], '{:.3g}')}</td><td>{' / '.join(f'{c:.0%}' for c in p['coverage'])}</td>"
            f"<td>{p['p50_ms']['mean']:.0f} / {p['p99_ms']['mean']:.0f}</td></tr>" for eps, p in sorted(data.items()))

    hdr = ("<tr><th>Target ε</th><th>Error</th><th>Met ε</th><th>Cost / 1k</th><th>Coverage per tier</th>"
           "<th>p50 / p99 ms</th></tr>")
    tables = (f"<table><caption>CASCADE-D (tiers: {esc(' → '.join(names))})</caption>{hdr}{cascade_rows(s['cascade'])}</table>"
              f"<table><caption>LLM-only cascade (no tier 0)</caption>{hdr}{cascade_rows(s['llm_only'])}</table>")
    tables += ("<table><caption>Single-tier baselines</caption><tr><th>Baseline</th><th>Error</th><th>Cost / 1k</th>"
               "<th>p50 / p99 ms</th></tr>" + "".join(
                   f"<tr><td>always-{esc(n)}</td><td>{_pm(p['error'])}</td><td>{_pm(p['cost_1k'], '{:.3g}')}</td>"
                   f"<td>{p['p50_ms']['mean']:.0f} / {p['p99_ms']['mean']:.0f}</td></tr>" for n, p in s["always"].items())
               + "</table>")
    tables += ("<table><caption>Per tier (held-out)</caption><tr><th>Tier</th><th>T</th><th>ECE raw</th><th>ECE calibrated</th>"
               "<th>AURC</th><th>Failures</th></tr>" + "".join(
                   f"<tr><td>{esc(n)}</td><td>{t['temperature']['mean']:.2f}</td><td>{_pm(t['ece_raw'])}</td>"
                   f"<td>{_pm(t['ece'])}</td><td>{_pm(t['aurc'])}</td><td>{t['failures']['mean']:.1%}</td></tr>"
                   for n, t in s["tiers"].items()) + "</table>")
    return (f"<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{esc(title)}</title><style>{CSS}</style>{head}<h2>Cost vs error</h2>{_chart(s, names)}{tables}</html>")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cache")
    ap.add_argument("--tiers", required=True, help="comma-separated tier names, cheapest first")
    ap.add_argument("--costs", required=True, help="comma-separated cost per request, same order")
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--title", default="CASCADE-D benchmark")
    ap.add_argument("--note", default="")
    ap.add_argument("--splits", type=int, default=10)
    a = ap.parse_args(argv)
    names, costs = a.tiers.split(","), [float(c) for c in a.costs.split(",")]
    if len(names) != len(costs):
        ap.error("--tiers and --costs need the same number of entries")
    result = run(load(a.cache), names, costs, splits=a.splits)
    with open(a.out, "w") as f:
        f.write(render(result, a.title, a.note))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
