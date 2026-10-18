"""M3: request ledger and a self-contained static HTML cost dashboard (no JS, no CDN)."""

import html
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Record:
    tier: int          # index of the tier that answered
    cost: float        # summed cost of every tier the request passed through
    latency_ms: float


class Ledger:
    # shortcut: in-memory ring buffer, lost on restart; persist to a file/DB once traffic matters
    def __init__(self, tier_names: list[str], costs: list[float], maxlen: int = 100_000):
        self.tier_names, self.costs = tier_names, costs
        self.records: deque[Record] = deque(maxlen=maxlen)

    def add(self, tier: int, cost: float, latency_ms: float) -> None:
        self.records.append(Record(tier, cost, latency_ms))

    def summary(self) -> dict:
        n = len(self.records)
        if n == 0:
            return {"requests": 0}
        tier = np.array([r.tier for r in self.records])
        cost = np.array([r.cost for r in self.records])
        lat = np.array([r.latency_ms for r in self.records])
        baseline = self.costs[-1]  # always-large: every request goes straight to the last tier
        return {
            "requests": n,
            "mean_cost": float(cost.mean()),
            "baseline_cost": baseline,
            "savings": float(1 - cost.mean() / baseline) if baseline else 0.0,
            "p50_ms": float(np.percentile(lat, 50)),
            "p99_ms": float(np.percentile(lat, 99)),
            "tiers": [
                {"name": name, "share": float(np.mean(tier == k)), "requests": int(np.sum(tier == k)),
                 "p50_ms": float(np.percentile(lat[tier == k], 50)) if np.any(tier == k) else None}
                for k, name in enumerate(self.tier_names)
            ],
        }


CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--bar:#4c78a8;--line:#ddd}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#eee;--muted:#999;--bar:#72a0d6;--line:#333}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.kpi{border:1px solid var(--line);border-radius:8px;padding:12px}.kpi b{display:block;font-size:1.6rem}
.kpi span,caption,th{color:var(--muted)}table{border-collapse:collapse;width:100%;margin-top:1.5rem}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}td.n{font-variant-numeric:tabular-nums}
.bar{height:10px;background:var(--bar);border-radius:3px}svg{max-width:100%;height:auto}
"""


def _ms(v: float | None) -> str:
    return "—" if v is None else f"{v:.0f} ms"


def render(ledger: Ledger, calibration: list[dict] | None = None) -> str:
    """Dashboard HTML. `calibration` is the rows from calibration.calibrate(), shown with their diagrams."""
    s = ledger.summary()
    esc = html.escape
    if s["requests"] == 0:
        body = "<p>No requests yet.</p>"
    else:
        kpis = [("Requests", f"{s['requests']:,}"), ("Mean cost / request", f"{s['mean_cost']:.4g}"),
                ("Saved vs always-large", f"{s['savings']:.1%}"), ("p50 / p99 latency", f"{s['p50_ms']:.0f} / {s['p99_ms']:.0f} ms")]
        rows = "".join(
            f"<tr><td>{esc(t['name'])}</td><td class=n>{t['requests']:,}</td><td class=n>{t['share']:.1%}</td>"
            f"<td><div class=bar style='width:{t['share'] * 100:.1f}%'></div></td>"
            f"<td class=n>{_ms(t['p50_ms'])}</td></tr>"
            for t in s["tiers"])
        body = ("<div class=kpis>" + "".join(f"<div class=kpi><span>{k}</span><b>{v}</b></div>" for k, v in kpis)
                + "</div><table><caption>Who answered</caption><tr><th>Tier</th><th>Requests</th><th>Share</th>"
                f"<th></th><th>p50</th></tr>{rows}</table>")
    if calibration:
        cal = "".join(
            f"<tr><td>{esc(r['tier'])}</td><td class=n>{r['temperature']:.2f}</td><td class=n>{r['ece_before']:.3f}</td>"
            f"<td class=n>{r['ece_after']:.3f}</td><td>{r['reliability'].svg(120)}</td></tr>" for r in calibration)
        body += ("<table><caption>Calibration (held-out)</caption><tr><th>Tier</th><th>T</th><th>ECE before</th>"
                 f"<th>ECE after</th><th>Reliability</th></tr>{cal}</table>")
    return (f"<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>CASCADE-D costs</title><style>{CSS}</style><h1>CASCADE-D costs</h1>{body}</html>")
