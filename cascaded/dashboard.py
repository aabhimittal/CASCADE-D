"""M3: request ledger and a self-contained static HTML cost dashboard (no JS, no CDN)."""

import html
import json
import math
import os
import threading
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Record:
    tier: int          # index of the tier that answered
    cost: float        # summed cost of every tier the request passed through
    latency_ms: float
    task: str | None = None


class Ledger:
    """Recent requests, optionally persisted as JSONL so stats survive restarts."""

    def __init__(self, tier_names: list[str], costs: list[float], maxlen: int = 100_000,
                 path: str | None = None, expected: dict | None = None):
        self.tier_names, self.costs, self.path = tier_names, costs, path
        self.expected = expected or {}  # {task: validation coverage per tier}, for drift checks
        self.records: deque[Record] = deque(maxlen=maxlen)
        self._lines = 0
        self._lock = threading.Lock()  # proxy endpoints run on a thread pool
        if path:
            try:
                with open(path) as f:
                    lines = [line for line in f if line.strip()]
            except FileNotFoundError:
                lines = []
            for line in lines:
                try:
                    self.records.append(Record(**json.loads(line)))
                except (ValueError, TypeError):  # a torn last line from a crash: skip it, keep the rest
                    pass
            self._lines = len(lines)

    def add(self, tier: int, cost: float, latency_ms: float, task: str | None = None) -> None:
        r = Record(tier, cost, latency_ms, task)
        with self._lock:
            self.records.append(r)
            if not self.path:
                return
            with open(self.path, "a") as f:  # shortcut: one open per request; batch writes at high QPS
                f.write(json.dumps(asdict(r)) + "\n")
            self._lines += 1
            if self._lines >= 2 * self.records.maxlen:  # compact: the file never outgrows twice the window
                tmp = self.path + ".tmp"
                with open(tmp, "w") as f:
                    f.writelines(json.dumps(asdict(x)) + "\n" for x in list(self.records))
                os.replace(tmp, self.path)  # atomic: a crash leaves the old file or the new one, never half
                self._lines = len(self.records)

    def drift(self, min_requests: int = 50, z_alarm: float = 3.0) -> list[dict]:
        """Live vs validation coverage per task and tier. Label-free: a big shift means traffic no longer
        looks like the validation set, so the error guarantee no longer applies. Normal approximation."""
        out = []
        for task, expected in self.expected.items():
            tiers = [r.tier for r in self.records if r.task == task]
            if len(tiers) < min_requests:
                continue
            for k, exp in enumerate(expected):
                obs = sum(t == k for t in tiers) / len(tiers)
                sd = math.sqrt(max(exp * (1 - exp), 1e-9) / len(tiers))
                z = (obs - exp) / sd
                out.append({"task": task, "tier": self.tier_names[k], "expected": exp, "observed": obs,
                            "n": len(tiers), "z": z, "alarm": abs(z) >= z_alarm})
        return out

    def summary(self) -> dict:
        n = len(self.records)
        if n == 0:
            return {"requests": 0}
        tier = np.array([r.tier for r in self.records])
        cost = np.array([r.cost for r in self.records])
        lat = np.array([r.latency_ms for r in self.records])
        baseline = self.costs[-1]  # always-large: every request goes straight to the last tier
        return {
            "drift": self.drift(),
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


def _temps(t) -> str:
    return ", ".join(f"{k} {v:.2f}" for k, v in t.items()) if isinstance(t, dict) else f"{t:.2f}"


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
    drift = ledger.drift()
    if drift:
        rows = "".join(
            f"<tr><td>{esc(str(d['task'] or 'all'))}</td><td>{esc(d['tier'])}</td><td class=n>{d['expected']:.1%}</td>"
            f"<td class=n>{d['observed']:.1%}</td><td class=n>{d['n']:,}</td>"
            f"<td>{'⚠ recalibrate: traffic has shifted' if d['alarm'] else 'ok'}</td></tr>" for d in drift)
        body += ("<table><caption>Drift: who answers now vs on validation</caption><tr><th>Task</th><th>Tier</th>"
                 f"<th>Validation</th><th>Live</th><th>Requests</th><th>Status</th></tr>{rows}</table>")
    if calibration:
        cal = "".join(
            f"<tr><td>{esc(r['tier'])}</td><td class=n>{_temps(r['temperature'])}</td><td class=n>{r['ece_before']:.3f}</td>"
            f"<td class=n>{r['ece_after']:.3f}</td><td>{r['reliability'].svg(120)}</td></tr>" for r in calibration)
        body += ("<table><caption>Calibration (held-out)</caption><tr><th>Tier</th><th>T</th><th>ECE before</th>"
                 f"<th>ECE after</th><th>Reliability</th></tr>{cal}</table>")
    return (f"<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>CASCADE-D costs</title><style>{CSS}</style><h1>CASCADE-D costs</h1>{body}</html>")
