"""Build the GitHub Pages demo into --out: index.html + data.json (real d1-3B runs) + a synthetic report.

    python site/build.py --out _site
"""

import argparse
import html
import json
import pathlib
import re
import shutil

from cascaded.benchmark import (SYNTHETIC_COSTS, SYNTHETIC_TIERS, _answers, load, render, run, synthetic_rows,
                                tier_summary)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATASETS = {  # cache file -> (name, what the decision is)
    "ag_news": ("AG News", "Which section does this news headline belong in? Four options."),
    "sst2": ("SST-2", "Is this movie review positive? Yes or no."),
    "emotion": ("Emotion", "Which emotion does this tweet express? Six options."),
}


def _clean(text: str, limit: int = 220) -> str:
    """AG News stores entities without '&' ("#39;", "quot;") and uses '\\' as a separator."""
    text = text.replace(" #39;", "#39;")  # "CBS #39;s" -> "CBS's"
    text = html.unescape(re.sub(r"(?<!&)(#\d+;|quot;|amp;|lt;|gt;)", r"&\1", text)).replace("\\", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def dataset_json(rows, name="d1-3B", text=True):
    items = []
    for row, ans in zip(rows, _answers(rows, name)):
        (q, truth), = row["truth"].items()
        a = ans[q]
        items.append([round(a.confidence, 4), a.label, truth, _clean(row.get("state", "")) if text else ""])
    items.sort(key=lambda x: -x[0])  # most confident first: the page sweeps a threshold down this list
    return {"summary": tier_summary(rows, name), "items": items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_site")
    ap.add_argument("--no-text", action="store_true",
                    help="omit dataset text from the page (all three datasets are research/non-commercial use only)")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = {}
    for key, (title, task) in DATASETS.items():
        path = ROOT / "data" / f"{key}_d1.jsonl"
        rows = load(str(path)) if path.exists() else []
        if rows:
            data[key] = {"title": title, "task": task, **dataset_json(rows, text=not args.no_text)}
    (out / "data.json").write_text(json.dumps(data, separators=(",", ":")))

    demo = run(synthetic_rows(3000), SYNTHETIC_TIERS, SYNTHETIC_COSTS, splits=5)
    (out / "synthetic-cascade.html").write_text(render(
        demo, "CASCADE-D full cascade (synthetic demo)",
        "Synthetic data. This page shows what the full report looks like; it is not a measurement. "
        "The LLM tiers have not been run on real data yet."))
    shutil.copy(ROOT / "site" / "index.html", out / "index.html")
    print(f"built {out} with datasets: {', '.join(data) or 'none'}")


if __name__ == "__main__":
    main()
