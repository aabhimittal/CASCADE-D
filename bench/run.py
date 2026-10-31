"""Benchmark CASCADE-D on public human-labelled datasets.

    # tier 0 only (no API keys needed): how much traffic can d1-3B answer alone?
    python bench/run.py ag_news --n 1000

    # full cascade: add OpenAI-compatible LLM tiers (need logprobs), cheapest first
    python bench/run.py ag_news --n 1000 --llm small=<model> --llm large=<model> \\
        --costs 0.00001,0.0001,0.002 --out report.html
"""

import argparse
import json
import os

from datasets import load_dataset

from cascaded.backends import D1Backend, LLMChoiceBackend
from cascaded.benchmark import collect, load, render, run, tier_summary
from cascaded.schema import DecisionRequest

# name -> (hf path, split, text field, question, option name per dataset label id)
DATASETS = {
    "ag_news": ("fancyzhx/ag_news", "test", "text", {
        "type": "choice", "instructions": "Which section does this news article belong in?",
        "criteria": {"world": "World news, politics, international affairs", "sports": "Sports",
                     "business": "Business, economy, markets, companies", "scitech": "Science and technology"}},
        ["world", "sports", "business", "scitech"]),
    "sst2": ("stanfordnlp/sst2", "validation", "sentence", {
        "type": "noul", "instructions": "Is this movie review positive?"}, ["false", "true"]),
    "emotion": ("dair-ai/emotion", "test", "text", {
        "type": "choice", "instructions": "Which emotion does the writer express?",
        "criteria": {"sadness": "Sadness", "joy": "Joy", "love": "Love", "anger": "Anger",
                     "fear": "Fear", "surprise": "Surprise"}},
        ["sadness", "joy", "love", "anger", "fear", "surprise"]),
}


def examples(name: str, n: int, seed: int):
    path, split, field, question, labels = DATASETS[name]
    ds = load_dataset(path, split=split).shuffle(seed=seed)
    ds = ds.select(range(min(n, len(ds))))
    return [(DecisionRequest(state=r[field], questions={"label": question}), {"label": labels[r["label"]]}) for r in ds]


def d1_backend():
    import torch
    from transformers import AutoModel
    # bf16 on CPU too: fp32 needs ~12.5 GB and the CPU latency difference is small
    model = AutoModel.from_pretrained("LiquidAI/d1-3B", trust_remote_code=True, dtype=torch.bfloat16)
    return D1Backend(model=model.to("cuda" if torch.cuda.is_available() else "cpu"))


def embedder(model_id="sentence-transformers/all-MiniLM-L6-v2"):
    """Sentence embeddings for the learned router (mean-pooled, L2-normalised), cached per text."""
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok, model, cache = AutoTokenizer.from_pretrained(model_id), AutoModel.from_pretrained(model_id).eval(), {}

    def featurize(texts):
        todo = [t for t in dict.fromkeys(map(str, texts)) if t not in cache]
        for i in range(0, len(todo), 64):
            batch = tok(todo[i:i + 64], padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                h = model(**batch).last_hidden_state
            m = batch["attention_mask"].unsqueeze(-1)
            v = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)
            cache.update(zip(todo[i:i + 64], v.numpy()))
        return np.stack([cache[str(t)] for t in texts])
    return featurize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", help="JSONL cache (default: <dataset>.jsonl); delete it to re-collect")
    ap.add_argument("--llm", action="append", default=[], help="name=model, cheapest first")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--costs", help="cost per request per tier, d1 first")
    ap.add_argument("--out", help="report path (default: <dataset>_report.html)")
    ap.add_argument("--method", choices=["bonferroni", "pareto"], default="bonferroni",
                    help="Learn-then-Test correction: bonferroni (safest) or pareto (cheaper)")
    ap.add_argument("--router-embeddings", action="store_true",
                    help="learned-router baseline on MiniLM sentence embeddings instead of hashed n-grams")
    a = ap.parse_args()
    cache = a.cache or f"{a.dataset}.jsonl"

    tiers = [d1_backend()] + [LLMChoiceBackend(m, a.base_url, name=n) for n, m in (s.split("=", 1) for s in a.llm)]
    if not os.path.exists(cache):
        collect(tiers, examples(a.dataset, a.n, a.seed), cache)
    rows = load(cache)
    if len(tiers) == 1:
        print(json.dumps(tier_summary(rows, tiers[0].name)))
        return
    out = a.out or f"{a.dataset}_report.html"
    with open(out, "w") as f:
        extra = {"featurize": embedder()} if a.router_embeddings else {}
        f.write(render(run(rows, [t.name for t in tiers], [float(c) for c in a.costs.split(",")],
                           method=a.method, **extra),
                       f"CASCADE-D on {a.dataset}"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
