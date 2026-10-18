"""Benchmark CASCADE-D on AG News (4-way topic routing, human labels).

    # tier 0 only (no API keys needed): how much traffic can d1-3B answer alone?
    python bench/ag_news.py --n 1000

    # full cascade: add OpenAI-compatible LLM tiers (need logprobs), cheapest first
    python bench/ag_news.py --n 1000 --llm small=gpt-x-mini --llm large=gpt-x \\
        --costs 0.00001,0.0001,0.002 --out report.html
"""

import argparse
import os

from datasets import load_dataset

from cascaded.backends import D1Backend, LLMChoiceBackend
from cascaded.benchmark import collect, load, render, run, tier_summary
from cascaded.schema import DecisionRequest

TOPICS = {"world": "World news, politics, international affairs", "sports": "Sports",
          "business": "Business, economy, markets, companies", "scitech": "Science and technology"}
QUESTION = {"topic": {"type": "choice", "instructions": "Which section does this news article belong in?",
                      "criteria": TOPICS}}


def examples(n: int, seed: int):
    ds = load_dataset("fancyzhx/ag_news", split="test").shuffle(seed=seed).select(range(n))
    names = list(TOPICS)  # dataset label order: World, Sports, Business, Sci/Tech
    return [(DecisionRequest(state=r["text"], questions=QUESTION), {"topic": names[r["label"]]}) for r in ds]


def d1_backend():
    import torch
    from transformers import AutoModel
    cuda = torch.cuda.is_available()
    # bf16 on CPU too: fp32 needs ~12.5 GB and the CPU latency difference is small
    model = AutoModel.from_pretrained("LiquidAI/d1-3B", trust_remote_code=True, dtype=torch.bfloat16)
    return D1Backend(model=model.to("cuda" if cuda else "cpu"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="ag_news.jsonl")
    ap.add_argument("--llm", action="append", default=[], help="name=model, cheapest first")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--costs", help="cost per request per tier, d1 first")
    ap.add_argument("--out", default="ag_news_report.html")
    a = ap.parse_args()

    tiers = [d1_backend()] + [LLMChoiceBackend(m, a.base_url, name=n) for n, m in (s.split("=", 1) for s in a.llm)]
    if not os.path.exists(a.cache):  # delete the cache to re-collect
        collect(tiers, examples(a.n, a.seed), a.cache)
    rows = load(a.cache)
    if len(tiers) == 1:
        print(tier_summary(rows, tiers[0].name))
        return
    costs = [float(c) for c in a.costs.split(",")]
    with open(a.out, "w") as f:
        f.write(render(run(rows, [t.name for t in tiers], costs), "CASCADE-D on AG News"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
