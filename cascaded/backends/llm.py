"""Tiers 1-2: a generative LLM forced into a one-token choice, scored by token logprobs.

Each option gets a letter. The model answers with one token; we read the top-20 logprobs
of that token and renormalise the mass over the valid letters. This gives a distribution
over options instead of the model's (poorly calibrated) self-reported confidence.

Works with any OpenAI-compatible /chat/completions server that returns logprobs
(OpenAI, vLLM, llama.cpp server). The Anthropic Messages API has no logprobs.
"""

import json
import math
import os
import string
import urllib.request

from cascaded.backends.base import DecisionBackend
from cascaded.schema import Answer, DecisionRequest, DecisionResponse, Question

TOP_LOGPROBS = 20  # OpenAI's maximum; an option outside the top 20 can never receive mass


class LLMChoiceBackend(DecisionBackend):
    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", api_key: str | None = None,
                 name: str | None = None, post=None):
        self.name = name or model
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._post = post or self._http_post  # injectable for tests

    def _http_post(self, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)

    def _decide(self, request: DecisionRequest) -> DecisionResponse:
        state = request.state if isinstance(request.state, str) else json.dumps(request.state)
        answers, usage = {}, {"input_tokens": 0, "output_tokens": 0}
        # shortcut: one API call per question; batch questions into one call if latency matters (M3)
        for key, q in request.questions.items():
            letters = _letters(q)
            resp = self._post({
                "model": self.model,
                "messages": [{"role": "user", "content": _prompt(state, q, letters)}],
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": True,
                "top_logprobs": TOP_LOGPROBS,
            })
            answers[key] = Answer(probabilities=_letter_probs(resp, letters))
            usage["input_tokens"] += resp.get("usage", {}).get("prompt_tokens", 0)
            usage["output_tokens"] += resp.get("usage", {}).get("completion_tokens", 0)
        return DecisionResponse(tier=self.name, answers=answers, usage=usage)


def _letters(q: Question) -> dict[str, str]:
    """Letter -> option name."""
    if len(q.options) > TOP_LOGPROBS:
        raise ValueError(f"{len(q.options)} options exceeds the {TOP_LOGPROBS} visible logprobs")
    return dict(zip(string.ascii_uppercase, q.options))


def _prompt(state: str, q: Question, letters: dict[str, str]) -> str:
    desc = q.describe()
    lines = "\n".join(f"{letter}. {desc[opt]}" for letter, opt in letters.items())
    return (f"State:\n{state}\n\nQuestion: {q.instructions}\n\nOptions:\n{lines}\n\n"
            f"Reply with exactly one letter ({', '.join(letters)}) and nothing else.")


def _letter_probs(resp: dict, letters: dict[str, str]) -> dict[str, float]:
    top = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    mass = dict.fromkeys(letters.values(), 0.0)
    for t in top:  # " A" and "A" are different tokens for the same answer
        letter = t["token"].strip().upper()
        if letter in letters:
            mass[letters[letter]] += math.exp(t["logprob"])
    total = sum(mass.values())
    if total == 0:
        raise ValueError("no option letter among the top logprobs; model ignored the format")
    return {k: v / total for k, v in mass.items()}
