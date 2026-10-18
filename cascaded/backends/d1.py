"""Tier 0: Liquid AI d1-3B decision model (open weights, zero output tokens)."""

from cascaded.backends.base import DecisionBackend
from cascaded.schema import Answer, DecisionRequest, DecisionResponse, Question


class D1Backend(DecisionBackend):
    def __init__(self, model=None, model_id: str = "LiquidAI/d1-3B", name: str = "d1-3B"):
        self.name = name
        self.model_id = model_id
        self._model = model  # inject a loaded model (or a fake in tests); otherwise loaded on first use

    @property
    def model(self):
        if self._model is None:
            import torch
            from transformers import AutoModel  # needs transformers>=5.14

            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            dtype = torch.float32 if device == "cpu" else torch.bfloat16
            self._model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True, dtype=dtype).to(device)
        return self._model

    def _decide(self, request: DecisionRequest) -> DecisionResponse:
        questions = {k: q.model_dump(exclude_none=True) for k, q in request.questions.items()}
        out = self.model.system_one(request.state, questions)
        answers = {k: Answer(probabilities=_to_probs(q, out["answers"][k])) for k, q in request.questions.items()}
        return DecisionResponse(tier=self.name, answers=answers, usage=out.get("usage", {}))


def _to_probs(q: Question, raw: dict) -> dict[str, float]:
    if q.type == "noul":
        return {"true": float(raw["noul"]), "false": 1.0 - float(raw["noul"])}
    # choice and score both return {option: p}; score keys are "0".."n-1", lowest level first
    return {k: float(raw["probabilities"][k]) for k in q.options}
