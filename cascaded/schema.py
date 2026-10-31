"""Typed decision schema shared by every tier.

Questions follow the Decision Index schema used by d1-3B (type / instructions / criteria).
Every answer, from any tier, is a probability distribution over the question's options,
so tiers are interchangeable and their confidences comparable.
"""

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Question(BaseModel):
    type: Literal["noul", "choice", "score"]
    instructions: str = Field(min_length=1)
    # noul: optional {"true": ..., "false": ...}; choice: {name: description}; score: levels, lowest first
    criteria: dict[str, str] | list[str] | None = None

    @model_validator(mode="after")
    def _check_criteria(self):
        c = self.criteria
        if self.type == "noul" and c is not None and (not isinstance(c, dict) or set(c) != {"true", "false"}):
            raise ValueError('noul criteria must be {"true": ..., "false": ...} or omitted')
        if self.type == "choice" and (not isinstance(c, dict) or len(c) < 2):
            raise ValueError("choice criteria must be a dict with at least 2 options")
        if self.type == "score" and (not isinstance(c, list) or not 2 <= len(c) <= 10):
            raise ValueError("score criteria must be a list of 2 to 10 levels")
        return self

    @property
    def options(self) -> list[str]:
        """Option names an answer distributes probability over. Score levels are '0'..'n-1'."""
        if self.type == "noul":
            return ["true", "false"]
        if self.type == "choice":
            return list(self.criteria)
        return [str(i) for i in range(len(self.criteria))]

    def describe(self) -> dict[str, str]:
        """Option name -> human description, for prompting generative tiers."""
        if self.type == "noul":
            return self.criteria or {"true": "Yes", "false": "No"}
        if self.type == "choice":
            return dict(self.criteria)
        return dict(zip(self.options, self.criteria))


class DecisionRequest(BaseModel):
    state: Any  # string, any JSON value, or None
    questions: dict[str, Question] = Field(min_length=1)
    task: str | None = None  # task family, for per-task thresholds (e.g. "ticket-routing")


class Answer(BaseModel):
    probabilities: dict[str, float]

    @model_validator(mode="after")
    def _check_distribution(self):
        p = self.probabilities
        if not p or any(not 0.0 <= v <= 1.0 for v in p.values()):
            raise ValueError("probabilities must be non-empty and each in [0, 1]")
        if not math.isclose(sum(p.values()), 1.0, abs_tol=1e-3):
            raise ValueError(f"probabilities must sum to 1, got {sum(p.values()):.4f}")
        return self

    @property
    def label(self) -> str:
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def confidence(self) -> float:
        return max(self.probabilities.values())


class DecisionResponse(BaseModel):
    tier: str
    answers: dict[str, Answer]
    usage: dict[str, int] = Field(default_factory=dict)

    def check_against(self, request: DecisionRequest) -> "DecisionResponse":
        """Raise unless every question is answered over exactly its own options."""
        if set(self.answers) != set(request.questions):
            raise ValueError(f"answered {sorted(self.answers)}, asked {sorted(request.questions)}")
        for name, q in request.questions.items():
            if set(self.answers[name].probabilities) != set(q.options):
                raise ValueError(f"{name}: options {sorted(self.answers[name].probabilities)} != {sorted(q.options)}")
        return self
