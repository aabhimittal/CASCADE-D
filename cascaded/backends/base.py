from abc import ABC, abstractmethod

from cascaded.schema import DecisionRequest, DecisionResponse


class DecisionBackend(ABC):
    """One cascade tier: answers typed questions with a probability per option."""

    name: str

    @abstractmethod
    def _decide(self, request: DecisionRequest) -> DecisionResponse: ...

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        return self._decide(request).check_against(request)
