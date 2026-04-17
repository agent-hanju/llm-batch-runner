from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from models import LLMRequest, LLMResponse


class BaseRunner(ABC):
    @abstractmethod
    def run(self, req: LLMRequest) -> LLMResponse:
        """Execute a single request and return its response."""

    @abstractmethod
    def stream(self, requests: list[LLMRequest]) -> Iterator[LLMResponse]:
        """Yield responses as they complete (completion order, not input order)."""
