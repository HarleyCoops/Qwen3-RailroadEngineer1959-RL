"""
Lightweight base classes used by the Dakota verifier implementations.

These mirror the abstract interfaces exposed by the upstream Prime Intellect
`verifiers` package while remaining dependency-free for local development and
testing.  When integrating with the full training stack the real interfaces can
subclass or replace these without changing caller code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Tuple


Messages = List[Dict[str, Any]]
State = Dict[str, Any]


class BaseEnv(ABC):
    """Generic environment base class that stores configuration kwargs."""

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs


class MultiTurnEnv(BaseEnv):
    """Interface for conversational, stateful verifier environments."""

    @abstractmethod
    async def is_completed(self, messages: Messages, state: State, **kwargs: Any) -> bool:
        """Return True when the conversation should stop."""

    @abstractmethod
    async def env_response(
        self, messages: Messages, state: State, **kwargs: Any
    ) -> Tuple[Messages, State]:
        """Validate the latest response and emit feedback plus the next state."""


class SingleTurnEnv(BaseEnv):
    """Interface for single-shot verifier tasks."""

    @abstractmethod
    async def check_answer(
        self, messages: Messages, **kwargs: Any
    ) -> Tuple[bool, Dict[str, Any]]:
        """Return a pass/fail flag and optional metadata."""


class Rubric(ABC):
    """Base rubric class used to compute scalar rewards."""

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs

    @abstractmethod
    def score(self, response: str, expected: str, **kwargs: Any) -> float:
        """Return a scalar reward for a given response."""

    def score_batch(
        self, responses: Iterable[str], expected_values: Iterable[str], **kwargs: Any
    ) -> List[float]:
        """Convenience helper for batch scoring."""
        return [
            self.score(response, expected, **kwargs)
            for response, expected in zip(responses, expected_values)
        ]
