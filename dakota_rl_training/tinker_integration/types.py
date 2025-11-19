from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

@dataclass(frozen=True)
class DakotaGrammarExample:
    example_id: str
    prompt: str
    answer: str
    info: Dict[str, Any]
    task: str

    @property
    def difficulty(self) -> str:
        return (self.info or {}).get("difficulty", "unknown")
