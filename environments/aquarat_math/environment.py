"""
Minimal AquaRat math reasoning environment and rubric stub.

This keeps a small, model-agnostic RL interface ready for future work on
multiple-choice algebra reasoning (e.g., deepmind/aqua-rat). It uses a simple
correct/incorrect reward with an optional brevity bonus.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import verifiers as vf
from datasets import Dataset
from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages

logger = logging.getLogger(__name__)

# Default paths are placeholders; provide an explicit dataset_path when loading.
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PACKAGE_ROOT / "data" / "aquarat_train.jsonl"
FALLBACK_EXAMPLE = {
    "id": "aquarat_demo_00001",
    "question": "If 3x + 5 = 20, what is x?",
    "options": {
        "A": "3",
        "B": "4",
        "C": "5",
        "D": "6",
        "E": "7",
    },
    "correct": "B",
    "rationale": "3x = 15 so x = 5; option B corresponds to 5.",
}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line %d in %s", lineno, path)
                continue
            records.append(payload)
    return records


def _build_records(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert raw AquaRat-style entries into dataset rows."""
    records: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        q = str(entry.get("question") or "").strip()
        opts = entry.get("options") or {}
        correct = str(entry.get("correct") or "").strip().upper()
        if not q or not opts or correct not in opts:
            continue
        # Flatten options into a prompt segment.
        options_str = " ".join(f"({k}) {v}" for k, v in opts.items())
        prompt = (
            f"Question: {q}\n"
            f"Options: {options_str}\n"
            "Reply with the correct option letter followed by a brief reason."
        )
        records.append(
            {
                "id": entry.get("id") or f"aquarat_{idx:05d}",
                "question": prompt,
                "answer": correct,
                "info": {
                    "options": opts,
                    "rationale": entry.get("rationale"),
                },
            }
        )
    return records


def _ensure_dataset(records: List[Dict[str, Any]]) -> Dataset:
    """Ensure a non-empty dataset, or fall back to a single demo item."""
    if not records:
        logger.warning("No AquaRat records found; using a single fallback example.")
        records = _build_records([FALLBACK_EXAMPLE])
    return Dataset.from_list(records)


class AquaRatParser(Parser):
    """Parser that extracts the first uppercase option letter."""

    def parse(self, text: str) -> str:
        return text.strip()

    def parse_answer(self, completion: Messages) -> str:
        parsed = super().parse_answer(completion) or ""
        match = re.search(r"\b([A-E])\b", parsed.upper())
        return match.group(1) if match else ""


class AquaRatRubric(Rubric):
    """Reward rubric: 1.0 for correct letter, optional brevity bonus."""

    def __init__(self, parser: Optional[Parser] = None):
        parser = parser or AquaRatParser()
        funcs = [self.correct_letter_reward, self.brevity_bonus]
        weights = [0.9, 0.1]
        super().__init__(funcs=funcs, weights=weights, parser=parser)

    def _prediction(self, completion: Messages, parser: Parser) -> str:
        return (parser.parse_answer(completion) or "").strip().upper()

    def correct_letter_reward(
        self, completion: Messages, answer: str, parser: Parser, **_: Any
    ) -> float:
        return float(self._prediction(completion, parser) == answer.strip().upper())

    def brevity_bonus(
        self, completion: Messages, answer: str, parser: Parser, max_lines: int = 3, **_: Any
    ) -> float:
        raw = parser.parse_answer(completion) or ""
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        return 1.0 if len(lines) <= max_lines else 0.0


class AquaRatEnv(SingleTurnEnv):
    """Single-turn environment for AquaRat multiple-choice algebra tasks."""

    def __init__(
        self,
        dataset: Dataset,
        eval_dataset: Optional[Dataset],
        rubric: AquaRatRubric,
        sampling_args: Optional[Dict[str, Any]] = None,
        message_type: str = "completion",
        **kwargs: Any,
    ):
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt="",  # Keep prompts literal for MCQ formats.
            parser=rubric.parser,
            rubric=rubric,
            sampling_args=sampling_args,
            message_type=message_type,
            **kwargs,
        )
        self.rubric = rubric


@dataclass
class DatasetBundle:
    train: Dataset
    eval: Optional[Dataset] = None


def build_dataset_bundle(
    dataset_path: Optional[Path] = None,
    eval_path: Optional[Path] = None,
    max_examples: int = -1,
    eval_fraction: float = 0.1,
    seed: Optional[int] = None,
) -> DatasetBundle:
    """Build train/eval datasets from JSONL files."""
    records: List[Dict[str, Any]] = []
    source_path = dataset_path or DEFAULT_DATASET
    records.extend(_build_records(_load_jsonl(source_path)))
    train_dataset = _ensure_dataset(records)

    eval_dataset: Optional[Dataset] = None
    if eval_path:
        eval_records = _build_records(_load_jsonl(eval_path))
        if eval_records:
            eval_dataset = Dataset.from_list(eval_records)
    elif eval_fraction > 0 and len(train_dataset) > 1:
        split = train_dataset.train_test_split(test_size=eval_fraction, seed=seed or 42)
        eval_dataset = split["test"]
        train_dataset = split["train"]

    if max_examples > 0:
        train_dataset = train_dataset.select(range(min(max_examples, len(train_dataset))))
    return DatasetBundle(train=train_dataset, eval=eval_dataset)


def load_environment(
    dataset_path: Optional[str | Path] = None,
    eval_path: Optional[str | Path] = None,
    max_examples: int = -1,
    eval_fraction: float = 0.1,
    seed: Optional[int] = None,
    sampling_args: Optional[Dict[str, Any]] = None,
    message_type: str = "completion",
) -> vf.Environment:
    """
    Load the AquaRat environment. Provide dataset_path pointing to a JSONL that
    mirrors deepmind/aqua-rat fields: {question, options: {A:...,B:...}, correct, rationale?}.
    """
    bundle = build_dataset_bundle(
        dataset_path=Path(dataset_path) if dataset_path else None,
        eval_path=Path(eval_path) if eval_path else None,
        max_examples=max_examples,
        eval_fraction=eval_fraction,
        seed=seed,
    )
    rubric = AquaRatRubric()
    env = AquaRatEnv(
        dataset=bundle.train,
        eval_dataset=bundle.eval,
        rubric=rubric,
        sampling_args=sampling_args,
        message_type=message_type,
    )
    logger.info(
        "Loaded AquaRat environment with %d train and %s eval examples.",
        len(env.dataset) if env.dataset else 0,
        len(env.eval_dataset) if env.eval_dataset else 0,
    )
    return env
