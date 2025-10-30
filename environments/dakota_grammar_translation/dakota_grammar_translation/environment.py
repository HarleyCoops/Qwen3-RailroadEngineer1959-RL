"""PrimeIntellect verifiers-compatible environment for Dakota grammar and translation tasks."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import verifiers as vf
from datasets import Dataset
from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a Dakota language expert specializing in the 1890 Dakota-English Dictionary grammar. "
    "Translate or explain each prompt concisely while preserving Dakota orthography exactly, "
    "including special characters (ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú, etc.) and cultural/grammatical nuance."
)

PACKAGE_ROOT = Path(__file__).resolve().parent
FALLBACK_DATASET = PACKAGE_ROOT / "data" / "sample_tasks.jsonl"
REPO_DATASET = Path(__file__).resolve().parents[3] / "dakota_rl_training" / "datasets" / "grammar_tasks_complete.jsonl"


def _normalize(text: str) -> str:
    """Normalize text for comparison while preserving Dakota special characters."""
    return " ".join(text.strip().lower().split())


def _char_f1(prediction: str, target: str) -> float:
    """Compute character-level F1 score, preserving Dakota special characters."""
    pred_chars = Counter(_normalize(prediction).replace(" ", ""))
    target_chars = Counter(_normalize(target).replace(" ", ""))
    if not target_chars:
        return 0.0
    overlap = sum(min(pred_chars[ch], target_chars[ch]) for ch in target_chars)
    precision = overlap / max(sum(pred_chars.values()), 1)
    recall = overlap / max(sum(target_chars.values()), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load entries from a JSONL file."""
    if not path.exists() or path.stat().st_size == 0:
        logger.info("Dataset file %s missing or empty.", path)
        return []

    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d of %s (%s).", line_no, path, exc)
                continue
            if "prompt" not in payload or "answer" not in payload:
                # Try alternative field names
                if "prompt" not in payload and "question" in payload:
                    payload["prompt"] = payload["question"]
                if "answer" not in payload and "ideal_answer" in payload:
                    payload["answer"] = payload["ideal_answer"]
                elif "answer" not in payload:
                    logger.debug(
                        "Skipping entry without prompt or answer on line %d of %s.",
                        line_no,
                        path,
                    )
                    continue
            entries.append(payload)
    return entries


def _prepare_records(
    entries: Sequence[dict[str, Any]],
    difficulties: Optional[Sequence[str]] = None,
    task_types: Optional[Sequence[str]] = None,
    include_hints: bool = True,
) -> list[dict[str, Any]]:
    """Prepare records for dataset creation with filtering."""
    allowed_difficulties = {d.lower() for d in difficulties or []}
    allowed_tasks = {t.lower() for t in task_types or []}
    records: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        difficulty = str(entry.get("difficulty", "medium"))
        task_type = str(entry.get("task_type", "default"))
        if allowed_difficulties and difficulty.lower() not in allowed_difficulties:
            continue
        if allowed_tasks and task_type.lower() not in allowed_tasks:
            continue
        prompt = str(entry.get("prompt", "")).strip()
        answer = str(entry.get("answer") or entry.get("ideal_answer", "")).strip()
        if not prompt or not answer:
            continue
        info = {
            "rule_id": entry.get("rule_id"),
            "verification_pattern": entry.get("verification_pattern"),
            "difficulty": difficulty,
            "task_type": task_type,
            "special_chars": entry.get("info", {}).get("special_chars", []),
            "required_affixes": entry.get("info", {}).get("required_affixes", []),
        }
        if include_hints:
            info["hints"] = entry.get("hints", [])
        records.append(
            {
                "id": entry.get("task_id") or f"dakota_task_{idx:05d}",
                "question": prompt,
                "answer": answer,
                "task": task_type,
                "info": info,
            }
        )
    return records


def _ensure_dataset(records: list[dict[str, Any]]) -> Dataset:
    """Ensure we have a valid dataset."""
    if not records:
        raise ValueError(
            "No usable Dakota grammar tasks were found. "
            "Ensure the RL dataset has been generated or provide `dataset_path`."
        )
    return Dataset.from_list(records)


@dataclass
class DatasetBundle:
    train: Dataset
    eval: Dataset | None = None


def _build_dataset_bundle(
    dataset_path: Optional[Path],
    eval_path: Optional[Path],
    max_examples: int,
    eval_examples: int,
    eval_fraction: float,
    seed: Optional[int],
    difficulty_filter: Optional[Sequence[str]],
    task_filter: Optional[Sequence[str]],
    include_hints: bool,
) -> DatasetBundle:
    """Build train/eval dataset bundle from JSONL files."""
    candidates: list[dict[str, Any]] = []
    explicit_path = dataset_path if dataset_path else REPO_DATASET
    candidates.extend(_prepare_records(_load_jsonl(explicit_path), difficulty_filter, task_filter, include_hints))

    if not candidates:
        fallback_entries = _load_jsonl(FALLBACK_DATASET)
        if fallback_entries:
            logger.warning(
                "Using bundled fallback tasks because no generated dataset was found at %s.",
                explicit_path,
            )
            candidates = _prepare_records(
                fallback_entries, difficulty_filter, task_filter, include_hints
            )
    if not candidates:
        raise ValueError(
            f"Unable to locate any Dakota grammar tasks. Checked: {explicit_path} and fallback sample."
        )

    if max_examples > 0:
        candidates = candidates[: max_examples]

    train_dataset = _ensure_dataset(candidates)

    if eval_path:
        eval_candidates = _prepare_records(
            _load_jsonl(eval_path), difficulty_filter, task_filter, include_hints
        )
        if eval_examples > 0:
            eval_candidates = eval_candidates[:eval_examples]
        eval_dataset = _ensure_dataset(eval_candidates) if eval_candidates else None
        return DatasetBundle(train=train_dataset, eval=eval_dataset)

    if eval_fraction > 0 and len(train_dataset) > 1:
        split = train_dataset.train_test_split(test_size=eval_fraction, seed=seed or 42)
        eval_dataset = split["test"]
        if eval_examples > 0:
            eval_dataset = eval_dataset.select(range(min(eval_examples, len(eval_dataset))))
        train_dataset = split["train"]
    else:
        eval_dataset = None
    return DatasetBundle(train=train_dataset, eval=eval_dataset)


class DakotaTranslationParser(Parser):
    """Parser that preserves Dakota orthography while normalizing whitespace."""

    def parse(self, text: str) -> str:
        """Parse and normalize text, preserving Dakota special characters."""
        return text.strip()

    def parse_answer(self, completion: Messages) -> str:
        """Extract answer from completion messages."""
        parsed = super().parse_answer(completion) or ""
        return parsed.strip()


class DakotaGrammarRubric(Rubric):
    """Reward rubric for Dakota grammar tasks with character preservation focus."""

    def __init__(self, parser: Optional[Parser] = None):
        parser = parser or DakotaTranslationParser()
        funcs = [
            self.exact_match_reward,
            self.char_overlap_reward,
            self.pattern_reward,
            self.affix_reward,
        ]
        weights = [0.5, 0.25, 0.15, 0.1]
        super().__init__(funcs=funcs, weights=weights, parser=parser, parallelize_scoring=False)
        self.special_chars = set("ćšŋḣṡáéíóúķśṅźėčžʼ")

    def _prediction(self, completion: vf.types.Messages, parser: Parser) -> str:
        """Extract prediction from completion."""
        response = parser.parse_answer(completion) or ""
        return response.strip()

    def exact_match_reward(
        self,
        completion: Messages,
        answer: str,
        parser: Parser,
        **_: Any,
    ) -> float:
        """Reward for exact match (normalized)."""
        prediction = self._prediction(completion, parser)
        return float(_normalize(prediction) == _normalize(answer))

    def char_overlap_reward(
        self,
        completion: Messages,
        answer: str,
        parser: Parser,
        **_: Any,
    ) -> float:
        """Reward for character-level overlap (critical for Dakota orthography)."""
        prediction = self._prediction(completion, parser)
        return float(_char_f1(prediction, answer))

    def affix_reward(
        self,
        completion: Messages,
        answer: str,
        parser: Parser,
        info: Dict[str, Any],
        **_: Any,
    ) -> float:
        """Reward for correct affix application."""
        prediction = self._prediction(completion, parser)
        required_affixes = (info or {}).get("required_affixes", []) or []
        if not required_affixes:
            return 1.0
        
        correct_count = 0
        for affix in required_affixes:
            affix_clean = affix.strip("-")
            if affix.startswith("-") and not affix.endswith("-"):
                # Suffix
                if re.search(rf'\w+{re.escape(affix_clean)}\b', prediction):
                    correct_count += 1
            elif affix.endswith("-") and not affix.startswith("-"):
                # Prefix
                if re.search(rf'\b{re.escape(affix_clean)}\w+', prediction):
                    correct_count += 1
            else:
                if affix_clean in prediction:
                    correct_count += 1
        
        return correct_count / len(required_affixes) if required_affixes else 1.0

    def pattern_reward(
        self,
        completion: Messages,
        answer: str,
        parser: Parser,
        info: Dict[str, Any],
        **_: Any,
    ) -> float:
        """Reward for pattern matching using verification patterns and hints."""
        prediction = self._prediction(completion, parser)
        score = 0.0
        pattern = (info or {}).get("verification_pattern")
        if pattern:
            try:
                if re.search(pattern, prediction, flags=re.IGNORECASE):
                    score = 1.0
            except re.error:
                if pattern.lower() in _normalize(prediction):
                    score = 1.0
        if score < 1.0:
            hints = (info or {}).get("hints", []) or []
            if hints:
                covered = sum(1 for hint in hints if hint.lower() in prediction.lower())
                score = max(score, covered / len(hints))
        return float(score)


class DakotaGrammarEnv(SingleTurnEnv):
    """Single-turn chat environment for Dakota grammar and translation tasks."""

    def __init__(
        self,
        dataset: Dataset,
        eval_dataset: Optional[Dataset],
        system_prompt: str,
        rubric: DakotaGrammarRubric,
        sampling_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            parser=rubric.parser,
            rubric=rubric,
            sampling_args=sampling_args,
            message_type="chat",
            **kwargs,
        )


def load_environment(
    dataset_path: str | Path | None = None,
    eval_path: str | Path | None = None,
    max_examples: int = -1,
    eval_examples: int = -1,
    eval_fraction: float = 0.1,
    difficulty_filter: Optional[Sequence[str]] = None,
    task_filter: Optional[Sequence[str]] = None,
    system_prompt: Optional[str] = None,
    sampling_args: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    include_hints: bool = True,
) -> vf.Environment:
    """
    Load the Dakota grammar translation environment.

    Args:
        dataset_path: Path to a JSONL file produced by the RL grammar pipeline.
        eval_path: Optional path to a JSONL evaluation split.
        max_examples: Cap on training examples (-1 uses full dataset).
        eval_examples: Cap on evaluation examples (-1 uses full split).
        eval_fraction: Fraction of training split reserved for evaluation if `eval_path` is not provided.
        difficulty_filter: Optional iterable of difficulty labels to keep (easy/medium/hard).
        task_filter: Optional iterable of task types to keep (morphology/translation/etc).
        system_prompt: Override the default system prompt.
        sampling_args: Optional sampling overrides passed to verifiers.
        seed: RNG seed used when performing internal splits.
        include_hints: Keep or drop hint metadata in the info payload.

    Returns:
        A configured verifiers `Environment` instance.
    """

    resolved_dataset = Path(dataset_path).expanduser().resolve() if dataset_path else None
    resolved_eval = Path(eval_path).expanduser().resolve() if eval_path else None

    bundle = _build_dataset_bundle(
        dataset_path=resolved_dataset,
        eval_path=resolved_eval,
        max_examples=max_examples,
        eval_examples=eval_examples,
        eval_fraction=eval_fraction,
        seed=seed,
        difficulty_filter=difficulty_filter,
        task_filter=task_filter,
        include_hints=include_hints,
    )

    rubric = DakotaGrammarRubric()
    env = DakotaGrammarEnv(
        dataset=bundle.train,
        eval_dataset=bundle.eval,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        rubric=rubric,
        sampling_args=sampling_args,
    )

    logger.info(
        "Loaded Dakota grammar environment with %d train and %s eval examples.",
        len(env.dataset) if env.dataset else 0,
        len(env.eval_dataset) if env.eval_dataset else 0,
    )
    return env

