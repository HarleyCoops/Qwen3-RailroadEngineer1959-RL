"""Helpers for turning dictionary entries into training tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class NormalizedEntry:
    """Lightweight representation of a dictionary row."""

    entry_id: str
    dakota: str
    english: str
    pos: Optional[str]
    morphological_notes: Optional[str]
    examples: List[str]
    notes: Optional[str]
    page_number: Optional[int]
    source_image: Optional[str]
    aliases: List[str]
    confidence: Optional[float]

    @property
    def context_block(self) -> str:
        """Context string that can be embedded in instructions."""
        lines: List[str] = []
        if self.pos:
            lines.append(f"Part of speech: {self.pos}")
        if self.morphological_notes:
            lines.append(f"Morphology: {self.morphological_notes}")
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        if self.examples:
            lines.append("Examples: " + " | ".join(self.examples))
        if self.aliases:
            lines.append("Variants: " + ", ".join(sorted(set(self.aliases))))
        if self.page_number is not None:
            lines.append(f"Source page: {self.page_number}")
        return "\n".join(lines)


def estimate_difficulty(entry: NormalizedEntry) -> str:
    """Infer a curriculum difficulty label."""
    definition_tokens = len(entry.english.split())
    morphology_bonus = 5 if entry.morphological_notes else 0
    example_bonus = 2 * len(entry.examples)
    score = definition_tokens + morphology_bonus + example_bonus

    if score <= 6:
        return "easy"
    if score <= 16:
        return "medium"
    return "hard"


def build_sft_example(entry: NormalizedEntry) -> Dict[str, object]:
    """Return an instruction-tuning style example."""
    instruction = "Translate the Dakota headword to English."
    context = entry.context_block or "No additional context available."
    return {
        "instruction": instruction,
        "input": f"Dakota: {entry.dakota}\n{context}",
        "output": entry.english,
        "metadata": {
            "entry_id": entry.entry_id,
            "pos": entry.pos,
            "difficulty": estimate_difficulty(entry),
        },
    }


def build_forward_task(entry: NormalizedEntry) -> Dict[str, object]:
    """Dakota → English translation task for RL."""
    prompt_lines = [
        "Translate the following Dakota headword into natural English.",
        f"Dakota: {entry.dakota}",
    ]
    if entry.pos:
        prompt_lines.append(f"Part of speech: {entry.pos}")
    if entry.morphological_notes:
        prompt_lines.append(f"Morphology: {entry.morphological_notes}")
    if entry.examples:
        prompt_lines.append("Example usage: " + entry.examples[0])

    prompt = "\n".join(prompt_lines) + "\n"
    return {
        "prompt": prompt,
        "answer": entry.english,
        "metadata": {
            "direction": "dakota_to_english",
            "entry_id": entry.entry_id,
            "difficulty": estimate_difficulty(entry),
            "page": entry.page_number,
            "source_image": entry.source_image,
        },
    }


def build_backward_task(entry: NormalizedEntry) -> Dict[str, object]:
    """English → Dakota reverse lookup task."""
    prompt_lines = [
        "Which Dakota headword best matches this English description?",
        f"Definition: {entry.english}",
    ]
    if entry.pos:
        prompt_lines.append(f"Target part of speech: {entry.pos}")
    if entry.examples:
        prompt_lines.append("Example clue: " + entry.examples[0])
    if entry.morphological_notes:
        prompt_lines.append(f"Morphological hints: {entry.morphological_notes}")

    prompt = "\n".join(prompt_lines) + "\n"
    return {
        "prompt": prompt,
        "answer": entry.dakota,
        "metadata": {
            "direction": "english_to_dakota",
            "entry_id": entry.entry_id,
            "difficulty": estimate_difficulty(entry),
            "page": entry.page_number,
            "source_image": entry.source_image,
        },
    }


def build_rl_tasks(entry: NormalizedEntry) -> List[Dict[str, object]]:
    """Return forward + backward RL tasks using the Stoney Nakoda pattern."""
    return [build_forward_task(entry), build_backward_task(entry)]
