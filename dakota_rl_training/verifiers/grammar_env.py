"""
Dakota Grammar Environment for RL Training with PrimeIntellect

This environment is based on actual extracted grammar from page 61:
- 13 grammar rules (morphology, syntax, semantics)
- 44+ RL tasks auto-generated
- Special character verification: ŋ, š, ć, ź, ž, ʼ
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import re

from .base import MultiTurnEnv, SingleTurnEnv


@dataclass
class DakotaState:
    """State tracking for multi-turn Dakota grammar learning"""
    special_chars_correct: bool = False
    affixes_correct: bool = False
    semantic_correct: bool = False
    attempts: int = 0
    last_error_type: Optional[str] = None
    partial_credit: float = 0.0


class DakotaGrammarEnv(MultiTurnEnv):
    """
    Multi-turn environment for Dakota grammar learning

    Supports task types from actual extraction:
    1. Morphology: Apply transformations (affixes, compounds)
    2. Word translation: Dakota → English
    3. Sentence translation: Dakota → English
    4. Reverse translation: English → Dakota (advanced)

    Verification based on:
    - Special character preservation
    - Required affix presence
    - Semantic accuracy
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_turns = kwargs.get("max_turns", 3)

        # Dakota special characters that MUST be preserved
        self.special_chars = set("ćšŋḣṡáéíóúķśṅźėčžʼ")

    async def is_completed(self, messages: List[Dict], state: Dict, **kwargs) -> bool:
        """
        Task is complete when:
        1. All criteria met (chars + affixes + semantics)
        2. Max attempts reached
        """
        task_info = kwargs.get("info", {})
        task_type = task_info.get("task_type", "morphology")

        # Check if perfect answer
        if task_type == "morphology":
            perfect = (
                state.get("special_chars_correct", False) and
                state.get("affixes_correct", False)
            )
        elif task_type in ["word_translation", "sentence_translation"]:
            perfect = state.get("semantic_correct", False)
        elif task_type == "reverse_translation":
            # Reverse translation requires both chars and semantics
            perfect = (
                state.get("special_chars_correct", False) and
                state.get("semantic_correct", False)
            )
        else:
            perfect = state.get("semantic_correct", False)

        # Complete if perfect or max attempts
        return perfect or state.get("attempts", 0) >= self.max_turns

    async def env_response(
        self,
        messages: List[Dict],
        state: Dict,
        **kwargs
    ) -> tuple[List[Dict], Dict]:
        """
        Verify student response and provide feedback
        """
        last_message = messages[-1]["content"]
        expected_answer = kwargs.get("answer", "")
        task_info = kwargs.get("info", {})

        # Initialize new state
        new_state = {
            "attempts": state.get("attempts", 0) + 1,
            "special_chars_correct": False,
            "affixes_correct": False,
            "semantic_correct": False,
            "partial_credit": 0.0
        }

        # 1. Verify special characters
        expected_chars = task_info.get("special_chars", [])
        chars_correct = self._verify_special_chars(
            last_message,
            expected_chars
        )
        new_state["special_chars_correct"] = chars_correct

        # 2. Verify affixes (for morphology tasks)
        if task_info.get("task_type") == "morphology":
            required_affixes = task_info.get("required_affixes", [])
            affixes_correct = self._verify_affixes(
                last_message,
                required_affixes
            )
            new_state["affixes_correct"] = affixes_correct
        else:
            new_state["affixes_correct"] = True  # N/A for non-morphology

        # 3. Verify semantic accuracy (exact or fuzzy match)
        semantic_correct, similarity = self._verify_semantic(
            last_message,
            expected_answer,
            task_info
        )
        new_state["semantic_correct"] = semantic_correct
        new_state["partial_credit"] = similarity

        # Generate feedback
        feedback = self._generate_feedback(
            last_message,
            expected_answer,
            new_state,
            task_info
        )

        return [{"role": "system", "content": feedback}], new_state

    def _verify_special_chars(
        self,
        response: str,
        expected_chars: List[str]
    ) -> bool:
        """Verify all required special characters are present"""
        if not expected_chars:
            return True  # No special chars required

        response_chars = set(c for c in response if c in self.special_chars)
        expected_set = set(expected_chars)

        return expected_set.issubset(response_chars)

    def _verify_affixes(
        self,
        response: str,
        required_affixes: List[str]
    ) -> bool:
        """Verify required affixes are present"""
        if not required_affixes:
            return True  # No affixes required

        # Check each affix is present in response
        for affix in required_affixes:
            # Strip prefix/suffix markers (-, -)
            affix_clean = affix.strip("-")

            # Check if affix appears in response
            if affix.startswith("-") and affix.endswith("-"):
                # Infix (rare) - just check presence
                if affix_clean not in response:
                    return False
            elif affix.startswith("-"):
                # Suffix - check at end of a word
                if not re.search(rf'\w+{re.escape(affix_clean)}\b', response):
                    return False
            elif affix.endswith("-"):
                # Prefix - check at start of a word
                if not re.search(rf'\b{re.escape(affix_clean)}\w+', response):
                    return False
            else:
                # Standalone affix marker - just check presence
                if affix_clean not in response:
                    return False

        return True

    def _verify_semantic(
        self,
        response: str,
        expected: str,
        task_info: Dict
    ) -> tuple[bool, float]:
        """
        Verify semantic correctness

        Returns:
            (is_correct, similarity_score)
        """
        # Normalize for comparison
        response_norm = response.strip().lower()
        expected_norm = expected.strip().lower()

        # Exact match
        if response_norm == expected_norm:
            return True, 1.0

        # For translation tasks, allow some flexibility
        task_type = task_info.get("task_type", "")
        if task_type in ["word_translation", "sentence_translation"]:
            # Simple word overlap for now (could use better similarity)
            response_words = set(response_norm.split())
            expected_words = set(expected_norm.split())

            if not expected_words:
                return False, 0.0

            overlap = len(response_words & expected_words)
            similarity = overlap / len(expected_words)

            # Accept if >80% overlap
            return similarity > 0.8, similarity

        # For morphology tasks, require exact match
        return False, 0.0

    def _generate_feedback(
        self,
        response: str,
        expected: str,
        state: Dict,
        task_info: Dict
    ) -> str:
        """Generate helpful feedback based on what's wrong"""

        # Perfect answer
        if (state["special_chars_correct"] and
            state["affixes_correct"] and
            state["semantic_correct"]):
            return "✓ Correct! Well done."

        # Build specific feedback
        feedback_parts = []

        # Check special characters
        if not state["special_chars_correct"]:
            expected_chars = task_info.get("special_chars", [])
            response_chars = set(c for c in response if c in self.special_chars)
            missing = set(expected_chars) - response_chars

            if missing:
                feedback_parts.append(
                    f"Missing special characters: {', '.join(sorted(missing))}"
                )
            else:
                feedback_parts.append(
                    "Special characters present but placement may be wrong"
                )

        # Check affixes (morphology tasks)
        if not state["affixes_correct"] and task_info.get("task_type") == "morphology":
            required_affixes = task_info.get("required_affixes", [])
            if required_affixes:
                feedback_parts.append(
                    f"Required affixes: {', '.join(required_affixes)}"
                )

        # Check semantic accuracy
        if not state["semantic_correct"]:
            similarity = state.get("partial_credit", 0.0)
            if similarity > 0.5:
                feedback_parts.append(
                    "Close, but not quite right. Check the exact form."
                )
            else:
                feedback_parts.append(
                    f"Incorrect. Expected: {expected}"
                )

        # Combine feedback
        if feedback_parts:
            return " | ".join(feedback_parts)
        else:
            return "Check your answer carefully."


class DakotaMorphologyEnv(SingleTurnEnv):
    """
    Single-turn environment for simple morphology tasks
    (faster than multi-turn for basic transformations)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.special_chars = set("ćšŋḣṡáéíóúķśṅźėčžʼ")

    async def check_answer(
        self,
        messages: List[Dict],
        **kwargs
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Check if morphological transformation is correct

        Returns:
            (is_correct, metadata)
        """
        response = messages[-1]["content"]
        expected = kwargs.get("answer", "")
        task_info = kwargs.get("info", {})

        # Exact match check
        response_norm = response.strip().lower()
        expected_norm = expected.strip().lower()

        if response_norm == expected_norm:
            return True, {
                "exact_match": True,
                "char_accuracy": 1.0,
                "affix_accuracy": 1.0
            }

        # Partial credit calculation
        metadata = {}

        # Character accuracy
        expected_chars = task_info.get("special_chars", [])
        if expected_chars:
            response_chars = set(c for c in response if c in self.special_chars)
            char_accuracy = len(set(expected_chars) & response_chars) / len(expected_chars)
            metadata["char_accuracy"] = char_accuracy
        else:
            metadata["char_accuracy"] = 1.0

        # Affix accuracy
        required_affixes = task_info.get("required_affixes", [])
        if required_affixes:
            affixes_present = sum(
                1 for affix in required_affixes
                if affix.strip("-") in response
            )
            affix_accuracy = affixes_present / len(required_affixes)
            metadata["affix_accuracy"] = affix_accuracy
        else:
            metadata["affix_accuracy"] = 1.0

        metadata["exact_match"] = False

        # Only accept if both metrics > 0.9
        is_correct = (
            metadata["char_accuracy"] > 0.9 and
            metadata["affix_accuracy"] > 0.9
        )

        return is_correct, metadata


if __name__ == "__main__":
    # Test the environment
    import asyncio

    async def test_env():
        env = DakotaGrammarEnv(max_turns=3)

        # Test morphology task (from actual extraction)
        task = {
            "prompt": "Apply morphological transformation to 'suŋka' meaning 'younger brother'",
            "answer": "Dawid suŋkaku",
            "info": {
                "task_type": "morphology",
                "base_form": "suŋka",
                "required_affixes": ["-ku"],
                "special_chars": ["ŋ"],
                "difficulty": "advanced"
            }
        }

        print("Task:", task["prompt"])
        print("Expected:", task["answer"])
        print()

        # Test correct answer
        messages = [
            {"role": "user", "content": task["prompt"]},
            {"role": "assistant", "content": "Dawid suŋkaku"}
        ]
        state = {}

        is_done = await env.is_completed(messages, state, **task)
        new_messages, new_state = await env.env_response(messages, state, **task)

        print("Correct answer:")
        print(f"  Complete: {is_done}")
        print(f"  State: {new_state}")
        print(f"  Feedback: {new_messages[0]['content']}")
        print()

        # Test wrong answer (missing special char)
        messages_wrong = [
            {"role": "user", "content": task["prompt"]},
            {"role": "assistant", "content": "Dawid sunkaku"}  # Missing ŋ
        ]
        state_wrong = {}

        new_messages_wrong, new_state_wrong = await env.env_response(
            messages_wrong, state_wrong, **task
        )

        print("Wrong answer (missing ŋ):")
        print(f"  State: {new_state_wrong}")
        print(f"  Feedback: {new_messages_wrong[0]['content']}")

    asyncio.run(test_env())
