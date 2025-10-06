"""
Dakota Grammar Schema for RL Training

This schema is designed to extract testable grammar rules from pages 1-88
of the Dakota Grammar & Dictionary, optimized for Reinforcement Learning
training with PrimeIntellect verifiers.

Key differences from dictionary_schema.py:
- Focus on extractable, verifiable grammar rules
- RL task generation metadata
- Morphological transformation patterns
- Multi-turn dialogue support
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
import json
from enum import Enum


class GrammarRuleType(Enum):
    """Types of grammar rules that can be extracted"""
    MORPHOLOGY = "morphology"  # Prefix/suffix transformations
    SYNTAX = "syntax"  # Word order, sentence structure
    PHONOLOGY = "phonology"  # Sound changes, vowel harmony
    SEMANTICS = "semantics"  # Meaning composition
    ORTHOGRAPHY = "orthography"  # Writing conventions


class TaskDifficulty(Enum):
    """Difficulty levels for RL curriculum"""
    BASIC = "basic"  # Single transformation, common chars
    INTERMEDIATE = "intermediate"  # Multiple transformations
    ADVANCED = "advanced"  # Complex rules, rare chars
    EXPERT = "expert"  # Multiple rules, edge cases


@dataclass
class MorphologicalTransformation:
    """
    A single morphological transformation example
    (e.g., root + affix → inflected form)
    """
    base_form: str  # Root word (e.g., "iŋhiŋ")
    transformed_form: str  # Result (e.g., "éiŋhiŋtku")
    affixes: List[str]  # Applied affixes (e.g., ["é-", "-ku"])
    gloss_base: str  # English meaning of base (e.g., "son")
    gloss_transformed: str  # English meaning transformed (e.g., "his son")
    special_chars: List[str] = field(default_factory=list)  # ["ŋ"]
    phonological_changes: Optional[str] = None  # "vowel harmony applied"

    def to_rl_task(self) -> Dict[str, Any]:
        """Convert to RL task format for verifiers"""
        return {
            "prompt": f"Apply morphological transformation to '{self.base_form}' meaning '{self.gloss_base}'",
            "answer": self.transformed_form,
            "info": {
                "task_type": "morphology",
                "base_form": self.base_form,
                "required_affixes": self.affixes,
                "special_chars": self.special_chars,
                "expected_gloss": self.gloss_transformed,
                "phonological_changes": self.phonological_changes
            }
        }


@dataclass
class InterlinearExample:
    """
    Interlinear translation with word-by-word alignment
    """
    dakota_text: str  # Full Dakota sentence
    word_glosses: List[str]  # Word-by-word glosses
    english_translation: str  # Full English translation
    dakota_words: Optional[List[str]] = None  # Split Dakota words
    morpheme_breakdown: Optional[List[List[str]]] = None  # Per-word morphemes
    special_characters_found: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.dakota_words is None:
            self.dakota_words = self.dakota_text.split()

    def to_translation_tasks(self) -> List[Dict[str, Any]]:
        """Generate multiple RL tasks from one interlinear example"""
        tasks = []

        # Word-level translation tasks
        for dakota_word, gloss in zip(self.dakota_words, self.word_glosses):
            tasks.append({
                "prompt": f"Translate Dakota word to English: {dakota_word}",
                "answer": gloss,
                "info": {
                    "task_type": "word_translation",
                    "context_sentence": self.dakota_text,
                    "full_translation": self.english_translation,
                    "special_chars": self._extract_special_chars(dakota_word)
                }
            })

        # Sentence-level translation
        tasks.append({
            "prompt": f"Translate Dakota sentence to English: {self.dakota_text}",
            "answer": self.english_translation,
            "info": {
                "task_type": "sentence_translation",
                "word_count": len(self.dakota_words),
                "word_glosses": self.word_glosses,
                "special_chars": self.special_characters_found
            }
        })

        # Reverse translation (English → Dakota)
        tasks.append({
            "prompt": f"Translate English to Dakota: {self.english_translation}",
            "answer": self.dakota_text,
            "info": {
                "task_type": "reverse_translation",
                "word_glosses": self.word_glosses,
                "special_chars": self.special_characters_found,
                "difficulty": "advanced"
            }
        })

        return tasks

    @staticmethod
    def _extract_special_chars(text: str) -> List[str]:
        special = set("ćšŋḣṡáéíóúķśṅźėčž")
        return sorted(set(c for c in text if c in special))


@dataclass
class GrammarRule:
    """
    A complete grammar rule with examples and verification criteria
    """
    rule_id: str  # Unique identifier
    rule_type: GrammarRuleType
    rule_name: str  # Human-readable name
    description: str  # Full description of the rule

    # Rule specification
    pattern: str  # Formal pattern (e.g., "{root} + -ku → {root}-ku")
    constraints: List[str] = field(default_factory=list)  # Conditions for rule
    exceptions: List[str] = field(default_factory=list)  # When rule doesn't apply

    # Examples
    transformations: List[MorphologicalTransformation] = field(default_factory=list)
    interlinear_examples: List[InterlinearExample] = field(default_factory=list)

    # Metadata
    chapter: Optional[str] = None
    page_number: Optional[int] = None
    difficulty: TaskDifficulty = TaskDifficulty.INTERMEDIATE
    testable: bool = True  # Can this be turned into RL task?

    # Verification
    verification_criteria: List[str] = field(default_factory=list)
    reward_function: Optional[str] = None  # Custom reward function name

    def generate_rl_tasks(self, n_samples: int = 10) -> List[Dict[str, Any]]:
        """Generate synthetic RL training tasks from this rule"""
        tasks = []

        # Morphological transformation tasks
        for i, transform in enumerate(self.transformations[:n_samples]):
            task = transform.to_rl_task()
            task["info"]["rule_id"] = self.rule_id
            task["info"]["rule_name"] = self.rule_name
            task["info"]["difficulty"] = self.difficulty.value
            task["info"]["verification_criteria"] = self.verification_criteria
            tasks.append(task)

        # Translation tasks from interlinear examples
        for example in self.interlinear_examples[:n_samples]:
            example_tasks = example.to_translation_tasks()
            for task in example_tasks:
                task["info"]["rule_id"] = self.rule_id
                task["info"]["rule_name"] = self.rule_name
            tasks.extend(example_tasks)

        return tasks

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data["rule_type"] = self.rule_type.value
        data["difficulty"] = self.difficulty.value
        return data

    def to_json(self, indent=2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class GrammarPage:
    """
    Complete extraction from one grammar page (pages 1-88)
    """
    page_number: int
    chapter: Optional[str] = None
    section_title: Optional[str] = None

    # Extracted content
    grammar_rules: List[GrammarRule] = field(default_factory=list)
    interlinear_examples: List[InterlinearExample] = field(default_factory=list)
    linguistic_notes: Optional[str] = None

    # Extraction metadata
    source_image: Optional[str] = None
    quality_issues: Optional[str] = None
    extraction_confidence: float = 1.0

    def generate_all_rl_tasks(self) -> List[Dict[str, Any]]:
        """Generate all RL tasks from this page"""
        all_tasks = []

        for rule in self.grammar_rules:
            if rule.testable:
                tasks = rule.generate_rl_tasks()
                all_tasks.extend(tasks)

        # Also generate tasks from standalone interlinear examples
        for example in self.interlinear_examples:
            tasks = example.to_translation_tasks()
            all_tasks.extend(tasks)

        return all_tasks

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self, indent=2) -> str:
        """Convert to JSON string"""
        data = self.to_dict()
        # Convert enums
        for rule in data.get("grammar_rules", []):
            if "rule_type" in rule:
                rule["rule_type"] = rule["rule_type"] if isinstance(rule["rule_type"], str) else rule["rule_type"].value
            if "difficulty" in rule:
                rule["difficulty"] = rule["difficulty"] if isinstance(rule["difficulty"], str) else rule["difficulty"].value
        return json.dumps(data, indent=indent, ensure_ascii=False)


def validate_grammar_rule(rule: GrammarRule) -> tuple[bool, List[str]]:
    """
    Validate a grammar rule for RL training

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    # Required fields
    if not rule.rule_id:
        issues.append("Missing rule_id")
    if not rule.rule_name:
        issues.append("Missing rule_name")
    if not rule.description:
        issues.append("Missing description")
    if not rule.pattern:
        issues.append("Missing pattern specification")

    # Must have examples if testable
    if rule.testable:
        if not rule.transformations and not rule.interlinear_examples:
            issues.append("Testable rule must have examples")

    # Verification criteria for RL
    if rule.testable and not rule.verification_criteria:
        issues.append("Warning: No verification criteria specified")

    return len(issues) == 0, issues


# Example grammar rules for testing
EXAMPLE_GRAMMAR_RULES = [
    GrammarRule(
        rule_id="dakota_possessive_01",
        rule_type=GrammarRuleType.MORPHOLOGY,
        rule_name="Third-person possessive suffix -ku",
        description="Add -ku suffix to noun to indicate 'his/her/its'",
        pattern="{noun} + -ku → {noun}-ku (his/her {noun})",
        constraints=["Applies to kinship terms", "May trigger vowel changes"],
        verification_criteria=[
            "Suffix -ku is present",
            "Special characters preserved",
            "Meaning includes possessive"
        ],
        transformations=[
            MorphologicalTransformation(
                base_form="iŋhiŋ",
                transformed_form="éiŋhiŋtku",
                affixes=["é-", "-ku"],
                gloss_base="son",
                gloss_transformed="his son",
                special_chars=["ŋ"],
                phonological_changes="é- prefix added"
            )
        ],
        difficulty=TaskDifficulty.BASIC,
        testable=True
    )
]


if __name__ == "__main__":
    # Test grammar rule creation
    rule = EXAMPLE_GRAMMAR_RULES[0]

    print("Grammar Rule:")
    print(rule.to_json())
    print("\n" + "="*80 + "\n")

    print("Generated RL Tasks:")
    tasks = rule.generate_rl_tasks(n_samples=3)
    for i, task in enumerate(tasks, 1):
        print(f"\nTask {i}:")
        print(json.dumps(task, indent=2, ensure_ascii=False))

    print("\n" + "="*80 + "\n")
    print("Validation:")
    is_valid, issues = validate_grammar_rule(rule)
    print(f"Valid: {is_valid}")
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")
