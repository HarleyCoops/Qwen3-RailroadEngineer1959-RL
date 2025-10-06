#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Extracted Dakota Grammar Rules to PrimeIntellect Task Format

Takes the 1,036 rules from data/rl_training_rules/ and converts them
to JSONL format for PrimeIntellect training.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import re


def extract_special_chars(text: str) -> List[str]:
    """Extract Dakota special characters from text"""
    special_chars = set("ćšŋḣṡáéíóúķśṅźėčžʼ")
    return sorted(list(set(c for c in text if c in special_chars)))


def extract_affixes(rule: Dict) -> List[str]:
    """Extract affixes from rule pattern"""
    pattern = rule.get('dakota_pattern', '')

    # Look for affix markers: -suffix, prefix-, or explicit mentions
    affixes = []

    # Find patterns like "-ku", "ta-", "-ću"
    affix_pattern = r'(?:^|\s)((?:\w+)?-\w+|-\w+(?:\w+)?|[\w]+-)(?:\s|$|,|\.)'
    matches = re.findall(affix_pattern, pattern)
    affixes.extend(matches)

    # Also check in constraints
    constraints = rule.get('constraints', '')
    matches = re.findall(affix_pattern, constraints)
    affixes.extend(matches)

    return list(set(affixes))


def create_morphology_task(rule: Dict, example: Dict) -> Optional[Dict]:
    """Create morphology task from rule and example"""

    dakota_text = example.get('dakota', '')
    if not dakota_text:
        return None

    # Create prompt
    prompt = f"Apply this Dakota grammar rule: {rule['rule_description']}\n\n"

    if rule.get('dakota_pattern'):
        prompt += f"Pattern: {rule['dakota_pattern']}\n\n"

    if example.get('notes'):
        prompt += f"Context: {example['notes']}\n\n"

    prompt += f"Transform or analyze: {dakota_text}"

    # Answer is the English explanation or gloss
    answer = example.get('english', dakota_text)

    return {
        "prompt": prompt,
        "answer": answer,
        "info": {
            "task_type": "morphology",
            "rule_id": rule['rule_id'],
            "rule_type": rule['rule_type'],
            "base_form": dakota_text,
            "required_affixes": extract_affixes(rule),
            "special_chars": extract_special_chars(dakota_text),
            "difficulty": rule['difficulty'],
            "source_pages": rule.get('source_pages', []),
            "confidence": rule.get('confidence', 0.0)
        }
    }


def create_translation_task(rule: Dict, example: Dict) -> Optional[Dict]:
    """Create translation task from rule and example"""

    dakota_text = example.get('dakota', '')
    english_text = example.get('english', '')

    if not dakota_text or not english_text:
        return None

    # Create prompt for Dakota → English translation
    prompt = f"Translate this Dakota sentence to English:\n\n{dakota_text}"

    if example.get('gloss'):
        prompt += f"\n\nGloss: {example['gloss']}"

    answer = english_text

    return {
        "prompt": prompt,
        "answer": answer,
        "info": {
            "task_type": "word_translation" if len(dakota_text.split()) <= 3 else "sentence_translation",
            "rule_id": rule['rule_id'],
            "rule_type": rule['rule_type'],
            "dakota_text": dakota_text,
            "special_chars": extract_special_chars(dakota_text),
            "difficulty": rule['difficulty'],
            "source_pages": rule.get('source_pages', []),
            "confidence": rule.get('confidence', 0.0)
        }
    }


def create_reverse_translation_task(rule: Dict, example: Dict) -> Optional[Dict]:
    """Create reverse translation task (English → Dakota)"""

    dakota_text = example.get('dakota', '')
    english_text = example.get('english', '')

    if not dakota_text or not english_text:
        return None

    # Create prompt for English → Dakota translation
    prompt = f"Translate this English sentence to Dakota:\n\n{english_text}"

    if example.get('notes'):
        prompt += f"\n\nNote: {example['notes']}"

    answer = dakota_text

    return {
        "prompt": prompt,
        "answer": answer,
        "info": {
            "task_type": "reverse_translation",
            "rule_id": rule['rule_id'],
            "rule_type": rule['rule_type'],
            "english_text": english_text,
            "special_chars": extract_special_chars(dakota_text),
            "difficulty": "advanced",  # Reverse translation is always hard
            "source_pages": rule.get('source_pages', []),
            "confidence": rule.get('confidence', 0.0)
        }
    }


def create_syntax_task(rule: Dict, example: Dict) -> Optional[Dict]:
    """Create syntax task from rule and example"""

    dakota_text = example.get('dakota', '')
    if not dakota_text:
        return None

    # Create prompt
    prompt = f"Analyze the syntax of this Dakota sentence:\n\n{dakota_text}\n\n"
    prompt += f"Rule: {rule['rule_description']}"

    if example.get('gloss'):
        prompt += f"\n\nGloss: {example['gloss']}"

    # Answer explains the syntax
    answer = example.get('english', '') or rule.get('english_explanation', '')

    return {
        "prompt": prompt,
        "answer": answer,
        "info": {
            "task_type": "syntax",
            "rule_id": rule['rule_id'],
            "rule_type": rule['rule_type'],
            "dakota_text": dakota_text,
            "special_chars": extract_special_chars(dakota_text),
            "difficulty": rule['difficulty'],
            "source_pages": rule.get('source_pages', []),
            "confidence": rule.get('confidence', 0.0)
        }
    }


def create_pattern_identification_task(rule: Dict) -> Optional[Dict]:
    """Create pattern identification task"""

    if not rule.get('dakota_pattern'):
        return None

    prompt = f"Identify the grammatical pattern in this Dakota rule:\n\n"
    prompt += f"{rule['rule_description']}\n\n"

    # Show examples if available
    if rule.get('positive_examples'):
        prompt += "Examples:\n"
        for ex in rule['positive_examples'][:2]:
            if ex.get('dakota'):
                prompt += f"  - {ex['dakota']}\n"

    answer = rule['dakota_pattern']

    return {
        "prompt": prompt,
        "answer": answer,
        "info": {
            "task_type": "identify_pattern",
            "rule_id": rule['rule_id'],
            "rule_type": rule['rule_type'],
            "pattern": rule['dakota_pattern'],
            "difficulty": rule['difficulty'],
            "source_pages": rule.get('source_pages', []),
            "confidence": rule.get('confidence', 0.0)
        }
    }


def convert_rule_to_tasks(rule: Dict) -> List[Dict]:
    """Convert a single rule to multiple training tasks"""

    tasks = []

    # Get examples
    positive_examples = rule.get('positive_examples', [])

    if not positive_examples:
        # If no examples, create a pattern identification task
        task = create_pattern_identification_task(rule)
        if task:
            tasks.append(task)
        return tasks

    # Create tasks based on rule type
    rule_type = rule.get('rule_type', '')

    for example in positive_examples:
        # Always create translation tasks if we have both Dakota and English
        if example.get('dakota') and example.get('english'):
            # Forward translation (Dakota → English)
            trans_task = create_translation_task(rule, example)
            if trans_task:
                tasks.append(trans_task)

            # Reverse translation (English → Dakota) - only for some difficulties
            if rule.get('difficulty') in ['medium', 'hard']:
                rev_task = create_reverse_translation_task(rule, example)
                if rev_task:
                    tasks.append(rev_task)

        # Type-specific tasks
        if rule_type == 'morphology':
            morph_task = create_morphology_task(rule, example)
            if morph_task:
                tasks.append(morph_task)

        elif rule_type == 'syntax':
            syntax_task = create_syntax_task(rule, example)
            if syntax_task:
                tasks.append(syntax_task)

    # Add pattern identification task
    if rule.get('dakota_pattern'):
        pattern_task = create_pattern_identification_task(rule)
        if pattern_task:
            tasks.append(pattern_task)

    return tasks


def main():
    print("\n" + "="*70)
    print(" CONVERTING RULES TO PRIMEINTELLECT FORMAT")
    print("="*70)

    # Load all rules
    rules_file = Path("data/rl_training_rules/all_rl_rules.json")

    if not rules_file.exists():
        print(f"\nERROR: Rules file not found: {rules_file}")
        print("Run: python organize_grammar_for_rl.py first")
        return

    with open(rules_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_rules = data['rules']
    print(f"\nLoaded {len(all_rules)} rules")

    # Convert all rules to tasks
    all_tasks = []
    stats = {
        'morphology': 0,
        'syntax': 0,
        'translation': 0,
        'reverse_translation': 0,
        'identify_pattern': 0,
        'word_translation': 0,
        'sentence_translation': 0
    }

    difficulty_stats = {
        'easy': 0,
        'medium': 0,
        'hard': 0
    }

    for rule in all_rules:
        tasks = convert_rule_to_tasks(rule)
        all_tasks.extend(tasks)

        # Track stats
        for task in tasks:
            task_type = task['info']['task_type']
            stats[task_type] = stats.get(task_type, 0) + 1

            difficulty = task['info']['difficulty']
            difficulty_stats[difficulty] = difficulty_stats.get(difficulty, 0) + 1

    print(f"\nGenerated {len(all_tasks)} training tasks")
    print(f"\nBy task type:")
    for task_type, count in sorted(stats.items()):
        if count > 0:
            print(f"  {task_type:25s}: {count:4d} tasks")

    print(f"\nBy difficulty:")
    for difficulty, count in sorted(difficulty_stats.items()):
        print(f"  {difficulty:10s}: {count:4d} tasks")

    # Save to JSONL
    output_dir = Path("dakota_rl_training/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "grammar_tasks_complete.jsonl"

    with open(output_file, 'w', encoding='utf-8') as f:
        for task in all_tasks:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')

    print(f"\nSaved to: {output_file}")

    # Also save by difficulty for curriculum learning
    for difficulty in ['easy', 'medium', 'hard']:
        difficulty_tasks = [t for t in all_tasks if t['info']['difficulty'] == difficulty]

        if difficulty_tasks:
            diff_file = output_dir / f"grammar_tasks_{difficulty}.jsonl"
            with open(diff_file, 'w', encoding='utf-8') as f:
                for task in difficulty_tasks:
                    f.write(json.dumps(task, ensure_ascii=False) + '\n')
            print(f"  Saved {len(difficulty_tasks)} {difficulty} tasks to: {diff_file.name}")

    # Generate sample tasks file for inspection
    sample_file = output_dir / "sample_tasks.json"
    sample_tasks = all_tasks[:10]

    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_tasks, f, indent=2, ensure_ascii=False)

    print(f"\nSample tasks (first 10): {sample_file}")

    # Print statistics
    print("\n" + "="*70)
    print(" CONVERSION COMPLETE")
    print("="*70)
    print(f"\nTotal tasks: {len(all_tasks)}")
    print(f"Source rules: {len(all_rules)}")
    print(f"Avg tasks per rule: {len(all_tasks) / len(all_rules):.1f}")

    print("\nNext steps:")
    print("  1. Review sample tasks: cat dakota_rl_training/datasets/sample_tasks.json")
    print("  2. Update training config: edit dakota_rl_training/configs/training_config.yaml")
    print("  3. Test verifier: python test_verifier_integration.py")
    print("  4. Run training: python dakota_rl_training/train.py")
    print()


if __name__ == "__main__":
    main()
