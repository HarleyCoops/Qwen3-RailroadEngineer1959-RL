#!/usr/bin/env python3
"""
Organize Extracted Grammar Rules for RL Training

This script processes extracted grammar rules and converts them into
training rules for the Dakota grammar RL environment.

RL Rule Format:
- Each rule defines a constraint that the RL agent must learn to follow
- Rules are structured as verifiable conditions with test cases
- Rules are organized by linguistic category (morphology, syntax, etc.)
"""

import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class RLTrainingRule:
    """A single training rule for the RL environment."""
    rule_id: str
    rule_type: str  # morphology, syntax, phonology, etc.
    rule_name: str
    rule_description: str

    # The pattern or constraint
    dakota_pattern: str
    english_explanation: str

    # Positive examples (correct usage)
    positive_examples: List[Dict[str, str]]

    # Negative examples (incorrect usage) - can be generated
    negative_examples: List[Dict[str, str]]

    # Verification function (as string for now)
    verification_pattern: str

    # Metadata
    source_pages: List[int]
    confidence: float
    difficulty: str  # easy, medium, hard

    # Additional constraints or notes
    constraints: str
    linguistic_notes: str


@dataclass
class RLRuleSet:
    """Collection of rules organized by category."""
    category: str  # morphology, syntax, phonology, particles
    rules: List[RLTrainingRule]
    total_examples: int
    avg_confidence: float
    description: str


def extract_positive_examples(grammar_rule: dict) -> List[Dict[str, str]]:
    """Extract positive examples from grammar rule."""
    examples = []

    for ex in grammar_rule.get('examples', []):
        if ex.get('dakota'):
            example = {
                'dakota': ex['dakota'],
                'english': ex.get('english', ''),
                'gloss': ex.get('gloss', ''),
                'notes': ex.get('notes', '')
            }
            examples.append(example)

    return examples


def generate_negative_examples(dakota_pattern: str, positive_examples: List[Dict]) -> List[Dict[str, str]]:
    """Generate negative examples by violating the pattern."""
    # This is a placeholder - in practice, we'd need pattern-specific generation
    negative = []

    # For now, just mark that negatives should be generated
    if positive_examples:
        negative.append({
            'dakota': '[TO BE GENERATED: violation of pattern]',
            'english': '[incorrect form]',
            'gloss': '',
            'notes': f'Should violate: {dakota_pattern}'
        })

    return negative


def estimate_difficulty(rule: dict) -> str:
    """Estimate rule difficulty based on complexity."""
    confidence = rule.get('confidence', 0.5)
    num_examples = len(rule.get('examples', []))

    # Low confidence or few examples = harder
    if confidence < 0.6 or num_examples < 2:
        return 'hard'
    elif confidence < 0.8 or num_examples < 4:
        return 'medium'
    else:
        return 'easy'


def create_verification_pattern(dakota_pattern: str, rule_type: str) -> str:
    """Create a verification pattern for the rule."""
    # This would be expanded to create actual regex or parsing patterns
    # For now, return the pattern as-is
    return f"verify_{rule_type}: {dakota_pattern}"


def convert_grammar_rule_to_rl(grammar_rule: dict, page_number: int) -> RLTrainingRule:
    """Convert extracted grammar rule to RL training format."""

    positive_examples = extract_positive_examples(grammar_rule)
    negative_examples = generate_negative_examples(
        grammar_rule.get('dakota_pattern', ''),
        positive_examples
    )

    return RLTrainingRule(
        rule_id=grammar_rule.get('rule_id', f'rule_p{page_number}_unknown'),
        rule_type=grammar_rule.get('rule_type', 'unknown'),
        rule_name=grammar_rule.get('rule_description', '').split('.')[0][:100],
        rule_description=grammar_rule.get('rule_description', ''),
        dakota_pattern=grammar_rule.get('dakota_pattern', ''),
        english_explanation=grammar_rule.get('english_explanation', ''),
        positive_examples=positive_examples,
        negative_examples=negative_examples,
        verification_pattern=create_verification_pattern(
            grammar_rule.get('dakota_pattern', ''),
            grammar_rule.get('rule_type', 'unknown')
        ),
        source_pages=[page_number],
        confidence=grammar_rule.get('confidence', 0.5),
        difficulty=estimate_difficulty(grammar_rule),
        constraints=grammar_rule.get('constraints', ''),
        linguistic_notes=''
    )


def extract_interlinear_examples(interlinear: dict, page_number: int) -> List[RLTrainingRule]:
    """Convert interlinear texts to example-based rules."""
    rules = []

    # Each interlinear text can be an example of multiple patterns
    text_id = interlinear.get('text_id', f'interlinear_p{page_number}')

    # Create a general translation rule from the interlinear
    if interlinear.get('dakota_lines'):
        dakota_text = ' '.join(interlinear['dakota_lines'])

        rule = RLTrainingRule(
            rule_id=f"{text_id}_translation",
            rule_type='translation',
            rule_name=f"Translation example from page {page_number}",
            rule_description="Complete sentence translation with interlinear gloss",
            dakota_pattern=dakota_text,
            english_explanation=interlinear.get('english_translation', ''),
            positive_examples=[{
                'dakota': dakota_text,
                'english': interlinear.get('english_translation', ''),
                'gloss': ' '.join(interlinear.get('gloss_lines', [])),
                'notes': interlinear.get('linguistic_notes', '')
            }],
            negative_examples=[],
            verification_pattern='translation_accuracy',
            source_pages=[page_number],
            confidence=interlinear.get('confidence', 0.7),
            difficulty='medium',
            constraints='',
            linguistic_notes=interlinear.get('linguistic_notes', '')
        )
        rules.append(rule)

    return rules


def organize_by_category(rl_rules: List[RLTrainingRule]) -> Dict[str, RLRuleSet]:
    """Organize rules by linguistic category."""

    categories = defaultdict(list)

    for rule in rl_rules:
        categories[rule.rule_type].append(rule)

    # Create RLRuleSet for each category
    rule_sets = {}

    for category, rules in categories.items():
        total_examples = sum(
            len(r.positive_examples) + len(r.negative_examples)
            for r in rules
        )

        avg_conf = sum(r.confidence for r in rules) / len(rules) if rules else 0.0

        rule_sets[category] = RLRuleSet(
            category=category,
            rules=rules,
            total_examples=total_examples,
            avg_confidence=avg_conf,
            description=f"{category.title()} rules extracted from Dakota grammar"
        )

    return rule_sets


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Organize extracted grammar rules for RL training"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/grammar_extracted",
        help="Directory with extracted grammar JSON files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/rl_training_rules",
        help="Output directory for RL training rules"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for rules"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print(" ORGANIZING GRAMMAR RULES FOR RL TRAINING")
    print("="*70)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all extracted grammar pages
    grammar_files = sorted(input_dir.glob("grammar_page_*.json"))

    if not grammar_files:
        print(f"\nNo grammar files found in {input_dir}")
        print("Run: python extract_grammar_pages.py --pages 1-88")
        return

    print(f"\nFound {len(grammar_files)} grammar pages")
    print(f"Minimum confidence: {args.min_confidence}")

    # Process all rules
    all_rl_rules = []

    stats = {
        'total_pages': 0,
        'total_rules': 0,
        'total_examples': 0,
        'total_interlinear': 0,
        'rules_filtered': 0
    }

    for grammar_file in grammar_files:
        with open(grammar_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        page_num = data.get('page_number', 0)
        stats['total_pages'] += 1

        # Convert grammar rules
        for rule in data.get('grammar_rules', []):
            stats['total_rules'] += 1

            if rule.get('confidence', 0) < args.min_confidence:
                stats['rules_filtered'] += 1
                continue

            rl_rule = convert_grammar_rule_to_rl(rule, page_num)
            all_rl_rules.append(rl_rule)
            stats['total_examples'] += len(rl_rule.positive_examples)

        # Convert interlinear texts
        for interlinear in data.get('interlinear_texts', []):
            stats['total_interlinear'] += 1

            if interlinear.get('confidence', 0) < args.min_confidence:
                continue

            interlinear_rules = extract_interlinear_examples(interlinear, page_num)
            all_rl_rules.extend(interlinear_rules)

    print(f"\nExtracted {len(all_rl_rules)} RL training rules")
    print(f"  Filtered out {stats['rules_filtered']} low-confidence rules")

    # Organize by category
    rule_sets = organize_by_category(all_rl_rules)

    print(f"\nOrganized into {len(rule_sets)} categories:")
    for category, rule_set in rule_sets.items():
        print(f"  {category:15s}: {len(rule_set.rules):3d} rules, "
              f"{rule_set.total_examples:3d} examples, "
              f"avg conf: {rule_set.avg_confidence:.2f}")

    # Save organized rules
    print(f"\nSaving to {output_dir}/")

    # Save by category
    for category, rule_set in rule_sets.items():
        category_file = output_dir / f"rules_{category}.json"

        with open(category_file, 'w', encoding='utf-8') as f:
            json.dump({
                'category': rule_set.category,
                'description': rule_set.description,
                'total_rules': len(rule_set.rules),
                'total_examples': rule_set.total_examples,
                'avg_confidence': rule_set.avg_confidence,
                'rules': [asdict(rule) for rule in rule_set.rules]
            }, f, indent=2, ensure_ascii=False)

        print(f"  Saved {category_file.name}")

    # Save complete combined file
    combined_file = output_dir / "all_rl_rules.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_rules': len(all_rl_rules),
            'categories': list(rule_sets.keys()),
            'statistics': stats,
            'rules': [asdict(rule) for rule in all_rl_rules]
        }, f, indent=2, ensure_ascii=False)

    print(f"  Saved {combined_file.name}")

    # Generate summary report
    summary_file = output_dir / "rl_rules_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Dakota Grammar RL Training Rules Summary\n")
        f.write("="*70 + "\n\n")

        f.write(f"Source pages: {stats['total_pages']}\n")
        f.write(f"Total rules extracted: {stats['total_rules']}\n")
        f.write(f"Rules after filtering: {len(all_rl_rules)}\n")
        f.write(f"Rules filtered (low confidence): {stats['rules_filtered']}\n")
        f.write(f"Total positive examples: {stats['total_examples']}\n")
        f.write(f"Interlinear texts: {stats['total_interlinear']}\n\n")

        f.write("Rules by Category:\n")
        f.write("-"*70 + "\n")
        for category, rule_set in sorted(rule_sets.items()):
            f.write(f"\n{category.upper()}\n")
            f.write(f"  Rules: {len(rule_set.rules)}\n")
            f.write(f"  Examples: {rule_set.total_examples}\n")
            f.write(f"  Avg confidence: {rule_set.avg_confidence:.2f}\n")
            f.write(f"  Description: {rule_set.description}\n")

            # List first 5 rules
            f.write("  Sample rules:\n")
            for rule in rule_set.rules[:5]:
                f.write(f"    - {rule.rule_name[:60]}\n")

        f.write("\n\nNext Steps:\n")
        f.write("-"*70 + "\n")
        f.write("1. Review rules in data/rl_training_rules/rules_*.json\n")
        f.write("2. Generate negative examples for each rule\n")
        f.write("3. Implement verification functions\n")
        f.write("4. Create RL environment with these rules\n")
        f.write("5. Train agent on Dakota grammar\n")

    print(f"  Saved {summary_file.name}")

    print("\n" + "="*70)
    print(" ORGANIZATION COMPLETE")
    print("="*70)
    print(f"\n{len(all_rl_rules)} rules ready for RL training")
    print(f"Output: {output_dir}/")
    print(f"\nReview: {summary_file}")
    print()


if __name__ == "__main__":
    main()
