#!/usr/bin/env python3
"""
Create Dakota Grammar RL Training Environment

This script creates the RL environment that uses extracted grammar rules
to train an agent on Dakota language structure.

The environment:
- Presents grammar challenges based on extracted rules
- Rewards correct application of Dakota grammar
- Penalizes rule violations
- Tracks learning progress across rule categories
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """Types of actions the agent can take."""
    APPLY_RULE = "apply_rule"
    TRANSLATE = "translate"
    CORRECT_ERROR = "correct_error"
    IDENTIFY_PATTERN = "identify_pattern"


@dataclass
class GrammarState:
    """Current state in the RL environment."""
    current_rule_id: str
    rule_category: str
    challenge_type: ActionType
    context: Dict
    available_actions: List[str]
    step_number: int
    score: float


@dataclass
class GrammarReward:
    """Reward structure for RL training."""
    base_reward: float
    accuracy_bonus: float
    difficulty_multiplier: float
    category_bonus: float
    total: float


class DakotaGrammarEnvironment:
    """RL Environment for Dakota Grammar Learning."""

    def __init__(self, rules_dir: str):
        """Initialize environment with extracted rules."""
        self.rules_dir = Path(rules_dir)
        self.rules_by_category = {}
        self.all_rules = []
        self.current_state = None
        self.episode_history = []

        # Load all rules
        self._load_rules()

        # Environment parameters
        self.max_steps_per_episode = 100
        self.difficulty_curve = "progressive"  # easy -> hard

        print(f"Environment initialized with {len(self.all_rules)} rules")
        print(f"Categories: {list(self.rules_by_category.keys())}")

    def _load_rules(self):
        """Load all RL training rules."""
        rule_files = list(self.rules_dir.glob("rules_*.json"))

        if not rule_files:
            raise ValueError(f"No rule files found in {self.rules_dir}")

        for rule_file in rule_files:
            with open(rule_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            category = data['category']
            rules = data['rules']

            self.rules_by_category[category] = rules
            self.all_rules.extend(rules)

        print(f"Loaded {len(self.all_rules)} rules from {len(rule_files)} categories")

    def reset(self) -> GrammarState:
        """Reset environment for new episode."""
        self.episode_history = []

        # Start with an easy rule
        easy_rules = [r for r in self.all_rules if r['difficulty'] == 'easy']

        if not easy_rules:
            easy_rules = self.all_rules

        rule = random.choice(easy_rules)

        # Create initial state
        self.current_state = self._create_state_from_rule(rule, step_number=0)

        return self.current_state

    def _create_state_from_rule(self, rule: Dict, step_number: int) -> GrammarState:
        """Create a state from a rule."""

        # Determine challenge type
        if rule.get('positive_examples'):
            challenge_type = random.choice([
                ActionType.APPLY_RULE,
                ActionType.IDENTIFY_PATTERN,
                ActionType.TRANSLATE
            ])
        else:
            challenge_type = ActionType.IDENTIFY_PATTERN

        # Get available actions (simplified for now)
        available_actions = self._generate_actions_for_rule(rule, challenge_type)

        return GrammarState(
            current_rule_id=rule['rule_id'],
            rule_category=rule['rule_type'],
            challenge_type=challenge_type,
            context={
                'rule_description': rule['rule_description'],
                'dakota_pattern': rule['dakota_pattern'],
                'english_explanation': rule['english_explanation'],
                'examples': rule.get('positive_examples', [])[:3],  # First 3 examples
                'constraints': rule.get('constraints', '')
            },
            available_actions=available_actions,
            step_number=step_number,
            score=0.0
        )

    def _generate_actions_for_rule(self, rule: Dict, challenge_type: ActionType) -> List[str]:
        """Generate possible actions for a rule."""

        actions = []

        if challenge_type == ActionType.APPLY_RULE:
            # Generate variations of applying the pattern
            pattern = rule.get('dakota_pattern', '')
            actions = [
                f"apply_pattern: {pattern}",
                f"apply_with_modification: {pattern}",
                "skip_rule",
                "request_example"
            ]

        elif challenge_type == ActionType.TRANSLATE:
            # Translation options
            examples = rule.get('positive_examples', [])
            if examples:
                actions = [
                    f"translate: {ex.get('dakota', '')}"
                    for ex in examples[:3]
                ]
            actions.append("request_hint")

        elif challenge_type == ActionType.IDENTIFY_PATTERN:
            # Pattern identification
            actions = [
                f"pattern_is: {rule.get('rule_type')}",
                "pattern_is: unknown",
                "request_examples"
            ]

        elif challenge_type == ActionType.CORRECT_ERROR:
            # Error correction
            negative_examples = rule.get('negative_examples', [])
            actions = [
                f"correct: {ex.get('dakota', '')}"
                for ex in negative_examples[:3]
            ]

        return actions[:5]  # Limit to 5 actions

    def step(self, action: str) -> Tuple[GrammarState, GrammarReward, bool, Dict]:
        """Take a step in the environment."""

        if self.current_state is None:
            raise ValueError("Environment not initialized. Call reset() first.")

        # Evaluate action
        reward = self._evaluate_action(action, self.current_state)

        # Update state
        self.episode_history.append({
            'state': self.current_state,
            'action': action,
            'reward': reward
        })

        # Check if episode is done
        done = (
            self.current_state.step_number >= self.max_steps_per_episode or
            self._is_terminal_state(self.current_state, action)
        )

        # Get next state
        if not done:
            next_rule = self._select_next_rule(self.current_state, reward)
            self.current_state = self._create_state_from_rule(
                next_rule,
                step_number=self.current_state.step_number + 1
            )
            self.current_state.score = self.current_state.score + reward.total
        else:
            self.current_state = None

        # Info dict
        info = {
            'episode_length': len(self.episode_history),
            'total_reward': sum(h['reward'].total for h in self.episode_history),
            'rules_seen': len(set(h['state'].current_rule_id for h in self.episode_history))
        }

        return self.current_state, reward, done, info

    def _evaluate_action(self, action: str, state: GrammarState) -> GrammarReward:
        """Evaluate the quality of an action."""

        # Get current rule
        rule = next(r for r in self.all_rules if r['rule_id'] == state.current_rule_id)

        # Base evaluation (simplified - would need actual verification)
        base_reward = 0.0
        accuracy_bonus = 0.0

        if action in state.available_actions:
            base_reward = 1.0

            # Check if action matches expected pattern
            if state.challenge_type == ActionType.APPLY_RULE:
                if rule['dakota_pattern'] in action:
                    accuracy_bonus = 1.0

            elif state.challenge_type == ActionType.TRANSLATE:
                # Would need actual translation verification
                accuracy_bonus = 0.5

            elif state.challenge_type == ActionType.IDENTIFY_PATTERN:
                if rule['rule_type'] in action:
                    accuracy_bonus = 1.0

        # Difficulty multiplier
        difficulty_map = {'easy': 1.0, 'medium': 1.5, 'hard': 2.0}
        difficulty_mult = difficulty_map.get(rule.get('difficulty', 'medium'), 1.0)

        # Category bonus (encourage learning diverse rules)
        category_bonus = 0.0
        categories_seen = set(h['state'].rule_category for h in self.episode_history)
        if state.rule_category not in categories_seen:
            category_bonus = 0.5

        total = (base_reward + accuracy_bonus) * difficulty_mult + category_bonus

        return GrammarReward(
            base_reward=base_reward,
            accuracy_bonus=accuracy_bonus,
            difficulty_multiplier=difficulty_mult,
            category_bonus=category_bonus,
            total=total
        )

    def _is_terminal_state(self, state: GrammarState, action: str) -> bool:
        """Check if this is a terminal state."""
        # Terminal if agent requests to end or makes critical error
        return action in ['end_episode', 'give_up'] or state.step_number >= self.max_steps_per_episode

    def _select_next_rule(self, current_state: GrammarState, reward: GrammarReward) -> Dict:
        """Select next rule based on curriculum."""

        # Progressive difficulty: if doing well, increase difficulty
        if reward.total > 1.5:
            # Agent is doing well, try harder rules
            harder_rules = [
                r for r in self.all_rules
                if r['difficulty'] in ['medium', 'hard']
                and r['rule_id'] != current_state.current_rule_id
            ]
            if harder_rules:
                return random.choice(harder_rules)

        # If struggling, stay at current level or go easier
        if reward.total < 0.5:
            easier_rules = [
                r for r in self.all_rules
                if r['difficulty'] == 'easy'
                and r['rule_id'] != current_state.current_rule_id
            ]
            if easier_rules:
                return random.choice(easier_rules)

        # Otherwise, random rule from same or similar category
        similar_rules = [
            r for r in self.all_rules
            if r['rule_type'] == current_state.rule_category
            and r['rule_id'] != current_state.current_rule_id
        ]

        if similar_rules:
            return random.choice(similar_rules)

        # Fallback: any rule
        return random.choice(self.all_rules)

    def get_statistics(self) -> Dict:
        """Get environment statistics."""
        if not self.episode_history:
            return {}

        total_reward = sum(h['reward'].total for h in self.episode_history)
        avg_reward = total_reward / len(self.episode_history)

        categories_seen = set(h['state'].rule_category for h in self.episode_history)
        rules_seen = set(h['state'].current_rule_id for h in self.episode_history)

        return {
            'episode_length': len(self.episode_history),
            'total_reward': total_reward,
            'average_reward': avg_reward,
            'categories_explored': len(categories_seen),
            'unique_rules_seen': len(rules_seen),
            'categories': list(categories_seen)
        }


def main():
    """Demo the Dakota Grammar RL Environment."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create and test Dakota Grammar RL Environment"
    )
    parser.add_argument(
        "--rules-dir",
        type=str,
        default="data/rl_training_rules",
        help="Directory with RL training rules"
    )
    parser.add_argument(
        "--demo-episodes",
        type=int,
        default=3,
        help="Number of demo episodes to run"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/rl_environment",
        help="Output directory for environment config"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print(" DAKOTA GRAMMAR RL ENVIRONMENT")
    print("="*70)

    # Create environment
    env = DakotaGrammarEnvironment(rules_dir=args.rules_dir)

    print(f"\nLoaded {len(env.all_rules)} grammar rules")
    print(f"Categories: {', '.join(env.rules_by_category.keys())}")

    # Run demo episodes
    print(f"\nRunning {args.demo_episodes} demo episodes...")

    for episode in range(args.demo_episodes):
        print(f"\n--- Episode {episode + 1} ---")

        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 10:  # Limit demo to 10 steps
            print(f"\nStep {steps + 1}")
            print(f"  Rule: {state.current_rule_id}")
            print(f"  Category: {state.rule_category}")
            print(f"  Challenge: {state.challenge_type.value}")
            try:
                print(f"  Pattern: {state.context['dakota_pattern']}")
            except UnicodeEncodeError:
                print(f"  Pattern: [Dakota text - encoding issue]")

            # Random action for demo
            action = random.choice(state.available_actions)
            try:
                print(f"  Action: {action}")
            except UnicodeEncodeError:
                print(f"  Action: [Action with Dakota text]")

            state, reward, done, info = env.step(action)
            print(f"  Reward: {reward.total:.2f} (base: {reward.base_reward:.2f}, "
                  f"accuracy: {reward.accuracy_bonus:.2f}, "
                  f"difficulty: {reward.difficulty_multiplier:.1f}x)")

            steps += 1

        stats = env.get_statistics()
        print(f"\nEpisode complete!")
        print(f"  Total steps: {stats['episode_length']}")
        print(f"  Total reward: {stats['total_reward']:.2f}")
        print(f"  Average reward: {stats['average_reward']:.2f}")
        print(f"  Rules explored: {stats['unique_rules_seen']}")
        print(f"  Categories: {', '.join(stats['categories'])}")

    # Save environment configuration
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'environment_type': 'dakota_grammar_rl',
        'total_rules': len(env.all_rules),
        'categories': list(env.rules_by_category.keys()),
        'rules_per_category': {
            cat: len(rules)
            for cat, rules in env.rules_by_category.items()
        },
        'action_types': [a.value for a in ActionType],
        'max_steps_per_episode': env.max_steps_per_episode,
        'difficulty_levels': ['easy', 'medium', 'hard'],
        'reward_structure': {
            'base_reward': 'Action validity',
            'accuracy_bonus': 'Correct pattern application',
            'difficulty_multiplier': 'Rule difficulty scaling',
            'category_bonus': 'Exploration reward'
        }
    }

    config_file = output_dir / "environment_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}")
    print(" ENVIRONMENT READY")
    print("="*70)
    print(f"\nConfiguration saved: {config_file}")
    print("\nNext steps:")
    print("  1. Implement verification functions for each rule type")
    print("  2. Create training curriculum (easy -> hard)")
    print("  3. Connect to PrimeIntellect verifier system")
    print("  4. Train RL agent on Dakota grammar")
    print()


if __name__ == "__main__":
    main()
