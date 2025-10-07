#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test PrimeIntellect Verifier Integration with New Grammar Tasks

This script tests that the DakotaGrammarEnv works correctly with the
5,657 tasks generated from the extracted grammar rules.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add dakota_rl_training to path
sys.path.insert(0, str(Path(__file__).parent / "dakota_rl_training"))

from verifiers.grammar_env import DakotaGrammarEnv
from verifiers.rubrics import DakotaGrammarRubric


async def test_task_loading():
    """Test that tasks load correctly"""
    print("\n" + "="*70)
    print(" TEST 1: Task Loading")
    print("="*70)

    task_file = Path("dakota_rl_training/datasets/grammar_tasks_complete.jsonl")

    if not task_file.exists():
        print(f"ERROR: Task file not found: {task_file}")
        return False

    # Load first 10 tasks
    tasks = []
    with open(task_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            tasks.append(json.loads(line))

    print(f"\nLoaded {len(tasks)} sample tasks")

    # Display task types
    task_types = {}
    for task in tasks:
        t_type = task['info']['task_type']
        task_types[t_type] = task_types.get(t_type, 0) + 1

    print("\nTask types in sample:")
    for t_type, count in task_types.items():
        print(f"  {t_type}: {count}")

    return True


async def test_multiturn_env():
    """Test multi-turn environment with actual task"""
    print("\n" + "="*70)
    print(" TEST 2: Multi-Turn Environment")
    print("="*70)

    env = DakotaGrammarEnv(max_turns=3)

    # Load a real task
    task_file = Path("dakota_rl_training/datasets/grammar_tasks_complete.jsonl")

    with open(task_file, 'r', encoding='utf-8') as f:
        # Get first morphology task
        for line in f:
            task = json.loads(line)
            if task['info']['task_type'] == 'morphology':
                break

    print(f"\nTask: {task['prompt'][:100]}...")
    print(f"Expected: {task['answer']}")
    print(f"Type: {task['info']['task_type']}")
    print(f"Difficulty: {task['info']['difficulty']}")

    # Test correct answer
    messages = [
        {"role": "user", "content": task["prompt"]},
        {"role": "assistant", "content": task["answer"]}
    ]
    state = {}

    is_complete = await env.is_completed(messages, state, **task)
    feedback_msgs, new_state = await env.env_response(messages, state, **task)

    print("\n✓ Correct Answer Test:")
    print(f"  Complete: {is_complete}")
    print(f"  Special chars correct: {new_state.get('special_chars_correct', False)}")
    print(f"  Affixes correct: {new_state.get('affixes_correct', False)}")
    print(f"  Semantic correct: {new_state.get('semantic_correct', False)}")
    print(f"  Feedback: {feedback_msgs[0]['content']}")

    # Test wrong answer
    messages_wrong = [
        {"role": "user", "content": task["prompt"]},
        {"role": "assistant", "content": "wrong answer"}
    ]
    state_wrong = {}

    feedback_wrong, new_state_wrong = await env.env_response(messages_wrong, state_wrong, **task)

    print("\n✓ Wrong Answer Test:")
    print(f"  Feedback: {feedback_wrong[0]['content']}")

    return True


async def test_rubric():
    """Test reward rubric"""
    print("\n" + "="*70)
    print(" TEST 3: Reward Rubric")
    print("="*70)

    rubric = DakotaGrammarRubric()

    # Load a task with Dakota special characters
    task_file = Path("dakota_rl_training/datasets/grammar_tasks_complete.jsonl")

    with open(task_file, 'r', encoding='utf-8') as f:
        # Get task with special chars
        for line in f:
            task = json.loads(line)
            if task['info'].get('special_chars'):
                break

    print(f"\nTask: {task['prompt'][:100]}...")
    print(f"Expected: {task['answer']}")
    print(f"Special chars: {task['info']['special_chars']}")

    # Test reward calculation
    reward = rubric.composite_reward(
        task['answer'],
        task['answer'],
        task['info']
    )

    print(f"\n✓ Perfect Answer Reward: {reward:.2f}")

    # Test with wrong answer
    wrong_reward = rubric.composite_reward(
        "wrong",
        task['answer'],
        task['info']
    )

    print(f"✓ Wrong Answer Reward: {wrong_reward:.2f}")

    return True


async def test_translation_task():
    """Test translation task"""
    print("\n" + "="*70)
    print(" TEST 4: Translation Task")
    print("="*70)

    env = DakotaGrammarEnv(max_turns=3)

    # Load translation task
    task_file = Path("dakota_rl_training/datasets/grammar_tasks_complete.jsonl")

    with open(task_file, 'r', encoding='utf-8') as f:
        for line in f:
            task = json.loads(line)
            if task['info']['task_type'] == 'word_translation':
                break

    print("\nTranslation Task:")
    print(f"Dakota: {task['info'].get('dakota_text', 'N/A')}")
    print(f"English: {task['answer']}")
    print(f"Special chars: {task['info'].get('special_chars', [])}")

    messages = [
        {"role": "user", "content": task["prompt"]},
        {"role": "assistant", "content": task["answer"]}
    ]
    state = {}

    feedback, new_state = await env.env_response(messages, state, **task)

    print("\n✓ Translation Test:")
    print(f"  Semantic correct: {new_state.get('semantic_correct', False)}")
    print(f"  Feedback: {feedback[0]['content']}")

    return True


async def test_difficulty_distribution():
    """Test distribution of difficulties"""
    print("\n" + "="*70)
    print(" TEST 5: Difficulty Distribution")
    print("="*70)

    task_file = Path("dakota_rl_training/datasets/grammar_tasks_complete.jsonl")

    difficulties = {}
    task_types = {}
    special_char_count = 0

    with open(task_file, 'r', encoding='utf-8') as f:
        for line in f:
            task = json.loads(line)

            diff = task['info']['difficulty']
            difficulties[diff] = difficulties.get(diff, 0) + 1

            t_type = task['info']['task_type']
            task_types[t_type] = task_types.get(t_type, 0) + 1

            if task['info'].get('special_chars'):
                special_char_count += 1

    print("\nDifficulty Distribution:")
    for diff, count in sorted(difficulties.items()):
        print(f"  {diff:10s}: {count:5d} tasks ({count/sum(difficulties.values())*100:.1f}%)")

    print("\nTask Type Distribution:")
    for t_type, count in sorted(task_types.items()):
        print(f"  {t_type:25s}: {count:5d} tasks ({count/sum(task_types.values())*100:.1f}%)")

    print(f"\nTasks with Dakota special characters: {special_char_count} ({special_char_count/sum(task_types.values())*100:.1f}%)")

    return True


async def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" PRIMEINTELLECT VERIFIER INTEGRATION TEST")
    print(" Testing 5,657 grammar tasks from images 31-92")
    print("="*70)

    tests = [
        ("Task Loading", test_task_loading),
        ("Multi-Turn Environment", test_multiturn_env),
        ("Reward Rubric", test_rubric),
        ("Translation Task", test_translation_task),
        ("Difficulty Distribution", test_difficulty_distribution),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
                print(f"\n✓ {name}: PASSED")
            else:
                failed += 1
                print(f"\n✗ {name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {name}: FAILED with error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    print(f"\nTotal tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n✓ All tests passed! Ready for training.")
        print("\nNext steps:")
        print("  1. Review config: cat dakota_rl_training/configs/training_config.yaml")
        print("  2. Run local training: cd dakota_rl_training && python train.py")
        print("  3. Run distributed: prime-rl train --config configs/training_config.yaml")
    else:
        print("\n✗ Some tests failed. Fix issues before training.")

    print()


if __name__ == "__main__":
    asyncio.run(main())
