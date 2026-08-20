# Complete Setup Plan for New Instance

## PART 1: Connect to New Instance

### Step 1: Get SSH Access
1. Go to Prime Intellect dashboard
2. Find your new instance
3. Click "SSH Connection" or download the "DakotaRL3" private key
4. If you get the key file, save it as `C:\Users\chris\.ssh\DakotaRL3`

### Step 2: Connect via SSH
```powershell
# If you downloaded the key:
ssh -i $env:USERPROFILE\.ssh\DakotaRL3 -p 1234 root@<new-instance-ip>

# Or if using web console, just click "SSH Connection"
```

---

## PART 2: Setup RL Environment on Instance

### Step 1: Clone Prime-RL (version .17)
```bash
cd ~
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl
git checkout v0.17  # Or whatever the latest .17 tag is

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install dependencies
uv sync
```

### Step 2: Create Project Directory
```bash
mkdir -p ~/dakota_rl_training/configs
mkdir -p ~/dakota_rl_training/outputs
```

### Step 3: Upload Config Files
From your Windows machine (PowerShell):
```powershell
# Upload configs based on wiki_search example format
scp dakota_rl_training\configs\train_30b.toml root@<instance-ip>:/root/dakota_rl_training/configs/
scp dakota_rl_training\configs\orch_30b.toml root@<instance-ip>:/root/dakota_rl_training/configs/
scp dakota_rl_training\configs\infer_30b.toml root@<instance-ip>:/root/dakota_rl_training/configs/
```

---

## PART 3: Update Environment with Verbose Penalty

### The Problem
Your verbose penalty is in `dakota_rl_training/verifiers/rubrics.py` but the **published environment** uses `environments/dakota_grammar_translation/dakota_grammar_translation/environment.py` which doesn't have it!

### Step 1: Add Length Penalty to Published Environment

Edit: `environments/dakota_grammar_translation/dakota_grammar_translation/environment.py`

Add this method to `DakotaGrammarRubric` class (around line 372):

```python
def length_penalty(
    self,
    completion: Messages,
    answer: str,
    parser: Parser,
    max_length_ratio: float = 3.0,
    **_: Any,
) -> float:
    """
    Penalize responses that are too long compared to expected answer.
    
    Prevents degenerate policies that generate long repetitive outputs.
    Returns penalty multiplier: 1.0 (no penalty) to 0.0 (severe penalty)
    """
    prediction = self._prediction(completion, parser)
    response_len = len(prediction.split())
    expected_len = max(len(answer.split()), 1)
    
    length_ratio = response_len / expected_len
    
    if length_ratio <= max_length_ratio:
        return 1.0
    else:
        # Exponential penalty for excessive length
        penalty = max_length_ratio / length_ratio
        return max(0.1, penalty)
```

### Step 2: Apply Length Penalty in Reward Functions

Update the reward functions to use length penalty. Modify `exact_match_reward`, `char_overlap_reward`, etc. to multiply by length penalty:

```python
def exact_match_reward(
    self,
    completion: Messages,
    answer: str,
    parser: Parser,
    **_: Any,
) -> float:
    """Reward for exact match (normalized) with length penalty."""
    prediction = self._prediction(completion, parser)
    base_reward = float(_normalize(prediction) == _normalize(answer))
    length_mult = self.length_penalty(completion, answer, parser)
    return base_reward * length_mult
```

Or better: Update the `__init__` weights to include length penalty as a separate function.

### Step 3: Update Version and Republish

Edit `environments/dakota_grammar_translation/pyproject.toml`:
```toml
version = "0.1.7"  # Or whatever version you want
```

Then republish:
```bash
cd environments/dakota_grammar_translation
python -m build
prime env push
```

---

## PART 4: Update Training Configs (wiki_search format)

Check the wiki_search example configs and update yours to match. The wiki_search example likely has:
- Different config structure
- Updated parameters for version .17
- Better defaults

---

## Quick Checklist

- [ ] Connect to new instance via SSH
- [ ] Clone prime-rl and checkout v0.17
- [ ] Install dependencies (uv sync)
- [ ] Upload config files
- [ ] Add length_penalty to environment.py
- [ ] Apply length penalty in reward functions
- [ ] Update version in pyproject.toml
- [ ] Republish environment
- [ ] Update training configs to match wiki_search format
- [ ] Launch training!

