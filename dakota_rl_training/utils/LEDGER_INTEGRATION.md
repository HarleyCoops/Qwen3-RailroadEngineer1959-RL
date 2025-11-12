"""
Integration Guide: Reward Ledger Logging

This guide explains how to integrate reward ledger logging into your RL training loop.

The ledger logging system exposes all internal reward components for transparency and debugging.
"""

# Example integration for PrimeIntellect prime-rl training

"""
Option 1: Custom Monitor Hook

Create a custom monitor that extracts ledger data from environment info dicts
and logs to W&B and CSV.

Add this to your training configuration or create a custom monitor class.
"""

from dakota_rl_training.utils.ledger_logging import log_step_ledger, extract_ledger_from_info


class LedgerMonitor:
    """Monitor that logs reward ledger data."""
    
    def __init__(self):
        self.step_infos = []
    
    def on_step_end(self, step: int, infos: List[Dict]):
        """
        Called after each training step.
        
        Args:
            step: Current training step
            infos: List of info dicts from environment steps
        """
        # Extract ledger data from infos
        ledger_infos = []
        for info in infos:
            ledger = extract_ledger_from_info(info)
            if ledger:
                ledger_infos.append(ledger)
        
        # If we have ledger data, log it
        if ledger_infos:
            log_step_ledger(step, ledger_infos, wandb_log=True)


"""
Option 2: Direct Integration in Training Loop

If you have access to the training loop, add this after each optimizer update:

```python
from dakota_rl_training.utils.ledger_logging import log_step_ledger, extract_ledger_from_info

# In your training loop, after collecting rewards and computing gradients:
step_infos = []  # Collect info dicts from environment steps
for batch in dataloader:
    # ... run environment, collect rewards ...
    step_infos.append(info)  # info dict from environment.step()

# After optimizer step:
log_step_ledger(current_step, step_infos, wandb_log=True)
```

Option 3: PrimeIntellect Integration

For PrimeIntellect prime-rl framework, you can create a custom callback:

```python
from prime_rl.utils.monitor import Monitor
from dakota_rl_training.utils.ledger_logging import log_step_ledger, extract_ledger_from_info

class LedgerMonitor(Monitor):
    def on_train_step_end(self, step: int, metrics: Dict, infos: List[Dict]):
        # Extract ledger data
        ledger_infos = [extract_ledger_from_info(i) for i in infos if extract_ledger_from_info(i)]
        if ledger_infos:
            log_step_ledger(step, ledger_infos, wandb_log=True)
```

Then add to your RLConfig:
```python
from dakota_rl_training.utils.ledger_logging import LedgerMonitor

config = RLConfig(
    # ... other config ...
    monitors=[LedgerMonitor()],
)
```

Option 4: Environment Wrapper

Wrap the environment to automatically include ledger in info:

```python
class LedgerEnvWrapper:
    def __init__(self, env):
        self.env = env
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Extract ledger from rubric if available
        if hasattr(self.env, 'get_reward_ledger'):
            ledger = self.env.get_reward_ledger()
            if ledger:
                info['ledger'] = ledger
        
        return obs, reward, done, info
```

Usage Notes:

1. The ledger is automatically computed by DakotaGrammarRubric.score()
2. The ledger is stored in the rubric instance and can be retrieved via get_last_ledger()
3. The environment's get_reward_ledger() method provides access to the ledger
4. The log_step_ledger() function aggregates across a batch and logs to W&B and CSV
5. CSV is saved to wandb_analysis/reward_ledger.csv
6. W&B logs are under the ledger/* namespace

After training, generate visualizations:

```bash
python scripts/analysis/plot_reward_ledger.py
python scripts/analysis/make_ledger_snippet.py
```

This will create:
- wandb_analysis/reward_ledger.png (visualization)
- wandb_analysis/reward_ledger_head_tail.md (markdown table for README)
"""

