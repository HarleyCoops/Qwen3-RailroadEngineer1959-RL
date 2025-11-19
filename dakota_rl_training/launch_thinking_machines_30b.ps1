# Launch Thinking Machines RL Training
# Model: Qwen3-30B-A3B-Instruct-2507
# Framework: Thinking Machines (Tinker)

# Ensure we are in the root directory
$ScriptDir = Split-Path $MyInvocation.MyCommand.Path
$RootDir = Resolve-Path "$ScriptDir\.."
Set-Location $RootDir

# Run the training script
python dakota_rl_training/tinker_train.py `
    --model-name "Qwen/Qwen3-30B-A3B-Instruct-2507" `
    --wandb-project "thinking-machines-qwen3-30b" `
    --log-path "dakota_rl_training/outputs/thinking_machines_30b" `
    --batch-size 32 `
    --group-size 16 `
    --max-examples 10000 `
    --eval-every 50 `
    --save-every 100
