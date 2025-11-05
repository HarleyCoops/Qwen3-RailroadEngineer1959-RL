# QUICK LAUNCH - Copy This When Instance Is Ready

## 1. SSH Into Instance
```bash
ssh user@your-instance-ip
# (Use the SSH command from Prime Intellect dashboard)
```

## 2. Upload Config Files (From Windows PowerShell)
```powershell
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\*.toml user@your-instance-ip:~/dakota_rl_training/configs/
```

## 3. Setup Prime-RL (On Instance)
```bash
mkdir -p ~/dakota_rl_training/configs
cd ~
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync
```

## 4. Launch Training
```bash
cd ~/prime-rl
uv run rl \
  --trainer @ ~/dakota_rl_training/configs/train_30b.toml \
  --orchestrator @ ~/dakota_rl_training/configs/orch_30b.toml \
  --inference @ ~/dakota_rl_training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota_rl_training/outputs/grpo_30b
```

## 5. Monitor
```bash
# In another terminal
tail -f ~/dakota_rl_training/outputs/grpo_30b/logs/trainer/rank_0.log
```

---

**That's it!** Training should start automatically.

