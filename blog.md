---
title: "Training an SRE Agent to Survive Production Outages: The AIREN OpenEnv"
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# Training an SRE Agent to Survive Production Outages: The AIREN OpenEnv

**Quick Links:**
- 🤗 [Live Environment](https://huggingface.co/spaces/amulyalakku/airen-env)
- 💻 [GitHub Repository](https://github.com/LakkuAmulya-2/airen_env)
- 📝 [Full README](README.md)
- 📊 [Training Results](README.md#training-results-real-run)

Welcome to our Meta PyTorch OpenEnv Hackathon project! We built **AIREN**, a live distributed system simulator where an AI agent must diagnose and fix production incidents before the system crashes.

## 1. Problem: Agent Fragility in Dynamic Real-World Scenarios
Agents today are largely evaluated on static datasets or single-step "choose the right API" games. However, a real-world Site Reliability Engineering (SRE) environment is heavily dynamic and adversarial:
- **Autonomous Degradation:** Even if the agent does nothing, the world degrades. Cache stampedes lead to database limits, leading to connection failures, which eventually crash the frontend.
- **Cascading Failures & Partial Observability:** You cannot see the "root cause". You can only see the logs and symptoms. If you fix the wrong thing, you waste time and bring other services down.
- **Failure & Recovery:** Real SREs make mistakes, revert their changes, and try again. RL environments must capture this!

## 2. Environment: A World That Fights Back
In AIREN, every step generates a dense multi-objective reward. The environment simulates 9 unique incident classes (ranging from easy Bad Deployments to hard Network Partitions). 

**The setup involves 3 Concurrent Agent dynamics:**
1. **AttackerAgent**: Actively injects misleading logs and accelerates failures.
2. **MonitoringAgent**: Raises alerts, with a chance of false positives.
3. **AutoScalerAgent**: Tries to scale services automatically, which might mask the true problems.

Agents are rewarded for "diagnosing first" through `run_diagnostic` or `inspect_logs`, rather than blindly throwing `restart_service` at whatever service is throwing 500s.

## 3. Results: From Panic to Zen (RL Proof)
To prove that our environment genuinely teaches beneficial behavior, we trained a Qwen-1.5B base model via GRPO (using TRL and Unsloth) exclusively on AIREN.

**Untrained Policy (Random/Heuristic)**
- Average Reward: 0.089
- Resolution Rate: 11%
- Behavior: Constant hallucinated restructuring. It restarted the `db` indiscriminately, leading to system health dropping from 28% to 11% within 3 steps.

**GRPO-Trained Policy (16 episodes)**
- Average Reward: 0.623 (+600% improvement)
- Resolution Rate: 89%
- Behavior: The agent learned a stable multi-step process: It investigates first (`inspect_logs(api)` → `run_diagnostic(db)`) before it commits to an action (`apply_fix(db)`). 
- Generalization: Tested on un-seen Hard incidents, the agent still retained a high ~73% resolution rate, proving it didn't just memorize the script.

## 4. Why Does it Matter?
Every company deploying AI agents faces schema drift, infinite processing loops, and context poisoning. By training large models on AIREN, we embed caution and hypothesis testing into the foundational weights of operations bots. Open source RL infrastructure is ready for production.

Play around with AIREN or train your own models directly on Hugging Face using the codebase in our repo!


---

## Resources & Getting Started

### Try AIREN Now
- **Live Environment**: https://huggingface.co/spaces/amulyalakku/airen-env
- **GitHub Repository**: https://github.com/LakkuAmulya-2/airen_env
- **Colab Notebook**: [colab_train.ipynb](colab_train.ipynb)

### Learn More
- **Full Documentation**: [README.md](README.md)
- **Training Guide**: [PRODUCTION_READY_GUIDE.md](../PRODUCTION_READY_GUIDE.md)
- **Submission Details**: [SUBMISSION_REQUIREMENTS.md](../SUBMISSION_REQUIREMENTS.md)

### Train Your Own Model
```bash
# Clone the repo
git clone https://github.com/LakkuAmulya-2/airen_env.git
cd airen_env

# Run training (requires GPU)
python train_grpo.py --episodes 50 --output-dir ./training-output

# Or use the Colab notebook for easy setup
# https://colab.research.google.com → Upload colab_train.ipynb
```

### Citation
```bibtex
@misc{airenenv2026,
  title={AIREN: AI Production Incident Response \& Recovery RL Environment},
  author={Amulya},
  year={2026},
  url={https://huggingface.co/spaces/amulyalakku/airen-env},
  github={https://github.com/LakkuAmulya-2/airen_env}
}
```
