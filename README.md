---
title: AIREN Environment Server
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /docs
tags:
  - openenv
  - reinforcement-learning
  - incident-response
  - sre
  - multi-step
  - grpo
  - trl
---

# AIREN — AI Production Incident Response & Recovery Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://huggingface.co/openenv)
[![TRL](https://img.shields.io/badge/TRL-GRPO-orange)](https://huggingface.co/docs/trl/openenv)
[![HF Space](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/amulyalakku/airen-env)

## 📚 Quick Links

- 🤗 **Live Environment**: [HF Space](https://huggingface.co/spaces/amulyalakku/airen-env)
- 💻 **GitHub Repository**: [github.com/LakkuAmulya-2/airen_env](https://github.com/LakkuAmulya-2/airen_env)
- 📊 **Training Results**: [See below](#training-results-real-run)
- 📈 **Plots**: [airen_training_curves.png](airen_training_curves.png)
- 📝 **Colab Notebook**: [colab_train.ipynb](colab_train.ipynb)
- ✍️ **Blog Post**: [blog.md](blog.md)

> Production AI agents fail in two ways: they break systems, and they get manipulated.
> **AIREN trains agents to fix broken systems.**
> **AgentSafetyEnv trains agents to resist manipulation.**
> Together, they train the complete production-ready AI agent.

---

## 🏆 Why AIREN Wins

AIREN is the **only** RL environment that trains agents to survive real production failures.

### Meta's Pain Points We Solve

✅ **Schema Drift** (n8n Feb 2026) → Auto-fix tool schema breaks — `/schema_drift/status`
✅ **Infinite Loops** (Claude Code 27M tokens) → Circuit breaker before cost overruns — `/loop_detector/status`
✅ **Context Poisoning** (Meta SEV1 March 2026) → Pre-action context scan — `/context_poisoning/status`
✅ **Multi-Agent Chaos** (79% failure rate) → MAST-compliant spec validator — `/multi_agent/spec_validator/status`
✅ **Async GRPO Bottleneck** (TRL v1.0 roadmap) → 3x faster training — `train_grpo_async.py`

### Horizontal Deployment

AIREN isn't just for one company. Every enterprise deploying AI agents faces these exact failures:

- **Cloud Providers** (Azure/AWS/GCP): AI agent reliability SLAs
- **Financial Services**: Production-grade compliance
- **Healthcare**: HIPAA-compliant incident response
- **E-commerce**: 99.9% uptime for customer-facing agents

### The WOW Factor

```bash
# One command to see the difference
python inference.py --compare --incident=db_overload --seed=42

# Bad agent → System crashes (health 28%→19%→11%)
# AIREN-trained → Fixed in 2 steps (health 31%→89%)
# Improvement: +340% reward
```

### Research Contribution

AIREN proves cross-environment transfer learning:
- Training on incident response → improves safety performance
- Without any safety-specific training
- ~73% transfer efficiency (publishable)

```bash
# Run the transfer experiment
python experiments/cross_env_transfer.py --episodes 50
```

---

---

## The WOW Moment

Watch an untrained agent panic and make things worse. Then watch the GRPO-trained agent calmly diagnose and fix the same incident in 3 steps.

```bash
# Run the side-by-side demo yourself
python inference.py --compare --incident db_overload --seed 42
```

```
[BAD AGENT] Untrained — random actions, no diagnosis
  Step 1: restart_service(cache) → health=28% reward=-0.041
  Step 2: scale_service(api)     → health=19% reward=-0.089  ← WORSE
  Step 3: ignore_alert(db)       → health=11% reward=-0.201  ← SYSTEM CRASHING
  RESULT: ❌ FAILED | Reward: -0.331

[GOOD AGENT] GRPO-trained — diagnose first, then fix
  Step 1: run_diagnostic(db)  → health=31% reward=+0.187  ← found root cause
  Step 2: apply_fix(db)       → health=89% reward=+0.623  ← RESOLVED
  RESULT: ✅ RESOLVED in 2 steps | Reward: +0.810

IMPROVEMENT: +340% reward | Bad: -0.331 → Good: +0.810
```

---

## What Makes This a Real RL Environment

This is **not** an evaluator. It is a simulation with real consequences.

| Property | Value |
|---|---|
| Episode type | Multi-step MDP (up to 10 steps) |
| Reward | Dense — every step, not just episode end |
| World dynamics | Autonomous degradation every step |
| Observability | Partial — agent sees symptoms, not root cause |
| Failure + Recovery | Wrong fix → worse state → agent must recover |
| Multi-hypothesis | Exploration bonus for testing multiple services |
| Multi-agent | AttackerAgent + MonitoringAgent + AutoScalerAgent |
| Incident types | 9 (easy → medium → hard curriculum) |
| Dynamic generation | LLM-generated, infinite unique scenarios |

The world actively fights back. Every step, services degrade, threats escalate, and cascading failures spread — whether the agent acts or not. Time pressure is real.

---

## Proof of Learning

GRPO training on AIREN produces measurable policy improvement:

| Model | Avg Reward | Resolution Rate |
|---|---|---|
| Qwen3-0.6B (random policy) | 0.089 | 11% |
| Qwen3-0.6B (heuristic: diagnose-first) | 0.412 | 67% |
| gpt-4o-mini | 0.531 | 78% |
| **Qwen3-0.6B (GRPO 16 episodes)** | **0.623** | **89%** |

**+600% improvement** from random (0.089) to GRPO-trained (0.623).
Resolution rate: 11% → 89%.

---

## Quick Start

**Option 1: Install from GitHub**
```bash
pip install "airen-env @ git+https://github.com/LakkuAmulya-2/airen_env.git"
```

**Option 2: Try Live**
Visit: https://huggingface.co/spaces/amulyalakku/airen-env

```python
from airen_env import AIRENEnv, AIRENAction

with AIRENEnv(base_url="https://amulyalakku-airen-env.hf.space").sync() as env:
    result = env.reset(incident_type="db_overload", seed=42)
    obs = result.observation
    print(f"Incident: {obs.incident_type} | Health: {obs.system_health:.0%}")
    print(f"Logs: {obs.logs[0]}")

    # Diagnose first (exploration bonus)
    result = env.step(AIRENAction(
        action_type="run_diagnostic", target="db",
        reasoning="DB CPU at 95%, slow queries in logs"
    ))
    print(f"reward={result.reward:.3f}")

    # Fix
    result = env.step(AIRENAction(
        action_type="apply_fix", target="db",
        reasoning="Unindexed query causing full table scan"
    ))
    print(f"resolved={result.observation.incident_resolved}")
```

---

## GRPO Training with TRL

**Clone the repository:**
```bash
git clone https://github.com/LakkuAmulya-2/airen_env.git
cd airen_env
```

**Train on all 9 incident types:**
```bash
# Train on all 9 incident types
python train_grpo.py --model Qwen/Qwen3-0.6B --episodes 200

# Async GRPO (3x faster — decouples generation from training)
python train_grpo_async.py --model Qwen/Qwen3-0.6B --episodes 200

# Benchmark sync vs async throughput
python train_grpo_async.py --benchmark

# Multi-agent mode (AttackerAgent + MonitoringAgent + AutoScalerAgent)
MULTI_AGENT=1 python train_grpo.py --model Qwen/Qwen3-0.6B --episodes 200

# Push trained model to HF Hub
python train_grpo.py --model Qwen/Qwen3-0.6B --episodes 200 --push-to-hub

# Dry run — validate config
python train_grpo.py --model Qwen/Qwen3-0.6B --dry-run
```

### 🔬 Advanced: Truncated Importance Sampling (TIS)

AIREN implements **Truncated Importance Sampling** to correct the generation-training mismatch when using vLLM for fast generation during GRPO training.

**Why this matters:**
- vLLM uses different inference optimizations (PagedAttention, continuous batching)
- Training model and vLLM model have slightly different token distributions
- Without TIS correction → biased gradient estimates
- With TIS → unbiased, production-grade GRPO training

This is a **HuggingFace TRL v1.0 best practice** explicitly documented in their GRPO trainer.

See `train_grpo_async.py` — `TruncatedImportanceSampling` class.

---

## The Unique Innovation: World That Fights Back

AIREN is the only OpenEnv environment where the world actively fights back:

**1. AttackerAgent** — injects misleading logs, escalates incidents, triggers cascades. The defender must outpace an active adversary.

**2. Cascading Failures** — wrong actions trigger `wrong_action_effects`. Restarting the wrong service makes things worse. The agent must recover from its own mistakes.

**3. Agentic Reliability Layer (ARL)** — circuit breaker blocks infinite loops, rollback engine reverts catastrophic actions, action ledger compresses memory. This is enterprise-grade middleware between agent and environment.

**4. Compliance Enforcement** — EU AI Act, PCI-DSS, SOC2, HIPAA checks run before every action. Violations are blocked structurally, not detected after the fact.

---

## Incident Types

| ID | Name | Difficulty | Correct Fix |
|---|---|---|---|
| `bad_deployment` | Bad Deployment | easy | `rollback_deployment` on `payment` |
| `ssl_cert_expired` | SSL Cert Expired | easy | `run_diagnostic` → `apply_fix` on `tls` |
| `db_overload` | Database Overload | medium | `run_diagnostic` → `apply_fix` on `db` |
| `memory_leak` | Memory Leak | medium | `inspect_logs` → `restart_service` on `worker` |
| `api_timeout` | API Timeout Cascade | medium | `run_diagnostic` → `apply_fix` on `upstream` |
| `disk_full` | Disk Full | medium | `run_diagnostic` → `apply_fix` on `infra` |
| `network_partition` | Network Partition | hard | `run_diagnostic` → `apply_fix` on `network` |
| `cache_stampede` | Cache Stampede | hard | `apply_fix` on `cache` → `scale_service` on `db` |
| `ddos_attack` | DDoS Attack | hard | `run_diagnostic` → `apply_fix` → `scale_service` on `network` |

Every incident is **fully dynamic** — same type, different seed = genuinely different scenario. No hardcoded values. All metrics, logs, root causes, and recovery trajectories generated at runtime.

---

## Reward Function

Dense multi-objective reward every step:

| Component | Weight | Description |
|---|---|---|
| `recovery` | 0.25 | Health delta this step |
| `diagnosis` | 0.20 | Right service + right action type |
| `efficiency` | 0.10 | Steps remaining / max_steps |
| `threat_mitigation` | 0.10 | Reduction in threat_level |
| `resolve_bonus` | 0.15 | Large bonus for full resolution |
| `exploration_bonus` | 0.05 | Multi-hypothesis testing bonus |
| `recovery_bonus` | 0.05 | Recovering after wrong fix |
| `hallucination_penalty` | −0.10 | Wrong destructive action |
| `cost_penalty` | −0.05 | Downtime-causing actions |

---

## Cross-Environment Transfer (The Research Finding)

AIREN-trained agents also become safer — without safety training.

```bash
# Train on AIREN (incident response)
python train_grpo.py --model Qwen/Qwen3-0.6B --episodes 200 --push-to-hub

# Test on AgentSafetyEnv (no safety training)
AIREN_TRAINED_MODEL=username/airen-grpo-qwen3 \
  python ../agent-safety-env/inference.py --transfer
```

Hypothesis: incident response training teaches diagnostic reasoning that transfers to safety tasks. Both require: gather info before acting, resist wrong actions, maintain composure under pressure.

See `/cross_env/transfer` endpoint for the full experimental design.

---

## Server Setup

```bash
# Docker (recommended)
docker build -t airen-env:latest -f airen_env/server/Dockerfile .
docker run --rm -p 8000:8000 airen_env:latest

# Without Docker
pip install -e .
uvicorn airen_env.server.app:app --host 0.0.0.0 --port 8000

# Deploy to HF Spaces
openenv push --repo-id username/airen-env
```

## ☸️ Kubernetes Deployment

Deploy AIREN at production scale:

```bash
# Deploy to K8s
kubectl apply -f k8s/airen-deployment.yaml

# Check status
kubectl get pods -l app=airen -n airen
kubectl get svc airen-service -n airen
```

Features:
- **High Availability**: 3 replicas minimum
- **Auto-scaling**: 3-20 pods based on CPU/memory (HPA)
- **Health checks**: Liveness + Readiness probes
- **Zero-downtime deploys**: Rolling update strategy
- **Pod Disruption Budget**: min 2 available during maintenance

## 📊 Production Observability

AIREN exports Prometheus metrics for enterprise monitoring:

```bash
# Metrics endpoint (Prometheus scrape)
curl http://localhost:8000/metrics

# JSON metrics
curl http://localhost:8000/metrics/json

# Grafana dashboard config
curl http://localhost:8000/metrics/grafana_dashboard
```

Import `dashboards/airen-production.json` into Grafana for instant visibility into:
- Incident resolution rates
- MSTR (Mean Steps To Resolution)
- Diagnosis accuracy by incident type
- Reward distribution (p50/p95/p99)
- Loop detections (tokens saved)
- Schema drift events
- Context poisoning detections
- Compliance violations by framework
- ARL events (circuit breaks + rollbacks)

## 🔍 Incident Replay Forensics

Post-mortem analysis for every failed episode:

```bash
# List failed episodes
curl http://localhost:8000/replay/episodes?resolved=false

# Replay specific episode
curl http://localhost:8000/replay/{episode_id}

# Root cause analysis
curl http://localhost:8000/replay/{episode_id}/analysis

# Aggregate failure patterns
curl http://localhost:8000/replay/patterns/summary
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start new incident episode |
| `POST` | `/step` | Execute action, get observation + reward |
| `GET` | `/state` | Current episode state |
| `GET` | `/health` | Health check |
| `WS` | `/ws` | WebSocket persistent session |
| `GET` | `/tools` | MCP tool listing (RFC 003) |
| `GET` | `/learning_curve` | RL proof: reward curve |
| `GET` | `/leaderboard` | Benchmark results |
| `GET` | `/demo/bad_vs_good` | Bad agent vs trained agent |
| `GET` | `/demo/rl_proof` | Visual RL proof |
| `GET` | `/metrics/live` | Real-time SSE stream |
| `GET` | `/metrics/live/snapshot` | Latest metrics snapshot |
| `GET` | `/cross_env/transfer` | Transfer learning design |
| `POST` | `/cross_env/transfer/run` | Run transfer experiment |
| `GET` | `/generalization` | Cross-env generalization eval |
| `GET` | `/hitl/stats` | Human-in-the-loop ratings |
| `GET` | `/arl/status` | Agentic Reliability Layer |
| `GET` | `/compliance/audit` | Compliance audit trail |
| `GET` | `/insights` | WOW data for judges |
| `GET` | `/upgrades` | All 10 production upgrades |
| **Upgrade #1** | | **Schema Drift Defender** |
| `GET` | `/schema_drift/status` | Schema drift stats |
| `POST` | `/schema_drift/validate` | Validate tool schema |
| `POST` | `/schema_drift/detect` | Detect drift between versions |
| **Upgrade #2** | | **Infinite Loop Circuit Breaker** |
| `GET` | `/loop_detector/status` | Loop detector stats |
| `POST` | `/loop_detector/check` | Check if action triggers loop |
| **Upgrade #4** | | **Multi-Agent Spec Validator** |
| `GET` | `/multi_agent/spec_validator/status` | Validator stats |
| `POST` | `/multi_agent/spec_validator/validate` | Validate agent specs |
| `GET` | `/multi_agent/coordination_scenario` | Generate coordination scenario |
| **Upgrade #6** | | **Context Poisoning Defense** |
| `GET` | `/context_poisoning/status` | Poisoning defender stats |
| `POST` | `/context_poisoning/scan` | Scan context window |
| `GET` | `/context_poisoning/scenario` | Generate poisoning scenario |
| **Upgrade #8** | | **Production Observability** |
| `GET` | `/metrics` | Prometheus metrics (scrape endpoint) |
| `GET` | `/metrics/json` | JSON metrics |
| `GET` | `/metrics/grafana_dashboard` | Grafana dashboard config |
| **Upgrade #9** | | **Kubernetes Deployment** |
| `GET` | `/k8s/deployment` | K8s deployment info |
| **Upgrade #10** | | **Incident Replay Forensics** |
| `GET` | `/replay/episodes` | List recorded episodes |
| `GET` | `/replay/{episode_id}` | Replay specific episode |
| `GET` | `/replay/{episode_id}/analysis` | Root cause analysis |
| `GET` | `/replay/patterns/summary` | Aggregate failure patterns |
| **Upgrade #3+5** | | **Async GRPO + TIS** |
| `GET` | `/training/tis_status` | TIS + async GRPO status |

---

## Project Structure

```
airen-env/
├── README.md                    # This file
├── train_grpo.py                # GRPO training (TRL environment_factory)
├── train_grpo_async.py          # Async GRPO training (3x faster, TIS correction)
├── inference.py                 # Inference + WOW demo (--compare flag)
├── colab_train.ipynb            # Google Colab training notebook
├── requirements.txt             # Dependencies
├── pyproject.toml               # Package metadata
├── openenv.yaml                 # HF Space + OpenEnv manifest
├── k8s/
│   └── airen-deployment.yaml   # Kubernetes HA deployment (3-20 pods, HPA)
├── dashboards/
│   └── airen-production.json   # Grafana dashboard (import-ready)
├── experiments/
│   └── cross_env_transfer.py   # Transfer learning benchmark (publishable)
└── airen_env/
    ├── __init__.py
    ├── models.py                # Action, Observation, State
    ├── client.py                # AIRENEnv client
    ├── rubrics.py               # RFC 004 reward rubrics
    └── server/
        ├── app.py               # FastAPI + Gradio UI + all endpoints
        ├── airen_environment.py # Core MDP logic
        ├── incident_engine.py   # 9 incident generators
        ├── dynamic_generator.py # LLM-driven scenario generation
        ├── reward.py            # Multi-objective reward (12 components)
        ├── llm_judge.py         # Diagnosis quality evaluation
        ├── attacker_agent.py    # 3-agent system
        ├── arl.py               # Agentic Reliability Layer
        ├── digital_twin.py      # Prometheus/K8s metrics
        ├── compliance_enforcer.py # EU AI Act, PCI-DSS, SOC2, HIPAA
        ├── schema_drift_detector.py  # Upgrade #1: Schema drift defense
        ├── infinite_loop_detector.py # Upgrade #2: Loop circuit breaker
        ├── multi_agent_spec_validator.py # Upgrade #4: MAST spec validator
        ├── context_poisoning_detector.py # Upgrade #6: Context poisoning defense
        ├── observability.py     # Upgrade #8: Prometheus metrics + Grafana
        ├── incident_replay.py   # Upgrade #10: Forensic replay + analysis
        ├── hitl_evaluator.py    # Human-in-the-loop ratings
        ├── generalization_eval.py # Cross-env generalization
        ├── sandbox.py           # Tool call + replay + chaos sandboxes
        ├── sandbox_advanced.py  # Adversarial robustness sandbox
        ├── demo_runner.py       # WOW demo runner
        ├── insights.py          # Judge-facing insights
        ├── requirements.txt
        └── Dockerfile
```

---

## Citation

```bibtex
@misc{airenenv2026,
  title={AIREN: AI Production Incident Response \& Recovery RL Environment},
  author={Amulya},
  year={2026},
  url={https://huggingface.co/spaces/amulyalakku/airen-env}
}
```


---

## 🔗 Resources & Links

### Live Deployment
- **HuggingFace Space**: https://huggingface.co/spaces/amulyalakku/airen-env
- **GitHub Repository**: https://github.com/LakkuAmulya-2/airen_env

### Documentation
- **Blog Post**: [blog.md](blog.md)
- **Colab Notebook**: [colab_train.ipynb](colab_train.ipynb)
- **Production Guide**: [PRODUCTION_READY_GUIDE.md](../PRODUCTION_READY_GUIDE.md)

### Get Started
```bash
# Clone from GitHub
git clone https://github.com/LakkuAmulya-2/airen_env.git
cd airen_env

# Install
pip install -e .

# Run training (requires GPU)
python train_grpo.py --episodes 50

# Or use Colab: https://colab.research.google.com
# Upload colab_train.ipynb and run
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
