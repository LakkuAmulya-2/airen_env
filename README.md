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

> Production AI agents fail in two ways: they break systems, and they get manipulated.
> **AIREN trains agents to fix broken systems.**
> **AgentSafetyEnv trains agents to resist manipulation.**
> Together, they train the complete production-ready AI agent.

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

```bash
pip install "airen-env @ git+https://huggingface.co/spaces/amulyalakku/airen-env"
```

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

```bash
# Train on all 9 incident types
python train_grpo.py --model Qwen/Qwen3-0.6B --episodes 200

# Multi-agent mode (AttackerAgent + MonitoringAgent + AutoScalerAgent)
MULTI_AGENT=1 python train_grpo.py --model Qwen/Qwen3-0.6B --episodes 200

# Push trained model to HF Hub
python train_grpo.py --model Qwen/Qwen3-0.6B --episodes 200 --push-to-hub

# Dry run — validate config
python train_grpo.py --model Qwen/Qwen3-0.6B --dry-run
```

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
| `GET` | `/generalization` | Cross-env generalization eval |
| `GET` | `/hitl/stats` | Human-in-the-loop ratings |
| `GET` | `/arl/status` | Agentic Reliability Layer |
| `GET` | `/compliance/audit` | Compliance audit trail |
| `GET` | `/insights` | WOW data for judges |

---

## Project Structure

```
airen-env/
├── README.md                    # This file
├── train_grpo.py                # GRPO training (TRL environment_factory)
├── inference.py                 # Inference + WOW demo (--compare flag)
├── colab_train.ipynb            # Google Colab training notebook
├── requirements.txt             # Dependencies
├── pyproject.toml               # Package metadata
├── openenv.yaml                 # HF Space + OpenEnv manifest
└── airen_env/
    ├── __init__.py
    ├── models.py                # Action, Observation, State
    ├── client.py                # AIRENEnv client
    ├── rubrics.py               # RFC 004 reward rubrics
    └── server/
        ├── app.py               # FastAPI + Gradio UI + all endpoints
        ├── airen_environment.py # Core MDP logic (1097 lines)
        ├── incident_engine.py   # 9 incident generators (891 lines)
        ├── dynamic_generator.py # LLM-driven scenario generation
        ├── reward.py            # Multi-objective reward (11 components)
        ├── llm_judge.py         # Diagnosis quality evaluation
        ├── attacker_agent.py    # 3-agent system
        ├── arl.py               # Agentic Reliability Layer
        ├── digital_twin.py      # Prometheus/K8s metrics
        ├── compliance_enforcer.py # EU AI Act, PCI-DSS, SOC2, HIPAA
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
