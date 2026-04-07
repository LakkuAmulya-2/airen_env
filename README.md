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
---

# AIREN — AI Production Incident Response & Recovery Environment

A **true multi-step RL environment** where an AI agent must diagnose and fix
production incidents in a live distributed system. The world evolves
autonomously every step — services degrade, threats escalate, cascading
failures spread. The agent must act efficiently or the system crashes.

This is **not** an evaluator. It is a simulation with real consequences.

## Quick Start

```python
from airen_env import AIRENEnv, AIRENAction

with AIRENEnv(base_url="http://localhost:8000").sync() as env:
    # Reset — get a fresh incident
    result = env.reset(incident_type="db_overload", seed=42)
    obs = result.observation
    print(f"Incident: {obs.incident_type} | Health: {obs.system_health:.0%}")
    print(f"Alerts: {[a['message'] for a in obs.alerts]}")

    # Step 1: diagnose
    result = env.step(AIRENAction(
        action_type="run_diagnostic",
        target="db",
        reasoning="DB CPU at 95%, slow queries in logs"
    ))
    print(f"reward={result.reward:.3f}, health={result.observation.system_health:.0%}")

    # Step 2: fix
    result = env.step(AIRENAction(
        action_type="apply_fix",
        target="db",
        reasoning="Unindexed query causing full table scan"
    ))
    print(f"resolved={result.observation.incident_resolved}, reward={result.reward:.3f}")
```

## Server Setup

### Docker (Recommended)

```bash
# From repo root — build base image first
docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .

# Build AIREN image
docker build -t airen-env:latest -f envs/airen_env/server/Dockerfile .

# Run
docker run --rm -p 8000:8000 airen-env:latest

# Verify
curl http://localhost:8000/health
# {"status":"healthy","service":"airen_env"}
```

### Without Docker

```bash
cd envs/airen_env
pip install -r server/requirements.txt
PYTHONPATH=../../src:.. uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## What Makes This a Real RL Environment

| Property | Value |
|---|---|
| Episode type | Multi-step MDP (up to 10 steps) |
| Reward | Dense — every step, not just episode end |
| World dynamics | Autonomous degradation each step |
| Observability | Partial — agent sees symptoms, not root cause |
| Transitions | Stochastic — cascading failures, random degradation |
| Adversarial mode | Optional attacker agent running concurrently |

The world gets worse every step regardless of what the agent does. Wrong
actions trigger cascading failures. The agent must diagnose before fixing
or it loses resolution credit.

## Incident Types

| ID | Name | Difficulty | Correct Fix |
|---|---|---|---|
| `db_overload` | Database Overload | medium | `run_diagnostic` → `apply_fix` on `db` |
| `memory_leak` | Memory Leak | medium | `inspect_logs` → `restart_service` on `worker` |
| `network_partition` | Network Partition | hard | `run_diagnostic` → `apply_fix` on `network` |
| `bad_deployment` | Bad Deployment | easy | `rollback_deployment` on `payment` |
| `cache_stampede` | Cache Stampede | hard | `apply_fix` on `cache` → `scale_service` on `db` |

## Action Space

9 actions, each targeting a specific service:

```
inspect_logs(target)       — read logs, reveals diagnostic clues
inspect_metrics(target)    — check CPU/memory/latency/error_rate
run_diagnostic(target)     — deep diagnostic, reveals root cause signals
restart_service(target)    — restart (brief downtime, use carefully)
scale_service(target)      — scale up to handle load
rollback_deployment(target)— rollback to previous stable version
apply_fix(target)          — apply targeted fix based on diagnosis
acknowledge_incident()     — acknowledge, reduces threat slightly
ignore_alert(target)       — always penalized, never correct
```

Services: `api`, `db`, `cache`, `worker`, `payment`, `network`

## Observation Space

```python
obs.incident_type    # "db_overload" | "memory_leak" | ...
obs.severity         # "critical" | "high" | "medium"
obs.system_health    # float 0.0–1.0 (decreases each step)
obs.threat_level     # float 0.0–1.0 (grows autonomously)
obs.services         # {name: {status, latency_ms, error_rate, cpu_pct, memory_pct}}
obs.metrics          # {metric_name: value}
obs.logs             # last 5 log lines (mix of real clues + red herrings)
obs.alerts           # active alerts [{service, severity, message}]

# Hidden from agent (partial observability):
# obs.root_cause     — true root cause
# obs.attack_progress — how far incident has progressed
```

## Reward Function

Multi-objective dense reward every step:

| Component | Weight | Description |
|---|---|---|
| recovery | 0.25 | Health delta this step (amplified 3×) |
| diagnosis | 0.20 | Right service + right action type |
| efficiency | 0.10 | Steps remaining / max_steps |
| threat_mitigation | 0.10 | Reduction in threat_level |
| resolve_bonus | 0.15 | Large bonus for full resolution |
| hallucination_penalty | −0.10 | Wrong destructive action |
| security_violation | −0.10 | Ignoring critical alerts |
| cost_penalty | −0.05 | Downtime-causing actions |

## State Transitions

Every step (regardless of agent action):
- `threat_level` increases by `degradation_rate × random(0.5, 1.5)`
- Degraded services worsen (error_rate, latency, CPU grow)
- When `attack_progress ≥ cascade_threshold`: a healthy service degrades and a new alert fires

Correct action on correct target:
- Service `error_rate *= 0.4`, `latency *= 0.5`, status → `"recovering"`
- `threat_level -= 0.25`, `attack_progress -= 1.0`

Wrong destructive action:
- Specific services worsen, `threat_level += 0.15`, `attack_progress += 0.5`

## Dynamic Scenario Generation

With `OPENAI_API_KEY` set, AIREN uses GPT-4o-mini to generate unique incidents
on every reset — different root causes, log patterns, and service states.
Falls back to a seed bank if the API is unavailable.

```bash
OPENAI_API_KEY=sk-... uvicorn server.app:app --port 8000
```

## RL Training with GRPO

```bash
# Train Qwen3-0.6B on AIREN incident response
python train_grpo.py \
    --model Qwen/Qwen3-0.6B \
    --env airen \
    --episodes 200 \
    --output-dir airen-grpo-out

# Dry run (validate config without training)
python train_grpo.py --env airen --dry-run
```

## Project Structure

```
airen_env/
├── __init__.py           # Exports: AIRENEnv, AIRENAction, AIRENObservation, AIRENState
├── models.py             # Action, Observation, State dataclasses
├── client.py             # AIRENEnv client (extends EnvClient)
├── openenv.yaml          # OpenEnv manifest
├── pyproject.toml        # Package metadata
├── README.md             # This file
└── server/
    ├── __init__.py
    ├── airen_environment.py  # Core environment logic
    ├── incident_engine.py    # Incident scenario generation (seed bank)
    ├── dynamic_generator.py  # LLM-driven unique scenario generation
    ├── reward.py             # Multi-objective reward computation
    ├── llm_judge.py          # LLM judge for diagnosis quality
    ├── attacker_agent.py     # Optional adversarial attacker
    ├── trace_logger.py       # Token/cost tracking
    ├── app.py                # FastAPI application
    ├── requirements.txt      # Server dependencies
    └── Dockerfile            # Container image
```

## Deploy to Hugging Face Spaces

```bash
# From the airen_env directory (where openenv.yaml lives)
cd envs/airen_env
openenv push

# Push to a specific repo
openenv push --repo-id username/airen-env

# Push as private
openenv push --repo-id username/airen-env --private
```

After deployment the space exposes:
- `/docs` — OpenAPI / Swagger UI
- `/web` — Interactive web interface
- `/health` — Health check
- `/ws` — WebSocket persistent session

Prerequisites: `pip install openenv-core` and `huggingface-cli login`.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start new incident episode |
| `POST` | `/step` | Submit action, get observation + reward |
| `GET` | `/state` | Current episode state |
| `GET` | `/health` | Health check |
| `WS` | `/ws` | WebSocket persistent session |
| `GET` | `/docs` | OpenAPI documentation |
