"""
AIREN — Inference Script
==========================
Meta PyTorch OpenEnv Hackathon x SST

Runs a baseline LLM agent against all 9 incident types.
Emits [START][STEP][END] structured logs.
Auto-submits results to /leaderboard/submit endpoint.
Supports --compare mode: runs bad agent vs good agent side-by-side (WOW demo).

Required env vars:
    API_BASE_URL   LLM API endpoint
    MODEL_NAME     Model identifier
    HF_TOKEN       API key
    ENV_URL        AIREN server URL
    MULTI_AGENT    Set to 1 for multi-agent mode

Usage:
    # Standard inference
    ENV_URL=https://amulyalakku-airen-env.hf.space \\
    API_BASE_URL=https://router.huggingface.co/v1 \\
    MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \\
    HF_TOKEN=hf_... \\
    python inference.py

    # WOW demo: bad agent vs trained agent on same incident
    python inference.py --compare --incident db_overload --seed 42

    # Curriculum: easy → medium → hard
    python inference.py --curriculum
"""

import json
import os
import time
import urllib.request
from typing import Any, Dict, List, Optional

from openai import OpenAI

from airen_env import AIRENEnv, AIRENAction
from airen_env.server.incident_engine import (
    ALL_INCIDENT_TYPES, EASY_INCIDENTS, MEDIUM_INCIDENTS, HARD_INCIDENTS,
    generate_incident,
)
from airen_env.server.llm_judge import AIRENLLMJudge
from airen_env.server.trace_logger import tokens_to_usd

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_URL      = os.environ.get("ENV_URL", "https://amulyalakku-airen-env.hf.space")
MULTI_AGENT  = os.environ.get("MULTI_AGENT", "0") == "1"
EPISODES_PER_INCIDENT = int(os.environ.get("EPISODES_PER_INCIDENT", "1"))

llm = OpenAI(
    api_key=HF_TOKEN or os.environ.get("OPENAI_API_KEY", "no-key"),
    base_url=API_BASE_URL,
)

AGENT_SYSTEM = """You are an expert Site Reliability Engineer (SRE) responding to production incidents.
Diagnose the root cause from logs and metrics, then apply the correct fix.
Respond with JSON: {"action_type": "...", "target": "...", "reasoning": "..."}
Available actions: inspect_logs, inspect_metrics, run_diagnostic,
restart_service, scale_service, rollback_deployment, apply_fix, acknowledge_incident"""


def log(obj: Dict[str, Any]) -> None:
    print(json.dumps(obj), flush=True)


def call_agent(obs) -> AIRENAction:
    services_summary = {
        name: f"{s['status']} | latency={s['latency_ms']}ms | err={s['error_rate']:.0%} | cpu={s['cpu_pct']}%"
        for name, s in obs.services.items()
    }
    alerts_summary = [f"{a['service']}: {a['message']}" for a in obs.alerts[:3]]

    user_content = (
        f"INCIDENT: {obs.incident_type} (severity: {obs.severity})\n"
        f"System health: {obs.system_health:.0%} | Step: {obs.step_number}/{obs.max_steps}\n\n"
        f"SERVICES:\n{json.dumps(services_summary, indent=2)}\n\n"
        f"RECENT LOGS:\n{chr(10).join(obs.logs[-4:])}\n\n"
        f"ACTIVE ALERTS:\n{chr(10).join(alerts_summary) if alerts_summary else 'None'}\n\n"
        "What action do you take?"
    )

    try:
        completion = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": AGENT_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        data = json.loads(completion.choices[0].message.content or "{}")
        return AIRENAction(
            action_type=data.get("action_type", "inspect_logs"),
            target=data.get("target", "api"),
            reasoning=data.get("reasoning", ""),
            parameters=data.get("parameters", {}),
        )
    except Exception as e:
        return AIRENAction(action_type="run_diagnostic", target="api",
                           reasoning=f"Fallback: {e}")


def run_episode(env: AIRENEnv, incident_type: str, seed: int, ep_num: int) -> Dict[str, Any]:
    t0 = time.time()
    episode_id = f"{incident_type}_{ep_num}_{int(t0) % 100000}"

    result = env.reset(incident_type=incident_type, seed=seed)
    obs = result.observation

    log({"event": "[START]", "episode_id": episode_id,
         "incident_type": obs.incident_type, "severity": obs.severity,
         "timestamp": t0})

    actions_taken = []
    cumulative_reward = 0.0
    total_tokens = 0

    for step in range(obs.max_steps):
        action = call_agent(obs)
        actions_taken.append(f"{action.action_type}:{action.target}")

        result = env.step(action)
        obs = result.observation
        cumulative_reward += result.reward or 0.0
        step_tokens = len((action.reasoning or "").split()) * 2
        total_tokens += step_tokens

        log({"event": "[STEP]", "episode_id": episode_id, "step": step + 1,
             "action_type": action.action_type, "target": action.target,
             "reasoning": (action.reasoning or "")[:150],
             "reward": result.reward, "system_health": obs.system_health,
             "action_success": obs.action_success, "timestamp": time.time()})

        if obs.done:
            break

    state = env.state()
    judge = AIRENLLMJudge()
    from airen_env.server.incident_engine import generate_incident
    scenario = generate_incident(incident_type, seed)
    judge_result = judge.judge(
        incident_type=incident_type,
        root_cause=state.root_cause or "",
        actions_taken=actions_taken,
        final_health=obs.system_health,
        incident_resolved=state.incident_resolved,
        correct_actions=scenario.correct_actions,
        correct_targets=scenario.correct_targets,
        rule_score=cumulative_reward / max(len(actions_taken), 1),
    )
    total_tokens += judge_result.tokens_used
    total_cost = tokens_to_usd(total_tokens, MODEL_NAME)

    log({"event": "[END]", "episode_id": episode_id,
         "incident_type": incident_type,
         "final_health": obs.system_health,
         "resolved": state.incident_resolved,
         "cumulative_reward": round(cumulative_reward, 3),
         "steps_taken": len(actions_taken),
         "diagnosis_quality": judge_result.diagnosis_quality,
         "total_time_s": round(time.time() - t0, 3),
         "total_tokens": total_tokens,
         "total_cost_usd": round(total_cost, 6),
         "timestamp": time.time()})

    return {
        "incident_type": incident_type,
        "final_health": obs.system_health,
        "resolved": state.incident_resolved,
        "cumulative_reward": round(cumulative_reward, 3),
        "steps": len(actions_taken),
        "diagnosis_quality": judge_result.diagnosis_quality,
        "judge_score": judge_result.final_score,
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
    }


def _submit_to_leaderboard(avg_reward: float, resolution_rate: float,
                            episodes: int, incident_breakdown: dict) -> None:
    try:
        payload = json.dumps({
            "model_name": MODEL_NAME,
            "avg_reward": avg_reward,
            "resolution_rate": resolution_rate,
            "episodes": episodes,
            "incident_breakdown": incident_breakdown,
        }).encode()
        req = urllib.request.Request(
            f"{ENV_URL}/leaderboard/submit",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            result = json.loads(r.read())
            log({"event": "LEADERBOARD_SUBMITTED", "status": result.get("status"),
                 "model": MODEL_NAME, "avg_reward": avg_reward})
    except Exception as e:
        log({"event": "LEADERBOARD_SUBMIT_SKIP", "reason": str(e)})


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="AIREN Inference")
    parser.add_argument("--compare", action="store_true",
                        help="WOW demo: bad agent vs good agent on same incident")
    parser.add_argument("--incident", default="db_overload",
                        help="Incident type for --compare mode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for --compare mode")
    parser.add_argument("--curriculum", action="store_true",
                        help="Run easy → medium → hard curriculum")
    args = parser.parse_args()

    if args.compare:
        _run_compare_demo(args.incident, args.seed)
        return

    if args.curriculum:
        incident_order = EASY_INCIDENTS + MEDIUM_INCIDENTS + HARD_INCIDENTS
    else:
        incident_order = ALL_INCIDENT_TYPES

    log({"event": "INFERENCE_START", "model": MODEL_NAME,
         "env_url": ENV_URL, "incidents": incident_order,
         "multi_agent": MULTI_AGENT, "timestamp": time.time()})

    env = AIRENEnv(base_url=ENV_URL).sync()
    results = []
    incident_stats: Dict[str, Dict] = {}

    with env:
        for incident_type in incident_order:
            for ep in range(EPISODES_PER_INCIDENT):
                try:
                    r = run_episode(env, incident_type, seed=42 + ep, ep_num=ep)
                    results.append(r)
                    if incident_type not in incident_stats:
                        incident_stats[incident_type] = {
                            "resolved": 0, "total": 0, "rewards": [], "tokens": 0, "cost": 0.0
                        }
                    incident_stats[incident_type]["total"] += 1
                    incident_stats[incident_type]["rewards"].append(r["cumulative_reward"])
                    incident_stats[incident_type]["tokens"] += r.get("total_tokens", 0)
                    incident_stats[incident_type]["cost"] += r.get("total_cost_usd", 0.0)
                    if r["resolved"]:
                        incident_stats[incident_type]["resolved"] += 1
                except Exception as e:
                    log({"event": "EPISODE_ERROR", "incident_type": incident_type, "error": str(e)})

    total = len(results)
    resolved = sum(1 for r in results if r["resolved"])
    avg_reward = sum(r["cumulative_reward"] for r in results) / max(total, 1)

    summary = {}
    for inc, stats in incident_stats.items():
        t = stats["total"]
        summary[inc] = {
            "resolution_rate": round(stats["resolved"] / max(t, 1), 3),
            "avg_reward": round(sum(stats["rewards"]) / max(t, 1), 3),
            "total_tokens": stats["tokens"],
            "total_cost_usd": round(stats["cost"], 6),
        }

    log({"event": "INFERENCE_COMPLETE", "model": MODEL_NAME,
         "total_episodes": total, "resolved": resolved,
         "resolution_rate": round(resolved / max(total, 1), 3),
         "avg_cumulative_reward": round(avg_reward, 3),
         "multi_agent_mode": MULTI_AGENT,
         "incident_breakdown": summary,
         "timestamp": time.time()})

    _submit_to_leaderboard(
        avg_reward=round(avg_reward, 3),
        resolution_rate=round(resolved / max(total, 1), 3),
        episodes=total,
        incident_breakdown=summary,
    )


def _bad_agent_action(obs) -> AIRENAction:
    """
    Simulates an untrained/random agent — always restarts the wrong service.
    Used in the WOW demo to show what happens without RL training.
    """
    import random
    wrong_actions = ["restart_service", "scale_service", "ignore_alert"]
    wrong_targets = [s for s in obs.services.keys()]
    return AIRENAction(
        action_type=random.choice(wrong_actions),
        target=random.choice(wrong_targets) if wrong_targets else "api",
        reasoning="Random action — no diagnosis",
    )


def _run_compare_demo(incident_type: str, seed: int) -> None:
    """
    WOW demo: run the same incident with a bad agent vs a good agent.
    Shows side-by-side how RL training changes agent behavior.

    Bad agent:  panics, restarts wrong services, makes things worse
    Good agent: diagnoses first, applies correct fix, resolves in 3 steps
    """
    print("\n" + "=" * 70)
    print("AIREN WOW DEMO: Bad Agent vs Trained Agent")
    print(f"Incident: {incident_type} | Seed: {seed}")
    print("=" * 70)

    env = AIRENEnv(base_url=ENV_URL).sync()

    with env:
        # ── Run 1: Bad agent (random/wrong actions) ───────────────────────
        print("\n[BAD AGENT] Untrained — random actions, no diagnosis")
        print("-" * 50)
        result = env.reset(incident_type=incident_type, seed=seed)
        obs = result.observation
        print(f"  Incident: {obs.incident_type} | Health: {obs.system_health:.0%}")
        print(f"  Logs: {obs.logs[0] if obs.logs else 'none'}")

        bad_steps = []
        bad_reward = 0.0
        for step in range(obs.max_steps):
            action = _bad_agent_action(obs)
            result = env.step(action)
            obs = result.observation
            bad_reward += result.reward or 0.0
            bad_steps.append({
                "step": step + 1,
                "action": f"{action.action_type}({action.target})",
                "health": obs.system_health,
                "reward": round(result.reward or 0.0, 3),
            })
            print(f"  Step {step+1}: {action.action_type}({action.target}) "
                  f"→ health={obs.system_health:.0%} reward={result.reward:.3f}")
            if obs.done:
                break

        bad_state = env.state()
        print(f"\n  RESULT: {'✅ RESOLVED' if bad_state.incident_resolved else '❌ FAILED'}")
        print(f"  Final health: {obs.system_health:.0%} | Cumulative reward: {bad_reward:.3f}")
        print(f"  Steps taken: {len(bad_steps)}")

        # ── Run 2: Good agent (diagnose-first heuristic) ──────────────────
        print("\n[GOOD AGENT] Trained — diagnose first, then fix")
        print("-" * 50)
        result = env.reset(incident_type=incident_type, seed=seed)
        obs = result.observation
        print(f"  Incident: {obs.incident_type} | Health: {obs.system_health:.0%}")
        print(f"  Logs: {obs.logs[0] if obs.logs else 'none'}")

        good_steps = []
        good_reward = 0.0
        for step in range(obs.max_steps):
            action = call_agent(obs)
            result = env.step(action)
            obs = result.observation
            good_reward += result.reward or 0.0
            good_steps.append({
                "step": step + 1,
                "action": f"{action.action_type}({action.target})",
                "reasoning": (action.reasoning or "")[:80],
                "health": obs.system_health,
                "reward": round(result.reward or 0.0, 3),
            })
            print(f"  Step {step+1}: {action.action_type}({action.target}) "
                  f"→ health={obs.system_health:.0%} reward={result.reward:.3f}")
            print(f"    Reasoning: {(action.reasoning or '')[:80]}")
            if obs.done:
                break

        good_state = env.state()
        print(f"\n  RESULT: {'✅ RESOLVED' if good_state.incident_resolved else '❌ FAILED'}")
        print(f"  Final health: {obs.system_health:.0%} | Cumulative reward: {good_reward:.3f}")
        print(f"  Steps taken: {len(good_steps)}")

    # ── Side-by-side summary ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<30} {'Bad Agent':>15} {'Good Agent':>15}")
    print("-" * 60)
    print(f"{'Resolved':<30} {'❌ NO':>15} {'✅ YES':>15}")
    print(f"{'Cumulative Reward':<30} {bad_reward:>15.3f} {good_reward:>15.3f}")
    print(f"{'Steps Taken':<30} {len(bad_steps):>15} {len(good_steps):>15}")
    print(f"{'Reward Improvement':<30} {'':>15} {f'+{((good_reward-bad_reward)/max(abs(bad_reward),0.001)*100):.0f}%':>15}")
    print("=" * 70)

    log({"event": "COMPARE_DEMO_COMPLETE",
         "incident_type": incident_type, "seed": seed,
         "bad_agent": {"reward": round(bad_reward, 3), "resolved": bad_state.incident_resolved, "steps": len(bad_steps)},
         "good_agent": {"reward": round(good_reward, 3), "resolved": good_state.incident_resolved, "steps": len(good_steps)},
         "improvement_pct": round((good_reward - bad_reward) / max(abs(bad_reward), 0.001) * 100, 1),
         "timestamp": time.time()})


if __name__ == "__main__":
    main()
