# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
LLM Judge for AIREN — evaluates agent's diagnosis quality.

Rule-based grader catches clear wins/losses.
LLM judge evaluates nuanced reasoning quality:
  - Did agent correctly identify root cause?
  - Was the diagnosis reasoning sound?
  - Were actions taken in logical order?
  - Did agent avoid red herrings?
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI


def _clamp_score(score: float) -> float:
    return max(0.001, min(0.999, round(score, 3)))


@dataclass
class AIRENJudgeResult:
    rule_score: float
    llm_score: Optional[float]
    final_score: float
    diagnosis_quality: str      # "excellent" | "good" | "poor" | "wrong"
    reasoning_feedback: str
    judge_used: str
    latency_ms: float
    tokens_used: int


_JUDGE_SYSTEM = """You are an expert Site Reliability Engineer (SRE) evaluating an AI agent's incident response.

You will receive:
1. The incident type and root cause
2. The actions the agent took (in order)
3. The final system health achieved

Score the agent's DIAGNOSIS QUALITY on 0.0-1.0:
  1.0 = Correctly identified root cause, took logical actions in right order
  0.8 = Mostly correct diagnosis, minor inefficiencies
  0.6 = Partially correct, some wrong actions but recovered
  0.4 = Confused diagnosis, many wrong actions
  0.2 = Wrong diagnosis, made things worse
  0.0 = Completely wrong, ignored obvious signals

Respond ONLY with valid JSON:
{
  "score": <float 0.0-1.0>,
  "diagnosis_quality": "excellent|good|poor|wrong",
  "reasoning": "<2-3 sentences explaining the score>",
  "key_insight": "<what the agent did right or wrong>",
  "confidence": "high|medium|low"
}"""


class AIRENLLMJudge:
    """
    LLM judge for AIREN incident response quality.
    Called after episode ends to evaluate diagnosis reasoning.
    Falls back gracefully if API unavailable.
    """

    AMBIGUOUS_LOW  = 0.35
    AMBIGUOUS_HIGH = 0.75
    RULE_WEIGHT    = 0.55
    LLM_WEIGHT     = 0.45

    def __init__(self):
        _key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
        _base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
        self._model = os.environ.get("JUDGE_MODEL") or os.environ.get("MODEL_NAME", "gpt-4o-mini")
        self._available = bool(_key)
        self._client = OpenAI(api_key=_key or "no-key", base_url=_base) if self._available else None
        self._call_count = 0
        self._total_tokens = 0

    def judge(
        self,
        incident_type: str,
        root_cause: str,
        actions_taken: List[str],
        final_health: float,
        incident_resolved: bool,
        correct_actions: List[str],
        correct_targets: List[str],
        rule_score: float,
    ) -> AIRENJudgeResult:
        t0 = time.time()

        # Fast path: rule score is confident
        if not self._available or not (self.AMBIGUOUS_LOW <= rule_score <= self.AMBIGUOUS_HIGH):
            return AIRENJudgeResult(
                rule_score=rule_score,
                llm_score=None,
                final_score=_clamp_score(rule_score),
                diagnosis_quality=self._quality_label(rule_score),
                reasoning_feedback="Rule-based score confident — LLM judge skipped.",
                judge_used="rules_only",
                latency_ms=round((time.time() - t0) * 1000, 1),
                tokens_used=0,
            )

        # LLM judge for ambiguous cases
        try:
            prompt = f"""Incident: {incident_type}
Root cause: {root_cause}
Agent actions (in order): {', '.join(actions_taken)}
Final system health: {final_health:.0%}
Incident resolved: {incident_resolved}
Correct approach was: {correct_actions} on {correct_targets}"""

            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=250,
                response_format={"type": "json_object"},
            )
            raw = completion.choices[0].message.content or "{}"
            tokens = completion.usage.total_tokens if completion.usage else 0
            data = json.loads(raw)
            llm_score = _clamp_score(float(data.get("score", rule_score)))
            final = _clamp_score(round(self.RULE_WEIGHT * rule_score + self.LLM_WEIGHT * llm_score, 3))
            self._call_count += 1
            self._total_tokens += tokens

            return AIRENJudgeResult(
                rule_score=rule_score,
                llm_score=llm_score,
                final_score=final,
                diagnosis_quality=data.get("diagnosis_quality", self._quality_label(final)),
                reasoning_feedback=data.get("reasoning", ""),
                judge_used="llm+rules",
                latency_ms=round((time.time() - t0) * 1000, 1),
                tokens_used=tokens,
            )
        except Exception as e:
            return AIRENJudgeResult(
                rule_score=rule_score,
                llm_score=None,
                final_score=_clamp_score(rule_score),
                diagnosis_quality=self._quality_label(rule_score),
                reasoning_feedback=f"LLM judge failed: {e}",
                judge_used="rules_only_fallback",
                latency_ms=round((time.time() - t0) * 1000, 1),
                tokens_used=0,
            )

    def _quality_label(self, score: float) -> str:
        if score >= 0.8: return "excellent"
        if score >= 0.6: return "good"
        if score >= 0.4: return "poor"
        return "wrong"

    def get_stats(self) -> Dict[str, Any]:
        return {
            "llm_calls": self._call_count,
            "total_tokens": self._total_tokens,
            "available": self._available,
        }


_judge: Optional[AIRENLLMJudge] = None

def get_judge() -> AIRENLLMJudge:
    global _judge
    if _judge is None:
        _judge = AIRENLLMJudge()
    return _judge
