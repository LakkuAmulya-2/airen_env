# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
Infinite Loop Circuit Breaker — Upgrade #2

Detects and breaks infinite loops before they consume millions of tokens.
Solves the Claude Code GitHub issue #15909: 27M tokens consumed in 4.6 hours
running npm install 300+ times.

Real pain point: LLM agents get stuck in repetitive action loops that cost
$400+ on $5 tasks. This circuit breaker catches the pattern early.

Integrated into the ARL (Agentic Reliability Layer) as an additional
detection layer beyond the existing DeterministicCircuitBreaker.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass
class LoopDetectionResult:
    loop_detected: bool
    pattern: str                    # same_tool_repetition | cyclic_action_pattern | token_spike
    tool: str = ""
    cycle: List[str] = field(default_factory=list)
    count: int = 0
    avg_tokens_per_step: float = 0.0
    recommendation: str = ""        # BREAK_LOOP | CIRCUIT_OPEN | NONE
    confidence: float = 0.0         # 0.0-1.0


@dataclass
class LoopBreakAction:
    action: str                     # escalate_to_human | suggest_alternative | warn
    reason: str
    suggestion: str = ""
    cooldown_steps: int = 0
    estimated_tokens_saved: int = 0


class InfiniteLoopCircuitBreaker:
    """
    Detect and break infinite loops before they consume millions of tokens.

    Three detection patterns:
    1. Same tool called N+ times consecutively (npm install loop)
    2. Cyclic pattern: A→B→A→B→A→B (oscillation loop)
    3. Token consumption spike (runaway generation)

    Circuit states:
    - CLOSED: normal operation
    - OPEN: loop detected, blocking actions
    - HALF_OPEN: testing if loop resolved
    """

    # Configurable thresholds
    SAME_TOOL_THRESHOLD = 5          # same tool N times in a row
    CYCLE_MIN_LENGTH = 2             # minimum cycle length to detect
    CYCLE_REPETITIONS = 3            # how many times cycle must repeat
    TOKEN_SPIKE_THRESHOLD = 50_000   # avg tokens/step above this = spike
    WINDOW_SIZE = 20                 # sliding window for action history

    def __init__(self) -> None:
        self._action_history: Deque[Dict[str, Any]] = deque(maxlen=self.WINDOW_SIZE)
        self._token_consumption: List[float] = []
        self._circuit_state: str = "CLOSED"
        self._cooldown_remaining: int = 0
        self._total_loops_detected: int = 0
        self._total_tokens_saved: int = 0
        self._loop_events: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self._action_history.clear()
        self._token_consumption.clear()
        self._circuit_state = "CLOSED"
        self._cooldown_remaining = 0

    def record_action(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        tokens_used: int = 0,
    ) -> None:
        """Record an action for loop detection analysis."""
        self._action_history.append({
            "tool": tool_name,
            "params_hash": _hash_params(parameters),
            "tokens": tokens_used,
            "timestamp": time.time(),
        })
        if tokens_used > 0:
            self._token_consumption.append(float(tokens_used))

        # Decrement cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining == 0:
                self._circuit_state = "HALF_OPEN"

    def detect_loop(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        tokens_used: int = 0,
    ) -> LoopDetectionResult:
        """
        Detect loop patterns in the action history.
        Call BEFORE recording the action to get pre-action detection.
        """
        # Circuit already open — block immediately
        if self._circuit_state == "OPEN":
            return LoopDetectionResult(
                loop_detected=True,
                pattern="circuit_open",
                tool=tool_name,
                recommendation="CIRCUIT_OPEN",
                confidence=1.0,
            )

        history = list(self._action_history)
        recent_tools = [a["tool"] for a in history]

        # Pattern 1: Same tool called N+ times consecutively
        if len(recent_tools) >= self.SAME_TOOL_THRESHOLD:
            last_n = recent_tools[-self.SAME_TOOL_THRESHOLD:]
            if len(set(last_n)) == 1 and last_n[0] == tool_name:
                return LoopDetectionResult(
                    loop_detected=True,
                    pattern="same_tool_repetition",
                    tool=tool_name,
                    count=self.SAME_TOOL_THRESHOLD + 1,
                    recommendation="BREAK_LOOP",
                    confidence=0.95,
                )

        # Pattern 2: Cyclic pattern (A→B→A→B)
        cycle = self._detect_cycle(recent_tools + [tool_name])
        if cycle:
            return LoopDetectionResult(
                loop_detected=True,
                pattern="cyclic_action_pattern",
                tool=tool_name,
                cycle=cycle,
                count=len(cycle),
                recommendation="BREAK_LOOP",
                confidence=0.85,
            )

        # Pattern 3: Token consumption spike
        if len(self._token_consumption) >= 5:
            avg = sum(self._token_consumption[-5:]) / 5
            if avg > self.TOKEN_SPIKE_THRESHOLD:
                return LoopDetectionResult(
                    loop_detected=True,
                    pattern="token_consumption_spike",
                    tool=tool_name,
                    avg_tokens_per_step=avg,
                    recommendation="CIRCUIT_OPEN",
                    confidence=0.75,
                )

        return LoopDetectionResult(
            loop_detected=False,
            pattern="none",
            recommendation="NONE",
            confidence=0.0,
        )

    def break_loop(
        self,
        detection: LoopDetectionResult,
        step_number: int,
        max_steps: int,
    ) -> LoopBreakAction:
        """
        Determine the circuit breaker action for a detected loop.
        Returns the action to take and updates circuit state.
        """
        self._total_loops_detected += 1

        # Estimate tokens saved (avg 10k tokens/step * remaining steps)
        steps_remaining = max(0, max_steps - step_number)
        estimated_saved = steps_remaining * 10_000
        self._total_tokens_saved += estimated_saved

        self._loop_events.append({
            "step": step_number,
            "pattern": detection.pattern,
            "tool": detection.tool,
            "tokens_saved_estimate": estimated_saved,
            "timestamp": time.time(),
        })

        if detection.recommendation == "CIRCUIT_OPEN":
            self._circuit_state = "OPEN"
            self._cooldown_remaining = 10
            return LoopBreakAction(
                action="escalate_to_human",
                reason=f"Infinite loop detected: {detection.pattern} on '{detection.tool}'",
                suggestion=(
                    "Token consumption spike detected. "
                    "Agent is stuck in a runaway loop. "
                    "Manual intervention required."
                ),
                cooldown_steps=10,
                estimated_tokens_saved=estimated_saved,
            )

        elif detection.recommendation == "BREAK_LOOP":
            self._circuit_state = "HALF_OPEN"
            self._cooldown_remaining = 3

            if detection.pattern == "same_tool_repetition":
                suggestion = (
                    f"You have called '{detection.tool}' {detection.count} times in a row. "
                    f"This approach is not working. Try a different tool or target."
                )
            elif detection.pattern == "cyclic_action_pattern":
                cycle_str = " → ".join(detection.cycle)
                suggestion = (
                    f"Cyclic loop detected: {cycle_str}. "
                    f"You are oscillating between the same actions. "
                    f"Escalate to human or try a completely different approach."
                )
            else:
                suggestion = "Loop detected. Try a different approach."

            return LoopBreakAction(
                action="suggest_alternative",
                reason=f"Loop detected: {detection.pattern}",
                suggestion=suggestion,
                cooldown_steps=3,
                estimated_tokens_saved=estimated_saved,
            )

        return LoopBreakAction(
            action="warn",
            reason="Potential loop forming",
            suggestion="Consider varying your approach",
            cooldown_steps=0,
            estimated_tokens_saved=0,
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "circuit_state": self._circuit_state,
            "total_loops_detected": self._total_loops_detected,
            "total_tokens_saved_estimate": self._total_tokens_saved,
            "cooldown_remaining": self._cooldown_remaining,
            "action_history_size": len(self._action_history),
            "recent_events": self._loop_events[-5:],
        }

    def _detect_cycle(self, tools: List[str]) -> List[str]:
        """
        Detect repeating cycle in tool sequence.
        Returns the cycle if found, empty list otherwise.
        """
        n = len(tools)
        # Try cycle lengths from 2 to n//3
        for cycle_len in range(self.CYCLE_MIN_LENGTH, n // self.CYCLE_REPETITIONS + 1):
            # Check if last (cycle_len * CYCLE_REPETITIONS) tools form a cycle
            needed = cycle_len * self.CYCLE_REPETITIONS
            if n < needed:
                continue
            tail = tools[-needed:]
            candidate = tail[:cycle_len]
            # Verify the cycle repeats
            is_cycle = all(
                tail[i * cycle_len:(i + 1) * cycle_len] == candidate
                for i in range(self.CYCLE_REPETITIONS)
            )
            if is_cycle:
                return candidate
        return []


def _hash_params(params: Dict[str, Any]) -> str:
    """Simple deterministic hash of parameters for comparison."""
    try:
        import json
        return str(hash(json.dumps(params, sort_keys=True)))
    except Exception:
        return str(hash(str(params)))


# Module-level singleton
_breaker: Optional[InfiniteLoopCircuitBreaker] = None


def get_loop_breaker() -> InfiniteLoopCircuitBreaker:
    global _breaker
    if _breaker is None:
        _breaker = InfiniteLoopCircuitBreaker()
    return _breaker
