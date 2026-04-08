# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
Human-in-the-Loop (HITL) Evaluator for AIREN.

Domain expert SREs can rate agent reasoning quality on a 5-point scale.
Ratings are stored and used to:
  1. Calibrate the LLM judge (compare human vs LLM scores)
  2. Provide richer training signal (human preference data for DPO/RLHF)
  3. Build a public leaderboard with human-validated scores
  4. Identify systematic failure modes (e.g. agent always misses DDoS)

This is the "Human-in-the-Loop evaluation" unsaturated area from the
Centific SF hackathon blog — most submissions have no human validation.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class HITLRating:
    """A single human expert rating of an agent episode."""
    rating_id: str
    episode_id: str
    incident_type: str
    rater_id: str               # anonymous rater identifier
    rater_role: str             # "sre" | "devops" | "developer" | "other"

    # 5-point Likert scale ratings
    diagnosis_accuracy: int     # 1-5: Did agent correctly identify root cause?
    action_quality: int         # 1-5: Were actions logical and in right order?
    efficiency: int             # 1-5: Did agent avoid unnecessary actions?
    recovery_handling: int      # 1-5: How well did agent handle wrong fixes?
    overall: int                # 1-5: Overall incident response quality

    # Free-text feedback
    what_went_well: str = ""
    what_went_wrong: str = ""
    suggested_improvement: str = ""

    # Episode context (for display)
    actions_taken: List[str] = field(default_factory=list)
    final_health: float = 0.0
    incident_resolved: bool = False
    llm_judge_score: float = 0.0
    llm_diagnosis_quality: str = ""

    timestamp: float = field(default_factory=time.time)

    @property
    def composite_score(self) -> float:
        """Weighted composite score 0.0-1.0. Weights configurable via env vars."""
        import os
        w_diag = float(os.environ.get("HITL_W_DIAGNOSIS", "0.35"))
        w_act  = float(os.environ.get("HITL_W_ACTION",    "0.30"))
        w_eff  = float(os.environ.get("HITL_W_EFFICIENCY","0.15"))
        w_rec  = float(os.environ.get("HITL_W_RECOVERY",  "0.20"))
        # Normalize weights to sum to 1.0
        total_w = w_diag + w_act + w_eff + w_rec
        raw = (
            self.diagnosis_accuracy  * (w_diag / total_w) +
            self.action_quality      * (w_act  / total_w) +
            self.efficiency          * (w_eff  / total_w) +
            self.recovery_handling   * (w_rec  / total_w)
        )
        return round((raw - 1) / 4, 3)  # normalize 1-5 → 0.0-1.0


@dataclass
class HITLStats:
    """Aggregated HITL statistics across all ratings."""
    total_ratings: int
    avg_diagnosis_accuracy: float
    avg_action_quality: float
    avg_efficiency: float
    avg_recovery_handling: float
    avg_overall: float
    avg_composite: float
    llm_human_correlation: float    # how well LLM judge matches human ratings
    by_incident_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_rater_role: Dict[str, int] = field(default_factory=dict)


# ── Storage ───────────────────────────────────────────────────────────────────

_RATINGS_FILE = Path(os.environ.get("HITL_RATINGS_FILE", "/tmp/airen_hitl_ratings.jsonl"))
_ratings_cache: List[HITLRating] = []


def _load_ratings() -> List[HITLRating]:
    global _ratings_cache
    if _ratings_cache:
        return _ratings_cache
    if not _RATINGS_FILE.exists():
        return []
    ratings = []
    try:
        with open(_RATINGS_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    ratings.append(HITLRating(**data))
    except Exception:
        pass
    _ratings_cache = ratings
    return ratings


def _save_rating(rating: HITLRating) -> None:
    global _ratings_cache
    _ratings_cache.append(rating)
    try:
        _RATINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_RATINGS_FILE, "a") as f:
            f.write(json.dumps(asdict(rating)) + "\n")
    except Exception:
        pass


# ── Public API ────────────────────────────────────────────────────────────────

def submit_rating(
    episode_id: str,
    incident_type: str,
    rater_id: str,
    rater_role: str,
    diagnosis_accuracy: int,
    action_quality: int,
    efficiency: int,
    recovery_handling: int,
    overall: int,
    what_went_well: str = "",
    what_went_wrong: str = "",
    suggested_improvement: str = "",
    actions_taken: Optional[List[str]] = None,
    final_health: float = 0.0,
    incident_resolved: bool = False,
    llm_judge_score: float = 0.0,
    llm_diagnosis_quality: str = "",
) -> HITLRating:
    """Submit a human expert rating for an episode."""
    import uuid
    # Validate ratings are 1-5
    for name, val in [
        ("diagnosis_accuracy", diagnosis_accuracy),
        ("action_quality", action_quality),
        ("efficiency", efficiency),
        ("recovery_handling", recovery_handling),
        ("overall", overall),
    ]:
        if not 1 <= val <= 5:
            raise ValueError(f"{name} must be 1-5, got {val}")

    rating = HITLRating(
        rating_id=str(uuid.uuid4())[:8],
        episode_id=episode_id,
        incident_type=incident_type,
        rater_id=rater_id,
        rater_role=rater_role,
        diagnosis_accuracy=diagnosis_accuracy,
        action_quality=action_quality,
        efficiency=efficiency,
        recovery_handling=recovery_handling,
        overall=overall,
        what_went_well=what_went_well,
        what_went_wrong=what_went_wrong,
        suggested_improvement=suggested_improvement,
        actions_taken=actions_taken or [],
        final_health=final_health,
        incident_resolved=incident_resolved,
        llm_judge_score=llm_judge_score,
        llm_diagnosis_quality=llm_diagnosis_quality,
    )
    _save_rating(rating)
    return rating


def get_stats() -> HITLStats:
    """Compute aggregated statistics across all human ratings."""
    ratings = _load_ratings()
    if not ratings:
        return HITLStats(
            total_ratings=0,
            avg_diagnosis_accuracy=0.0,
            avg_action_quality=0.0,
            avg_efficiency=0.0,
            avg_recovery_handling=0.0,
            avg_overall=0.0,
            avg_composite=0.0,
            llm_human_correlation=0.0,
        )

    n = len(ratings)
    avg = lambda field: round(sum(getattr(r, field) for r in ratings) / n, 2)

    # LLM-human correlation (Pearson-like, simplified)
    human_scores = [r.composite_score for r in ratings]
    llm_scores = [r.llm_judge_score for r in ratings]
    if len(set(human_scores)) > 1 and len(set(llm_scores)) > 1:
        mean_h = sum(human_scores) / n
        mean_l = sum(llm_scores) / n
        cov = sum((h - mean_h) * (l - mean_l) for h, l in zip(human_scores, llm_scores)) / n
        std_h = (sum((h - mean_h) ** 2 for h in human_scores) / n) ** 0.5
        std_l = (sum((l - mean_l) ** 2 for l in llm_scores) / n) ** 0.5
        corr = round(cov / (std_h * std_l + 1e-9), 3)
    else:
        corr = 0.0

    # By incident type
    by_type: Dict[str, Dict[str, float]] = {}
    for r in ratings:
        if r.incident_type not in by_type:
            by_type[r.incident_type] = {"count": 0, "composite_sum": 0.0, "overall_sum": 0.0}
        by_type[r.incident_type]["count"] += 1
        by_type[r.incident_type]["composite_sum"] += r.composite_score
        by_type[r.incident_type]["overall_sum"] += r.overall
    for itype, stats in by_type.items():
        c = stats["count"]
        by_type[itype] = {
            "count": c,
            "avg_composite": round(stats["composite_sum"] / c, 3),
            "avg_overall": round(stats["overall_sum"] / c, 2),
        }

    # By rater role
    by_role: Dict[str, int] = {}
    for r in ratings:
        by_role[r.rater_role] = by_role.get(r.rater_role, 0) + 1

    return HITLStats(
        total_ratings=n,
        avg_diagnosis_accuracy=avg("diagnosis_accuracy"),
        avg_action_quality=avg("action_quality"),
        avg_efficiency=avg("efficiency"),
        avg_recovery_handling=avg("recovery_handling"),
        avg_overall=avg("overall"),
        avg_composite=round(sum(r.composite_score for r in ratings) / n, 3),
        llm_human_correlation=corr,
        by_incident_type=by_type,
        by_rater_role=by_role,
    )


def get_recent_ratings(n: int = 10) -> List[Dict[str, Any]]:
    """Get the N most recent ratings for display."""
    ratings = _load_ratings()
    recent = sorted(ratings, key=lambda r: r.timestamp, reverse=True)[:n]
    return [asdict(r) for r in recent]


def get_calibration_data() -> Dict[str, Any]:
    """
    Returns data for calibrating the LLM judge against human ratings.
    Shows where LLM over/under-estimates compared to human experts.
    """
    ratings = _load_ratings()
    if not ratings:
        return {"calibration_pairs": [], "bias": 0.0, "rmse": 0.0}

    pairs = [
        {
            "episode_id": r.episode_id,
            "incident_type": r.incident_type,
            "human_score": r.composite_score,
            "llm_score": r.llm_judge_score,
            "delta": round(r.composite_score - r.llm_judge_score, 3),
        }
        for r in ratings
    ]
    deltas = [p["delta"] for p in pairs]
    bias = round(sum(deltas) / len(deltas), 3)
    rmse = round((sum(d ** 2 for d in deltas) / len(deltas)) ** 0.5, 3)

    return {
        "calibration_pairs": pairs[-20:],  # last 20
        "bias": bias,  # positive = LLM underestimates, negative = overestimates
        "rmse": rmse,
        "n": len(pairs),
        "interpretation": (
            "LLM judge is well-calibrated" if abs(bias) < 0.1
            else f"LLM judge {'underestimates' if bias > 0 else 'overestimates'} by {abs(bias):.2f}"
        ),
    }
