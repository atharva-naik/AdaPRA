"""
Schema dataclasses for AdaPRA trajectory collection and labeling.

TrajectoryRecord is the central object flowing through the pipeline:
  collect_trajectories → label_outcomes → run_programmatic_detectors
  → run_llm_judge → export_prm_dataset
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

@dataclass
class TaskInfo:
    task_id: str                      # e.g. "titanic" or competition slug
    challenge_name: str
    split: str                        # "easy" or "hard"
    description: str                  # full competition description / prompt
    docker_challenge_path: str        # path inside Docker container
    data_files: list[str] = field(default_factory=list)
    source: str = "dspredict"
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-turn step inside a trajectory
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryStep:
    turn: int
    role: str                   # "assistant" | "user" | "system"
    content: str                # raw message text
    # code extracted from <python>/<code> blocks (may be empty)
    code: str = ""
    # stdout / stderr from code execution (may be empty)
    observation: str = ""
    done: bool = False
    step_time: float = 0.0
    has_error: bool = False
    error_type: Optional[str] = None  # e.g. "KeyError", "ImportError"
    reasoning_content: str = ""  # internal reasoning from thinking models
    tool_type: str = ""  # "python", "bash", "file_read", "file_write", or ""


# ---------------------------------------------------------------------------
# Outcome labels (from DSGym evaluators / logs)
# ---------------------------------------------------------------------------

@dataclass
class OutcomeLabels:
    valid_submission: Optional[bool] = None
    final_score: Optional[float] = None           # Kaggle leaderboard score
    local_validation_score: Optional[float] = None
    improvement_over_first_baseline: Optional[float] = None
    first_valid_step_index: Optional[int] = None
    num_execution_errors: Optional[int] = None
    completed: Optional[bool] = None
    timeout: Optional[bool] = None
    above_median: Optional[bool] = None           # above leaderboard median
    percentile: Optional[float] = None            # leaderboard percentile rank
    medal: Optional[str] = None                   # "gold"/"silver"/"bronze"/null
    rank: Optional[int] = None                    # public leaderboard rank (from score_submissions.py)
    private_score: Optional[float] = None         # Kaggle private leaderboard score
    private_percentile: Optional[float] = None    # private leaderboard percentile
    private_above_median: Optional[bool] = None   # above median on private leaderboard
    private_medal: Optional[str] = None           # medal on private leaderboard


# ---------------------------------------------------------------------------
# Single rubric label (one per rubric per trajectory)
# ---------------------------------------------------------------------------

@dataclass
class RubricLabel:
    rubric_id: str
    label: Optional[str] = None       # "agent_failure"|"agent_success"|"not_applicable"|"insufficient_evidence"|None
    confidence: float = 0.0           # 0.0–1.0
    evidence: list[str] = field(default_factory=list)
    rationale: str = ""
    detector_version: str = "0.1.0"
    source: str = "programmatic"      # "programmatic" | "llm_judge"


# ---------------------------------------------------------------------------
# Full trajectory record (central pipeline object)
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryRecord:
    # Identity
    trajectory_id: str                # unique run ID, e.g. "{task_id}__{model}__{timestamp}"
    task: TaskInfo

    # Agent config
    agent_model: str
    agent_backend: str                # "litellm" | "openhands" | "mock"
    run_config: dict[str, Any] = field(default_factory=dict)

    # Trajectory content
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_answer: str = ""
    submission_path: Optional[str] = None   # local path to submission.csv if saved
    raw_log_path: Optional[str] = None      # path to raw log file

    # Labels (filled progressively by labeling scripts)
    outcome: OutcomeLabels = field(default_factory=OutcomeLabels)
    rubric_labels: dict[str, RubricLabel] = field(default_factory=dict)

    # LLM judge output (filled by run_llm_judge)
    overall_scalar_quality: Optional[float] = None
    short_critique: Optional[str] = None
    suggested_next_actions: list[str] = field(default_factory=list)
    raw_judge_response: Optional[str] = None

    # Run metadata
    num_turns: int = 0
    total_tokens: int = 0
    execution_time: float = 0.0
    errors: list[str] = field(default_factory=list)
    timed_out: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------------------
    # Serialization helpers
    # ---------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for step in d.get("steps", []):
            if not step.get("reasoning_content"):
                step.pop("reasoning_content", None)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrajectoryRecord":
        task = TaskInfo(**d["task"])
        steps = [TrajectoryStep(**{k: v for k, v in s.items() if k != "reward"})
                 for s in d.get("steps", [])]
        # Filter to declared OutcomeLabels fields: extra keys (e.g.
        # `selected_scoring`, written into the JSON by the --score-selected
        # scorer) are preserved on disk by the raw-JSON writeback but must not
        # be passed to the dataclass ctor.
        _outcome_fields = {f.name for f in fields(OutcomeLabels)}
        outcome = OutcomeLabels(**{k: v for k, v in d.get("outcome", {}).items()
                                   if k in _outcome_fields})
        rubric_labels = {
            k: RubricLabel(**v)
            for k, v in d.get("rubric_labels", {}).items()
        }
        return cls(
            trajectory_id=d["trajectory_id"],
            task=task,
            agent_model=d["agent_model"],
            agent_backend=d["agent_backend"],
            run_config=d.get("run_config", {}),
            steps=steps,
            final_answer=d.get("final_answer", ""),
            submission_path=d.get("submission_path"),
            raw_log_path=d.get("raw_log_path"),
            outcome=outcome,
            rubric_labels=rubric_labels,
            overall_scalar_quality=d.get("overall_scalar_quality"),
            short_critique=d.get("short_critique"),
            suggested_next_actions=d.get("suggested_next_actions", []),
            raw_judge_response=d.get("raw_judge_response"),
            num_turns=d.get("num_turns", 0),
            total_tokens=d.get("total_tokens", 0),
            execution_time=d.get("execution_time", 0.0),
            errors=d.get("errors", []),
            timed_out=d.get("timed_out", False),
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "TrajectoryRecord":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
