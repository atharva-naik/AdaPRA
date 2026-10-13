"""
LLM-as-judge v2: importance-weighted probabilistic scoring.

Replaces the 4-class label scheme with continuous outputs per rubric:
  - p_agent_success: P(agent followed this rubric correctly) in [0, 1]
  - importance_score: I(r) in [0, 1] — how relevant this rubric is to THIS trajectory

Composite scoring formula:
  Score_τ = Σ(I(r) × [p_r × reward_success + (1-p_r) × reward_failure]) / Σ(I(r))

With default rewards (+1 success, -1 failure):
  Score_τ = Σ(I(r) × [2×p_r - 1]) / Σ(I(r))

This replaces hard dataset-level rubric selection with soft per-trajectory weighting.
"""

from __future__ import annotations

import json
import os
import re
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from .rubrics import RUBRIC_REGISTRY, RubricDef, list_rubrics
from .schema import TrajectoryRecord

_JSON_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)```|(\{[\s\S]+\})", re.DOTALL)


def _extract_json(text: str) -> Optional[dict]:
    m = _JSON_RE.search(text)
    if m:
        raw = m.group(1) or m.group(2)
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def _compress_trajectory(
    record: TrajectoryRecord,
    max_chars: int = 150000,
    code_chars: int = 5000,
    obs_chars: int = 2000,
) -> str:
    """Compact trajectory text with 150k char limit for v2 (minimal truncation)."""
    lines = []
    lines.append(f"TASK: {record.task.challenge_name} [{record.task.split}]")
    lines.append(f"MODEL: {record.agent_model}")
    lines.append(
        f"TURNS: {record.num_turns}  ERRORS: {record.outcome.num_execution_errors}  "
        f"VALID_SUB: {record.outcome.valid_submission}"
    )
    lines.append("")
    for step in record.steps:
        if step.role == "assistant" and step.code:
            lines.append(f"[T{step.turn}:code] {step.code[:code_chars].replace(chr(10), ' | ')}")
        if step.role == "user" and step.observation:
            lines.append(f"[T{step.turn}:obs]  {step.observation[:obs_chars].replace(chr(10), ' | ')}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text


def _build_rubric_prompt_v2(trajectory_summary: str, task_desc: str, rubric: RubricDef) -> str:
    """Single-rubric v2 prompt: outputs importance + probability instead of label + confidence."""
    instruction = rubric.natural_language_instruction or rubric.description
    return f"""You are an expert data science competition judge evaluating an agent trajectory.

=== TASK DESCRIPTION ===
{task_desc}

=== TRAJECTORY ===
{trajectory_summary}

=== RUBRIC ===
{instruction}

Your task is to evaluate this rubric for this specific trajectory. Produce TWO scores:

1. **importance_score** (0.0 to 1.0): How relevant is this rubric to THIS trajectory?
   - 0.0 = completely irrelevant (the rubric topic never comes up, or the task doesn't involve this aspect)
   - 0.3-0.5 = marginally relevant (the topic is tangentially present but not central)
   - 0.6-0.8 = clearly relevant (the agent actively engages with the behavior this rubric measures)
   - 0.9-1.0 = highly relevant (this rubric targets a core aspect of what the agent did)

2. **p_agent_success** (0.0 to 1.0): Probability that the agent SUCCEEDED at following the good practice described by this rubric.
   - 0.0 = certain failure (clear, unambiguous violation of the rubric)
   - 0.1-0.3 = likely failure (strong evidence of violation, minor ambiguity)
   - 0.4-0.6 = uncertain (insufficient evidence or mixed signals)
   - 0.7-0.9 = likely success (good evidence the agent handled this correctly)
   - 1.0 = certain success (clear, unambiguous adherence)

   When the rubric is not applicable to this trajectory, set importance_score close to 0 and p_agent_success to 0.5 (maximum uncertainty).

Respond with ONLY valid JSON:
{{
  "p_agent_success": <float 0.0-1.0>,
  "importance_score": <float 0.0-1.0>,
  "evidence": ["[T<turn>:<code|obs>] <brief verbatim quote from that turn>"],
  "rationale": "one sentence explaining the scores"
}}

IMPORTANT: Each evidence entry MUST start with a turn prefix like [T5:code] or [T12:obs] matching the trajectory format, followed by a brief verbatim quote from that turn. This makes evidence grounding verifiable. If the rubric is not applicable, evidence can be empty."""


def _build_batched_rubric_prompt_v2(
    trajectory_summary: str, task_desc: str, rubrics: list[RubricDef]
) -> str:
    """Batched v2 prompt: score ALL rubrics in a single LLM call.

    Same per-rubric scoring semantics as `_build_rubric_prompt_v2`, but the model
    evaluates every rubric in one pass and returns a JSON object keyed by
    rubric_id. This trades N calls for 1 (big cost/latency win) at the risk of
    slightly shallower per-rubric reasoning; the schema forces one evidence quote
    + one-sentence rationale per rubric to keep it grounded.
    """
    rubric_block = "\n\n".join(
        f"### RUBRIC: {r.rubric_id}\n{r.natural_language_instruction or r.description}"
        for r in rubrics
    )
    ids = [r.rubric_id for r in rubrics]
    example_id = ids[0]
    return f"""You are an expert data science competition judge evaluating an agent trajectory.

=== TASK DESCRIPTION ===
{task_desc}

=== TRAJECTORY ===
{trajectory_summary}

=== RUBRICS ({len(rubrics)}) ===
Evaluate EVERY rubric below against this trajectory.
{rubric_block}

For EACH rubric, produce TWO scores:

1. **importance_score** (0.0 to 1.0): How relevant is this rubric to THIS trajectory?
   - 0.0 = completely irrelevant (the rubric topic never comes up, or the task doesn't involve this aspect)
   - 0.3-0.5 = marginally relevant (the topic is tangentially present but not central)
   - 0.6-0.8 = clearly relevant (the agent actively engages with the behavior this rubric measures)
   - 0.9-1.0 = highly relevant (this rubric targets a core aspect of what the agent did)

2. **p_agent_success** (0.0 to 1.0): Probability that the agent SUCCEEDED at following the good practice described by this rubric.
   - 0.0 = certain failure (clear, unambiguous violation of the rubric)
   - 0.1-0.3 = likely failure (strong evidence of violation, minor ambiguity)
   - 0.4-0.6 = uncertain (insufficient evidence or mixed signals)
   - 0.7-0.9 = likely success (good evidence the agent handled this correctly)
   - 1.0 = certain success (clear, unambiguous adherence)

   When a rubric is not applicable to this trajectory, set importance_score close to 0 and p_agent_success to 0.5 (maximum uncertainty).

3. **suggested_actions** (string): ONLY when the agent is likely FAILING this rubric (low p_agent_success), give ONE concrete, imperative next action to fix it, grounded in the specific evidence (name the exact file / model / step). No generic advice. If the agent is succeeding, set this to "".

Respond with ONLY valid JSON — a single object mapping EACH rubric_id to its scores. Include ALL {len(rubrics)} rubric_ids exactly as written above:
{{
  "{example_id}": {{
    "p_agent_success": <float 0.0-1.0>,
    "importance_score": <float 0.0-1.0>,
    "evidence": ["[T<turn>:<code|obs>] <brief verbatim quote from that turn>"],
    "rationale": "one sentence explaining the scores",
    "suggested_actions": "one concrete imperative fix if failing, else \\"\\""
  }},
  ...one entry per rubric_id...
}}

IMPORTANT: Each evidence entry MUST start with a turn prefix like [T5:code] or [T12:obs] matching the trajectory format, followed by a brief verbatim quote from that turn. This makes evidence grounding verifiable. If a rubric is not applicable, its evidence can be empty. Return every rubric_id; do not omit any."""


def compute_trajectory_score(
    rubric_scores: dict[str, dict[str, float]],
    reward_success: float = 1.0,
    reward_failure: float = -1.0,
) -> float:
    """Compute composite trajectory score from per-rubric importance + probability.

    Args:
        rubric_scores: {rubric_id: {"p_agent_success": float, "importance_score": float}}
        reward_success: reward for agent_success (default +1)
        reward_failure: reward for agent_failure (default -1)

    Returns:
        Score in [reward_failure, reward_success], or 0.0 if no rubrics have importance > 0.
    """
    numerator = 0.0
    denominator = 0.0
    for scores in rubric_scores.values():
        importance = scores.get("importance_score", 0.0)
        p_success = scores.get("p_agent_success", 0.5)
        numerator += importance * (p_success * reward_success + (1 - p_success) * reward_failure)
        denominator += importance
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


class LLMJudgeV2:
    """
    V2 judge: outputs importance_score + p_agent_success per rubric.
    One API call per rubric (same parallelization as v1).
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        mock: bool = False,
        temperature: Optional[float] = None,
        workers: int = 8,
        client_type: str = "litellm",
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        max_chars: int = 300000,
        code_chars: int = 20000,
        obs_chars: int = 8000,
        max_retries: int = 3,
        **litellm_kwargs: Any,
    ):
        self.model = model
        if api_key:
            self.api_key = api_key
        elif not model.startswith("bedrock/"):
            self.api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        else:
            self.api_key = None
        self.base_url = base_url
        self.mock = mock
        self.temperature = temperature
        self.workers = workers
        self.client_type = client_type
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.max_chars = max_chars
        self.code_chars = code_chars
        self.obs_chars = obs_chars
        self.max_retries = max_retries
        self.litellm_kwargs = litellm_kwargs
        # Bedrock region routing with fallback: a 503 on the primary region
        # (shared preview-model serving capacity) rotates to a fallback region
        # on the next retry rather than re-hammering the same overloaded
        # endpoint. aws_region_name = primary; aws_region_fallbacks = list (or
        # comma-string) of fallbacks. Pulled out of litellm_kwargs so we can
        # inject the chosen region per-attempt instead of passing it verbatim.
        _primary = self.litellm_kwargs.pop("aws_region_name", None)
        _fallbacks = self.litellm_kwargs.pop("aws_region_fallbacks", None) or []
        if isinstance(_fallbacks, str):
            _fallbacks = [r.strip() for r in _fallbacks.split(",") if r.strip()]
        _regions: list[Optional[str]] = []
        for _r in [_primary, *_fallbacks]:
            if _r and _r not in _regions:
                _regions.append(_r)
        # [None] => region unset here (env-driven AWS_REGION_NAME).
        self.aws_regions: list[Optional[str]] = _regions or [None]
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    def _call(self, prompt: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            region = self.aws_regions[attempt % len(self.aws_regions)]
            try:
                return self._call_once(prompt, aws_region_name=region)
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(self._retry_delay_seconds(exc, attempt))
        assert last_exc is not None
        raise last_exc

    def _retry_delay_seconds(self, exc: Exception, attempt: int) -> float:
        """Honor provider Retry-After hints when present; otherwise exponential backoff."""
        response = getattr(exc, "response", None)
        if response is not None:
            retry_after = response.headers.get("retry-after") if hasattr(response, "headers") else None
            if retry_after:
                try:
                    return min(float(retry_after) + 1.0, 300.0)
                except ValueError:
                    pass

        match = re.search(r"Retry after (\d+) seconds", str(exc), re.IGNORECASE)
        if match:
            return min(float(match.group(1)) + 1.0, 300.0)
        return min(float(2 ** attempt), 60.0)

    def _call_once(self, prompt: str, aws_region_name: Optional[str] = None) -> str:
        try:
            import litellm
        except ImportError as e:
            raise ImportError("litellm not installed. pip install litellm") from e

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        kwargs.update(self.litellm_kwargs)
        if aws_region_name is not None:
            kwargs["aws_region_name"] = aws_region_name
        response = litellm.completion(**kwargs)
        if hasattr(response, 'usage') and response.usage:
            self.total_prompt_tokens += getattr(response.usage, 'prompt_tokens', 0) or 0
            self.total_completion_tokens += getattr(response.usage, 'completion_tokens', 0) or 0
        self.total_calls += 1
        return response.choices[0].message.content

    def _judge_one_rubric(
        self, trajectory_summary: str, task_desc: str, rubric: RubricDef
    ) -> tuple[str, dict[str, Any]]:
        if self.mock:
            return rubric.rubric_id, {
                "p_agent_success": 0.5,
                "importance_score": 0.5,
                "evidence": ["mock mode"],
                "rationale": "Mock response",
            }
        prompt = _build_rubric_prompt_v2(trajectory_summary, task_desc, rubric)
        raw = self._call(prompt)
        parsed = _extract_json(raw)
        if parsed is None:
            return rubric.rubric_id, {
                "p_agent_success": 0.5,
                "importance_score": 0.0,
                "evidence": ["JSON parse failed"],
                "rationale": "",
            }
        return rubric.rubric_id, parsed

    def _judge_all_rubrics(
        self, trajectory_summary: str, task_desc: str, rubrics: list[RubricDef]
    ) -> dict[str, dict[str, Any]]:
        """Score ALL rubrics in a SINGLE LLM call (batched).

        Returns {rubric_id: scores} for every requested rubric. Any rubric the
        model omits or returns malformed is filled with a neutral fallback
        (importance 0.0, p_success 0.5) so downstream scoring is unaffected.
        """
        if self.mock:
            return {
                r.rubric_id: {
                    "p_agent_success": 0.5, "importance_score": 0.5,
                    "evidence": ["mock mode"], "rationale": "Mock response",
                }
                for r in rubrics
            }
        prompt = _build_batched_rubric_prompt_v2(trajectory_summary, task_desc, rubrics)
        raw = self._call(prompt)
        parsed = _extract_json(raw)
        out: dict[str, dict[str, Any]] = {}
        for r in rubrics:
            entry = parsed.get(r.rubric_id) if isinstance(parsed, dict) else None
            if isinstance(entry, dict) and "p_agent_success" in entry:
                out[r.rubric_id] = entry
            else:
                out[r.rubric_id] = {
                    "p_agent_success": 0.5, "importance_score": 0.0,
                    "evidence": ["missing from batched response"], "rationale": "",
                }
        return out

    def score_record(
        self,
        record: TrajectoryRecord,
        reward_success: float = 1.0,
        reward_failure: float = -1.0,
    ) -> dict[str, Any]:
        """Score all applicable rubrics and compute composite.

        Returns:
            {
                "rubric_scores": {rubric_id: {p_agent_success, importance_score, evidence, rationale}},
                "composite_score": float,
                "total_importance": float,
            }
        """
        dataset_name = record.task.split
        rubrics = list_rubrics(dataset=dataset_name)

        task_desc = record.task.description[:3000]
        trajectory_summary = _compress_trajectory(
            record,
            max_chars=self.max_chars,
            code_chars=self.code_chars,
            obs_chars=self.obs_chars,
        )

        rubric_scores: dict[str, dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self._judge_one_rubric, trajectory_summary, task_desc, r): r
                for r in rubrics
            }
            for future in as_completed(futures):
                rubric_id, scores = future.result()
                rubric_scores[rubric_id] = scores

        composite = compute_trajectory_score(
            rubric_scores, reward_success=reward_success, reward_failure=reward_failure
        )
        total_importance = sum(
            s.get("importance_score", 0.0) for s in rubric_scores.values()
        )

        return {
            "rubric_scores": rubric_scores,
            "composite_score": composite,
            "total_importance": total_importance,
        }
