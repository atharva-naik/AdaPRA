"""
PRM feedback generation for inter-attempt sequential scaling.

Scores a completed trajectory against rubrics and produces a structured
feedback report that the agent reads on the next attempt.
"""

from __future__ import annotations

from typing import Any, Optional

from dsgym.adapra.schema import TrajectoryRecord


SKIP_RUBRICS = {
    "leakage_risk",
    "no_error_analysis",
    "stops_after_first_valid_submission",
    "no_ablation_or_comparison",
    "metric_mismatch",
    "no_hyperparameter_search_when_warranted",
    # Almost-always-successful on dspredict-easy — the agent passes these ~95%+
    # of the time (mean p_success >= 0.90 across 108 scored v3 attempts), so they
    # only ever surface as "strengths" and add no actionable feedback. Dropping
    # them saves tokens without losing signal. See analysis in the batched-PRM
    # work. Kept fragile_pipeline_or_non_reproducible (0.87) as the
    # borderline case just under the 0.90 cut.
    "no_validation_split_or_cv",
    "misreads_dataset_schema",
    "no_final_sanity_check",
    "missing_feature_handling",
    "poor_preprocessing_for_modality",
    # Exact duplicate of correct_submission_format: identical definition
    # ("Agent fails to produce a submission matching the competition format
    # specification") and +0.99 success-correlation across the v3 PRM data.
    # Keep correct_submission_format (clearer name); drop this one.
    "submission_format_error_risk",
    # ── n10-v2 rubric-set pruning (target ≤20 rubrics) ──
    # Cut as DEAD/low-signal from the inline-PRM run's applicable-firing weakness
    # rates (importance>=0.5, p<0.6): these fire ~0-6% weakness = almost always
    # "strength" = no actionable signal. Kept the 14 discovered rubrics + 5
    # high-signal survivors (inconsistent_cross_validation, no_comparison_to_baseline,
    # weak_baseline_only, incorrect_library_use, correct_submission_format).
    "overfits_local_validation",              # 2% weak
    "submission_scale_mismatch",              # 2% weak (n=236)
    "target_transform_inconsistency",         # 0% weak
    "wrong_model_family_for_task_type",       # 0% weak
    "ignores_domain_specific_representations", # 0% weak
    "fragile_pipeline_or_non_reproducible",   # 6% weak — reproducibility hygiene the agent already meets
    "incompatible_input_output_formatting",   # 12% weak but overlaps error_root_cause_resolution (discovered)
    # MECE: leakage_free_meta_modeling already lists "ensemble weight search" fit
    # on the same rows as a failure condition, so this discovered rubric would
    # double-penalize the same behavior. Fold it into the main leakage rubric.
    "ensemble_weight_search_leakage_free",
    # CUT for n10-v2: no task ships the original source dataset in the mount (0/49),
    # so this rubric pushes the agent toward a fragile network download (kagglehub
    # auth / arbitrary slug). The mined evidence shows that often became wasted
    # motion (KeyErrors merging original+competition features, failed downloads) —
    # the exact meander this PRM redesign aims to REDUCE. Lowest-support emergent
    # rubric (6), niche (PS-only). Revisit if the original datasets get mounted.
    "original_dataset_augmentation",
}


def generate_feedback_report(
    rubric_scores: dict[str, dict],
    final_answer: str = "",
    min_importance: float = 0.5,
    failure_threshold: float = 0.6,
    extra_skip_rubrics: Optional[set[str]] = None,
    mid_run: bool = False,
    rubric_meta: Optional[dict[str, tuple[str, str]]] = None,
    apply_skip: bool = True,
) -> str:
    """Generate a natural language PRM feedback report from rubric scores.

    WEAKNESSES-ONLY: the inline-PRM analysis
    showed ~47% of firings were pure praise and that strength/"keep doing this"
    text does not steer the agent — while a specific, actionable weakness does.
    So we drop strengths entirely and surface only the rubrics the agent is
    failing, each with a concrete `suggested_actions` fix. Anything the agent was
    doing well simply won't be flagged; if it later regresses on it, the next PRM
    firing will catch it.

    Args:
        rubric_scores: Dict mapping rubric_id to score dict with keys:
            p_agent_success, importance_score, rationale, evidence, suggested_actions
        final_answer: The agent's approach summary from the attempt.
        min_importance: Omit rubrics with importance below this.
        failure_threshold: p_success < this = weakness (flagged). Default 0.6 —
            0.5 was too lenient; a rubric at 0.55 is still shaky enough to surface.
        extra_skip_rubrics: Additional rubric IDs to skip (merged with SKIP_RUBRICS).
        rubric_meta: Optional {rubric_id: (display_name, description)} fallback used
            when a rubric is NOT in RUBRIC_REGISTRY (e.g. the launch-selected
            "discovered" set, which is intentionally not auto-registered). Lets the
            unified reviewer render its store rubrics through this SAME renderer.
        apply_skip: When False, SKIP_RUBRICS is NOT applied — for a curated
            launch-selected set (already pruned), exactly like PRMScorer bypassing
            SKIP_RUBRICS when rubric_set is given.

    Returns:
        Formatted markdown feedback report, or "" if there are no weaknesses.
    """
    from dsgym.adapra.rubrics import RUBRIC_REGISTRY

    if not rubric_scores:
        return ""

    skip = SKIP_RUBRICS | (extra_skip_rubrics or set())
    rubric_meta = rubric_meta or {}

    failures = []

    for rubric_id, scores in rubric_scores.items():
        if apply_skip and rubric_id in skip:
            continue

        p_success = scores.get("p_agent_success", 0.5)
        importance = scores.get("importance_score", 0.5)
        rationale = scores.get("rationale", "")
        suggested = (scores.get("suggested_actions") or "").strip()

        if importance < min_importance:
            continue
        if p_success >= failure_threshold:
            continue  # strengths are intentionally NOT reported

        rubric_def = RUBRIC_REGISTRY.get(rubric_id)
        meta_display, meta_desc = rubric_meta.get(rubric_id, ("", ""))
        description = ((rubric_def.display_description if rubric_def else "")
                       or (rubric_def.description if rubric_def else "")
                       or meta_desc)
        display = ((rubric_def.display_name if rubric_def else "")
                   or meta_display
                   or rubric_id.replace("_", " ").title())

        failures.append({
            "rubric": rubric_id,
            "display": display,
            "p_success": p_success,
            "importance": importance,
            "rationale": rationale,
            "description": description,
            "suggested": suggested,
        })

    if not failures:
        return ""

    failures.sort(key=lambda x: x["importance"], reverse=True)

    lines = []
    lines.append("## Expert Feedback (Process Reward Model)")
    lines.append("")
    if mid_run:
        # Trajectory is STILL ONGOING — the agent will keep working after reading
        # this, so frame it as in-progress guidance (not a post-mortem).
        lines.append(
            "A process reward model reviewed your progress SO FAR on this "
            "in-progress attempt. The attempt is NOT finished — this is mid-run "
            "guidance, not a verdict on a completed run. It found the following "
            "issues. In your VERY NEXT step you MUST explicitly acknowledge this "
            "feedback and, for EACH issue, either (a) take the suggested action, or "
            "(b) state in one line why it does not apply yet given where you are — "
            "then continue. Do not silently ignore it. Start with the "
            "highest-importance issue."
        )
    else:
        lines.append(
            "A process reward model evaluated this attempt and found the "
            "following issues. Fix each one on the next attempt, starting with "
            "the suggested action."
        )
    lines.append("")

    weak_hdr = ("### ISSUES TO FIX (address these in your remaining steps):"
                if mid_run else "### ISSUES TO FIX (address these on the next attempt):")

    lines.append(weak_hdr)
    for i, f in enumerate(failures, 1):
        rubric_name = f.get("display") or f["rubric"].replace("_", " ").title()
        lines.append(f"{i}. **{rubric_name}** [Importance: {f['importance']:.2f} | Success: {f['p_success']:.2f}]:")
        if f["description"]:
            lines.append(f"   - Definition: {f['description']}")
        lines.append(f"   - Problem: {f['rationale']}")
        if f["suggested"]:
            lines.append(f"   - Suggested action: {f['suggested']}")
    lines.append("")

    return "\n".join(lines)


class PRMScorer:
    """Scores a trajectory against rubrics using an LLM judge.

    This is the inter-attempt PRM scorer: after each attempt completes,
    it evaluates the trajectory and produces feedback for the next attempt.

    Uses LLMJudge.apply_to_record() which scores all applicable rubrics
    in parallel, then converts the labels to a feedback report.
    """

    def __init__(
        self,
        judge_model: str = "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        rubric_ids: Optional[list[str]] = None,
        exclude_rubric_ids: Optional[list[str]] = None,
        rubric_set: Optional[str] = None,
        min_importance: float = 0.5,
        failure_threshold: float = 0.5,
        api_key: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        aws_region_fallbacks: Optional[list[str]] = None,
        workers: int = 8,
        batched: bool = True,
    ):
        """
        Args:
            judge_model: LLM model for rubric scoring (litellm format).
            rubric_ids: Specific rubric IDs to evaluate. None = all applicable.
            exclude_rubric_ids: Rubric IDs to exclude (overrides rubric_ids if both set).
            min_importance: Omit rubrics with importance below this in feedback.
            failure_threshold: Used to classify labels as weakness vs strength.
            api_key: API key for the judge model.
            aws_region_name: AWS region for Bedrock judge models.
            workers: Number of parallel rubric scoring threads (per-rubric mode only).
            batched: If True (default), score all rubrics in ONE LLM call instead
                of one call per rubric — ~19x fewer calls. Falls back to the
                per-rubric parallel path when False.
        """
        self.judge_model = judge_model
        self.rubric_ids = rubric_ids
        self.exclude_rubric_ids = set(exclude_rubric_ids or [])
        # rubric_set: optional path to a rubric-set JSON (e.g. the discovered
        # unified set). When set, the scored rubrics come entirely from it,
        # bypassing the registry + SKIP_RUBRICS (a launch-selected set is already
        # curated). See dsgym.adapra.rubrics.resolve_rubric_set.
        self.rubric_set = rubric_set
        self.min_importance = min_importance
        self.failure_threshold = failure_threshold
        self.api_key = api_key
        self.aws_region_name = aws_region_name
        self.aws_region_fallbacks = aws_region_fallbacks
        self.workers = workers
        self.batched = batched

        # Cost tracking (accumulated across all calls)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    def generate_feedback(self, record: TrajectoryRecord) -> str:
        """Score trajectory with LLM judge V2 and generate feedback report.

        Args:
            record: The completed trajectory to evaluate.

        Returns:
            Markdown feedback report string.
        """
        from dsgym.adapra.llm_judge_v2 import LLMJudgeV2, _compress_trajectory
        from dsgym.adapra.rubrics import resolve_rubric_set
        from concurrent.futures import ThreadPoolExecutor, as_completed

        judge_kwargs: dict[str, Any] = {
            "model": self.judge_model,
            "workers": self.workers,
        }
        if self.api_key:
            judge_kwargs["api_key"] = self.api_key
        if self.aws_region_name:
            judge_kwargs["aws_region_name"] = self.aws_region_name
        if self.aws_region_fallbacks:
            judge_kwargs["aws_region_fallbacks"] = self.aws_region_fallbacks

        judge = LLMJudgeV2(**judge_kwargs)

        # Get applicable rubrics and filter out skipped ones before scoring.
        # rubric_set (a launch-selected JSON) fixes the set and bypasses the
        # registry + SKIP_RUBRICS; otherwise fall back to the registry minus skips.
        dataset_name = record.task.source or f"dspredict-{record.task.split}"
        all_skip = (SKIP_RUBRICS if not self.rubric_set else set()) | self.exclude_rubric_ids
        rubrics = resolve_rubric_set(
            dataset=dataset_name, rubric_set=self.rubric_set, skip=all_skip,
            include_ids=set(self.rubric_ids) if self.rubric_ids else None)
        assert len(rubrics) >= 10, (
            f"Only {len(rubrics)} rubrics for dataset={dataset_name!r} "
            f"(source={record.task.source!r}, split={record.task.split!r}, "
            f"rubric_set={self.rubric_set!r}). Expected >=10. skip={all_skip}"
        )

        task_desc = record.task.description[:3000]
        trajectory_summary = _compress_trajectory(
            record, max_chars=judge.max_chars, code_chars=judge.code_chars, obs_chars=judge.obs_chars,
        )

        mode = "batched (1 call)" if self.batched else f"{len(rubrics)} calls"
        print(f"      [PRM] Scoring {len(rubrics)} rubrics for task={record.task.task_id} [{mode}]...")
        rubric_scores: dict[str, dict[str, Any]] = {}
        if self.batched:
            rubric_scores = judge._judge_all_rubrics(trajectory_summary, task_desc, rubrics)
            for rubric_id, scores in rubric_scores.items():
                p_s = scores.get("p_agent_success", "?")
                imp = scores.get("importance_score", "?")
                print(f"      [PRM]   {rubric_id}: p_success={p_s}, importance={imp}")
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {
                    pool.submit(judge._judge_one_rubric, trajectory_summary, task_desc, r): r
                    for r in rubrics
                }
                for future in as_completed(futures):
                    rubric_id, scores = future.result()
                    rubric_scores[rubric_id] = scores
                    p_s = scores.get("p_agent_success", "?")
                    imp = scores.get("importance_score", "?")
                    print(f"      [PRM]   {rubric_id}: p_success={p_s}, importance={imp}")

        # Accumulate cost
        self.total_prompt_tokens += judge.total_prompt_tokens
        self.total_completion_tokens += judge.total_completion_tokens
        self.total_calls += judge.total_calls

        return generate_feedback_report(
            rubric_scores=rubric_scores,
            final_answer=record.final_answer or "",
            min_importance=self.min_importance,
            failure_threshold=self.failure_threshold,
            extra_skip_rubrics=self.exclude_rubric_ids,
        )

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def cost_summary(self) -> str:
        return (
            f"PRM cost: {self.total_calls} calls, "
            f"{self.total_prompt_tokens:,} prompt + {self.total_completion_tokens:,} completion "
            f"= {self.total_tokens:,} total tokens"
        )
