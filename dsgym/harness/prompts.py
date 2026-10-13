"""
Prompt templates for the OpenHands-based DSGym harness.

Extends the OpenHands system prompt with data science task-specific
instructions, attempt history, and time budget awareness.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# System prompt suffix (appended to OpenHands default system prompt)
# ---------------------------------------------------------------------------

DS_SYSTEM_PROMPT_SUFFIX = """\

<DATA_SCIENCE_TASK>
You are solving a data science / machine learning competition task.

## Environment
- Training data is at `/data/train.csv` (or similar files in `/data/`)
- Test data is at `/data/test.csv`
- You must write your predictions to `/submission/submission.csv`
- Your workspace is at `/app/workspace/` — use it for intermediate files, models, etc.
- The Jupyter kernel persists state across python tool calls (variables, imports survive)

## Workflow
1. **Explore**: Read the data, understand the target variable and features
2. **Baseline**: Train a simple model quickly to establish a working pipeline
3. **Iterate**: Improve the model (feature engineering, hyperparameter tuning, ensembles)
4. **Submit**: Write predictions to `/submission/submission.csv` in the required format

## Important Rules
- ALWAYS ensure your submission.csv has the correct format (check sample_submission.csv if available)
- Validate your predictions before writing to submission.csv (no NaN, correct number of rows, correct columns)
- Use local cross-validation to estimate performance before submitting
- When done, use the finish tool with a summary of your approach and validation score
</DATA_SCIENCE_TASK>
"""


# ---------------------------------------------------------------------------
# Attempt preamble (injected as first user message for attempts > 0)
# ---------------------------------------------------------------------------

ATTEMPT_PREAMBLE_TEMPLATE = """\
This is attempt {attempt_num} of {total_attempts} for this task.

{previous_attempts_section}

## Instructions for this attempt
- Review previous attempt summaries at `/app/workspace/ATTEMPT_*.md` for details
- Your goal is to IMPROVE on the best validation score achieved so far
- The workspace retains all files from previous attempts (models, features, etc.)
- You can build on previous work or try a completely different approach
- Focus on what didn't work before and try something new

Best validation score so far: {best_score}
"""

PREVIOUS_ATTEMPTS_SECTION = """\
## Previous Attempts Summary
{summaries}
"""


# ---------------------------------------------------------------------------
# Attempt summary template (written to ATTEMPT_i.md after each attempt)
# ---------------------------------------------------------------------------

ATTEMPT_SUMMARY_TEMPLATE = """\
# Attempt {attempt_num}

## Approach
{approach_summary}

## Validation Score
{val_score}

## Key Observations
{observations}

## Submission
- Valid: {submission_valid}
- Path: /submission/submission.csv

## What to Try Next
{next_suggestions}
"""


# ---------------------------------------------------------------------------
# Time budget prompt (injected into system prompt when time constraints exist)
# ---------------------------------------------------------------------------

TIME_BUDGET_PROMPT = """\

<TIME_BUDGET>
You have approximately {minutes_remaining} minutes remaining for this attempt.
Plan your work accordingly — prioritize completing a valid submission over perfection.
If you're running low on time, submit your current best model rather than starting something new.
</TIME_BUDGET>
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_system_prompt_kwargs(
    task_description: str,
    time_budget_minutes: Optional[int] = None,
) -> dict[str, str]:
    """Build system_prompt_kwargs for the Agent.

    The OpenHands Agent uses Jinja2 templates. We inject our DS-specific
    content via the system_prompt_kwargs mechanism.
    """
    suffix = DS_SYSTEM_PROMPT_SUFFIX

    if time_budget_minutes:
        suffix += TIME_BUDGET_PROMPT.format(minutes_remaining=time_budget_minutes)

    return {
        "ds_task_suffix": suffix,
        "task_description": task_description,
    }


def build_attempt_preamble(
    attempt_index: int,
    total_attempts: int,
    previous_summaries: list[dict[str, str]],
    best_score: Optional[float] = None,
) -> str:
    """Build the user message preamble for a multi-attempt run.

    Args:
        attempt_index: 0-indexed attempt number.
        total_attempts: Total number of attempts.
        previous_summaries: List of dicts with 'attempt_num' and 'summary' keys.
        best_score: Best validation score achieved so far.

    Returns:
        Formatted preamble string.
    """
    if attempt_index == 0:
        return ""

    summaries_text = ""
    if previous_summaries:
        lines = []
        for s in previous_summaries:
            lines.append(f"- Attempt {s['attempt_num']}: {s['summary']}")
        summaries_text = PREVIOUS_ATTEMPTS_SECTION.format(summaries="\n".join(lines))

    score_str = f"{best_score:.4f}" if best_score is not None else "None (no valid submission yet)"

    return ATTEMPT_PREAMBLE_TEMPLATE.format(
        attempt_num=attempt_index + 1,
        total_attempts=total_attempts,
        previous_attempts_section=summaries_text,
        best_score=score_str,
    )


def build_attempt_summary(
    attempt_index: int,
    approach_summary: str = "See trajectory for details.",
    val_score: Optional[float] = None,
    observations: str = "",
    submission_valid: bool = False,
    next_suggestions: str = "",
) -> str:
    """Build the ATTEMPT_i.md content to write to the workspace.

    Args:
        attempt_index: 0-indexed attempt number.
        approach_summary: Description of the approach taken.
        val_score: Validation score achieved (None if not computed).
        observations: Key observations from this attempt.
        submission_valid: Whether a valid submission was produced.
        next_suggestions: Suggestions for the next attempt.

    Returns:
        Formatted markdown string.
    """
    return ATTEMPT_SUMMARY_TEMPLATE.format(
        attempt_num=attempt_index + 1,
        approach_summary=approach_summary,
        val_score=f"{val_score:.4f}" if val_score is not None else "Not computed",
        observations=observations or "None recorded.",
        submission_valid="Yes" if submission_valid else "No",
        next_suggestions=next_suggestions or "Try a different modeling approach or feature engineering strategy.",
    )
