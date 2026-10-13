"""
Agentic PRM: score an in-progress (or finished) trajectory against the rubric
set with a SINGLE LLM call that emits one JSON for ALL rubrics.

Design:
- Input = the agent's own conversation replayed as context (turn-annotated
  transcript, tool outputs truncated), plus the task description. This is the
  "whole agent context" the PRM judges — it can be truncated to the first K
  steps to score the trajectory AS IT STOOD at step K (for mid-run PRM).
- A COMMON system prompt carries the scoring-dimension instructions
  (p_agent_success + importance_score) and the JSON schema, so the individual
  rubric definitions (still in 4-label language) don't each have to.
- Output = {rubric_id: {p_agent_success, importance_score, evidence, rationale}}
  for every scored rubric; missing/malformed rubrics get a neutral fallback.
- Cost = exact input/output tokens for THAT call (a fresh judge per call), so a
  per-call JSON can record it and an attempt's total is the sum across calls.

Call-strategy hooks (every-N-steps, every-new-CSV) live in the caller; this
module just scores a given cutoff and returns a self-contained record.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from dsgym.adapra.llm_judge_v2 import (
    LLMJudgeV2, _extract_json, compute_trajectory_score,
)
from dsgym.adapra.rubrics import RubricDef, list_rubrics


# The common scoring-dimension + schema instruction. Rubric bodies stay in
# 4-label (agent_failure/success/not_applicable/insufficient_evidence) language;
# this prompt tells the judge how to MAP that to the two continuous scores.
PRM_SYSTEM_PROMPT = """You are an expert data-science process reward model (PRM).

Below is the transcript of an agent solving a data-science competition — every
action it took and the (truncated) tool output each produced, turn-annotated as
[Turn N]. You will judge this trajectory against a set of rubrics.

Each rubric describes a good practice, written with 'agent_failure' /
'agent_success' / 'not_applicable' / 'insufficient_evidence' language. For EACH
rubric, translate that into TWO continuous scores:

1. importance_score (0.0-1.0): how RELEVANT this rubric is to the trajectory AS
   IT STANDS AT THIS POINT. The transcript may be a PARTIAL trajectory (the agent
   is still working); score relevance to what has happened SO FAR, not to a
   hypothetical finished run.
   - 0.0  = not applicable YET or ever: either the task doesn't involve this
            aspect, OR the agent has not yet reached the stage this rubric
            measures (e.g. it is still doing EDA and has not built any model, so
            'best_model_not_selected' / 'correct_submission_format' cannot apply
            yet). When the relevant behavior has not occurred, importance is LOW
            — do NOT penalize the agent for work it has not gotten to.
   - 0.3-0.5 = marginally relevant, or the behavior is just starting to come up
   - 0.6-0.8 = clearly relevant (the agent has actively engaged this behavior)
   - 0.9-1.0 = central to what the agent has done so far

2. p_agent_success (0.0-1.0): probability the agent SUCCEEDED at the good
   practice (i.e. would be labeled 'agent_success'), based on evidence so far.
   - 0.0 = certain failure (unambiguous violation)
   - 0.1-0.3 = likely failure
   - 0.4-0.6 = uncertain / mixed / insufficient_evidence
   - 0.7-0.9 = likely success
   - 1.0 = certain success

   If the rubric is not applicable (or not yet reached), set importance_score
   ~0.0 and p_agent_success = 0.5 (maximum uncertainty).

Judge ONLY what is visible in the transcript so far — do not assume or reward/
penalize work the agent has not yet done. A rubric whose behavior lies in a
later stage the agent has not reached should get LOW importance, not a low
success score. Ground every judgment in the trajectory.

3. suggested_actions (string): if — and ONLY if — the agent is likely FAILING
   this rubric (low p_agent_success), give ONE concrete, imperative next action
   the agent should take to fix it, grounded in the specific evidence (name the
   exact file / model / step, e.g. "restore preds/cat_v4.csv to
   /submission/submission.csv — it has the better CV RMSE 12.216 vs the current
   12.395"). Do NOT give generic advice ("keep tuning", "try more models"). If
   the agent is succeeding at the rubric, set this to "".

Respond with ONLY valid JSON: a single object mapping EACH rubric_id to its
scores, including ALL rubric_ids exactly as written, none omitted:
{
  "<rubric_id>": {
    "p_agent_success": <float 0.0-1.0>,
    "importance_score": <float 0.0-1.0>,
    "evidence": ["[T<turn>:<code|obs>] <brief verbatim quote>"],
    "rationale": "one sentence explaining the two scores",
    "suggested_actions": "one concrete imperative fix if failing, else \\"\\""
  },
  ... one entry per rubric_id ...
}
Each evidence entry MUST start with a [T<turn>:...] marker matching the
transcript. If a rubric is not applicable its evidence may be empty."""


# ALTERNATE system prompt for the p_success instruction style. Used when the
# rubric bodies are phrased directly on the 0-1 p_agent_success scale (see
# rubrics.prm_instruction(style="p_success")), so there is no 4-label->score
# translation step. Selected via _build_prm_prompt(..., style="p_success");
# additive + reversible — the default PRM_SYSTEM_PROMPT path is unchanged.
PRM_SYSTEM_PROMPT_PSUCCESS = """You are an expert data-science process reward model (PRM).

Below is the transcript of an agent solving a data-science competition — every
action it took and the (truncated) tool output each produced, turn-annotated as
[Turn N]. You will judge this trajectory against a set of rubrics.

Each rubric describes a good practice and tells you directly what makes
p_agent_success high vs low. For EACH rubric, produce THREE outputs:

1. importance_score (0.0-1.0): how RELEVANT this rubric is to the trajectory AS
   IT STANDS AT THIS POINT. The transcript may be a PARTIAL trajectory (the agent
   is still working); score relevance to what has happened SO FAR, not to a
   hypothetical finished run.
   - 0.0  = not applicable YET or ever: either the task doesn't involve this
            aspect, OR the agent has not yet reached the stage this rubric
            measures (e.g. still doing EDA, no model built yet, so a
            submission-selection rubric cannot apply). When the relevant behavior
            has not occurred, importance is LOW — do NOT penalize the agent for
            work it has not gotten to.
   - 0.3-0.5 = marginally relevant, or the behavior is just starting to come up
   - 0.6-0.8 = clearly relevant (the agent has actively engaged this behavior)
   - 0.9-1.0 = central to what the agent has done so far

2. p_agent_success (0.0-1.0): how well the agent satisfies this rubric's good
   practice, based on evidence so far. Each rubric states which behaviors push
   this toward 1.0 and which push it toward 0.0; score by those criteria.
   - 1.0 = clearly does the good thing the rubric describes
   - 0.7-0.9 = likely does it
   - 0.4-0.6 = mixed / partial / too little evidence to tell
   - 0.1-0.3 = likely does the bad thing
   - 0.0 = clearly does the bad thing
   If the rubric is not applicable (or not yet reached), set importance_score
   ~0.0 and p_agent_success = 0.5 (maximum uncertainty).

Judge ONLY what is visible in the transcript so far — do not assume or reward/
penalize work the agent has not yet done. A rubric whose behavior lies in a
later stage the agent has not reached should get LOW importance, not a low
success score. Ground every judgment in the trajectory.

3. suggested_actions (string): if — and ONLY if — p_agent_success is low, give
   ONE concrete, imperative next action the agent should take to fix it, grounded
   in the specific evidence (name the exact file / model / step, e.g. "restore
   preds/cat_v4.csv to /submission/submission.csv — it has the better CV RMSE
   12.216 vs the current 12.395"). Do NOT give generic advice ("keep tuning",
   "try more models"). If p_agent_success is high, set this to "".

Respond with ONLY valid JSON: a single object mapping EACH rubric_id to its
scores, including ALL rubric_ids exactly as written, none omitted:
{
  "<rubric_id>": {
    "p_agent_success": <float 0.0-1.0>,
    "importance_score": <float 0.0-1.0>,
    "evidence": ["[T<turn>:<code|obs>] <brief verbatim quote>"],
    "rationale": "one sentence explaining the two scores",
    "suggested_actions": "one concrete imperative fix if p_agent_success is low, else \\"\\""
  },
  ... one entry per rubric_id ...
}
Each evidence entry MUST start with a [T<turn>:...] marker matching the
transcript. If a rubric is not applicable its evidence may be empty."""


def _txt(c: Any) -> str:
    if isinstance(c, list):
        return "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
    return c or ""


def _action_text(act: dict) -> str:
    """Full rendering of one action's arguments (truncated_transcript detail="full")."""
    kind = act.get("kind", "")
    if kind == "ThinkAction":
        return _txt(act.get("thought"))
    if kind == "FinishAction":
        return _txt(act.get("message"))
    if kind == "FileEditorAction":
        parts = [" ".join(str(x) for x in (act.get("command"), act.get("path")) if x)]
        for key in ("file_text", "old_str", "new_str"):
            if act.get(key) is not None:
                parts.append(f"<{key}>\n{act[key]}\n</{key}>")
        for key in ("insert_line", "view_range"):
            if act.get(key) is not None:
                parts.append(f"{key}: {act[key]}")
        return "\n".join(parts)
    return act.get("command") or ""


def truncated_transcript(events: list[dict], up_to_step: Optional[int],
                         max_chars: int = 200_000,
                         from_step: Optional[int] = None,
                         detail: str = "command") -> tuple[str, int]:
    """Render the FULL trajectory (user + assistant messages, actions, and tool
    outputs) with [Turn N] markers, stopping AFTER `up_to_step` action steps
    (None = whole trajectory). Returns (transcript, n_action_steps_included).

    `from_step` (optional): only emit turns with index > from_step — the turn
    counter still runs from 1 (absolute [Turn N] numbering is preserved), but
    earlier turns are dropped from the rendered text. Used to render the PUSH
    delta (the agent's NEW turns since the supervisor's last invocation) without
    re-sending the whole history each wake.

    This is the trajectory AS THE AGENT SAW IT:
    - The initial task message and any user/assistant text are included (the
      task description is [Turn 0] USER — no separate task-desc section needed).
    - Per tool output we apply OpenHands' OWN truncation (`maybe_truncate` at
      DEFAULT_TEXT_CONTENT_LIMIT=50k, head+tail), so the PRM sees exactly what
      the agent saw — not a more aggressive re-truncation that would drop CV
      tables / model comparisons the rubrics need.
    - One [Turn N] per ActionEvent (same counter the manifest / selector use),
      so 'Turn 37' in evidence maps to a real action. up_to_step lets the PRM
      judge the trajectory as it stood at that step (mid-run scoring).

    `detail`: "command" (default) renders each action as its command field only,
    so ThinkAction text, FileEditorAction path/contents and the agent's reasoning
    are not shown. "full" also renders the agent's reasoning (THOUGHT line), the
    ThinkAction text, the FileEditorAction path and file_text/old_str/new_str, and
    the FinishAction message.
    """
    from openhands.sdk.utils import DEFAULT_TEXT_CONTENT_LIMIT, maybe_truncate
    lines: list[str] = []
    t = 0
    def _keep(turn: int) -> bool:
        return from_step is None or turn > from_step
    for ev in events:
        kind = ev.get("kind") or ev.get("type", "")
        if kind == "ActionEvent":
            if up_to_step is not None and t >= up_to_step:
                break
            t += 1
            if not _keep(t):
                continue
            act = ev.get("action") or {}
            if detail == "full":
                thought = _txt(ev.get("thought")).strip()
                if thought:
                    lines.append(f"[Turn {t}] THOUGHT: "
                                 f"{maybe_truncate(thought, DEFAULT_TEXT_CONTENT_LIMIT)}")
                cmd = _action_text(act)
            else:
                cmd = act.get("command") or _txt(ev.get("thought")) or ""
            lines.append(f"[Turn {t}] ACTION {act.get('kind','')}: "
                         f"{maybe_truncate(cmd, DEFAULT_TEXT_CONTENT_LIMIT)}")
        elif kind == "ObservationEvent":
            if not _keep(t):
                continue
            obs = ev.get("observation")
            ot = _txt(obs.get("content") if isinstance(obs, dict) else obs)
            if ot.strip():
                lines.append(f"[Turn {t}] OUTPUT: "
                             f"{maybe_truncate(ot, DEFAULT_TEXT_CONTENT_LIMIT)}")
        elif kind == "MessageEvent":
            if not _keep(t):
                continue
            src = ev.get("source", "")
            mt = _txt((ev.get("llm_message") or {}).get("content"))
            if mt.strip() and src != "environment":
                # [Turn 0] USER = the task message (before any action fires).
                lines.append(f"[Turn {t}] {src.upper()}: "
                             f"{maybe_truncate(mt, DEFAULT_TEXT_CONTENT_LIMIT)}")
    text = "\n".join(lines)
    if len(text) > max_chars:  # safety cap; keep the tail (later turns matter)
        text = "…[earlier turns truncated]…\n" + text[-max_chars:]
    return text, t


def _build_prm_prompt(transcript: str, rubrics: list[RubricDef],
                      style: str = "labels") -> str:
    """Assemble the PRM prompt. The task description is already [Turn 0] USER in
    the transcript (full-trajectory replay), so there is no separate task
    section.

    style selects both the per-rubric instruction field and the matching system
    prompt: "labels" (default) = 4-label instructions + translate-to-scores
    framing; "p_success" = 0-1-framed instructions + the direct-scoring framing.
    """
    from dsgym.adapra.rubrics import prm_instruction
    system = PRM_SYSTEM_PROMPT_PSUCCESS if style == "p_success" else PRM_SYSTEM_PROMPT
    rubric_block = "\n\n".join(
        f"### RUBRIC: {r.rubric_id}\n{prm_instruction(r, style=style)}"
        + _grounding_hint(r.grounding)
        for r in rubrics)
    grounding_note = ""
    if any(getattr(r, "grounding", "") for r in rubrics):
        grounding_note = (
            "\n\nSome rubrics carry a suggested evidence source "
            "(artifact = the persistent workspace files; trajectory = how the "
            "agent worked over turns). This is a RECOMMENDATION about where the "
            "signal usually lives, NOT a restriction: use whichever of the "
            "trajectory or the workspace you find decisive for each rubric."
        )
    return (
        system
        + grounding_note
        + "\n\n===== TRAJECTORY (turn-annotated, as the agent saw it) =====\n"
        + transcript
        + f"\n\n===== RUBRICS ({len(rubrics)}) =====\n" + rubric_block
        + f"\n\nScore ALL {len(rubrics)} rubrics. Respond with ONLY the JSON object."
    )


_GROUNDING_HINTS = {
    "artifact": ("Suggested evidence source (recommendation only): the persistent "
                 "WORKSPACE artifacts (submission files, code, CV registries, "
                 "cached predictions) — but check the trajectory too if decisive."),
    "trajectory": ("Suggested evidence source (recommendation only): the PROCESS "
                   "over turns (how the agent worked) — but check the workspace "
                   "too if decisive."),
}


def _grounding_hint(grounding: str) -> str:
    """Advisory evidence-source line appended to a rubric in the PRM prompt. Empty
    for rubrics with no grounding tag (the hand-authored set), so their rendering
    is unchanged."""
    hint = _GROUNDING_HINTS.get((grounding or "").strip())
    return f"\n{hint}" if hint else ""


def score_trajectory_at_step(
    *, llm_kwargs: dict, events: list[dict], task_desc: str = "",
    rubrics: Optional[list[RubricDef]] = None, up_to_step: Optional[int] = None,
    trigger: str = "manual", attempt_index: int = 0,
    out_path: Optional[Path] = None, instruction_style: str = "labels",
    exclude_rubric_ids: Optional[set[str]] = None,
    rubric_set: Optional[str] = None, dataset: str = "dspredict-easy",
) -> dict:
    """Run ONE PRM call over the trajectory truncated to `up_to_step`.

    The full trajectory (user + assistant messages, actions, outputs) is
    replayed as context. `task_desc`, if given, is injected as the [Turn 0] USER
    message — needed for the OFFLINE path where reconstructed events omit the
    initial task message; in the LIVE path the task message is already the first
    event, so pass task_desc="".

    Returns a self-contained record (rubric_scores, composite, cost, metadata).
    If `out_path` is given, writes the record there as prm_call JSON. A FRESH
    judge is used per call so the cost fields are exactly this call's usage.
    """
    if rubrics is None:
        from dsgym.harness.prm_feedback import SKIP_RUBRICS
        from dsgym.adapra.rubrics import resolve_rubric_set
        # exclude_rubric_ids drops rubrics from SCORING entirely (not just
        # injection) — used by the stacked-v2 ownership split so the static PRM
        # spends no judge tokens on the axes the dynamic reviewer owns.
        # rubric_set (a JSON path) fixes the set explicitly from a launch command;
        # SKIP_RUBRICS only applies to the registry fallback (a launch-selected
        # JSON set is already curated, so we don't re-prune it).
        skip = (SKIP_RUBRICS if not rubric_set else set()) | (exclude_rubric_ids or set())
        rubrics = resolve_rubric_set(dataset=dataset, rubric_set=rubric_set, skip=skip)

    # Prepend the task message as [Turn 0] USER when it isn't already in events.
    if task_desc and not any(
        (e.get("kind") or e.get("type")) == "MessageEvent"
        and e.get("source") == "user" for e in events
    ):
        events = [{"kind": "MessageEvent", "source": "user",
                   "llm_message": {"content": task_desc}}] + events

    transcript, n_steps = truncated_transcript(events, up_to_step)
    prompt = _build_prm_prompt(transcript, rubrics, style=instruction_style)

    judge = LLMJudgeV2(**{"workers": 1, **llm_kwargs})
    raw = judge._call(prompt)
    parsed = _extract_json(raw)

    rubric_scores: dict[str, dict[str, Any]] = {}
    for r in rubrics:
        entry = parsed.get(r.rubric_id) if isinstance(parsed, dict) else None
        if isinstance(entry, dict) and "p_agent_success" in entry:
            rubric_scores[r.rubric_id] = entry
        else:
            rubric_scores[r.rubric_id] = {
                "p_agent_success": 0.5, "importance_score": 0.0,
                "evidence": ["missing from PRM response"], "rationale": ""}

    composite = compute_trajectory_score(rubric_scores)
    record = {
        "attempt_index": attempt_index,
        "step": n_steps,
        "up_to_step": up_to_step,
        "trigger": trigger,
        "n_rubrics": len(rubrics),
        "rubric_scores": rubric_scores,
        "composite_score": composite,
        "cost": {
            "input_tokens": judge.total_prompt_tokens,
            "output_tokens": judge.total_completion_tokens,
            "total_tokens": judge.total_prompt_tokens + judge.total_completion_tokens,
            "model": judge.model,
        },
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2))
    return record
