"""
Agentic final-submission SELECTION sub-agent (V4, flag-gated).

After an attempt's main loop, spawn a SEPARATE OpenHands sub-agent — same model,
seeded with the just-finished trajectory, but with a RESTRICTED action space
(only a `select_submission` tool) and a different system prompt — whose sole job
is to pick the single best of the submission CSVs produced during this attempt.
The chosen CSV is deterministically copied to the canonical submission path.

Design:
- Runs OUTSIDE the attempt time/turn budget (fresh conversation).
- Candidate list = CSVs the manifest recorded as produced THIS attempt (already
  deduped by md5 by the snapshotter), filtered to n_lines == expected, numbered
  1..N and tagged with the producing turn + a model/CV hint.
- Context = the main run's events replayed into the sub-agent, each ANNOTATED
  with the same [Turn N] counter the manifest uses, so "Turn 37" in the list
  matches a "[Turn 37]" marker in the transcript the agent reads.
- Action = select_submission(choice: 1..N, reason). Harness copies candidate
  #choice -> submission_path. Fallback: keep last-written on failure.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from pydantic import Field

from openhands.sdk import Agent, LocalConversation, LocalWorkspace, register_tool
from openhands.sdk.llm import LLM
from openhands.sdk.llm.message import Message, TextContent
from openhands.sdk.tool import (
    Action, Observation, ToolAnnotations, ToolDefinition, ToolExecutor,
)


# Filled per-invocation (single-threaded selection); the executor reads it.
_SELECTION: dict[str, Any] = {}


class SelectSubmissionAction(Action):
    choice: int = Field(description="The list number (1..N) of the submission to submit.")
    reason: str = Field(default="", description="One-line justification for the choice.")


class SelectSubmissionObservation(Observation):
    @property
    def visualize(self):
        from rich.text import Text
        return Text(self.to_text() if hasattr(self, "to_text") else "")


class SelectSubmissionExecutor(ToolExecutor):
    def __call__(self, action: SelectSubmissionAction, conversation=None):
        cands = _SELECTION.get("candidates", [])
        dst = Path(_SELECTION["submission_path"])
        n = len(cands)
        idx = action.choice - 1  # 1-based -> 0-based
        if not (0 <= idx < n):
            return SelectSubmissionObservation.from_text(
                text=f"Invalid choice {action.choice}; must be 1..{n}. Try again.")
        chosen = cands[idx]
        src = Path(chosen["csv_path"])
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        except OSError as e:
            return SelectSubmissionObservation.from_text(text=f"Copy failed: {e}")
        _SELECTION["result"] = {
            "choice": action.choice, "reason": action.reason,
            "turn": chosen.get("turn"), "md5": chosen.get("md5"),
            "script": chosen.get("script"),
        }
        return SelectSubmissionObservation.from_text(
            text=f"Selected submission #{action.choice} (turn {chosen.get('turn')}, "
                 f"{chosen.get('script')}) and wrote it to {dst}. Done — call finish.")


class SelectSubmissionTool(ToolDefinition[SelectSubmissionAction, SelectSubmissionObservation]):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["SelectSubmissionTool"]:
        return [cls(
            action_type=SelectSubmissionAction,
            observation_type=SelectSubmissionObservation,
            description=(
                "Select the single best submission CSV to submit, by its list "
                "number (1..N) from the candidate list you were shown. This "
                "copies that exact CSV to the competition's submission path. "
                "Call this exactly once, then finish."
            ),
            executor=SelectSubmissionExecutor(),
            annotations=ToolAnnotations(title="select_submission", readOnlyHint=False,
                                        destructiveHint=False, idempotentHint=True,
                                        openWorldHint=False),
        )]


SELECTOR_SYSTEM_PROMPT = """You are the final-submission SELECTOR for a data-science competition attempt.

The transcript above is the COMPLETE record of an attempt you just finished:
every script you wrote and ran, and the cross-validation / evaluation output
each run produced. Each event is tagged with a [Turn N] marker.

You maintained a "git-history" of every submission CSV you produced. Below is
the list of the DISTINCT submissions produced during THIS attempt whose row
count matches the required format — each numbered and tagged with the turn that
produced it. To recall the model and its cross-validation score for a candidate,
FIND that turn ([Turn N]) in the transcript above and read the CV/eval output
the run produced there. (No pre-computed score hints are given — derive them
yourself from the transcript so you aren't misled by a mis-attributed number.)

YOUR ONLY JOB: pick the ONE submission most likely to score best on the hidden
test set, and call `select_submission(choice=<list number>, reason=...)`.

Guidance:
- Cross-reference each candidate's [Turn N] against what you did at that turn in
  the transcript to recall the model and its cross-validation score. Quote the
  actual CV number you found for each candidate in your reason.
- Compare the AGGREGATE cross-validation score (mean across folds), NOT a single
  lucky fold. A per-fold array like [0.70 0.66 0.76 ...] is noisy; use the mean.
- Small CV differences are usually within noise. Compute the gap between the top
  candidates' mean CV and compare it to the fold-to-fold spread (std across
  folds): if the gap is SMALLER than the std, treat them as TIED and prefer the
  SIMPLER / more ROBUST model (fewer moving parts, less hyperparameter-tuned,
  less likely to overfit), rather than chasing a marginally-higher number.
  Do the arithmetic carefully — do not claim a gap exceeds the std unless it
  numerically does.
- Watch the metric DIRECTION: for error metrics (RMSE, MAE, LogLoss, RMSLE)
  LOWER is better; for AUC/accuracy/F1 HIGHER is better. Pick accordingly.
- Beware overfitting to the local split, leakage, or a suspiciously-good fold.
- `choice` is the LIST NUMBER (1..N), NOT a turn number. Double-check that the
  number you pass matches the candidate you intend.
- You cannot run code or write files. Select once, then finish.
"""


def _annotated_transcript(events: list[dict], max_chars: int = 60_000) -> str:
    """Render the main run's events with [Turn N] markers using the same
    per-action-step counter as the CSV manifest (one bump per ActionEvent)."""
    def _txt(c):
        if isinstance(c, list):
            return "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
        return c or ""
    lines = []
    t = 0
    for ev in events:
        kind = ev.get("kind") or ev.get("type", "")
        if kind == "ActionEvent":
            t += 1
            act = ev.get("action") or {}
            cmd = act.get("command") or _txt(ev.get("thought")) or ""
            lines.append(f"[Turn {t}] ACTION {act.get('kind','')}: {cmd[:600]}")
        elif kind == "ObservationEvent":
            obs = ev.get("observation")
            ot = _txt(obs.get("content") if isinstance(obs, dict) else obs)
            if ot.strip():
                lines.append(f"[Turn {t}] OUTPUT: {ot[:800]}")
        elif kind == "MessageEvent":
            src = ev.get("source", "")
            mt = _txt((ev.get("llm_message") or {}).get("content"))
            if mt.strip() and src != "environment":
                lines.append(f"[Turn {t}] {src.upper()}: {mt[:400]}")
    text = "\n".join(lines)
    if len(text) > max_chars:  # keep the tail (later turns = the good models)
        text = "…[earlier turns truncated]…\n" + text[-max_chars:]
    return text


def _model_hint(events: list[dict], turn: int, window: int = 8) -> str:
    """Best-effort model/CV hint for a candidate.

    Prioritizes SUMMARY CV lines (e.g. "BEST CV SCORE: 0.7190", "mean CV RMSE
    0.123", "Final CV: ...") over raw per-fold arrays like
    "CV F1 scores: [0.70 0.66 0.76 ...]", which are noisy and mislead the
    selector into chasing a marginally-higher fold. Fold-array lines are
    dropped when a summary/aggregate line is available for that turn window.
    """
    import re as _re
    def _txt(c):
        if isinstance(c, list):
            return "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
        return c or ""
    obs_by_turn: dict[int, str] = {}
    t = 0
    for ev in events:
        kind = ev.get("kind") or ev.get("type", "")
        if kind == "ActionEvent":
            t += 1
        elif kind == "ObservationEvent":
            obs = ev.get("observation")
            obs_by_turn[t] = obs_by_turn.get(t, "") + _txt(
                obs.get("content") if isinstance(obs, dict) else obs)

    KEYS = ("CV", "RMSE", "RMSLE", "MAE", "R2", "Best", "score", "Score",
            "accuracy", "AUC", "LogLoss", "mean", "Mean", "Final", "FINAL")
    # A "summary" line reports a single aggregate number, not a fold array.
    def is_fold_array(ln):
        return bool(_re.search(r"\[\s*[0-9]*\.?[0-9]+[\s,]", ln))  # "[0.70 0.66..."
    def is_summary(ln):
        low = ln.lower()
        return (any(k in low for k in ("best cv", "mean cv", "final cv", "overall",
                                        "average cv", "cv score", "mean rmse",
                                        "best score", "final score"))
                and not is_fold_array(ln))
    summary, other = [], []
    for tt in range(max(1, turn - window), turn + 1):
        for ln in obs_by_turn.get(tt, "").splitlines():
            ln = ln.strip()
            if not any(k in ln for k in KEYS):
                continue
            if is_summary(ln):
                summary.append(ln)
            elif not is_fold_array(ln):  # keep non-fold CV mentions
                other.append(ln)
    # Prefer summary lines; fall back to non-fold CV mentions; never raw folds.
    cv = summary if summary else other
    seen = set(); uniq = []
    for ln in reversed(cv):
        if ln and ln not in seen:
            seen.add(ln); uniq.append(ln)
    return " | ".join(reversed(uniq[:6]))[:400] or "(no CV line found near this turn)"


def _load_candidates(csv_history_dir: Path, expected_lines: int | None,
                     events: list[dict],
                     expected_columns: list[str] | None = None) -> list[dict]:
    """Load distinct submission candidates from the CSV-history manifest.

    Normal (prediction) tasks: filter to n_lines == expected_lines (right row
    count). DATA-CENTRIC tasks (e.g. s3e21, where the submission is a TRAINING
    dataset that "may have fewer rows than sample_submission"): row count is NOT
    fixed, so pass expected_columns instead and filter by COLUMN-SCHEMA match —
    the candidate must contain exactly the sample's columns (features + target).
    """
    manifest = json.loads((csv_history_dir / "manifest.json").read_text())
    subs = [e for e in manifest
            if "submission" in Path(e.get("source_path", "")).name.lower()
            and Path(e.get("source_path", "")).name.lower().endswith(".csv")]
    cands = []
    for e in sorted(subs, key=lambda x: x.get("turn") or 0):
        csv_path = csv_history_dir / e["archived_as"]
        if expected_columns is not None:
            # data-centric: match COLUMNS, ignore row count
            try:
                cols = csv_path.open().readline().strip().split(",")
            except OSError:
                continue
            if cols != list(expected_columns):
                continue  # wrong schema — skip
        elif expected_lines is not None and e.get("n_lines") not in (None, expected_lines):
            continue  # wrong row count — skip
        cands.append({
            "turn": e.get("turn"),
            "script": Path(e.get("source_path", "")).name,
            "md5": e.get("md5"),
            "n_lines": e.get("n_lines"),
            "csv_path": str(csv_history_dir / e["archived_as"]),
            # No pre-extracted CV "hint": the heuristic mis-attributed CV lines
            # across nearby turns (fed the model wrong numbers). The model has
            # the full turn-annotated transcript and each candidate's [Turn N],
            # so it maps candidate -> CV itself. `events` kept for signature
            # compatibility; intentionally unused here now.
        })
    return cands


def select_submission_subagent(*, llm_kwargs: dict, events: list[dict],
                               csv_history_dir: Path, submission_path: str,
                               expected_lines: int | None, workspace_dir: str,
                               max_iterations: int = 6,
                               expected_columns: list[str] | None = None) -> dict:
    """Run the selection sub-agent. Returns the recorded decision (or a status
    dict on failure). Copies the chosen CSV to submission_path as a side effect.

    For DATA-CENTRIC tasks pass expected_columns (the sample_submission header):
    candidates are then filtered by COLUMN-SCHEMA match rather than exact row
    count (a data-centric training-dataset submission may have fewer rows).
    """
    cands = _load_candidates(csv_history_dir, expected_lines, events,
                             expected_columns=expected_columns)
    if not cands:
        return {"status": "no valid candidates; kept last submission"}
    if len(cands) == 1:
        shutil.copyfile(cands[0]["csv_path"], submission_path)
        return {"status": "single candidate", "choice": 1, "turn": cands[0]["turn"]}

    _SELECTION.clear()
    _SELECTION["candidates"] = cands
    _SELECTION["submission_path"] = submission_path

    register_tool("select_submission", SelectSubmissionTool)
    from openhands.tools.preset.default import get_default_tools
    # restricted toolset: our selector + finish only (drop terminal/file_editor)
    from openhands.sdk.tool import Tool
    tools = [Tool(name="select_submission")]

    # Tool-calling mode must match the model family, or the select_submission
    # call arrives with no `choice` arg:
    #   - Anthropic/Claude: NATIVE tool calling is reliable; the text-parsing
    #     path (native_tool_calling=False) drops the argument -> "Missing
    #     required parameters: {'choice'}" every iteration.
    #   - Qwen/others via Bedrock: text-parsing (False) is what the harness is
    #     tuned for.
    # Auto-pick from the model name, but let an explicit caller value win.
    model_name = str(llm_kwargs.get("model", "")).lower()
    is_anthropic = ("claude" in model_name or "anthropic" in model_name
                    or "sonnet" in model_name or "opus" in model_name
                    or "haiku" in model_name)
    resolved = dict(llm_kwargs)
    if "native_tool_calling" not in resolved:
        resolved["native_tool_calling"] = is_anthropic
    # Bedrock Qwen/etc. reject stopSequences → disable stop words.
    llm = LLM(**{**resolved, "usage_id": "submission-selector",
                 "disable_stop_word": True})

    transcript = _annotated_transcript(events)
    cand_list = "\n".join(
        f"{i+1}) Turn {c['turn']}: {c['script']}"
        for i, c in enumerate(cands))
    first_msg = (
        SELECTOR_SYSTEM_PROMPT
        + "\n\n===== ATTEMPT TRANSCRIPT (turn-annotated) =====\n"
        + transcript
        + "\n\n===== SUBMISSION CANDIDATES (produced this attempt) =====\n"
        + cand_list
        + "\n\nReview the transcript, then call select_submission(choice=<1.."
        + str(len(cands)) + ">, reason=...)."
    )

    agent = Agent(llm=llm, tools=tools)
    conv = LocalConversation(agent=agent, workspace=LocalWorkspace(working_dir=workspace_dir),
                             max_iteration_per_run=max_iterations, delete_on_close=True)
    try:
        conv.send_message(first_msg)
        conv.run()
    finally:
        conv.close()

    result = _SELECTION.get("result")
    if result:
        (csv_history_dir / "final_selection.json").write_text(json.dumps(
            {**result, "n_candidates": len(cands)}, indent=2))
        return result
    return {"status": "selector made no valid choice; kept last submission",
            "n_candidates": len(cands)}
