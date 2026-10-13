#!/usr/bin/env python3
"""
V4 Agent Runner: executed inside the Docker container via `docker exec`.

Pure OpenHands agent loop with standard TerminalTool + FileEditorTool.
The host writes an attempt_config.json with:
  - task_message (preamble already injected for attempts > 0)
  - summary_request (forced summary turn prompt)
  - llm_kwargs, system_prompt_kwargs, max_iterations, condenser_max_size
  - events_output_path (where to write events JSONL)
  - workspace_dir (agent working directory)

This runner:
  1. Reads config
  2. Creates Agent with DS system prompt
  3. Creates fresh LocalConversation (message stream is empty each attempt)
  4. Sends task_message, runs agent loop
  5. Sends summary_request, runs again (forced summary)
  6. Serializes events to JSONL for host to read back
  7. Writes meta.json with cost/token usage
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"


PROMPTS_DIR = Path("/app/prompts")
SUMMARY_REQUEST = (
    "Your attempt is now complete. Please provide a structured summary of your work "
    "using the finish tool. Your summary MUST include these sections:\n\n"
    "## Validation Score\n<single number — your best cross-validation score>\n\n"
    "## Approach\n<brief description of what you did>\n\n"
    "## What Worked\n<what produced good results>\n\n"
    "## What Didn't Work\n<what failed or didn't help>\n\n"
    "## Suggestions for Next Attempt\n<what to try next time>"
)

# Injected ONCE when the task deadline is within one command-timeout — the
# agent's next action may not finish, so it must persist its best submission now.
FINAL_SAVE_WARNING = (
    "🚨 FINAL WARNING — TIME IS ALMOST UP. Your next action may be your last. "
    "You are judged ONLY on /submission/submission.csv exactly as it exists when "
    "time runs out. STOP all experiments now. If your best model's predictions are "
    "not already written to that exact path, re-write them THIS TURN. A missing or "
    "stale submission = zero credit for all your work."
)


def _select_best_submission(llm_kwargs, events, csv_history_dir, submission_path,
                            expected_lines, workspace_dir):
    """Thin wrapper: run the agentic submission-selection sub-agent. See
    dsgym/harness/submission_selector.py for the full design."""
    from dsgym.harness.submission_selector import select_submission_subagent
    return select_submission_subagent(
        llm_kwargs=llm_kwargs, events=events,
        csv_history_dir=Path(csv_history_dir), submission_path=submission_path,
        expected_lines=expected_lines, workspace_dir=workspace_dir,
    )


def _run_reviewer_mode(reviewer_config_path):
    """--reviewer-config entry: run ONE wake of the agentic submission reviewer
    and write its result back for the host watcher to pick up.

    The request JSON (written by the host-side watcher, openhands_harness_v4.py)
    carries paths + the triggering-submission context. In Option B this runs
    INSIDE the reviewer's own apptainer instance, where the base workspace is
    mounted :ro at base_ws_dir, per-task data :ro at data_dir, and the reviewer's
    writable memory :rw at rubric_store_dir. Paths only → the reviewer module is
    arch-independent. Must NEVER raise: always writes a result file so the
    base agent's _fire_reviewer block-poll is released.
    """
    req = json.loads(Path(reviewer_config_path).read_text())

    # Redirect HOME + caches into a writable dir (root FS + /home are :ro), same
    # rationale as the base runner: openhands writes ~/.openhands at init.
    _home = req.get("home_dir") or os.path.join(
        req.get("rubric_store_dir", "/reviewer"), ".home")
    for _d in (_home, os.path.join(_home, ".cache"), os.path.join(_home, ".config")):
        os.makedirs(_d, exist_ok=True)
    os.environ["HOME"] = _home
    os.environ["XDG_CACHE_HOME"] = os.path.join(_home, ".cache")
    os.environ["XDG_CONFIG_HOME"] = os.path.join(_home, ".config")
    os.environ["MPLCONFIGDIR"] = os.path.join(_home, ".config", "matplotlib")

    result_path = req.get("result_path")
    # Select the PRA implementation: "adapra" (default) or "dataprm". Each branch
    # imports its own module so only one reviewer registers its OpenHands tools.
    impl = req.get("impl", "adapra")
    try:
        if impl == "dataprm":
            # DataPRM baseline (step-level generative PRM served over an
            # OpenAI-compatible endpoint); no rubric store, no OpenHands tools.
            from dsgym.harness.prm_dataprm import run_dataprm_reviewer
            out = run_dataprm_reviewer(
                llm_kwargs=req["llm_kwargs"],
                base_ws_dir=req["base_ws_dir"],
                data_dir=req["data_dir"],
                prm_reviews_dir=req.get("prm_reviews_dir"),
                task_desc=req.get("task_desc", ""),
                trajectory_path=req.get("trajectory_path"),
                convo_persist_dir=req.get("convo_persist_dir"),
                submission_index=req.get("submission_index", 0),
                attempt_index=req.get("attempt_index", 0),
                time_budget_seconds=req.get("time_budget_seconds", 1200.0),
            )
        else:
            from dsgym.harness.prm_reviewer import run_submission_reviewer
            out = run_submission_reviewer(
                llm_kwargs=req["llm_kwargs"],
                events=req.get("events", []),
                base_ws_dir=req["base_ws_dir"],
                submission_path=req["submission_path"],
                data_dir=req["data_dir"],
                rubric_store_dir=req["rubric_store_dir"],
                prm_reviews_dir=req.get("prm_reviews_dir"),
                task_desc=req.get("task_desc", ""),
                submission_index=req.get("submission_index", 0),
                attempt_index=req.get("attempt_index", 0),
                command_timeout=req.get("command_timeout", 600),
                max_iterations=req.get("max_iterations", 10),
                time_budget_seconds=req.get("time_budget_seconds", 1200.0),
                convo_persist_dir=req.get("convo_persist_dir"),
                convo_id=req.get("convo_id"),
                rubric_set=req.get("rubric_set"),
                rubric_set_exclude_ids=req.get("rubric_set_exclude_ids"),
                evolution=req.get("evolution", "evolving"),
                operationalization=req.get("operationalization", "dynamic"),
                history=req.get("history", "pull"),
                trajectory_path=req.get("trajectory_path"),
                pushed_transcript=req.get("pushed_transcript"),
                prior_trajectory_paths=req.get("prior_trajectory_paths"),
                n_attempts=req.get("n_attempts", 5),
                min_importance=req.get("min_importance"),
                failure_threshold=req.get("failure_threshold"),
            )
    except Exception as e:  # never leave the base agent block-polling forever
        import traceback
        out = {"status": "ERROR", "error": str(e),
               "traceback": traceback.format_exc()}

    if result_path:
        try:
            p = Path(result_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write so the host watcher never reads a partial file.
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(out, indent=2))
            tmp.replace(p)
        except OSError:
            pass


def main():
    import argparse

    parser = argparse.ArgumentParser(description="V4 agent runner")
    parser.add_argument("--config", help="Path to attempt config JSON")
    parser.add_argument("--reviewer-config",
                        help="Path to a submission-reviewer request JSON. When "
                             "set, run ONE wake of the agentic submission "
                             "reviewer (Option B: in its own apptainer instance) "
                             "instead of the base agent, and exit.")
    args = parser.parse_args()

    if args.reviewer_config:
        _run_reviewer_mode(args.reviewer_config)
        return

    if not args.config:
        parser.error("--config is required unless --reviewer-config is given")

    config = json.loads(Path(args.config).read_text())

    workspace_dir = config["workspace_dir"]

    # Redirect HOME + caches into the WRITABLE workspace before importing
    # openhands. With the read-only root + masked /home, the default
    # HOME=/home/<user> is read-only, and openhands' LocalConversation writes
    # ~/.openhands (LLMProfileStore) at init — which crashed every attempt at
    # turn 0 with "Read-only file system: /home/<user>". Pointing HOME at the
    # workspace fixes it and keeps those writes on a host-visible mount.
    _home = os.path.join(workspace_dir, ".home")
    for _d in (_home, os.path.join(_home, ".cache"), os.path.join(_home, ".config")):
        os.makedirs(_d, exist_ok=True)
    os.environ["HOME"] = _home
    os.environ["XDG_CACHE_HOME"] = os.path.join(_home, ".cache")
    os.environ["XDG_CONFIG_HOME"] = os.path.join(_home, ".config")
    os.environ["MPLCONFIGDIR"] = os.path.join(_home, ".config", "matplotlib")

    events_output_path = Path(config["events_output_path"])
    events_output_path.parent.mkdir(parents=True, exist_ok=True)

    task_message = config["task_message"]
    summary_request = config.get("summary_request", SUMMARY_REQUEST)
    max_iterations = config.get("max_iterations", 200)
    condenser_max_size = config.get("condenser_max_size", 80)
    llm_kwargs = config["llm_kwargs"]
    system_prompt_kwargs = config.get("system_prompt_kwargs", {})
    # CSV-history: dir where per-turn snapshots of every CSV are archived, and
    # the roots to scan for CSVs (workspace + submission mount).
    csv_history_dir = config.get("csv_history_dir")
    csv_scan_roots = config.get("csv_scan_roots", [workspace_dir, "/submission"])
    attempt_index = config.get("system_prompt_kwargs", {}).get("attempt_index", 0)
    # Where to write the home-access red-flag report (host-visible).
    home_flag_path = config.get("home_flag_path")
    # Final-submission SELECTION phase (flag-gated). When true, after the main
    # loop we run a separate phase — OUTSIDE the attempt time/turn budget — that
    # shows the agent every submission CSV it produced (deduped, each tagged with
    # producing turn + script + CV context) and asks it to SELECT the single best
    # one via one structured LLM call. The chosen CSV is copied to
    # submission_path. No code execution — the action space is just the choice.
    final_submission_select = config.get("final_submission_select", False)
    submission_path = config.get("submission_path", "/submission/submission.csv")
    # Resume-on-error: after a ConversationRunError (e.g. a transient Bedrock
    # ServiceUnavailable burst that exhausts the SDK's in-call retries), re-run()
    # the state-persisting conversation to resume mid-work instead of losing the
    # whole attempt. Bounded so a genuine hard outage can't loop forever.
    run_resume_retries = int(config.get("run_resume_retries", 3))
    run_resume_wait = int(config.get("run_resume_wait", 30))
    # Inline PRM (flag-gated). Fires the agentic PRM DURING the attempt on two
    # optional triggers (either/both): every N action-steps, and/or every time a
    # new submission CSV is saved. Each firing writes a self-contained
    # prm_call_NN.json into prm_calls_dir (host-visible), and — when
    # prm_inject_feedback is on — injects the rendered feedback into the agent's
    # conversation to steer the rest of the attempt.
    prm_config = config.get("prm_inline", {}) or {}
    prm_enabled = bool(prm_config.get("enabled", False))
    prm_every_n_steps = prm_config.get("every_n_steps")          # int or None
    prm_on_new_submission = bool(prm_config.get("on_new_submission", False))
    prm_inject_feedback = bool(prm_config.get("inject_feedback", True))
    # p_agent_success below this = weakness worth surfacing. 0.6 default (0.5 was
    # too lenient — a rubric at 0.55 is still shaky). Tunable hyperparameter.
    prm_failure_threshold = float(prm_config.get("failure_threshold", 0.6))
    # Rubric instruction style: "labels" (default, 4-label scheme) or "p_success"
    # (0-1-framed rubric bodies + matching system prompt). Additive/reversible.
    prm_instruction_style = prm_config.get("instruction_style", "labels")
    prm_calls_dir = config.get("prm_calls_dir")  # host-visible dir for prm_call_NN.json
    prm_llm_kwargs = prm_config.get("llm_kwargs") or config.get("llm_kwargs", {})
    # Rubric IDs the static PRM does NOT score at all (dropped from the judge call,
    # so they never appear in prm_call_NN.json and are never injected). Used by the
    # stacked run to hand submission/CV-artifact-truth rubrics to the dynamic
    # reviewer, which owns them (verifies by execution), so the static PRM stops
    # issuing contradictory directives on the same artifact (the conflicting-
    # supervisor failure — see the rubric-ownership sweep,
    # the rubric-ownership sweep). Empty by default → single-PRM inline
    # runs are unchanged. The v1 stacked run still has these scored for the
    # counterfactual, so re-scoring them here would only waste judge tokens.
    prm_exclude_rubrics = set(prm_config.get("exclude_rubric_ids", []) or [])
    # Explicit rubric-set selection (launch-fixed). Path to a rubric-set JSON (e.g.
    # the bottom-up discovered set, dsgym/adapra/rubric_sets/dspredict_discovered.json).
    # When set, the inline PRM scores exactly that set — bypassing the registry +
    # SKIP_RUBRICS — so which rubrics a run uses is chosen by the launch command,
    # not a code-level registry swap (see rubrics.resolve_rubric_set). None → the
    # legacy registry-minus-skip behavior, unchanged.
    prm_rubric_set = prm_config.get("rubric_set")
    prm_dataset = prm_config.get("dataset") or config.get("dataset") or "dspredict-easy"
    # Agentic submission REVIEWER (flag-gated, Option B). On a new submission CSV,
    # write a review-request sentinel into reviewer_requests_dir and block-poll for
    # the host watcher's result (which ran the reviewer in its OWN apptainer
    # instance with the base workspace :ro). The reviewer's wall-clock is EXCLUDED
    # from the base agent's budget (BudgetTracker.add_paused). Independent of the
    # inline PRM above — either, both, or neither may be enabled.
    rev_config = config.get("prm_reviewer", {}) or {}
    rev_enabled = bool(rev_config.get("enabled", False))
    rev_inject_feedback = bool(rev_config.get("inject_feedback", True))
    rev_inject_on_pass = bool(rev_config.get("inject_on_pass", True))
    rev_poll_timeout = float(rev_config.get("poll_timeout_seconds", 1800.0))
    rev_requests_dir = config.get("reviewer_requests_dir")   # host-visible sentinel dir
    rev_reviews_dir = config.get("prm_reviews_dir")          # host-visible reviewer_call_NN.json dir
    rev_data_dir = config.get("reviewer_data_dir")           # per-task data :ro path (in reviewer instance)
    rev_max_iterations = rev_config.get("max_turns", 10)
    # Turn-cadence trigger (in ADDITION to the submission-content trigger): fire a
    # reviewer wake every rev_cadence base-agent turns even if no new submission
    # landed. Reviewer analysis found the
    # submission-only trigger left 10/38 tasks ZERO-fire — the reviewer never woke on
    # agents that stalled early or one-shot a weak submission and stopped. Cadence
    # closes that coverage hole. 0 disables the cadence trigger.
    rev_cadence = int(rev_config.get("cadence_turns", 0) or 0)
    # Mandatory pre-finish review: always run the reviewer once at finish, even if
    # no submission changed since the last wake, so the agent gets a last-line audit
    # before ending (independent of gate_finish_on_fail, which only forces a
    # refinement loop on a submission-triggered FAIL).
    rev_review_on_finish = bool(rev_config.get("review_on_finish", True))
    # Gate-on-finish: when the base agent writes a submission and calls finish in
    # the SAME turn, the normal post-turn reviewer hook would inject the verdict
    # into a conversation that is already finalizing (the agent never acts on it —
    # the "clobber-and-quit" hole). With the gate ON, that finish is intercepted:
    # the reviewer runs, and on a FAIL verdict we force ONE more real loop
    # iteration (verdict injected) so the agent can fix the submission before
    # finishing. Bounded by rev_max_refinements so it can never loop forever.
    rev_gate_finish = bool(rev_config.get("gate_finish_on_fail", True))
    rev_max_refinements = int(rev_config.get("max_finish_refinements", 1))
    # Stacked-run collision handling: when BOTH the inline PRM and the reviewer are
    # enabled with fixed cadences (e.g. inline every 10, reviewer every 20), they
    # deterministically fire on the SAME turn every LCM (turns 20/40/60…), injecting
    # two environment messages into one turn. With rev_defer_on_prm_collision ON, a
    # reviewer cadence wake that lands on a turn the inline PRM also fired is pushed
    # to the NEXT turn: the agent reads the PRM (process feedback) going into turn
    # N+1, then the reviewer verdict (submission audit) going into turn N+2 — one
    # signal per turn, no merge. The deferred wake re-anchors the reviewer cadence to
    # the +1 offset, so after a single deferral the two fixed grids permanently
    # interleave (inline 10/20/30…, reviewer 21/41/61…) and never collide again.
    rev_defer_on_prm_collision = bool(rev_config.get("defer_on_prm_collision", False))
    # History-delivery axis (paper §sec:axes). The trajectory τ is ALWAYS available
    # to the supervisor (no config is blind to the process — otherwise trajectory-
    # grounded rubrics could never be evaluated); this only picks HOW it is
    # delivered: "pull" (default) = the full trajectory is written to a read-only
    # artifact the reviewer reads on demand; "push" = only the agent's NEW turns
    # since the last review are inserted into the reviewer's context each wake.
    rev_history = (rev_config.get("history") or "pull").lower()
    # Transcript rendering detail (prm_agent.truncated_transcript): "command"
    # (default) shows each action's command only; "full" adds the agent's
    # reasoning, ThinkAction text and FileEditorAction contents.
    rev_tau_detail = (rev_config.get("transcript_detail") or "command").lower()
    # Adaptive final-warning threshold: fire when remaining time drops to within
    # one full command timeout (+2 min safety) of the task deadline, i.e. the
    # agent's next action could be its last. Derived from config, not hardcoded.
    command_timeout = config.get("system_prompt_kwargs", {}).get("command_timeout", 600)
    final_warning_threshold = command_timeout + 120
    # HARD per-command timeout CLAMP (see the TerminalExecutor monkeypatch below).
    # The no-output guard (no_change_timeout_seconds) only fires when the agent does
    # NOT set an explicit per-command timeout: an explicit action.timeout puts the
    # SDK terminal into BLOCKING mode, which SKIPS the no-output check entirely. So a
    # single chatty runaway command that sets a huge timeout (observed live:
    # action.timeout=300000s on `python train_lgbm.py`) bypasses the 600s guard and
    # burns the whole attempt until the host watchdog SIGKILLs the process group at
    # task_timeout+120. This clamp caps the agent-supplied action.timeout so such a
    # command fails fast and the agent can recover. Derived from config, default
    # task_timeout//3 (mirrors the v2 collector convention).
    max_command_timeout = int(
        config.get("system_prompt_kwargs", {}).get("max_command_timeout", 3600)
    )

    # Imports (heavy, so inside main)
    import time as _time
    import hashlib as _hashlib
    import shutil as _shutil
    from openhands.sdk import Agent, LocalConversation, LocalWorkspace
    from openhands.sdk.llm import LLM
    from openhands.sdk.context.condenser import LLMSummarizingCondenser
    from openhands.sdk.event import Event, ActionEvent, MessageEvent, ObservationEvent
    from openhands.sdk.llm.message import Message, TextContent
    from openhands.tools.preset.default import register_default_tools, get_default_tools

    # Register standard tools (no browser)
    register_default_tools(enable_browser=False)
    tools = get_default_tools(enable_browser=False)
    # Enforce a per-command NO-OUTPUT timeout on the terminal tool. Without this,
    # get_default_tools() uses the SDK default and a hung command (e.g. a training
    # script that spins with no output) is never killed until the 2h task_timeout
    # watchdog — which burns a worker slot for 2h AND yields an empty (0-turn)
    # trajectory (observed: s3e11 attempt_1 hit 7320s). no_change_timeout_seconds
    # kills a command after `command_timeout` (600s) of NO new output, so a wedged
    # command fails fast and the agent can recover. command_timeout comes from the
    # config (system_prompt_kwargs.command_timeout, default 600).
    from openhands.sdk.tool.spec import Tool as _ToolSpec
    from openhands.tools.terminal import TerminalTool as _TerminalTool
    tools = [
        _ToolSpec(name=_TerminalTool.name,
                  params={"no_change_timeout_seconds": int(command_timeout)})
        if t.name == _TerminalTool.name else t
        for t in tools
    ]

    # HARD per-command timeout CLAMP. The tool-spec params above only control the
    # no-OUTPUT timeout, which the SDK SKIPS when a command runs in blocking mode
    # (i.e. when the LLM sets an explicit action.timeout). TerminalAction is a
    # frozen pydantic model, so we cannot mutate action.timeout in place; instead we
    # wrap the executor's __call__ to swap in a model_copy with a clamped timeout
    # BEFORE the command runs. Only bites when the agent set an explicit timeout
    # ABOVE the cap; commands with timeout=None keep the no-output guard untouched.
    try:
        from openhands.tools.terminal.impl import TerminalExecutor as _TermExec
        _orig_term_call = _TermExec.__call__

        def _clamped_term_call(self, action, conversation=None):
            _t = getattr(action, "timeout", None)
            if _t is not None and _t > max_command_timeout:
                print(f"   [clamp] per-command timeout {_t}s -> "
                      f"{max_command_timeout}s (hard cap)", flush=True)
                action = action.model_copy(update={"timeout": float(max_command_timeout)})
            return _orig_term_call(self, action, conversation)

        _TermExec.__call__ = _clamped_term_call
        print(f"[agent_runner] per-command timeout clamp active: cap="
              f"{max_command_timeout}s (no-output guard={int(command_timeout)}s)", flush=True)
    except Exception as _clamp_e:
        print(f"[agent_runner] WARNING: could not install per-command timeout "
              f"clamp: {_clamp_e}", flush=True)

    # --- CSV history: archive every new/changed CSV per turn -----------------
    # After each turn we scan csv_scan_roots for *.csv, and for any file whose
    # content hash changed since we last saw it, copy it into csv_history_dir as
    # turn{N}__{flattened_path}.csv and append a manifest row. This captures the
    # FULL submission history (not just the final file) so a PRM/human can map a
    # scored/lost CSV back to the turn that produced it and its val-score obs.
    _csv_seen: dict[str, str] = {}  # path -> last content hash
    _csv_seen_md5: set[str] = set()  # content hashes already archived (any path)
    _csv_manifest: list[dict] = []
    _hist_dir = Path(csv_history_dir) if csv_history_dir else None
    if _hist_dir is not None:
        _hist_dir.mkdir(parents=True, exist_ok=True)

    def _seed_preexisting_csvs() -> None:
        """Record CSVs already present at attempt start (carried over from prior
        attempts via the persistent workspace) so we DON'T re-archive them. Each
        attempt's manifest should contain ONLY CSVs produced during THAT attempt
        — otherwise a file made in attempt 0 reappears in every later attempt's
        manifest, bloating history and causing duplicate Kaggle re-scoring."""
        if _hist_dir is None:
            return
        for root in csv_scan_roots:
            rp = Path(root)
            if not rp.exists():
                continue
            for csv in rp.rglob("*.csv"):
                try:
                    if _hist_dir in csv.parents:
                        continue
                    raw = csv.read_bytes()
                except OSError:
                    continue
                h = _hashlib.md5(raw).hexdigest()
                _csv_seen[str(csv)] = h
                _csv_seen_md5.add(h)

    def _snapshot_csvs(turn: int) -> bool:
        """Archive new/changed CSVs this turn. Returns True if a NEW submission
        CSV was archived this turn (drives the PRM on_new_submission trigger)."""
        if _hist_dir is None:
            return False
        _new_submission = False
        for root in csv_scan_roots:
            rp = Path(root)
            if not rp.exists():
                continue
            for csv in rp.rglob("*.csv"):
                # Skip our own archive to avoid recursion.
                try:
                    if _hist_dir in csv.parents:
                        continue
                    raw = csv.read_bytes()
                except OSError:
                    continue
                h = _hashlib.md5(raw).hexdigest()
                key = str(csv)
                if _csv_seen.get(key) == h:
                    continue  # unchanged since last snapshot
                _csv_seen[key] = h
                if h in _csv_seen_md5:
                    continue  # identical content already archived (e.g. carryover)
                _csv_seen_md5.add(h)
                flat = key.replace("/", "__").lstrip("_")
                dest = _hist_dir / f"turn{turn:03d}__{flat}"
                try:
                    _shutil.copyfile(csv, dest)
                except OSError:
                    continue
                nrows = raw.count(b"\n")
                _csv_manifest.append({
                    "turn": turn,
                    "attempt_index": attempt_index,
                    "source_path": key,
                    "archived_as": dest.name,
                    "md5": h,
                    "n_lines": nrows,
                })
                if "submission" in Path(key).name.lower():
                    _new_submission = True
        # Rewrite manifest each time (cheap, small).
        try:
            (_hist_dir / "manifest.json").write_text(json.dumps(_csv_manifest, indent=2))
        except OSError:
            pass
        return _new_submission

    # --- Home-access red flag -------------------------------------------------
    # On some clusters /home is an automount that apptainer cannot fully mask, so
    # an agent CAN read the host home (source, ~/.aws) even though it cannot write
    # outside the bind mounts. We can't prevent the read, but we flag any turn
    # whose action text touches /home, ~/.aws, or credential paths, so a reviewer
    # can quarantine/discard trajectories that snooped. Written to a host-visible
    # sidecar the harness folds into trajectory metadata.
    import re as _re
    _HOME_FLAG_RE = _re.compile(
        r"(/home/[A-Za-z0-9._-]+|~/|\.aws|/etc/aws|credentials|aws_secret|aws_access)",
        _re.IGNORECASE,
    )
    _home_flags: list[dict] = []

    def _action_texts(action_events) -> list[str]:
        texts: list[str] = []
        for ae in action_events or []:
            for attr in ("command", "path", "file_text", "new_str", "old_str", "content"):
                v = getattr(getattr(ae, "action", ae), attr, None)
                if isinstance(v, str) and v:
                    texts.append(v)
            # thought may be a list of content blocks
            th = getattr(ae, "thought", None)
            if isinstance(th, str):
                texts.append(th)
        return texts

    def _check_home_access(turn: int, action_events) -> None:
        if home_flag_path is None:
            return
        for text in _action_texts(action_events):
            m = _HOME_FLAG_RE.search(text)
            if m:
                _home_flags.append({
                    "turn": turn,
                    "attempt_index": attempt_index,
                    "match": m.group(0),
                    "snippet": text[:200],
                })
                print(f"   🚩 HOME-ACCESS RED FLAG (turn {turn}): {m.group(0)}", flush=True)
        if _home_flags:
            try:
                Path(home_flag_path).parent.mkdir(parents=True, exist_ok=True)
                Path(home_flag_path).write_text(json.dumps(_home_flags, indent=2))
            except OSError:
                pass

    # LLM
    llm = LLM(**llm_kwargs)

    # Condenser
    condenser = LLMSummarizingCondenser(
        llm=llm.model_copy(update={"usage_id": "condenser"}),
        max_size=condenser_max_size,
        keep_first=4,
    )

    # Events collected during the run (populated by on_event); defined here so
    # the inline-PRM hook in DSAgent._execute_actions can read the transcript.
    collected_events: list = []

    # --- Inline PRM ----------------------------------------------------------
    _prm_calls: list[dict] = []        # in-memory log of this attempt's PRM calls
    _prm_last_step_fired = [0]         # mutable holder for every-N-steps bookkeeping

    def _fire_prm(turn: int, trigger: str, conversation) -> None:
        """Run one agentic-PRM call over the trajectory SO FAR, write
        prm_call_NN.json to the host-visible prm_calls_dir, and (if enabled)
        inject the rendered feedback into the conversation to steer the agent."""
        if not prm_calls_dir:
            return
        try:
            from dsgym.harness.prm_agent import score_trajectory_at_step
            from dsgym.harness.prm_feedback import generate_feedback_report
        except Exception as e:
            print(f"   [prm] import failed: {e}", flush=True)
            return
        _events = [e.model_dump(mode="json") if hasattr(e, "model_dump") else e
                   for e in collected_events]
        idx = len(_prm_calls)
        out_path = Path(prm_calls_dir) / f"prm_call_{idx:02d}.json"
        try:
            rec = score_trajectory_at_step(
                llm_kwargs=prm_llm_kwargs, events=_events, task_desc=task_message,
                up_to_step=turn, trigger=trigger, attempt_index=attempt_index,
                out_path=out_path, instruction_style=prm_instruction_style,
                exclude_rubric_ids=prm_exclude_rubrics,
                rubric_set=prm_rubric_set, dataset=prm_dataset)
        except Exception as e:
            print(f"   [prm] call failed at turn {turn}: {e}", flush=True)
            return
        _prm_calls.append(rec)
        cost = rec.get("cost", {})
        print(f"   [prm] call {idx} @turn{turn} ({trigger}): "
              f"composite={rec.get('composite_score'):.3f} "
              f"{cost.get('total_tokens')}tok", flush=True)
        if prm_inject_feedback:
            # mid_run=True: the report is framed as guidance for an ONGOING
            # attempt (address the flagged issues in remaining steps), NOT a
            # post-mortem of a finished attempt. Weaknesses-only — no strengths
            # are injected, and if there are no weaknesses fb is "" (nothing sent).
            # Excluded rubrics were never scored (see prm_exclude_rubrics above),
            # so they are absent from rubric_scores and can't be injected.
            fb = generate_feedback_report(rec.get("rubric_scores", {}), mid_run=True,
                                          failure_threshold=prm_failure_threshold)
            if fb.strip():
                try:
                    conversation.send_message(fb)
                except Exception as e:
                    print(f"   [prm] feedback injection failed: {e}", flush=True)

    # --- Agentic submission reviewer (Option B: own instance) ----------------
    _reviews: list[dict] = []          # in-memory log of this attempt's reviews
    _rev_refinements = [0]             # gate-on-finish forced-refinement count (this attempt)
    _rev_last_md5 = [""]               # md5 of the canonical submission at last review
    _rev_last_cadence_turn = [0]       # last base-agent turn a cadence wake fired
    _rev_fired_on_finish = [False]     # pre-finish review already run this attempt?
    _rev_deferred_pending = [False]    # a cadence wake was deferred off a PRM-collision turn
    _rev_last_pushed_turn = [0]        # last agent turn already PUSHed to the reviewer (history=push delta cursor)

    def _canonical_submission_md5() -> str:
        """md5 of the CANONICAL submission file (the one that is actually
        scored). Empty string if it does not exist / can't be read."""
        try:
            sp = Path(submission_path)
            if sp.exists():
                return _hashlib.md5(sp.read_bytes()).hexdigest()
        except OSError:
            pass
        return ""

    def _reviewer_should_fire() -> bool:
        """Gate the reviewer on the CANONICAL submission's CONTENT changing since
        the last review. `_snapshot_csvs`' `_new_sub` fires on ANY submission-
        NAMED csv anywhere in the scan roots (e.g. per-model `*_submission.csv`
        scratch files), and the agent rewrites the canonical file many times per
        attempt — together that woke the reviewer ~18x/attempt, burying the one
        real FAIL in a stream of PASSes on transient/non-scored files (alarm
        fatigue). Firing only when the file that actually gets scored changes
        concentrates the reviewer's budget on the submission that matters and is
        also cheaper. Returns False (no re-fire) if the canonical file is missing
        or byte-identical to what we last reviewed."""
        md5 = _canonical_submission_md5()
        if not md5 or md5 == _rev_last_md5[0]:
            return False
        _rev_last_md5[0] = md5
        return True

    def _fire_reviewer(turn: int, conversation, inject: bool = True):
        """On a new submission CSV, request a reviewer wake via a sentinel file,
        block-poll for the host watcher's result (the reviewer ran in its OWN
        apptainer instance with the base workspace :ro), then inject the verdict.

        Returns the reviewer result dict (or None on any failure / no result) so
        the gate-on-finish path can act on the verdict WITHOUT injecting it into a
        conversation that is mid-finalize. When ``inject`` is False the verdict is
        NOT sent via send_message (the caller handles it) — everything else (the
        pause accounting, the _reviews record, the telemetry print) is identical.

        The reviewer's wall-clock is EXCLUDED from the base agent's budget: we
        record the pause and add it via budget_tracker.add_paused in ALL paths.
        Fully guarded — a reviewer failure must never break the base attempt."""
        if not (rev_requests_dir and rev_reviews_dir):
            return None
        import hashlib
        _pause_start = _time.time()
        sub_idx = len(_reviews)
        try:
            reqs = Path(rev_requests_dir); reqs.mkdir(parents=True, exist_ok=True)
            # md5 of the triggering submission (host-visible via csv_history or
            # the /submission mount) — telemetry only; best-effort.
            md5 = ""
            try:
                sp = Path(submission_path)
                if sp.exists():
                    md5 = hashlib.md5(sp.read_bytes()).hexdigest()
            except OSError:
                pass
            result_path = str(reqs / f"review_result_{turn}.json")
            # History-delivery axis (paper §sec:axes): the trajectory τ is ALWAYS
            # made available to the reviewer (no config is blind to the process,
            # else trajectory-grounded rubrics could never be evaluated). `history`
            # only picks HOW: "pull" = write the FULL turn-annotated transcript to a
            # host-visible file next to this sentinel, and let the watcher expose it
            # read-only in the reviewer's workspace (read on demand); "push" = inline
            # only the agent's NEW turns since the last wake (a bounded delta).
            traj_file = None
            pushed_transcript = None
            try:
                from dsgym.harness.prm_agent import truncated_transcript
                _rev_events = [e.model_dump(mode="json") if hasattr(e, "model_dump")
                               else e for e in collected_events]
                if rev_history == "push":
                    delta, n_steps = truncated_transcript(
                        _rev_events, up_to_step=None,
                        from_step=_rev_last_pushed_turn[0], detail=rev_tau_detail)
                    pushed_transcript = delta
                    _rev_last_pushed_turn[0] = n_steps
                else:  # pull (default): full transcript to a file for the watcher
                    full, _ = truncated_transcript(_rev_events, up_to_step=None,
                                                   detail=rev_tau_detail)
                    traj_name = f"review_traj_{turn}.txt"
                    (reqs / traj_name).write_text(full)
                    traj_file = traj_name
            except Exception as e:
                print(f"   [reviewer] transcript render failed @turn{turn}: {e}",
                      flush=True)
            req = {
                "turn": turn, "attempt_index": attempt_index,
                "submission_index": sub_idx, "submission_md5": md5,
                "submission_path": submission_path,
                "task_desc": task_message,
                "max_iterations": rev_max_iterations,
                # Trajectory delivery (see above). traj_file is a basename in
                # rev_requests_dir; the watcher copies it into the reviewer's
                # workspace and fills trajectory_path. pushed_transcript rides
                # inline (bounded delta). history is echoed so the reviewer renders
                # the right access block.
                "history": rev_history,
                "traj_file": traj_file,
                "pushed_transcript": pushed_transcript,
                "result_path": result_path,
                # Per-attempt conversation continuity: stable key → same convo_id
                # resumes the message stream across wakes WITHIN this attempt.
                "convo_id": f"attempt{attempt_index}",
                # data_dir + rubric_store_dir + convo_persist_dir + trajectory_path
                # are filled by the host watcher (it knows the reviewer-instance
                # mount paths).
            }
            # Atomic sentinel write so the watcher never reads a partial request.
            tmp = reqs / f"review_request_{turn}.json.tmp"
            tmp.write_text(json.dumps(req))
            tmp.replace(reqs / f"review_request_{turn}.json")
            print(f"   [reviewer] requested review @turn{turn} "
                  f"(sub #{sub_idx}); polling for result…", flush=True)
        except Exception as e:
            print(f"   [reviewer] request write failed @turn{turn}: {e}", flush=True)
            budget_tracker.add_paused(_time.time() - _pause_start)
            return None

        # Block-poll for the result file (own timeout so we never deadlock).
        rp = Path(result_path)
        result = None
        deadline = _time.time() + rev_poll_timeout
        while _time.time() < deadline:
            if rp.exists():
                try:
                    result = json.loads(rp.read_text())
                except (OSError, json.JSONDecodeError):
                    result = None
                break
            _time.sleep(2.0)

        paused = _time.time() - _pause_start
        budget_tracker.add_paused(paused)
        if result is None:
            print(f"   [reviewer] no result within {rev_poll_timeout:.0f}s "
                  f"@turn{turn}; resuming (paused {paused:.0f}s)", flush=True)
            return None
        _reviews.append(result)
        status = result.get("status")
        print(f"   [reviewer] verdict {status} @turn{turn} "
              f"(forced={result.get('forced')}, paused {paused:.0f}s)", flush=True)
        verdict = result.get("verdict_text") or ""
        if not verdict or status == "ERROR":
            return result
        # Weaknesses-only gate (mirrors the static PRM): suppress a CLEAN PASS —
        # sound submission with nothing to act on. But a PASS that carries MEASURED
        # modeling SUGGESTIONS is still actionable feedback the agent should see, so
        # do NOT suppress those (that would silently drop the reviewer's modeling
        # half). WARN/FAIL are always injected.
        if status == "PASS" and not rev_inject_on_pass:
            if not (result.get("suggestions") or []):
                return result
        if inject and rev_inject_feedback:
            try:
                conversation.send_message(verdict)
            except Exception as e:
                print(f"   [reviewer] verdict injection failed: {e}", flush=True)
        return result

    # Agent with custom DS prompt + budget tracker
    class DSAgent(Agent):
        @property
        def prompt_dir(self) -> Path:
            return PROMPTS_DIR

        def _check_iterative_refinement(self, conversation, action_event):
            """Gate the base agent's FINISH on the submission reviewer's verdict.

            The SDK's finalize() calls this for a FinishTool action AFTER the
            turn's other tool actions have already executed (so a submission
            written THIS turn is on disk) but BEFORE it marks the conversation
            FINISHED. Returning (True, followup) injects `followup` and forces
            one more loop instead of finishing.

            This closes the "clobber-and-quit" hole: an agent that writes a
            submission and calls finish in the same turn would otherwise have the
            reviewer's verdict injected into an already-finalizing conversation
            (never acted on). Here, if that finish co-occurs with a NEW submission
            and the reviewer returns FAIL, we force ONE more real iteration with
            the verdict injected so the agent can fix it.

            Fully guarded: any failure falls through to the SDK default (finish).
            Bounded by rev_max_refinements so it can never loop forever."""
            # The gate does two things at finish: (a) intercept a write-then-finish
            # so a FAIL forces a refinement (gate_finish_on_fail), and (b) run a
            # MANDATORY last-line review before ending even if nothing changed
            # (review_on_finish) — the coverage backstop for agents that never
            # triggered a submission-based wake.
            if not (rev_enabled and (rev_gate_finish or rev_review_on_finish)):
                return super()._check_iterative_refinement(conversation, action_event)
            try:
                # Did a NEW submission land this turn? _snapshot_csvs de-dupes by
                # content hash, so calling it here (before the post-super hook)
                # both DETECTS the new submission AND marks it seen — the normal
                # post-turn hook will then NOT re-fire the reviewer for it.
                turn = budget_tracker.turn_count + 1  # tick() runs post-super
                _new_this_turn = _snapshot_csvs(turn)
                _budget_left = _rev_refinements[0] < rev_max_refinements
                # (a) write-then-finish gate: a new submission co-occurring with
                # finish, refinement budget remaining.
                _gate_now = rev_gate_finish and _new_this_turn and _budget_left
                # (b) pre-finish backstop: no gate fired, we haven't already done a
                # pre-finish review this attempt, and a canonical submission exists.
                _prefinish_now = (rev_review_on_finish and not _gate_now
                                  and not _rev_fired_on_finish[0]
                                  and bool(_canonical_submission_md5()))
                if not (_gate_now or _prefinish_now):
                    return super()._check_iterative_refinement(conversation, action_event)
                if _prefinish_now:
                    _rev_fired_on_finish[0] = True
                # Review now, inline, WITHOUT injecting (the conversation is
                # mid-finalize — we return the verdict as the refinement followup
                # instead, or inject it below if we allow the finish).
                result = _fire_reviewer(turn, conversation, inject=False)
                if not result:
                    return super()._check_iterative_refinement(conversation, action_event)
                status = result.get("status")
                verdict = result.get("verdict_text") or ""
                if status == "FAIL" and verdict and _budget_left:
                    _rev_refinements[0] += 1
                    print(f"   [reviewer] GATE: FAIL @turn{turn} "
                          f"({'new-sub' if _new_this_turn else 'pre-finish'}); "
                          f"forcing refinement {_rev_refinements[0]}/{rev_max_refinements}",
                          flush=True)
                    return True, verdict
                # PASS/WARN/ERROR → allow finish, but still surface the verdict for
                # the record. Suppress only a CLEAN PASS (weaknesses-only gate); a
                # PASS carrying measured modeling SUGGESTIONS is still injected.
                _clean_pass = (status == "PASS" and not rev_inject_on_pass
                               and not (result.get("suggestions") or []))
                if verdict and status != "ERROR" and not _clean_pass:
                    if rev_inject_feedback:
                        try:
                            conversation.send_message(verdict)
                        except Exception:
                            pass
                return super()._check_iterative_refinement(conversation, action_event)
            except Exception as e:
                print(f"   [reviewer] GATE error (allowing finish): {e}", flush=True)
                return super()._check_iterative_refinement(conversation, action_event)

        def _execute_actions(self, conversation, action_events, on_event):
            super()._execute_actions(conversation, action_events, on_event)
            budget_tracker.tick()
            turn = budget_tracker.turn_count
            # Archive any new/changed CSVs produced this turn, tagged by turn.
            _new_sub = _snapshot_csvs(turn)
            # Red-flag: did this turn try to read/mount the host home or creds?
            _check_home_access(turn, action_events)
            # Inline PRM triggers: every-N-steps and/or on-new-submission.
            _prm_fired_this_turn = False
            if prm_enabled:
                fire = False
                trig = ""
                if prm_every_n_steps and turn - _prm_last_step_fired[0] >= prm_every_n_steps:
                    fire, trig = True, "every_n_steps"
                    _prm_last_step_fired[0] = turn
                elif prm_on_new_submission and _new_sub:
                    fire, trig = True, "new_submission"
                if fire:
                    _fire_prm(turn, trig, conversation)
                    _prm_fired_this_turn = True
            # Agentic submission reviewer (Option B): ONE of two MUTUALLY EXCLUSIVE
            # wake regimes, selected by rev_cadence (the flag; 0 = off).
            #   * Cadence mode (rev_cadence > 0): wake every rev_cadence base-agent
            #     turns regardless of submissions (reaches agents that stall /
            #     one-shot — the coverage hole
            #     #2) PLUS one mandatory pre-finish review. Deliberately does NOT
            #     also fire per-submission: mixing a fixed cadence with a
            #     content-triggered wake gives a ragged, submission-count-dependent
            #     fire pattern that muddies the ablation. Fixed cadence = clean.
            #   * Submission mode (rev_cadence == 0): the v5 behavior — wake only
            #     when the CANONICAL (scored) submission's CONTENT changes since the
            #     last review (NOT every submission-named scratch csv).
            # Runs after the inline PRM so both may coexist; pauses the base clock.
            if rev_enabled:
                if rev_cadence > 0:
                    _fire = turn - _rev_last_cadence_turn[0] >= rev_cadence
                    # A pending deferred wake (slipped off a prior PRM-collision turn)
                    # fires on this next turn regardless of the cadence grid.
                    if _rev_deferred_pending[0]:
                        _fire = True
                    if _fire:
                        _rev_last_cadence_turn[0] = turn
                    # Stacked-run collision: if the inline PRM also fired THIS turn and
                    # we are not already firing a deferred wake, push this wake to the
                    # next turn so the agent reads one signal per turn (PRM now,
                    # reviewer next). The deferred wake (fired next turn) re-anchors the
                    # cadence to that odd offset, so with inline every 10 / reviewer
                    # every 20 the grids permanently interleave (reviewer → 21/41/61…)
                    # after a SINGLE deferral, never colliding again.
                    if (_fire and rev_defer_on_prm_collision and _prm_fired_this_turn
                            and not _rev_deferred_pending[0]):
                        _rev_deferred_pending[0] = True
                        _fire = False
                        print(f"   [reviewer] cadence wake @turn{turn} deferred "
                              f"(inline PRM fired same turn); will fire next turn",
                              flush=True)
                    elif _fire:
                        _rev_deferred_pending[0] = False
                else:
                    _fire = _new_sub and _reviewer_should_fire()
                if _fire:
                    _fire_reviewer(turn, conversation)
            status = budget_tracker.status_line()
            print(status, flush=True)
            content = status
            # Fire the final-save warning ONCE, when the deadline is within one
            # command-timeout (+safety) — the next action may not complete.
            if (not budget_tracker.final_warning_sent
                    and budget_tracker.remaining() <= final_warning_threshold):
                budget_tracker.final_warning_sent = True
                content = status + "\n\n" + FINAL_SAVE_WARNING
                print("   [FINAL-SAVE WARNING issued]", flush=True)
            msg = MessageEvent(
                source="environment",
                llm_message=Message(
                    role="user",
                    content=[TextContent(text=content)],
                ),
            )
            on_event(msg)

    def _cgroup_mem_gb():
        """Return (limit_gb, current_gb) from the cgroup v2 memory controller,
        or (None, None) if unavailable.

        This is the TRUE ceiling — under a batch scheduler the allocation can cap
        memory far below the physical node, and /proc/meminfo
        reports the PHYSICAL node, which would badly mislead the agent. The
        cgroup limit is shared across ALL workers in the allocation, so
        memory.current reflects real aggregate pressure. Walk up the cgroup tree
        and take the smallest numeric memory.max found (the binding limit)."""
        try:
            with open("/proc/self/cgroup") as f:
                rel = f.readline().strip().split(":")[-1]  # v2: 0::/path
        except OSError:
            return None, None
        limit = None
        current = None
        p = "/sys/fs/cgroup" + rel
        base = "/sys/fs/cgroup"
        while True:
            try:
                with open(p + "/memory.max") as f:
                    v = f.read().strip()
                if v != "max":
                    iv = int(v) / 1024 ** 3
                    limit = iv if limit is None else min(limit, iv)
                    # current at the same (binding) level is the best signal
                    try:
                        with open(p + "/memory.current") as cf:
                            current = int(cf.read().strip()) / 1024 ** 3
                    except OSError:
                        pass
            except OSError:
                pass
            if p == base or "/" not in rel:
                break
            rel = rel.rsplit("/", 1)[0]
            p = base + rel if rel else base
            if not rel:
                p = base
                # one more pass at the root then stop
                rel = ""
                if limit is not None:
                    break
        return limit, current

    def _memory_status():
        """Best-effort memory footprint line for the footer.

        Soft awareness only (no hard limit): reports RAM used by this attempt's
        process tree and how much memory is still available under the REAL limit,
        so the agent can moderate its NEXT action (e.g. not loading every model
        into RAM at once for a stacking ensemble). Prefers the cgroup limit (the
        allocation ceiling, shared across workers) over /proc/meminfo (the
        physical node, which overstates headroom ~10x here). Can't prevent an OOM
        WITHIN a single tool call — the footer only shows between turns — but it
        nudges away from memory-hungry choices. Returns "" on any read failure."""
        try:
            # RSS of this process tree (self + children), from /proc/*/status.
            rss_gb = 0.0
            try:
                import os as _os
                pgid = _os.getpgrp()
                for pid in _os.listdir("/proc"):
                    if not pid.isdigit():
                        continue
                    try:
                        with open(f"/proc/{pid}/stat") as sf:
                            fields = sf.read().rsplit(")", 1)[1].split()
                        if int(fields[2]) != pgid:  # field is pgrp
                            continue
                        with open(f"/proc/{pid}/status") as sf:
                            for line in sf:
                                if line.startswith("VmRSS:"):
                                    rss_gb += int(line.split()[1]) / 1024 / 1024
                                    break
                    except (OSError, ValueError, IndexError):
                        continue
            except OSError:
                pass

            # Prefer the cgroup (allocation) limit; fall back to /proc/meminfo.
            limit_gb, current_gb = _cgroup_mem_gb()
            if limit_gb is not None:
                free_gb = max(0.0, limit_gb - (current_gb or 0.0))
                return (f" | Mem: this task ~{rss_gb:.0f}G; "
                        f"{free_gb:.0f}G free of {limit_gb:.0f}G "
                        f"(shared across workers)")
            avail_gb = total_gb = None
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail_gb = int(line.split()[1]) / 1024 / 1024
                    elif line.startswith("MemTotal:"):
                        total_gb = int(line.split()[1]) / 1024 / 1024
                    if avail_gb is not None and total_gb is not None:
                        break
            if avail_gb is None:
                return ""
            return (f" | Mem: this task ~{rss_gb:.0f}G, "
                    f"{avail_gb:.0f}G free of {total_gb:.0f}G")
        except OSError:
            return ""

    class BudgetTracker:
        def __init__(self, max_turns, time_budget_seconds):
            self.max_turns = max_turns
            self.time_budget_seconds = time_budget_seconds
            self.start_time = _time.time()
            self.turn_count = 0
            self.final_warning_sent = False
            # Total seconds spent in reviewer wakes — EXCLUDED from the base
            # agent's elapsed time so its footer/final-warning stay accurate.
            self.paused_seconds = 0.0

        def tick(self):
            self.turn_count += 1

        def add_paused(self, seconds):
            self.paused_seconds += max(0.0, seconds)

        def _elapsed(self):
            return (_time.time() - self.start_time) - self.paused_seconds

        def remaining(self):
            return max(0, self.time_budget_seconds - self._elapsed())

        def status_line(self):
            elapsed = self._elapsed()
            remaining = max(0, self.time_budget_seconds - elapsed)
            return (
                f"[Turn {self.turn_count}/{self.max_turns} | "
                f"Time {elapsed/60:.0f}m/{self.time_budget_seconds/60:.0f}m | "
                f"{remaining/60:.0f}m remaining"
                f"{_memory_status()}]"
            )

    time_budget = config.get("time_budget_seconds", 7200)
    budget_tracker = BudgetTracker(max_iterations, time_budget)

    agent = DSAgent(
        llm=llm,
        tools=tools,
        condenser=condenser,
        system_prompt_kwargs=system_prompt_kwargs,
    )

    # Event collection — appends to the `collected_events` list defined above the
    # DSAgent class (same object, so the inline-PRM hook sees the live transcript).
    #
    # Also STREAM each event to the events JSONL as it arrives (line-buffered +
    # flush). The batch dump at the end of main() (see "Serialize events to JSONL")
    # is the only other write, and a host-watchdog force-kill terminates the
    # process mid-run before it ever runs — so a killed ("wedged") attempt used to
    # read back as an EMPTY event stream (num_turns=0), discarding every completed
    # turn. Streaming means whatever turns finished before the kill are already on
    # disk for the host to reconstruct. (A turn stuck INSIDE one never-returning
    # command still emits no event for that turn — this recovers the prior turns,
    # not the in-flight one.) flush() pushes the Python buffer to the OS page cache,
    # which survives process death (SIGKILL), so no fsync is needed.
    _events_stream = open(events_output_path, "w", buffering=1)

    def on_event(event: Event):
        collected_events.append(event)
        try:
            _events_stream.write(json.dumps(event.model_dump(mode="json")) + "\n")
        except Exception:
            _events_stream.write(
                json.dumps({"type": "SerializationError", "repr": repr(event)}) + "\n"
            )
        _events_stream.flush()

    # Workspace
    workspace = LocalWorkspace(working_dir=workspace_dir)

    # Conversation (fresh message stream — resets each attempt)
    conv = LocalConversation(
        agent=agent,
        workspace=workspace,
        max_iteration_per_run=max_iterations,
        callbacks=[on_event],
        delete_on_close=True,
        stuck_detection=True,
    )

    def _run_with_resume(label: str):
        """Run the conversation, resuming after a transient ConversationRunError.

        A Bedrock ServiceUnavailable burst that exhausts the SDK's in-call retries
        raises ConversationRunError and sets the conversation status to ERROR. The
        conversation STATE persists (messages + iteration position), and run()
        resumes from ERROR (local_conversation.py:590), so re-running continues the
        attempt mid-work instead of discarding it. Bounded by run_resume_retries so
        a genuine hard outage still terminates. Non-LLM errors re-raise immediately.
        """
        from openhands.sdk.conversation.exceptions import ConversationRunError
        attempts = 0
        while True:
            # The SDK resets its per-run iteration counter to 0 at the top of
            # every run() (local_conversation.py:598) and caps on
            # `iteration >= max_iteration_per_run`. On a resume that would hand
            # the agent a FRESH full turn budget, so a 503 burst could let one
            # attempt run run_resume_retries × max_iterations turns (observed:
            # 375 turns against a 200 cap). Clamp the per-run cap to the turns
            # REMAINING in the global budget (budget_tracker.turn_count is the
            # true cross-resume turn count) so the global cap holds.
            remaining_turns = max_iterations - budget_tracker.turn_count
            if remaining_turns <= 0:
                print(f"   [resume] {label}: global turn budget exhausted "
                      f"({budget_tracker.turn_count}/{max_iterations}); stopping",
                      flush=True)
                return
            conv.max_iteration_per_run = remaining_turns
            try:
                conv.run()
                return
            except ConversationRunError as e:
                attempts += 1
                if attempts > run_resume_retries:
                    print(f"   [resume] {label}: exhausted {run_resume_retries} "
                          f"resume attempts; giving up ({e})", flush=True)
                    raise
                print(f"   [resume] {label}: ConversationRunError "
                      f"({attempts}/{run_resume_retries}) — resuming in "
                      f"{run_resume_wait}s: {e}", flush=True)
                time.sleep(run_resume_wait)

    try:
        # Record CSVs carried over from prior attempts so this attempt's history
        # only contains CSVs actually produced during THIS attempt.
        _seed_preexisting_csvs()
        # Main agent loop
        conv.send_message(task_message)
        _run_with_resume("main")

        # Final-submission SELECTION phase (flag-gated, outside the time budget).
        # One structured LLM call picking the best of the CSVs already produced;
        # harness-side deterministic copy to submission_path. See _select_best.
        if final_submission_select and csv_history_dir:
            try:
                # serialize events for the sub-agent's turn-annotated transcript
                _events = [e.model_dump(mode="json") if hasattr(e, "model_dump") else e
                           for e in collected_events]
                # expected row count = most-frequent n_lines among this attempt's
                # submission versions (canonical shape; malformed stubs are rarer)
                _mp = Path(csv_history_dir) / "manifest.json"
                _exp = None
                try:
                    from collections import Counter as _C
                    _nl = [e.get("n_lines") for e in json.loads(_mp.read_text())
                           if "submission" in Path(e.get("source_path", "")).name.lower()
                           and e.get("n_lines")]
                    _exp = _C(_nl).most_common(1)[0][0] if _nl else None
                except Exception:
                    pass
                sel = _select_best_submission(
                    llm_kwargs=llm_kwargs, events=_events,
                    csv_history_dir=Path(csv_history_dir),
                    submission_path=submission_path,
                    expected_lines=_exp, workspace_dir=workspace_dir,
                )
                print(f"   [final-select] {sel}", flush=True)
            except Exception as e:
                print(f"   [final-select] FAILED (keeping last submission): {e}", flush=True)

        # Forced summary turn
        conv.send_message(summary_request)
        _run_with_resume("summary")
    except Exception as e:
        error_info = {"error": str(e), "traceback": traceback.format_exc()}
        # Attempt-keyed so a crashed attempt's error can't overwrite (or be
        # confused with) a prior attempt's. events_output_path is
        # events_attempt_{i}.jsonl; derive the matching error/meta suffix.
        _stem = events_output_path.stem  # e.g. "events_attempt_3"
        _sfx = _stem.replace("events_attempt_", "") if "events_attempt_" in _stem else ""
        _err_name = f"error_attempt_{_sfx}.json" if _sfx else "error.json"
        (events_output_path.parent / _err_name).write_text(
            json.dumps(error_info, indent=2)
        )
        # Keep the un-suffixed name too for backward-compat log scraping.
        (events_output_path.parent / "error.json").write_text(
            json.dumps(error_info, indent=2)
        )
    finally:
        conv.close()
        # Close the streaming handle before the authoritative batch rewrite below
        # (avoids two open write-handles to the same path). On a force-kill this
        # finally never runs, but the streamed lines are already flushed to disk.
        try:
            _events_stream.close()
        except Exception:
            pass

    # Serialize events to JSONL (authoritative, fully-ordered rewrite on a clean
    # exit; identical content to what on_event already streamed).
    with open(events_output_path, "w") as f:
        for event in collected_events:
            try:
                f.write(json.dumps(event.model_dump(mode="json")) + "\n")
            except Exception:
                f.write(json.dumps({"type": "SerializationError", "repr": repr(event)}) + "\n")

    # Write metadata. accumulated_token_usage is a Pydantic TokenUsage object,
    # not JSON-serializable directly — dump it to a dict first.
    token_usage = getattr(llm.metrics, "accumulated_token_usage", {})
    if hasattr(token_usage, "model_dump"):
        token_usage = token_usage.model_dump(mode="json")
    elif not isinstance(token_usage, dict):
        token_usage = {}
    meta = {
        "num_events": len(collected_events),
        "cost": getattr(llm.metrics, "accumulated_cost", 0),
        "token_usage": token_usage,
    }
    # Attempt-keyed meta so a crashed attempt (which may not rewrite meta before
    # dying) can't leave the PREVIOUS attempt's cost/tokens for the host to read
    # back and mis-attribute (the identical-token duplication bug). The host
    # reads meta_attempt_{i}.json first, falling back to meta.json.
    _meta_stem = events_output_path.stem
    _meta_sfx = _meta_stem.replace("events_attempt_", "") if "events_attempt_" in _meta_stem else ""
    _meta_name = f"meta_attempt_{_meta_sfx}.json" if _meta_sfx else "meta.json"
    (events_output_path.parent / _meta_name).write_text(json.dumps(meta, indent=2))
    (events_output_path.parent / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_info = {"error": str(e), "traceback": traceback.format_exc()}
        # Write to a known location so host can always find the error
        error_path = Path("/workspace/.harness/runner_error.json")
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(json.dumps(error_info, indent=2))
        sys.exit(1)
