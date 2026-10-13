"""
Agentic submission REVIEWER sub-agent (V4, flag-gated) — the "agentic PRM".

When the base agent writes/updates a submission CSV, a SEPARATE OpenHands
sub-agent wakes ("submission-triggered reviewer"). Unlike the inline rubric PRM
(a single scoring call, no execution), this reviewer can EXECUTE code-based
checks against the base agent's repository + the competition data, and maintains
an evolving per-task store of reusable checks.

Design:
- Runs in its OWN apptainer instance (Option B): the base agent's workspace is
  bind-mounted READ-ONLY at /base_ws, the per-task data READ-ONLY at /data, and
  the reviewer's writable memory at /reviewer (its cwd). This module is
  arch-independent — it takes only paths, so it runs identically in-container
  or in a separate instance.
- Rubric store persists across all 5 attempts (on disk, in /reviewer); the
  message stream resets per attempt (the base agent's cross-attempt memory is
  likewise the repository, not the conversation).
- Warm-started with 3 executable seed checks: cv_reproduce,
  submission_vs_target_dist, submission_coverage (§1b.i).
- Emits a <verdict> (STATUS/FINDINGS/EVIDENCE) that the harness injects verbatim
  into the base agent's trajectory.
- Has its OWN footer (turn + wall-clock budget), mirroring the base DSAgent.

All rubric-store operations go through TOOL CALLS (not raw file writes): schema
validation on every mutation + a structured op-log (rubric lifetime / fire count
/ overturn rate = the paper's rubric-dynamics signal). Store ops append to
rubric_oplog.jsonl (kept separate so a store rewrite can't lose it).
"""
from __future__ import annotations

import json
import time as _time
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Optional

from pydantic import BaseModel, Field

from openhands.sdk import Agent, LocalConversation, LocalWorkspace, register_tool
from openhands.sdk.llm import LLM
from openhands.sdk.llm.message import Message, TextContent
from openhands.sdk.event import Event, MessageEvent
from openhands.sdk.tool import (
    Action, Observation, ToolAnnotations, ToolDefinition, ToolExecutor,
)


# Filled per-invocation (single reviewer runs at a time in its instance); the
# tool executors read/write it. Mirrors submission_selector._SELECTION.
_REVIEW: dict[str, Any] = {}

# The active reviewer's budget tracker, set by _run_reviewer_conversation before
# the loop. Module-global (not a closure) so ReviewerAgent can live at module
# scope — a local class has a "<locals>" qualname and the SDK's ConversationState
# refuses to serialize it when persistence is enabled (per-attempt resume).
_ACTIVE_BUDGET: "ReviewerBudgetTracker | None" = None


# =====================================================================
# Rubric store — schema + on-disk backend
# =====================================================================

STORE_FILE = "rubric_store.json"
OPLOG_FILE = "rubric_oplog.jsonl"
CHECKS_SUBDIR = "checks"

_VALID_KINDS = {"executable", "nl"}
_VALID_RESULTS = {"pass", "warn", "fail"}
_VALID_STATUS = {"active", "retired"}

# Unified PRM scoring thresholds (paper §sec:axes). The reviewer scores every
# rubric it assesses on TWO 0-1 axes — importance_score (how relevant the rubric
# is to the trajectory as it stands) and p_agent_success (probability the agent
# satisfied it) — EXACTLY the interface the inline / inter-attempt PRM emits
# (prm_agent.py + prm_feedback.py). A rubric is a surfaced WEAKNESS iff
# importance >= _MIN_IMPORTANCE AND p_agent_success < _FAILURE_THRESHOLD; the
# injected feedback + the finish-gate are both driven by that single rule (so the
# reviewer's output is byte-compatible with the other PRM mechanisms). Overridable
# per-run via _REVIEW["min_importance"] / _REVIEW["failure_threshold"].
_MIN_IMPORTANCE = 0.5
_FAILURE_THRESHOLD = 0.6


def _thresholds() -> tuple[float, float]:
    """(min_importance, failure_threshold) for THIS wake — config overrides, else
    the module defaults. Read by the verdict/forced-verdict rendering + gate."""
    mi = _REVIEW.get("min_importance")
    ft = _REVIEW.get("failure_threshold")
    return (float(mi) if mi is not None else _MIN_IMPORTANCE,
            float(ft) if ft is not None else _FAILURE_THRESHOLD)


def _derive_status(rubric_scores: dict[str, dict]) -> str:
    """Derive the SOUNDNESS-style STATUS the finish-gate consumes purely from the
    thresholded scores (confirmed design): FAIL if ANY assessed rubric is a
    surfaced weakness (importance >= min_importance AND p_success < failure_thr),
    else PASS. No separate soundness call — gating falls out of the same thresholds
    the injected feedback uses."""
    min_imp, fail_thr = _thresholds()
    for s in (rubric_scores or {}).values():
        try:
            imp = float(s.get("importance_score", 0.0))
            p = float(s.get("p_agent_success", 1.0))
        except (TypeError, ValueError):
            continue
        if imp >= min_imp and p < fail_thr:
            return "FAIL"
    return "PASS"


def _store_path(store_dir: Path) -> Path:
    return store_dir / STORE_FILE


def _oplog_path(store_dir: Path) -> Path:
    return store_dir / OPLOG_FILE


def _load_store(store_dir: Path) -> list[dict]:
    """Read the rubric store array. Re-seeds on a corrupt/missing file so the
    reviewer always has the 3 seed checks (never crashes the base attempt)."""
    p = _store_path(store_dir)
    try:
        data = json.loads(p.read_text())
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError, FileNotFoundError):
        pass
    # Corrupt or missing -> re-seed.
    _ensure_store(store_dir, force=True)
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError, FileNotFoundError):
        return []


def _save_store(store_dir: Path, rubrics: list[dict]) -> None:
    """Atomic-ish read-modify-write with a .bak (attempts are sequential per
    task, so there is no concurrent writer)."""
    p = _store_path(store_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        try:
            (store_dir / (STORE_FILE + ".bak")).write_text(p.read_text())
        except OSError:
            pass
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(rubrics, indent=2))
    tmp.replace(p)


def _append_oplog(store_dir: Path, op: dict) -> None:
    """Append one line to the op-log (durable telemetry, separate from store)."""
    try:
        rec = {"ts": _time.time(), **op}
        with open(_oplog_path(store_dir), "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass
    _REVIEW.setdefault("ops", []).append(op)


def _next_id(rubrics: list[dict]) -> str:
    n = 0
    for r in rubrics:
        rid = str(r.get("id", ""))
        if rid.startswith("r") and rid[1:].isdigit():
            n = max(n, int(rid[1:]))
    return f"r{n + 1}"


# The 3 warm-start seed rubrics. These are pure
# NATURAL-LANGUAGE specifications: WHAT to verify + WHY it matters + what a
# failure looks like. The reviewer writes BESPOKE code contextual to each task
# (its columns, metric, transform) rather than running fixed scripts — the
# fixed-script model was brittle (wrong thresholds, wrong arg templating). Any
# script the reviewer writes lives under checks/ and persists across attempts
# (the store dir is its :rw memory); checks/README.md is its cross-attempt index.
_SEED_RUBRICS = [
    {
        "id": "r1",
        "name": "cv_reproduce",
        "description": (
            "Reproduce the reported CV: recompute the competition metric from the "
            "archived out-of-fold (OOF) predictions + the true train.csv target; it "
            "must match cv_scores.json within tolerance. A reported CV is a CLAIM "
            "until you reproduce it. Foundation for trusting everything else — if "
            "you cannot reproduce it, the number is not evidence."
        ),
        "origin": "seed",
        "created_at_submission": 0,
        "created_at_attempt": 0,
        "status": "active",
        "history": [],
    },
    {
        "id": "r2",
        "name": "submission_vs_oof_consistency",
        "description": (
            "THE money check for CV-invisible bugs. The submitted prediction column "
            "must be distributionally consistent with the OOF predictions the CV was "
            "computed on (mean/std/min/max/nunique). A divergence between submission "
            "and OOF — scale shift, a missing inverse-transform, rounding a "
            "continuous target to discrete values — is the failure the reported CV "
            "CANNOT see, because CV is computed on the OOF preds, not the submission. "
            "Compare submission vs OOF FIRST (the direct test); then sanity-check OOF "
            "vs the train target. If submission and OOF agree but both differ from "
            "the target, that is a model-quality signal, not a pipeline bug."
        ),
        "origin": "seed",
        "created_at_submission": 0,
        "created_at_attempt": 0,
        "status": "active",
        "history": [],
    },
    {
        "id": "r3",
        "name": "submission_coverage",
        "description": (
            "The submitted file must have exactly the required rows, all test IDs "
            "present exactly once (no dup/missing), columns matching the sample "
            "submission header, and no NaN/inf in the prediction column. Catches "
            "invalid-submission collapses and malformed files that score zero."
        ),
        "origin": "seed",
        "created_at_submission": 0,
        "created_at_attempt": 0,
        "status": "active",
        "history": [],
    },
    # --- Modeling seeds (M1/M2/M3'+3): execution-grounded checks that a
    # code-reading PRM structurally cannot run. Chosen because they recur across
    # BOTH the no-PRM and static-PRM runs and are runnable from artifacts that
    # exist on disk.
    {
        "id": "r4",
        "name": "submission_distribution_health",
        "description": (
            "Compare the submitted prediction column against the train target "
            "using ONLY submission.csv + train.csv (no OOF, no retraining — ALWAYS "
            "runnable). REGRESSION: dispersion ratio std(pred)/std(target) — FAIL "
            "if < ~0.5 (near-constant collapse / broken-scale transform), WARN if "
            "< ~0.85 (under-dispersion, likely underfit); plus mean offset "
            "|mean(pred)-mean(target)|/std(target). A collapse or scale-shift is "
            "OFTEN a missing/broken INVERSE-TRANSFORM on a log-modeled target — "
            "check the round-trip (e.g. a log1p target that was never expm1'd back). "
            "CLASSIFICATION/PROBABILITY: mean(pred) or positive-class rate vs the "
            "train base rate; minority-class share vs prior; class collapse; "
            "probability hygiene for logloss/AUC (unclipped near-0/1 probs; "
            "degenerate nunique causing tie-induced ranking loss). A hard collapse "
            "is a soundness FAIL; a milder mismatch is a modeling SUGGESTION. The "
            "reported CV is BLIND to this because CV is computed on OOF, not the "
            "submission file."
        ),
        "origin": "seed",
        "created_at_submission": 0,
        "created_at_attempt": 0,
        "status": "active",
        "history": [],
    },
    {
        "id": "r5",
        "name": "cv_selection_reliability",
        "description": (
            "Audit the MODEL-SELECTION decision from the per-fold CV numbers the "
            "agent ALREADY reported (cv_scores.json / final_selection.json / step "
            "logs) — no retraining. FAIL/WARN when: (a) the winning model's CV edge "
            "over the runner-up is SMALLER than the fold-to-fold std (selection is "
            "noise-driven — e.g. 'gap 0.00002 < fold std 0.000756'); (b) per-fold "
            "CV coefficient of variation is high (the point estimate is "
            "untrustworthy, e.g. AUC 0.893 ± 0.028); (c) a fold metric jumps "
            "implausibly vs its siblings (leakage signature, e.g. an accuracy fold "
            "at 0.985 among 0.86s). SUGGESTION: prefer the simpler / more robust "
            "candidate, use repeated CV, do not chase a sub-noise gap."
        ),
        "origin": "seed",
        "created_at_submission": 0,
        "created_at_attempt": 0,
        "status": "active",
        "history": [],
    },
    {
        "id": "r6",
        "name": "submission_ledger_and_regression_guard",
        "description": (
            "Maintain a persistent ledger at /reviewer/submission_ledger.jsonl "
            "(your writable memory persists across all 5 attempts; the raw prior "
            "submission CSVs are NOT mounted, so this ledger IS your history). On "
            "every wake append ONE line for the submission under review: submission "
            "index, its md5, your REPRODUCED CV/metric (or best estimate), key "
            "distribution stats (dispersion ratio, mean/base-rate), and your "
            "verdict. BEFORE verdicting, read the ledger: if the current submission "
            "is WORSE on your reproduced metric than an earlier one you already "
            "logged this rollout, that is a best-submission regression — FAIL/WARN "
            "with a SUGGESTION to revert to the better earlier submission. Also "
            "flag a byte-identical resubmit (same md5 as a prior entry — wasted "
            "budget)."
        ),
        "origin": "seed",
        "created_at_submission": 0,
        "created_at_attempt": 0,
        "status": "active",
        "history": [],
    },
    {
        "id": "r7",
        "name": "train_test_drift",
        "description": (
            "Adversarial validation — runnable TODAY from train.csv + test.csv, no "
            "OOF needed. Label train rows 0 and test rows 1, train a quick "
            "classifier to separate them, and read the CV AUC. AUC ≈ 0.5 means "
            "train and test are drawn alike (CV should track the leaderboard); AUC "
            "≫ 0.5 (say > ~0.7) means the feature space DRIFTS between train and "
            "test, so the agent's CV will NOT reflect the LB — WARN, and report the "
            "most drift-inducing features (top adversarial importances) as a "
            "SUGGESTION (drop/normalize them, or use a validation split that mimics "
            "the test distribution). Explains high-CV / low-LB gaps (e.g. titanic "
            "CV 0.88 landing bottom-half)."
        ),
        "origin": "seed",
        "created_at_submission": 0,
        "created_at_attempt": 0,
        "status": "active",
        "history": [],
    },
    {
        "id": "r8",
        "name": "residual_vs_feature",
        "description": (
            "Needs OOF (oof_predictions.csv) — skip with a WARN if absent. Compute "
            "residuals (regression: target - oof_pred; classification: target - "
            "oof_prob) and look for structure a code-reader cannot see: correlate "
            "residuals against each feature and against the prediction itself, and "
            "slice mean |residual| by segments of high-signal features. If a "
            "feature or segment carries large systematic residuals, the model is "
            "UNDERUSING signal there — SUGGESTION: add an interaction / nonlinearity "
            "/ target-encoding for that feature, or a segment-specific model. This "
            "is execution-grounded modeling advice (the static PRM cannot compute "
            "residuals)."
        ),
        "origin": "seed",
        "created_at_submission": 0,
        "created_at_attempt": 0,
        "status": "active",
        "history": [],
    },
    {
        "id": "r9",
        "name": "single_feature_leakage",
        "description": (
            "Detect a too-good-to-be-true predictor. For each feature, compute a "
            "cheap univariate association with the target on train.csv (e.g. "
            "single-feature AUC / correlation / a 1-split stub). If ONE raw feature "
            "alone nearly reproduces the full model's CV (or a categorical maps "
            "near-perfectly to the target), that feature is a leakage / "
            "identifier / target-echo suspect — WARN with a SUGGESTION to verify it "
            "is legitimately available at test time and is not an index or a "
            "post-outcome artifact. Distinct from good signal: leakage is "
            "IMPLAUSIBLY strong for a single raw column."
        ),
        "origin": "seed",
        "created_at_submission": 0,
        "created_at_attempt": 0,
        "status": "active",
        "history": [],
    },
]

# Seed contents for checks/README.md — the reviewer's durable, cross-attempt
# index of the bespoke check scripts it writes (message history resets per
# attempt, so this file is how a later wake knows what scripts already exist).
_CHECKS_README_SEED = """\
# Reviewer check scripts — cross-attempt index

This directory holds check scripts YOU (the reviewer) write. It persists across
all attempts on this task; your message history does NOT. So this README is your
memory of what already exists.

CONVENTION — whenever you write a new check script here, append a row to the
table below (script filename, the rubric it serves, one-line purpose, and the
EXACT command to run it). Before writing a new script, read this table and reuse
an existing script if one already fits — only write fresh if none does or the
pipeline changed.

| script | rubric | purpose | run command |
|--------|--------|---------|-------------|
| (none yet) | | | |
"""

# STATIC-operationalization seed: no terminal, so there are no scripts to index;
# the README is instead the reviewer's cross-attempt notes on what it concluded.
_CHECKS_README_SEED_STATIC = """\
# Reviewer notes — cross-attempt index

This file persists across all attempts on this task; your message history does
NOT. You have no terminal (you read artifacts + code and reason), so this is your
memory of what earlier wakes CONCLUDED.

CONVENTION — after you assess a rubric, append a row to the table below (the
rubric, which files you read, and the one-line finding). Read this table first so
you do not re-derive what an earlier wake already established.

| rubric | files read | finding |
|--------|------------|---------|
| (none yet) | | |
"""



def _seeds_from_rubric_set(
    rubric_set: str,
    exclude_ids: Sequence[str] | None = None,
) -> list[dict]:
    """Build reviewer store seeds from a rubric-set JSON (the launch-selected set,
    e.g. the bottom-up "discovered" unified set). Each RubricDef becomes an NL-spec
    seed: the neutral good-practice description + the label-language instruction
    (what success vs failure looks like) + the advisory grounding hint, so the
    reviewer knows WHAT to verify without shipping fixed check scripts. This is how
    the SAME discovered rubric content drives both the static PRM and the dynamic
    reviewer — the two operationalizations of one rubric set (paper §sec:axes)."""
    from dsgym.adapra.rubrics import resolve_rubric_set
    excluded = set(exclude_ids or [])
    rubrics = [
        r for r in resolve_rubric_set(rubric_set=rubric_set)
        if r.rubric_id not in excluded
    ]
    seeds: list[dict] = []
    for r in rubrics:
        desc = (getattr(r, "display_description", "") or r.description or "").strip()
        instr = (r.natural_language_instruction or "").strip()
        body = desc
        if instr:
            body = (body + "\n\n" + instr).strip()
        g = (getattr(r, "grounding", "") or "").strip()
        if g == "artifact":
            body += ("\n\n[grounding: usually checkable from the WORKSPACE artifacts "
                     "(submission, code, CV registries, cached preds) — recommendation, "
                     "not a restriction.]")
        elif g == "trajectory":
            body += ("\n\n[grounding: usually about the PROCESS over turns (how the "
                     "agent worked) — recommendation, not a restriction.]")
        seeds.append({
            "id": r.rubric_id,
            "name": r.rubric_id,
            "description": body,
            "origin": "seed",
            "created_at_submission": 0,
            "created_at_attempt": 0,
            "status": "active",
            "history": [],
        })
    return seeds


def _ensure_store(store_dir: Path, force: bool = False,
                  seeds: Optional[list[dict]] = None, static: bool = False) -> None:
    """Create the rubric store (NL seeds) + the checks/ dir + README index if
    absent (or on force). Persists across attempts, so this is a no-op after the
    first attempt on a task. The reviewer writes bespoke check scripts into
    checks/ itself — we no longer ship fixed seed scripts."""
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / CHECKS_SUBDIR).mkdir(parents=True, exist_ok=True)
    p = _store_path(store_dir)
    if force or not p.exists():
        seed_rubrics = seeds if seeds is not None else _SEED_RUBRICS
        _save_store(store_dir, [dict(r) for r in seed_rubrics])
        _append_oplog(store_dir, {"op": "seed", "n_rubrics": len(seed_rubrics),
                                  "source": "rubric_set" if seeds is not None else "builtin"})
    # Seed the cross-attempt script index (self-heals if missing).
    readme = store_dir / CHECKS_SUBDIR / "README.md"
    if force or not readme.exists():
        readme.write_text(_CHECKS_README_SEED_STATIC if static else _CHECKS_README_SEED)


def _validate_rubric(store_dir: Path, name: str, description: str,
                     origin: str) -> tuple[bool, str]:
    """Validate an add_rubric request. Rubrics are NL specs, so all that's
    required is a name + description. The RUN-ONCE GATE still applies: the
    reviewer must have recorded a matching check via record_check_result in THIS
    wake before persisting it — a rubric enters the store only after the reviewer
    has actually written+run a check for it (no hallucinated specs), and the
    bespoke script it wrote lives under checks/ + is indexed in checks/README.md."""
    if not name or not name.strip():
        return False, "name is required"
    if not description or not description.strip():
        return False, "description is required"
    ran = {c.get("name") for c in _REVIEW.get("recorded", [])}
    if name not in ran:
        return False, (f"run-once gate: run a check for this and log it with "
                       f"record_check_result(name='{name}', ...) BEFORE adding it "
                       f"to the store (no unverified rubrics)")
    return True, ""


# =====================================================================
# Seed check scripts (shipped into /reviewer/checks/) — small argparse CLIs
# that print stats + a final "RESULT: PASS|WARN|FAIL — <reason>" line.
# =====================================================================

_SEED_SCRIPT_CV_REPRODUCE = '''\
#!/usr/bin/env python3
"""Seed check: reproduce the reported CV from OOF preds + train target.

Recomputes the competition metric from an out-of-fold prediction file and the
true train target, and compares it to the reported cv_scores.json. A mismatch
means the reported CV is not what the code actually produced.

Usage:
  python cv_reproduce.py --oof <oof.csv> --train <train.csv> --target <col> \\
      --metric <mae|rmse|rmsle|logloss|auc|accuracy> [--reported <float>] \\
      [--id-col <col>] [--pred-col <col>] [--tol 0.02]
This is a STARTING POINT — the reviewer should adapt paths/columns per task.
"""
import argparse, sys, math

def _load_csv(path):
    """Load a table as a list of {col: str} row dicts. Handles .csv and
    .parquet (OOF preds are often parquet); values are stringified so the
    rest of the script (which floats them) is format-agnostic."""
    if str(path).endswith((".parquet", ".pq")):
        import pandas as pd
        df = pd.read_parquet(path)
        return [{k: ("" if v is None else str(v)) for k, v in r.items()}
                for r in df.to_dict(orient="records")]
    import csv
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows

def _metric(name, y, p):
    n = len(y)
    if name == "mae":
        return sum(abs(a - b) for a, b in zip(y, p)) / n
    if name == "rmse":
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(y, p)) / n)
    if name == "rmsle":
        return math.sqrt(sum((math.log1p(max(a, 0)) - math.log1p(max(b, 0))) ** 2
                             for a, b in zip(y, p)) / n)
    if name == "logloss":
        eps = 1e-15
        return -sum(a * math.log(min(max(b, eps), 1 - eps))
                    + (1 - a) * math.log(1 - min(max(b, eps), 1 - eps))
                    for a, b in zip(y, p)) / n
    if name == "accuracy":
        return sum(1 for a, b in zip(y, p) if round(b) == a) / n
    raise SystemExit(f"metric {name} not implemented in seed; adapt the script")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--metric", required=True)
    ap.add_argument("--reported", type=float, default=None)
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--pred-col", default=None)
    ap.add_argument("--tol", type=float, default=0.02)
    a = ap.parse_args()

    oof = _load_csv(a.oof)
    train = _load_csv(a.train)
    pred_col = a.pred_col or [c for c in oof[0].keys() if c != a.id_col][-1]

    if a.id_col:
        tgt = {r[a.id_col]: float(r[a.target]) for r in train}
        y, p = [], []
        for r in oof:
            k = r.get(a.id_col)
            if k in tgt:
                y.append(tgt[k]); p.append(float(r[pred_col]))
    else:
        # positional align (assumes oof rows correspond to train rows)
        y = [float(r[a.target]) for r in train][:len(oof)]
        p = [float(r[pred_col]) for r in oof][:len(y)]

    if not y:
        print("RESULT: WARN — could not align OOF to train (check --id-col/--target)")
        return
    got = _metric(a.metric, y, p)
    print(f"reproduced_{a.metric}={got:.6f}  n={len(y)}  pred_col={pred_col}")
    if a.reported is not None:
        rel = abs(got - a.reported) / (abs(a.reported) + 1e-12)
        print(f"reported={a.reported:.6f}  rel_diff={rel:.4f}  tol={a.tol}")
        if rel <= a.tol:
            print(f"RESULT: PASS — reproduced CV matches reported ({got:.4f} vs {a.reported:.4f})")
        else:
            print(f"RESULT: FAIL — reproduced CV {got:.4f} != reported {a.reported:.4f} "
                  f"(rel {rel:.1%} > tol {a.tol:.1%}); reported CV is not what the code produced")
    else:
        print(f"RESULT: WARN — no --reported given; computed {a.metric}={got:.4f} for comparison")

if __name__ == "__main__":
    main()
'''

_SEED_SCRIPT_SUB_VS_TARGET = '''\
#!/usr/bin/env python3
"""Seed check: submitted predictions' distribution vs the train target.

Compares {mean,std,min,max,nunique} of the submitted prediction column against
the train target. Catches the CV-invisible failures: continuous target rounded
to few discrete values (s3e16), wrong inverse-transform / scale shift (s4e12),
predictions on the wrong range.

Usage:
  python submission_vs_target_dist.py --submission <sub.csv> --train <train.csv> \\
      --target <col> [--pred-col <col>] [--kind regression|proba] \\
      [--mean-tol 0.5] [--nunique-frac 0.5]
Adapt --pred-col/--target/--kind per task (read cv_scores.json / the metric).
"""
import argparse, csv, math

def _col(path, name):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    vals = []
    for r in rows:
        try:
            vals.append(float(r[name]))
        except (ValueError, KeyError, TypeError):
            pass
    return vals

def _stats(v):
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / n) if n > 1 else 0.0
    return {"n": n, "mean": m, "std": sd, "min": min(v), "max": max(v),
            "nunique": len(set(v))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--pred-col", default=None)
    ap.add_argument("--kind", default="regression", choices=["regression", "proba"])
    ap.add_argument("--mean-tol", type=float, default=0.5,
                    help="allowed |sub_mean-train_mean| in units of train std")
    ap.add_argument("--nunique-frac", type=float, default=0.5,
                    help="min sub_nunique/train_nunique for a continuous target")
    a = ap.parse_args()

    with open(a.submission) as f:
        header = f.readline().strip().split(",")
    pred_col = a.pred_col or header[-1]
    sub = _col(a.submission, pred_col)
    tgt = _col(a.train, a.target)
    if not sub or not tgt:
        print(f"RESULT: WARN — empty column (sub[{pred_col}]={len(sub)} tgt[{a.target}]={len(tgt)})")
        return
    s, t = _stats(sub), _stats(tgt)
    print(f"submission[{pred_col}]: {s}")
    print(f"train_target[{a.target}]: {t}")

    fails = []
    if a.kind == "proba":
        if s["min"] < -1e-6 or s["max"] > 1 + 1e-6:
            fails.append(f"proba out of [0,1]: min={s['min']:.4f} max={s['max']:.4f}")
        if s["nunique"] <= 1:
            fails.append("all predictions identical")
    else:
        mean_gap = abs(s["mean"] - t["mean"]) / (t["std"] + 1e-12)
        if mean_gap > a.mean_tol:
            fails.append(f"mean off by {mean_gap:.2f} train-std "
                         f"(sub {s['mean']:.3f} vs train {t['mean']:.3f}) — scale/transform bug?")
        # continuous target collapse (rounding). Compare unique-FRACTIONS
        # (nunique/n), not raw counts — the submission and train are different
        # sizes, and a continuous target has nunique/n ~ 1 at any size.
        t_frac = t["nunique"] / max(t["n"], 1)
        s_frac = s["nunique"] / max(s["n"], 1)
        if t_frac > 0.5 and s_frac < a.nunique_frac * t_frac:
            fails.append(f"nunique collapse: sub={s['nunique']}/{s['n']} "
                         f"({s_frac:.2f} unique) vs train {t_frac:.2f} unique "
                         f"— continuous target rounded to discrete values?")
        if s["max"] > t["max"] * 5 or s["min"] < t["min"] - 5 * (t["std"] + 1e-12):
            fails.append(f"range far outside train "
                         f"(sub [{s['min']:.2f},{s['max']:.2f}] vs train [{t['min']:.2f},{t['max']:.2f}])")
    if fails:
        print("RESULT: FAIL — " + "; ".join(fails))
    else:
        print("RESULT: PASS — submitted distribution consistent with train target")

if __name__ == "__main__":
    main()
'''

_SEED_SCRIPT_COVERAGE = '''\
#!/usr/bin/env python3
"""Seed check: submission coverage / schema integrity.

Verifies the submitted file has exactly the sample-submission rows, all IDs
present once (no dup/missing), matching columns, and no NaN/inf in predictions.

Usage:
  python submission_coverage.py --submission <sub.csv> --sample <sample_submission.csv> \\
      [--id-col <col>]
"""
import argparse, csv, math

def _read(path):
    with open(path) as f:
        r = csv.DictReader(f)
        rows = list(r)
        return r.fieldnames, rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--id-col", default=None)
    a = ap.parse_args()

    sub_cols, sub = _read(a.submission)
    smp_cols, smp = _read(a.sample)
    id_col = a.id_col or (smp_cols[0] if smp_cols else None)
    print(f"submission: {len(sub)} rows, cols={sub_cols}")
    print(f"sample:     {len(smp)} rows, cols={smp_cols}")

    fails = []
    if sub_cols != smp_cols:
        fails.append(f"columns mismatch: sub={sub_cols} vs sample={smp_cols}")
    if len(sub) != len(smp):
        fails.append(f"row count {len(sub)} != required {len(smp)}")
    if id_col and id_col in (sub_cols or []) and id_col in (smp_cols or []):
        sub_ids = [r[id_col] for r in sub]
        want = set(r[id_col] for r in smp)
        got = set(sub_ids)
        if len(sub_ids) != len(got):
            fails.append(f"duplicate ids: {len(sub_ids)-len(got)} dupes")
        missing = want - got
        extra = got - want
        if missing:
            fails.append(f"{len(missing)} missing ids (e.g. {list(missing)[:3]})")
        if extra:
            fails.append(f"{len(extra)} unexpected ids (e.g. {list(extra)[:3]})")
    # NaN/inf in non-id columns
    pred_cols = [c for c in (sub_cols or []) if c != id_col]
    bad = 0
    for r in sub:
        for c in pred_cols:
            try:
                v = float(r[c])
                if math.isnan(v) or math.isinf(v):
                    bad += 1
            except (ValueError, TypeError, KeyError):
                bad += 1
    if bad:
        fails.append(f"{bad} NaN/inf/non-numeric prediction cells")

    if fails:
        print("RESULT: FAIL — " + "; ".join(fails))
    else:
        print("RESULT: PASS — coverage/schema OK")

if __name__ == "__main__":
    main()
'''

_SEED_SCRIPTS = {
    "cv_reproduce.py": _SEED_SCRIPT_CV_REPRODUCE,
    "submission_vs_target_dist.py": _SEED_SCRIPT_SUB_VS_TARGET,
    "submission_coverage.py": _SEED_SCRIPT_COVERAGE,
}


# =====================================================================
# Rubric-store TOOLS (one Action/Observation/Executor/ToolDefinition quad each)
# =====================================================================

def _store_dir() -> Path:
    return Path(_REVIEW["store_dir"])


def _sub_idx() -> int:
    return int(_REVIEW.get("submission_index", 0))


def _attempt_idx() -> int:
    return int(_REVIEW.get("attempt_index", 0))


class _RubricObs(Observation):
    @property
    def visualize(self):
        from rich.text import Text
        return Text(self.text)


# --- list_rubrics --------------------------------------------------------
class ListRubricsAction(Action):
    status: str = Field(default="active",
                        description="Filter: 'active' (default), 'retired', or 'all'.")


class ListRubricsExecutor(ToolExecutor):
    def __call__(self, action: ListRubricsAction, conversation=None):
        rubrics = _load_store(_store_dir())
        want = action.status or "active"
        rows = [r for r in rubrics if want == "all" or r.get("status") == want]
        if not rows:
            return _RubricObs.from_text(text=f"(no rubrics with status={want})")
        lines = []
        for r in rows:
            hist = r.get("history", [])
            last = hist[-1] if hist else None
            fires = f"{len(hist)} fires" + (f", last={last['result']}" if last else "")
            lines.append(
                f"[{r['id']}] {r['name']} ({r.get('status')}, {fires})\n"
                f"    {r.get('description','')}\n"
                f"    origin={r.get('origin','-')}")
        return _RubricObs.from_text(text="CURRENT RUBRIC STORE:\n" + "\n".join(lines))


class ListRubricsTool(ToolDefinition[ListRubricsAction, _RubricObs]):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["ListRubricsTool"]:
        return [cls(
            action_type=ListRubricsAction, observation_type=_RubricObs,
            description=("List the checks in your persistent rubric store (persists "
                         "across all attempts on this task). Start here to see your "
                         "3 seed checks + any you added previously."),
            executor=ListRubricsExecutor(),
            annotations=ToolAnnotations(title="list_rubrics", readOnlyHint=True,
                                        destructiveHint=False, idempotentHint=True,
                                        openWorldHint=False),
        )]


# --- add_rubric ----------------------------------------------------------
class AddRubricAction(Action):
    name: str = Field(description="Short snake_case name, e.g. age_predictions_continuous.")
    description: str = Field(description="The NL spec: WHAT this check verifies, WHY it "
                             "matters, and what a failure looks like. The reviewer writes "
                             "bespoke code for it per task — do NOT hardcode column names "
                             "or thresholds here; describe the invariant.")
    origin: str = Field(default="discovered",
                        description="'discovered' or 'specialized_from:<seed_id>'.")


class AddRubricExecutor(ToolExecutor):
    def __call__(self, action: AddRubricAction, conversation=None):
        sd = _store_dir()
        ok, err = _validate_rubric(sd, action.name, action.description, action.origin)
        if not ok:
            return _RubricObs.from_text(text=f"add_rubric rejected: {err}. Fix and retry.")
        rubrics = _load_store(sd)
        if any(r.get("name") == action.name and r.get("status") == "active"
               for r in rubrics):
            return _RubricObs.from_text(
                text=f"a rubric named '{action.name}' is already active — use "
                     f"update_rubric to change it instead.")
        rid = _next_id(rubrics)
        entry = {
            "id": rid, "name": action.name,
            "description": action.description,
            "origin": action.origin or "discovered",
            "created_at_submission": _sub_idx(),
            "created_at_attempt": _attempt_idx(),
            "status": "active", "history": [],
        }
        rubrics.append(entry)
        _save_store(sd, rubrics)
        _append_oplog(sd, {"op": "add", "id": rid, "name": action.name,
                           "origin": entry["origin"],
                           "submission": _sub_idx(), "attempt": _attempt_idx()})
        return _RubricObs.from_text(
            text=f"Added rubric {rid} ({action.name}). If you wrote a script for it, "
                 f"make sure it's indexed in checks/README.md.")


class AddRubricTool(ToolDefinition[AddRubricAction, _RubricObs]):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["AddRubricTool"]:
        return [cls(
            action_type=AddRubricAction, observation_type=_RubricObs,
            description=("Add a reusable NL-spec check to the store so future "
                         "submissions on this task are guarded. Describe the invariant "
                         "(not task-specific columns). You MUST have already written + "
                         "run + recorded a check for it this wake (run-once gate)."),
            executor=AddRubricExecutor(),
            annotations=ToolAnnotations(title="add_rubric", readOnlyHint=False,
                                        destructiveHint=False, idempotentHint=False,
                                        openWorldHint=False),
        )]


# --- update_rubric -------------------------------------------------------
class UpdateRubricAction(Action):
    id: str = Field(description="The rubric id to edit, e.g. r4.")
    description: str = Field(description="New NL spec for the rubric (refine what it "
                             "verifies / why / what a failure looks like).")


class UpdateRubricExecutor(ToolExecutor):
    def __call__(self, action: UpdateRubricAction, conversation=None):
        sd = _store_dir()
        rubrics = _load_store(sd)
        r = next((x for x in rubrics if x.get("id") == action.id), None)
        if r is None:
            return _RubricObs.from_text(text=f"no rubric with id {action.id}. Retry.")
        if not action.description or not action.description.strip():
            return _RubricObs.from_text(text="nothing to update (pass a new description).")
        r["description"] = action.description
        _save_store(sd, rubrics)
        _append_oplog(sd, {"op": "update", "id": action.id, "changed": ["description"],
                           "submission": _sub_idx(), "attempt": _attempt_idx()})
        return _RubricObs.from_text(text=f"Updated rubric {action.id} (description).")


class UpdateRubricTool(ToolDefinition[UpdateRubricAction, _RubricObs]):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["UpdateRubricTool"]:
        return [cls(
            action_type=UpdateRubricAction, observation_type=_RubricObs,
            description="Refine an existing rubric's NL spec (e.g. sharpen the invariant "
                        "or note a task-specific nuance). If the check script changed, "
                        "update checks/README.md instead — scripts live there, not here.",
            executor=UpdateRubricExecutor(),
            annotations=ToolAnnotations(title="update_rubric", readOnlyHint=False,
                                        destructiveHint=False, idempotentHint=True,
                                        openWorldHint=False),
        )]


# --- retire_rubric -------------------------------------------------------
class RetireRubricAction(Action):
    id: str = Field(description="The rubric id to retire.")
    reason: str = Field(description="Why it's being retired (required; kept in the log).")


class RetireRubricExecutor(ToolExecutor):
    def __call__(self, action: RetireRubricAction, conversation=None):
        sd = _store_dir()
        if not action.reason or not action.reason.strip():
            return _RubricObs.from_text(text="retire_rubric requires a reason. Retry.")
        rubrics = _load_store(sd)
        r = next((x for x in rubrics if x.get("id") == action.id), None)
        if r is None:
            return _RubricObs.from_text(text=f"no rubric with id {action.id}. Retry.")
        r["status"] = "retired"
        r["retire_reason"] = action.reason
        _save_store(sd, rubrics)
        _append_oplog(sd, {"op": "retire", "id": action.id, "reason": action.reason,
                           "submission": _sub_idx(), "attempt": _attempt_idx()})
        return _RubricObs.from_text(text=f"Retired rubric {action.id} (kept in log).")


class RetireRubricTool(ToolDefinition[RetireRubricAction, _RubricObs]):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["RetireRubricTool"]:
        return [cls(
            action_type=RetireRubricAction, observation_type=_RubricObs,
            description=("Retire a check that never discriminates (sets status:retired; "
                         "never hard-deleted — the op log is preserved for analysis)."),
            executor=RetireRubricExecutor(),
            annotations=ToolAnnotations(title="retire_rubric", readOnlyHint=False,
                                        destructiveHint=True, idempotentHint=True,
                                        openWorldHint=False),
        )]


# --- record_check_result -------------------------------------------------
class RecordCheckResultAction(Action):
    id: str = Field(default="",
                    description="Rubric id whose result you're recording (for a store "
                    "rubric). Leave blank if recording a not-yet-added check by name.")
    name: str = Field(default="",
                      description="Check name (required if id is blank — this satisfies "
                      "the run-once gate before add_rubric).")
    importance_score: float = Field(default=-1.0,
                      description="0.0-1.0 (REQUIRED): how RELEVANT this rubric is to the "
                      "trajectory AS IT STANDS. A rubric for a stage the agent has not "
                      "reached gets LOW importance (do NOT penalize it) — never a low "
                      "p_agent_success.")
    p_agent_success: float = Field(default=-1.0,
                      description="0.0-1.0 (REQUIRED): probability the agent SATISFIED "
                      "this rubric's good behavior. DYNAMIC op — this MUST be GROUNDED in "
                      "the check you just ran (the measured numbers below). STATIC op — "
                      "you ASSIGN it by reasoning over what you read. 0.5 = maximum "
                      "uncertainty / the rubric is not yet reachable.")
    result: str = Field(default="",
                      description="Optional legacy label pass|warn|fail. If omitted it is "
                      "derived from p_agent_success (< failure_threshold → fail). Kept for "
                      "the run-once add_rubric gate + telemetry.")
    evidence: str = Field(description="The numbers you actually computed (dynamic), or the "
                          "concrete artifact facts you read (static), e.g. "
                          "'nunique=16 (train target nunique=49368)'.")


class RecordCheckResultExecutor(ToolExecutor):
    def __call__(self, action: RecordCheckResultAction, conversation=None):
        sd = _store_dir()
        imp, p = action.importance_score, action.p_agent_success
        if imp is None or imp < 0.0 or imp > 1.0:
            return _RubricObs.from_text(
                text="importance_score is REQUIRED in [0.0, 1.0] (how relevant this "
                     "rubric is to the trajectory as it stands). Retry.")
        if p is None or p < 0.0 or p > 1.0:
            return _RubricObs.from_text(
                text="p_agent_success is REQUIRED in [0.0, 1.0] (probability the agent "
                     "satisfied this rubric). Retry.")
        if not action.evidence or not action.evidence.strip():
            return _RubricObs.from_text(text="evidence is required (the numbers you "
                                             "computed, or — in static mode — the "
                                             "artifact facts you read). Retry.")
        _min_imp, fail_thr = _thresholds()
        result = (action.result or "").strip().lower()
        if result and result not in _VALID_RESULTS:
            return _RubricObs.from_text(
                text=f"result, if given, must be one of {sorted(_VALID_RESULTS)}. Retry.")
        if not result:
            result = "fail" if p < fail_thr else "pass"
        fire = {"submission": _sub_idx(), "attempt": _attempt_idx(),
                "result": result, "importance_score": imp, "p_agent_success": p,
                "evidence": action.evidence}
        # Track for the run-once gate + the verdict synthesis fallback.
        _REVIEW.setdefault("recorded", []).append(
            {"id": action.id, "name": action.name, **fire})
        if action.id:
            rubrics = _load_store(sd)
            r = next((x for x in rubrics if x.get("id") == action.id), None)
            if r is None:
                return _RubricObs.from_text(
                    text=f"no rubric with id {action.id}; if this is a new check, pass "
                         f"name= instead (then add_rubric). Recorded for this wake.")
            r.setdefault("history", []).append(fire)
            _save_store(sd, rubrics)
            _append_oplog(sd, {"op": "record", "id": action.id, "result": action.result,
                               "submission": _sub_idx(), "attempt": _attempt_idx()})
            return _RubricObs.from_text(
                text=f"Recorded {action.result} for {action.id}. You may now cite it.")
        # No id: a pre-add recording (satisfies the run-once gate for add_rubric).
        if not action.name:
            return _RubricObs.from_text(text="pass either id (store rubric) or name "
                                             "(new check). Retry.")
        _append_oplog(sd, {"op": "record", "name": action.name, "result": action.result,
                           "submission": _sub_idx(), "attempt": _attempt_idx()})
        return _RubricObs.from_text(
            text=f"Recorded {action.result} for check '{action.name}' (run-once gate "
                 f"satisfied — you may now add_rubric it).")


class RecordCheckResultTool(ToolDefinition[RecordCheckResultAction, _RubricObs]):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["RecordCheckResultTool"]:
        return [cls(
            action_type=RecordCheckResultAction, observation_type=_RubricObs,
            description=("Log what a check actually ran and produced BEFORE you cite it "
                         "in your verdict (anti-hallucination: a verdict claim must trace "
                         "to a recorded fire). Also satisfies the run-once gate for "
                         "add_rubric on a new executable check."),
            executor=RecordCheckResultExecutor(),
            annotations=ToolAnnotations(title="record_check_result", readOnlyHint=False,
                                        destructiveHint=False, idempotentHint=False,
                                        openWorldHint=False),
        )]


# --- submit_verdict ------------------------------------------------------
class RubricScore(BaseModel):
    """One per-rubric score. This is the UNIFIED PRM output — the exact
    {importance_score, p_agent_success} interface the inline / inter-attempt PRM
    emits (prm_agent.py + prm_feedback.py), now spoken by the reviewer too. You
    score every applicable rubric; the HARNESS thresholds them (importance >=
    min_importance AND p_agent_success < failure_threshold → surfaced weakness) and
    renders the injected block — you do NOT pick which to show or write the block,
    and passing / low-importance rubrics are dropped automatically.

    Every score MUST cite a `rubric_id` that already exists in the store (a seed or
    a discovered rubric). The verdict is REJECTED if it cites an id not in the store
    — this anchors every verdict to a fixed, named rubric set. The ONLY way to grow
    the set is add_rubric / update_rubric (its check must have run once first)."""
    rubric_id: str = Field(description="The rubric store id, e.g. 'r1' or "
                           "'cv_score_registry'. MUST already exist (list_rubrics); "
                           "add_rubric a genuinely new failure mode FIRST, then score it.")
    importance_score: float = Field(description="0.0-1.0: how RELEVANT this rubric is to "
                           "the trajectory AS IT STANDS. A rubric for a stage the agent "
                           "hasn't reached → LOW importance (not a low p_agent_success).")
    p_agent_success: float = Field(description="0.0-1.0: probability the agent SATISFIED "
                           "this rubric. DYNAMIC op — GROUNDED in the checks you ran "
                           "(recorded numbers). STATIC op — ASSIGNED by reasoning over "
                           "what you read. 0.5 = maximum uncertainty.")
    rationale: str = Field(default="", description="Why — the numbers you measured "
                           "(dynamic) or the artifact facts you read (static). Required "
                           "when this is a weakness (low p, high importance).")
    suggested_action: str = Field(default="", description="Concrete imperative fix, cited "
                           "ONLY when p_agent_success is low; else \"\".")


class SubmitVerdictAction(Action):
    scores: list[RubricScore] = Field(default_factory=list,
                        description="Per-rubric scores over the rubrics you assessed — "
                        "cover EVERY applicable rubric in the store (list_rubrics), one "
                        "entry each. The harness thresholds them into the injected "
                        "feedback + the finish-gate (importance >= min_importance AND "
                        "p_agent_success < failure_threshold → surfaced); you do NOT pick "
                        "which to surface, set a STATUS, or write the block. Every "
                        "rubric_id MUST exist in the active store.")


class SubmitVerdictExecutor(ToolExecutor):
    def __call__(self, action: SubmitVerdictAction, conversation=None):
        # Anchor every score to a rubric that exists in the store. The reviewer may
        # GROW the set (add_rubric), but a verdict can only cite ids that already
        # exist — reject otherwise so it must add_rubric first, then re-submit.
        store_ids = {r.get("id") for r in _load_store(_store_dir())
                     if r.get("status", "active") == "active"}
        scores = list(action.scores or [])
        if not scores:
            return _RubricObs.from_text(
                text="submit_verdict needs at least one per-rubric score. list_rubrics, "
                     "then score EACH applicable rubric with importance_score + "
                     "p_agent_success. Retry.")
        unknown = sorted({(s.rubric_id or "").strip() for s in scores
                          if (s.rubric_id or "").strip() not in store_ids})
        if unknown:
            return _RubricObs.from_text(
                text=f"Rejected: scores cite rubric_id(s) not in the active store: "
                     f"{unknown}. Score only existing rubrics (list_rubrics). If one is a "
                     f"genuinely new failure mode, add_rubric it first (its check must "
                     f"have run once), then re-call submit_verdict. Do NOT invent ids.")
        bad = sorted({(s.rubric_id or "").strip() for s in scores
                      if not (0.0 <= float(s.importance_score) <= 1.0
                              and 0.0 <= float(s.p_agent_success) <= 1.0)})
        if bad:
            return _RubricObs.from_text(
                text=f"Rejected: importance_score and p_agent_success must be in "
                     f"[0.0, 1.0] for {bad}. Retry.")
        rubric_scores: dict[str, dict] = {}
        for s in scores:  # duplicate ids: last score wins
            rubric_scores[(s.rubric_id or "").strip()] = {
                "importance_score": float(s.importance_score),
                "p_agent_success": float(s.p_agent_success),
                "rationale": (s.rationale or "").strip(),
                "suggested_actions": (s.suggested_action or "").strip(),
            }
        status = _derive_status(rubric_scores)
        _REVIEW["result"] = {
            "status": status,
            "scores": rubric_scores,
            "findings": _weakness_findings(rubric_scores),  # thresholded, telemetry
            "verdict_text": _render_verdict(rubric_scores),
        }
        return _RubricObs.from_text(
            text=f"Verdict recorded (STATUS={status}, {len(rubric_scores)} rubrics "
                 f"scored) — call finish.")


def _store_meta() -> dict[str, tuple[str, str]]:
    """{rubric_id: (name, description)} from the active store — passed to the shared
    prm_feedback renderer so discovered rubrics (not in RUBRIC_REGISTRY) still get a
    human name + definition in the injected block."""
    meta: dict[str, tuple[str, str]] = {}
    try:
        for r in _load_store(_store_dir()):
            rid = r.get("id")
            if rid:
                meta[rid] = (r.get("name") or rid, r.get("description") or "")
    except Exception:
        pass
    return meta


def _weakness_findings(rubric_scores: dict[str, dict]) -> list[dict]:
    """The thresholded weaknesses (importance >= min_importance AND p < failure_thr),
    most-important first — persisted for telemetry / analysis. The injected text
    itself is rendered by the shared prm_feedback renderer (_render_verdict)."""
    min_imp, fail_thr = _thresholds()
    meta = _store_meta()
    out: list[dict] = []
    for rid, s in rubric_scores.items():
        imp = float(s.get("importance_score", 0.0))
        p = float(s.get("p_agent_success", 1.0))
        if imp >= min_imp and p < fail_thr:
            name, _desc = meta.get(rid, (rid, ""))
            out.append({"rubric_id": rid, "rubric": name,
                        "importance_score": imp, "p_agent_success": p,
                        "rationale": s.get("rationale", ""),
                        "suggested_action": s.get("suggested_actions", "")})
    out.sort(key=lambda x: x["importance_score"], reverse=True)
    return out


def _render_verdict(rubric_scores: dict[str, dict]) -> str:
    """Render the injected feedback through the SHARED prm_feedback renderer, so the
    reviewer's output is byte-identical to the inline / inter-attempt PRM (unified
    output format). Weaknesses-only, thresholded, mid-run framing; discovered
    rubrics get their names/definitions from the store via rubric_meta; SKIP_RUBRICS
    is NOT applied (the launch-selected set is already curated). Returns "" when no
    rubric is both important and failing (a clean pass → nothing to inject)."""
    from dsgym.harness.prm_feedback import generate_feedback_report
    min_imp, fail_thr = _thresholds()
    return generate_feedback_report(
        rubric_scores=rubric_scores or {},
        min_importance=min_imp, failure_threshold=fail_thr,
        mid_run=True, rubric_meta=_store_meta(), apply_skip=False,
    )


class SubmitVerdictTool(ToolDefinition[SubmitVerdictAction, _RubricObs]):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["SubmitVerdictTool"]:
        return [cls(
            action_type=SubmitVerdictAction, observation_type=_RubricObs,
            description=("Emit your final verdict as PER-RUBRIC SCORES over the rubrics "
                         "you assessed — each {rubric_id, importance_score, "
                         "p_agent_success, rationale, suggested_action}. The harness "
                         "thresholds them into the injected block (importance >= "
                         "min_importance AND p_agent_success < failure_threshold → "
                         "surfaced) and derives the finish-gate; you do NOT write the "
                         "block or set a STATUS. Call once, then finish."),
            executor=SubmitVerdictExecutor(),
            annotations=ToolAnnotations(title="submit_verdict", readOnlyHint=False,
                                        destructiveHint=False, idempotentHint=True,
                                        openWorldHint=False),
        )]


# =====================================================================
# System prompt
# =====================================================================

REVIEWER_SYSTEM_PROMPT = """You are a skeptical, execution-grounded reviewer for a data-science competition agent.
A separate agent is iteratively building a solution in a repository you can read
but NOT modify. You are woken PERIODICALLY, in the MIDDLE of an ONGOING attempt —
each wake is a mid-run checkpoint, NOT a post-mortem. The agent KEEPS WORKING after
you return; the artifacts and trajectory you see are a SNAPSHOT at the current turn.
The transcript ending at the latest turn does NOT mean the agent finished, crashed,
gave up, or called finish() — it only reflects the present moment. NEVER infer
termination or abandonment from the transcript ending. The agent may still be early
(loading data, EDA, first model): a rubric for a stage it has NOT YET reached (a
valid submission exists, submission integrity verified, a CV registry maintained) is
NOT-YET-APPLICABLE, not a defect — score it LOW importance (never a penalizing
importance≈1.0 with p_success≈0.0). Reserve high importance + low p_success for what
is WRONG or missing GIVEN how far the agent has ACTUALLY progressed. Your edge over a
code-reading reviewer is that you can WRITE AND RUN small diagnostic scripts over its artifacts
(out-of-fold predictions, residuals, features, the submission file, train/test
data). Use it. Your job has two equally-important halves, and you maintain a store
of reusable checks across both:
  (1) SOUNDNESS — verify the submitted file faithfully reflects the cross-validated
      model (the CV is real and reproduces, the submission matches the OOF preds,
      the file is well-formed).
  (2) MODELING — surface concrete, MEASURED ways the model is leaving performance
      on the table (underfitting, leftover signal in residuals, miscalibration,
      unstable CV, leakage), each backed by a diagnostic you actually ran.

WHAT YOUR JOB IS — AND IS NOT
- SOUNDNESS is the last line of defense against a good model being wrecked by a
  submission-pipeline bug (wrong scale, missing inverse-transform, rounding, wrong
  columns/rows) that the reported CV cannot see. A submission-pipeline bug is a
  FAIL.
- MODELING feedback is where you do what a code-reading reviewer CANNOT: you write
  a script and compute the number. Generic advice with no measurement ("try
  XGBoost", "add regularization", "use k-fold") is FORBIDDEN — a code-reading PRM
  already gives that, and you would just be a worse version of it. What ONLY you
  can do, and therefore MUST prefer: compute OOF residuals and correlate them with
  each feature (incl. dropped ones) to find leftover signal; compare the
  prediction distribution to the target (under/over-dispersion → under/overfit);
  check calibration (mean pred vs mean target; a reliability curve for
  probabilities); recompute the metric PER FOLD to expose high OOF variance
  (unstable CV the leaderboard will punish); test whether any single feature is
  near-perfectly predictive of the target (leakage). Every modeling point MUST
  cite a number you computed — exactly like a soundness finding.
- You are NOT the modeler. Do NOT re-run TRAINING, re-fit models, or grid-search —
  that is the base agent's job and it has its own budget. You run lightweight
  DIAGNOSTICS over existing artifacts (read 2-3 files, compute a statistic); you
  never train a better model or hand the agent a finished one.
- SOUNDNESS governs the STATUS; MODELING does not. A submission that is SOUND is
  never FAIL just because the model could be better — soundness and model-quality
  are separate axes. Report both, but a modeling suggestion can never, by itself,
  flip a sound submission to FAIL (see STATUS guidance below).

HARD RULES
- Trust nothing the agent claims. A cross-validation (CV) number is real ONLY if
  you reproduce it yourself from the out-of-fold (OOF) predictions and the true
  training target. The agent's reported CV is a claim, not evidence.
- The most dangerous failures are invisible in the code and the reported CV: the
  submitted file DIVERGES from what was validated. The direct test is
  submission-vs-OOF: the CV is computed on the OOF preds, so if the submission's
  distribution diverges from the OOF preds, the reported CV does not describe the
  submission. Check submission-vs-OOF FIRST.
- You may read and run code in the base repo and data dir (both READ-ONLY). You
  write scratch and your check scripts in your own workspace (your cwd).
- You may only cite a check result in your verdict AFTER you have run it and
  recorded it with record_check_result. No claimed numbers you did not compute.

YOUR WORKSPACE (exact absolute paths are in the RESOLVED FILE PATHS block below —
use those, they are authoritative)
- base repo  (READ-ONLY): the agent's repository — code, cv_scores.json, preds/,
  the submission file.
- data dir   (READ-ONLY): train.csv (with target), test.csv (no target), sample
  submission. No test labels — you verify methodology, not the answer.
- your cwd   (WRITABLE): your rubric store (via the rubric tools) + your check
  scripts (checks/) + checks/README.md (your cross-attempt script index) + scratch.

HOW CHECKS WORK — rubrics are SPECS, you write the CODE
- Your rubrics are natural-language SPECIFICATIONS of what to verify (see
  list_rubrics). There are NO pre-written check scripts. For each spec you write
  a SMALL bespoke script contextual to THIS task (its real column names, metric,
  and any transform) — a fixed generic script gets the thresholds and columns
  wrong. Keep each check to a few lines that read 2-3 files and print numbers +
  a one-line "RESULT: PASS|WARN|FAIL — <reason>".
- Your message history resets each attempt, but checks/ persists across all
  attempts. checks/README.md is your memory of what scripts already exist. READ
  it first; REUSE a fitting script (just re-run it) instead of rewriting; when
  you write a NEW script, APPEND a row to checks/README.md (filename, rubric,
  purpose, exact run command) so the next attempt finds it.
- KEEP checks/README.md ACCURATE — it is a contract, not a scratch note. If you
  EDIT a script's behavior (change a column, threshold, weight, or path) you MUST
  edit its README row in the SAME turn so the description + run command still
  match what the script does. A stale row that claims different weights/columns
  than the code is worse than no row — a future wake will trust it and skip
  re-reading the script. When in doubt, keep the row's purpose generic (WHAT it
  checks) and let specifics live in the code, so the row cannot drift.

PROCEDURE (you have up to {k} turns — spend them EXECUTING, not exploring)
1. list_rubrics, then read checks/README.md — know your specs and existing scripts.
2. SOUNDNESS FIRST. For each soundness spec: reuse-or-write its script, run it, then
   IMMEDIATELY call record_check_result — before the next check — with the numbers AND
   your two scores for that rubric (importance_score + p_agent_success; see SCORING
   below). A check you ran but did not record is WASTED. Do the fast, always-applicable
   ones FIRST (submission_coverage, then submission_vs_oof_consistency — the money
   check), THEN cv_reproduce.
3. cv_reproduce / OOF-consistency / most modeling diagnostics need the OOF
   predictions. Look ONCE (preds/, oof*.csv/.parquet). If not found in one command,
   record the OOF-dependent rubrics with LOW importance (or p_agent_success=0.5, max
   uncertainty) and MOVE ON — do not keep hunting.
4. MODELING DIAGNOSTICS. With turns remaining, run the modeling specs that apply to
   this task's metric/target type — residual-vs-feature signal, prediction dispersion
   vs. target, calibration, per-fold metric variance, single-feature leakage. Same
   discipline: write/reuse a small script, run it, record_check_result with the number
   + the two scores. Prefer diagnostics whose result would change what the agent does.
5. If you discover a task-specific failure mode OR a recurring modeling weakness,
   add_rubric an NL spec for it (and leave its script indexed in checks/README.md)
   so future submissions on this task are guarded. OPTIONAL — skip if short on turns.
6. Call submit_verdict — MANDATORY. Do it while you still have turns left (by turn
   {k}-1 at the latest), scoring EVERY applicable rubric, grounded ONLY in results you
   recorded.

SCORING — the unified PRM interface (identical to the inline / inter-attempt PRM). For
EVERY rubric you assess you output TWO 0-1 scores, and the HARNESS decides what the
agent sees (you do NOT set a STATUS or write the injected block):
- importance_score (0.0-1.0): how RELEVANT this rubric is to the trajectory AS IT
  STANDS. A rubric for a stage the agent has not reached → LOW importance. NEVER
  penalize an unreached rubric with a low p_agent_success — lower its importance.
- p_agent_success (0.0-1.0): the probability the agent SATISFIED this rubric's good
  behavior. HOW you get this number is the whole point of your operationalization:
    • DYNAMIC (you have a terminal): p_agent_success must be GROUNDED in a check you
      actually wrote+ran — the measured numbers decide the probability (a passed test
      → high, a failed test → low). No number, no score.
    • STATIC (no terminal): you ASSIGN p_agent_success by READING the artifacts + code
      and reasoning; cite the concrete facts you read as evidence.
  0.5 = maximum uncertainty (not reached / not enough evidence).
The harness surfaces a rubric to the agent ONLY when importance_score >=
min_importance AND p_agent_success < failure_threshold, rendering the weaknesses as a
numbered issue list in the SAME format the agent already acts on:

  ### ISSUES TO FIX (address these in your remaining steps):
  1. **CV Reproduce** [Importance: 0.90 | Success: 0.20]:
     - Definition: the reported CV metric recomputes from the OOF preds + target
     - Problem: reproduced RMSE 12.8 vs reported 11.9 — CV optimistic
     - Suggested action: re-run CV with the exact fold seed; do not trust 11.9

submit_verdict arguments:
- scores: a list, ONE ENTRY PER RUBRIC you assessed (cover every APPLICABLE rubric —
  strengths included; the harness drops passing / low-importance ones automatically).
  Each entry = {rubric_id (an id that EXISTS in the store — see below),
  importance_score, p_agent_success, rationale (the numbers you measured / facts you
  read), suggested_action (a concrete imperative fix; cite ONLY when p_agent_success
  is low)}. (The rubric NAME + DEFINITION shown to the agent are filled in
  automatically from the store by rubric_id — you do not write them.)

EVERY score MUST cite a `rubric_id` that already exists in the store (a seed like `r1`
or a discovered rubric). The verdict is REJECTED if it cites an id not in the store —
you cannot invent a rubric in the verdict. If you found a genuinely new failure mode,
add_rubric it FIRST (its check must have run once), THEN score the new id. This anchors
every verdict to a fixed, named rubric set; the ONLY way to grow the set is the
add_rubric / update_rubric channels.

Fold the numbers you measured (dynamic) or the artifact facts you read (static)
directly into each entry's `rationale`. A weakness claim with no grounding is not
allowed. The STATUS + finish-gate are DERIVED by the harness from your scores (any
important-and-failing rubric → the agent is asked to fix it before finishing); you do
not set them.
"""


# Static-operationalization replacement for the whole "HOW CHECKS WORK" + "PROCEDURE"
# region (which in dynamic mode is all about writing+running check scripts). In STATIC
# mode the reviewer has NO terminal, so it READS artifacts + code and REASONS. The
# strict per-wake static workflow is (re)stated by _workflow_block(static=True).
_STATIC_CHECKS_AND_PROCEDURE = """HOW YOU ASSESS — rubrics are SPECS, you READ + REASON (no code)
- Your rubrics are natural-language SPECIFICATIONS of what to verify (see
  list_rubrics). You have NO terminal, so you do NOT write or run check scripts.
  For each spec, OPEN the relevant artifacts (the submission csv, cv_scores.json,
  oof_predictions.csv/.parquet, the modeling code) with the file-reading tool and
  reason about whether the rubric is satisfied from what you can SEE.
- checks/README.md persists across attempts as your memory of what earlier wakes
  concluded; read it first for prior reasoning, and record fresh findings with
  record_check_result so the next wake inherits them.

PROCEDURE (you have up to {k} turns — spend them READING + REASONING, not hunting)
1. read /reviewer/submission_ledger.jsonl (your prior verdicts) + list_rubrics +
   checks/README.md — know your specs and what earlier wakes found.
2. SOUNDNESS FIRST. For each soundness spec, open the artifacts it concerns and
   reason: does the submission match the OOF preds (submission-vs-OOF — the money
   check)? Is coverage/schema right? Does the reported CV look reproducible from
   the OOF preds + target as written? record_check_result for each, with the
   concrete facts you read + your two scores (see SCORING).
3. OOF-dependent specs: look ONCE for oof*.csv/.parquet in ./, preds/. If absent,
   record those rubrics LOW importance (or p_agent_success=0.5) and MOVE ON.
4. MODELING DIAGNOSTICS. With turns left, read the modeling code + artifacts and
   reason about residual signal, prediction-vs-target dispersion, calibration,
   per-fold variance, single-feature leakage — cite the facts you read.
5. If you discover a task-specific failure mode, add_rubric an NL spec for it (index
   it in checks/README.md). OPTIONAL — skip if short on turns.
6. Call submit_verdict — MANDATORY, by turn {k}-1 at the latest — scoring EVERY
   applicable rubric, grounded ONLY in facts you read.

"""


def _reviewer_system_prompt(static: bool) -> str:
    """The reviewer's leading instructions. DYNAMIC (operationalization='dynamic')
    returns REVIEWER_SYSTEM_PROMPT verbatim — unchanged legacy behavior. STATIC
    (operationalization='static', the AdaPRA − PRA / − code-generation ablation)
    rewrites the code-execution framing: the reviewer has NO terminal, so it READS
    artifacts + code and REASONS to a verdict (a linter over read-only W). Same
    rubric set, same scoring interface — only the operationalization differs
    (paper §sec:axes). Kept as a post-hoc rewrite (not a second literal) so the two
    prompts cannot silently drift apart."""
    if not static:
        return REVIEWER_SYSTEM_PROMPT
    p = REVIEWER_SYSTEM_PROMPT
    # (needle, static replacement) for the self-contained code-execution claims.
    subs = [
        ("Your edge over a\n"
         "code-reading reviewer is that you can WRITE AND RUN small diagnostic scripts over its artifacts\n"
         "(out-of-fold predictions, residuals, features, the submission file, train/test\n"
         "data). Use it.",
         "You have NO terminal and CANNOT run code: you are a STATIC\n"
         "analyzer. You READ its artifacts (out-of-fold predictions, residuals, features,\n"
         "the submission file, train/test data) and its modeling code, and REASON to a\n"
         "verdict from what you can SEE — cite the concrete facts you read as evidence."),
        ("  (2) MODELING — surface concrete, MEASURED ways the model is leaving performance\n"
         "      on the table (underfitting, leftover signal in residuals, miscalibration,\n"
         "      unstable CV, leakage), each backed by a diagnostic you actually ran.",
         "  (2) MODELING — surface concrete, GROUNDED ways the model is leaving performance\n"
         "      on the table (underfitting, leftover signal in residuals, miscalibration,\n"
         "      unstable CV, leakage), each backed by a concrete fact you READ."),
        # The whole MODELING bullet: in dynamic mode it is a list of statistics to
        # COMPUTE; in static mode the same diagnostics become things to READ for.
        ("- MODELING feedback is where you do what a code-reading reviewer CANNOT: you write\n"
         "  a script and compute the number. Generic advice with no measurement (\"try\n"
         "  XGBoost\", \"add regularization\", \"use k-fold\") is FORBIDDEN — a code-reading PRM\n"
         "  already gives that, and you would just be a worse version of it. What ONLY you\n"
         "  can do, and therefore MUST prefer: compute OOF residuals and correlate them with\n"
         "  each feature (incl. dropped ones) to find leftover signal; compare the\n"
         "  prediction distribution to the target (under/over-dispersion → under/overfit);\n"
         "  check calibration (mean pred vs mean target; a reliability curve for\n"
         "  probabilities); recompute the metric PER FOLD to expose high OOF variance\n"
         "  (unstable CV the leaderboard will punish); test whether any single feature is\n"
         "  near-perfectly predictive of the target (leakage). Every modeling point MUST\n"
         "  cite a number you computed — exactly like a soundness finding.",
         "- MODELING feedback: surface concrete ways the model leaves performance on the\n"
         "  table, read directly from the artifacts + code (you cannot run anything).\n"
         "  Generic advice with no grounding (\"try XGBoost\", \"add regularization\", \"use\n"
         "  k-fold\") is FORBIDDEN. Look for: features built or loaded but dropped from the\n"
         "  model; the prediction values you can see vs the target's range/scale (under/\n"
         "  over-dispersion, clipping, a missing inverse-transform); per-fold scores in\n"
         "  cv_scores.json or the logs (high fold variance = unstable CV the leaderboard\n"
         "  will punish); a single feature that the code or EDA output shows is near-\n"
         "  perfectly predictive of the target (leakage). Every modeling point MUST cite a\n"
         "  concrete fact you READ in the artifacts/code — exactly like a soundness finding."),
        ("You run lightweight\n"
         "  DIAGNOSTICS over existing artifacts (read 2-3 files, compute a statistic); you\n"
         "  never train",
         "You READ existing\n"
         "  artifacts + code (2-3 files per check) and reason about them; you\n"
         "  never train"),
        ("your rubric store (via the rubric tools) + your check\n"
         "  scripts (checks/) + checks/README.md (your cross-attempt script index) + scratch.",
         "your rubric store (via the rubric tools) +\n"
         "  checks/README.md (your cross-attempt notes on what earlier wakes concluded)."),
        ("add_rubric it FIRST (its check must have run once), THEN score the new id.",
         "add_rubric it FIRST (after recording a check result for it), THEN score the new id."),
        ("- Trust nothing the agent claims. A cross-validation (CV) number is real ONLY if\n"
         "  you reproduce it yourself from the out-of-fold (OOF) predictions and the true\n"
         "  training target. The agent's reported CV is a claim, not evidence.",
         "- Trust nothing the agent claims. Treat a reported cross-validation (CV) number as\n"
         "  a CLAIM you cannot re-run: corroborate or doubt it by READING the OOF preds, the\n"
         "  fold code, and the target handling — inconsistencies you can see are evidence."),
        ("- You may read and run code in the base repo and data dir (both READ-ONLY). You\n"
         "  write scratch and your check scripts in your own workspace (your cwd).\n"
         "- You may only cite a check result in your verdict AFTER you have run it and\n"
         "  recorded it with record_check_result. No claimed numbers you did not compute.",
         "- You may READ any file in the base repo and data dir (both READ-ONLY). You have\n"
         "  NO terminal — do NOT attempt to execute anything.\n"
         "- Ground every verdict entry in a concrete artifact/code fact you READ, and record\n"
         "  it with record_check_result before you cite it. No numbers you did not observe."),
    ]
    for needle, repl in subs:
        if needle not in p:
            print("[prm_reviewer] WARN static prompt: needle not found (dynamic "
                  f"text drifted?): {needle[:60]!r}", flush=True)
            continue
        p = p.replace(needle, repl)
    # Replace the whole dynamic "HOW CHECKS WORK … PROCEDURE" region (short, unique
    # anchors) with the static read+reason procedure.
    a1 = "HOW CHECKS WORK — rubrics are SPECS, you write the CODE\n"
    a2 = "SCORING — the unified PRM interface"
    if a1 in p and a2 in p:
        head, _, rest = p.partition(a1)
        _, _, tail = rest.partition(a2)
        p = head + _STATIC_CHECKS_AND_PROCEDURE + a2 + tail
    else:
        print("[prm_reviewer] WARN static prompt: HOW-CHECKS/SCORING anchors not "
              "found — leaving dynamic procedure text in place", flush=True)
    return p


# =====================================================================
# Reviewer BudgetTracker + Agent subclass (its OWN footer)
# =====================================================================

class ReviewerBudgetTracker:
    """Mirror of the base DSAgent's BudgetTracker so the reviewer gets its own
    turn + wall-clock footer, independent of the base agent's clock."""
    def __init__(self, max_turns: int, time_budget_seconds: float):
        self.max_turns = max_turns
        self.time_budget_seconds = time_budget_seconds
        self.start_time = _time.time()
        self.turn_count = 0

    def tick(self):
        self.turn_count += 1

    def remaining(self):
        return max(0, self.time_budget_seconds - (_time.time() - self.start_time))

    def status_line(self):
        elapsed = _time.time() - self.start_time
        remaining = max(0, self.time_budget_seconds - elapsed)
        line = (f"[REVIEWER Turn {self.turn_count}/{self.max_turns} | "
                f"Time {elapsed/60:.0f}m/{self.time_budget_seconds/60:.0f}m | "
                f"{remaining/60:.0f}m remaining]")
        # Nudge to close out cleanly before turns/time run out.
        turns_left = self.max_turns - self.turn_count
        if turns_left <= 2 or remaining <= 60:
            line += (" ⚠ Almost out of turns — record any un-recorded check "
                     "results NOW and call submit_verdict THIS TURN.")
        return line


# Subclassing Agent makes prompt_dir resolve relative to THIS module (which has
# no prompts/ dir), so point it back at the SDK's own prompts dir.
import openhands.sdk.agent.agent as _sdk_agent_mod  # noqa: E402
_SDK_PROMPTS_DIR = str(Path(_sdk_agent_mod.__file__).parent / "prompts")


class ReviewerAgent(Agent):
    """Base Agent + the reviewer's own turn/wall-clock footer. Defined at module
    scope (NOT inside the run function) so its qualname has no "<locals>" — the
    SDK's ConversationState rejects local classes when persistence is enabled.
    Reads the active budget from the module global _ACTIVE_BUDGET."""

    @property
    def prompt_dir(self) -> str:
        return _SDK_PROMPTS_DIR

    def _execute_actions(self, conversation, action_events, on_event):
        super()._execute_actions(conversation, action_events, on_event)
        budget = _ACTIVE_BUDGET
        if budget is None:
            return
        budget.tick()
        status = budget.status_line()
        print(status, flush=True)
        on_event(MessageEvent(
            source="environment",
            llm_message=Message(role="user",
                                content=[TextContent(text=status)]),
        ))


def run_submission_reviewer(
    *,
    llm_kwargs: dict,
    events: list[dict],
    base_ws_dir: str,
    submission_path: str,
    data_dir: str,
    rubric_store_dir: str,
    prm_reviews_dir: str | None = None,
    task_desc: str = "",
    submission_index: int = 0,
    attempt_index: int = 0,
    command_timeout: int = 600,
    max_iterations: int = 10,
    time_budget_seconds: float = 1200.0,
    convo_persist_dir: str | None = None,
    convo_id: str | None = None,
    rubric_set: str | None = None,
    rubric_set_exclude_ids: Sequence[str] | None = None,
    evolution: str = "evolving",
    operationalization: str = "dynamic",
    history: str = "pull",
    trajectory_path: str | None = None,
    pushed_transcript: str | None = None,
    prior_trajectory_paths: Sequence[str] | None = None,
    n_attempts: int = 5,
    min_importance: float | None = None,
    failure_threshold: float | None = None,
) -> dict:
    """Run one wake of the submission reviewer. Paths only → runs identically
    in-container or in its own apptainer instance.

    Returns a self-contained record:
      {"status": "PASS|WARN|FAIL", "verdict_text": ..., "scores": {...},
       "findings": [...], "checks_run": [...], "rubric_ops": [...],
       "turns_used": ..., "wall_seconds": ..., "forced": bool}

    `scores` is the unified per-rubric map {rubric_id: {importance_score,
    p_agent_success, rationale, suggested_actions}} — the SAME interface the inline
    / inter-attempt PRM emits; `findings` is its importance-thresholded weakness
    subset (what got injected). min_importance/failure_threshold override the module
    defaults (0.5 / 0.6) for the threshold+gate; None → defaults.
    or {"status": "ERROR", "error": ...} on failure (caller injects nothing).
    Never raises — must not crash the base attempt.

    Per-attempt message-stream continuity: pass a stable `convo_persist_dir` +
    `convo_id` for the attempt. LocalConversation does create-or-resume, so each
    submission-event wake within the SAME attempt resumes the prior message
    history (delete_on_close=False). A NEW attempt uses a new convo_id → fresh
    stream. This works even though Option B spawns a fresh process per wake
    (an in-memory conversation handle can't cross the process boundary). If
    convo_persist_dir is None, a throwaway conversation is used (per-wake reset).
    """
    t0 = _time.time()
    store_dir = Path(rubric_store_dir)
    try:
        # rubric_set (launch-selected JSON, e.g. "discovered"): seed the store
        # from it instead of the 9 built-in _SEED_RUBRICS, so the reviewer and the
        # static PRM operationalize the SAME rubric content. A launch-fixed
        # exclusion list supports controlled subset ablations without modifying a
        # packaged source JSON. No-op after the first wake on a task (the store
        # persists), so a mid-run change never re-seeds.
        seeds = (
            _seeds_from_rubric_set(rubric_set, rubric_set_exclude_ids)
            if rubric_set else None
        )
        _ensure_store(store_dir, seeds=seeds,
                      static=(operationalization or "dynamic").lower() == "static")
    except Exception as e:  # never crash the base attempt
        return {"status": "ERROR", "error": f"store init failed: {e}"}

    # Reset per-wake capture state (keep nothing across wakes except the store).
    _REVIEW.clear()
    _REVIEW["store_dir"] = str(store_dir)
    _REVIEW["submission_index"] = submission_index
    _REVIEW["attempt_index"] = attempt_index
    _REVIEW["n_attempts"] = int(n_attempts)
    _REVIEW["recorded"] = []
    _REVIEW["ops"] = []
    # Scoring thresholds for THIS wake (drive both the injected-feedback filter and
    # the finish-gate); None → module defaults via _thresholds().
    if min_importance is not None:
        _REVIEW["min_importance"] = float(min_importance)
    if failure_threshold is not None:
        _REVIEW["failure_threshold"] = float(failure_threshold)

    try:
        rec = _run_reviewer_conversation(
            llm_kwargs=llm_kwargs, base_ws_dir=base_ws_dir,
            submission_path=submission_path, data_dir=data_dir,
            store_dir=store_dir, task_desc=task_desc,
            command_timeout=command_timeout, max_iterations=max_iterations,
            time_budget_seconds=time_budget_seconds,
            convo_persist_dir=convo_persist_dir, convo_id=convo_id,
            evolution=evolution, operationalization=operationalization,
            history=history, trajectory_path=trajectory_path,
            pushed_transcript=pushed_transcript,
            prior_trajectory_paths=prior_trajectory_paths,
            n_attempts=int(n_attempts), attempt_index=attempt_index,
        )
    except Exception as e:
        import traceback
        return {"status": "ERROR", "error": str(e),
                "traceback": traceback.format_exc()}

    result = _REVIEW.get("result")
    recorded = _REVIEW.get("recorded", [])
    ops = _REVIEW.get("ops", [])
    if not result:
        # Turns exhausted with no explicit verdict — synthesize a forced WARN
        # grounded in whatever checks were recorded.
        result = _synthesize_forced_verdict(recorded)
        result["forced"] = True
    else:
        result["forced"] = False

    steps = rec.get("steps", [])
    out = {
        **result,
        "checks_run": recorded,
        "rubric_ops": ops,
        "turns_used": rec.get("turns_used"),
        "wall_seconds": round(_time.time() - t0, 1),
        "submission_index": submission_index,
        "attempt_index": attempt_index,
        "cost": rec.get("cost", {}),
    }
    if prm_reviews_dir:
        try:
            d = Path(prm_reviews_dir)
            d.mkdir(parents=True, exist_ok=True)
            idx = len(list(d.glob("reviewer_call_*.json")))
            # Compact verdict summary (as before).
            (d / f"reviewer_call_{idx:02d}.json").write_text(json.dumps(out, indent=2))
            # Full reviewer trajectory, base-agent-style (steps[] + metadata) so
            # the whole message stream is inspectable in outputs/ like an attempt.
            traj = {
                "trajectory_id": f"reviewer-a{attempt_index}-s{submission_index}",
                "agent_model": llm_kwargs.get("model"),
                "role": "submission_reviewer",
                "attempt_index": attempt_index,
                "submission_index": submission_index,
                "verdict_status": out.get("status"),
                "forced": out.get("forced"),
                "num_turns": rec.get("turns_used"),
                "wall_seconds": out.get("wall_seconds"),
                "cost": out.get("cost", {}),
                "checks_run": recorded,
                "rubric_ops": ops,
                "verdict_text": out.get("verdict_text"),
                "steps": steps,
            }
            (d / f"reviewer_trajectory_{idx:02d}.json").write_text(
                json.dumps(traj, indent=2))
        except OSError:
            pass
    return out


def _synthesize_forced_verdict(recorded: list[dict]) -> dict:
    """Build a verdict from the recorded per-rubric scores when the reviewer ran out
    of turns without calling submit_verdict. Reconstructs the SAME rubric_scores map
    an explicit verdict would (last recorded fire per store rubric id, carrying its
    importance_score + p_agent_success), then thresholds + renders through the shared
    path — so a forced verdict is byte-identical in format to a submitted one."""
    by_id: dict[str, dict] = {}
    for c in recorded:
        rid = (c.get("id") or "").strip()
        if not rid or "importance_score" not in c or "p_agent_success" not in c:
            continue
        by_id[rid] = c  # last fire wins
    rubric_scores: dict[str, dict] = {
        rid: {
            "importance_score": float(c["importance_score"]),
            "p_agent_success": float(c["p_agent_success"]),
            "rationale": (c.get("evidence") or "").strip(),
            "suggested_actions": "",
        }
        for rid, c in by_id.items()
    }
    if not rubric_scores:
        # Reviewer never scored any store rubric — emit a bare WARN so the caller
        # has something, without inventing rubric-level findings.
        txt = ("## Expert Feedback (Process Reward Model)\n\n"
               "The reviewer ran out of turns before scoring any rubric; treat "
               "unverified checks as unconfirmed and sanity-check the submission "
               "before finishing.")
        return {"status": "WARN", "scores": {}, "findings": [], "verdict_text": txt}
    status = _derive_status(rubric_scores)
    return {"status": status, "scores": rubric_scores,
            "findings": _weakness_findings(rubric_scores),
            "verdict_text": _render_verdict(rubric_scores)}


def _build_first_message(base_ws_dir: str, submission_path: str, data_dir: str,
                         store_dir: Path, task_desc: str, k: int,
                         static: bool = False, history: str = "pull",
                         trajectory_path: str | None = None,
                         pushed_transcript: str | None = None,
                         prior_trajectory_paths: Sequence[str] | None = None,
                         n_attempts: int = 5,
                         attempt_index: int = 0) -> str:
    """Prompt + a compact artifacts-first context block + a trajectory-access
    section (paper §sec:axes: τ is ALWAYS available; `history` only picks HOW it
    is delivered — "pull" = a read-only file the reviewer reads on demand, "push"
    = the agent's new turns since the last wake inlined here)."""
    import os
    def _listing(root: str, limit: int = 60) -> str:
        out = []
        try:
            for dp, dn, fn in os.walk(root):
                dn[:] = [d for d in dn if not d.startswith(".")][:20]
                for f in fn:
                    rel = os.path.relpath(os.path.join(dp, f), root)
                    out.append(rel)
                    if len(out) >= limit:
                        return "\n".join(sorted(out)) + f"\n… (>{limit} files)"
        except OSError:
            pass
        return "\n".join(sorted(out)) or "(empty)"

    cv_txt = ""
    for cand in ("cv_scores.json",):
        p = Path(base_ws_dir) / cand
        if p.exists():
            try:
                cv_txt = f"\n{cand}:\n{p.read_text()[:2000]}"
            except OSError:
                pass

    # Resolve concrete data-file paths NOW (host-side, cheap) so the reviewer
    # doesn't waste turns running `find` inside the container (which hung on a
    # broad /data scan in the smoke test). Give it exact absolute paths.
    def _find(root: str, names) -> str:
        for n in names:
            hit = Path(root) / n
            if hit.exists():
                return str(hit)
        try:
            for dp, dn, fn in os.walk(root):
                dn[:] = [d for d in dn if not d.startswith(".")]
                for f in fn:
                    if f in names:
                        return os.path.join(dp, f)
        except OSError:
            pass
        return ""
    train_csv = _find(data_dir, ("train.csv",))
    test_csv = _find(data_dir, ("test.csv",))
    sample_csv = _find(data_dir, ("sample_submission.csv", "sampleSubmission.csv"))
    paths = "\n".join(
        f"  {label}: {val or '(not found — locate it yourself)'}"
        for label, val in (
            ("YOUR CWD / rubric store (writable — run checks from here)", str(store_dir)),
            ("base repo (read-only)", base_ws_dir),
            ("submission (under review)", submission_path),
            ("train.csv", train_csv), ("test.csv", test_csv),
            ("sample_submission", sample_csv)))

    # Infer id-col + prediction-col from the sample submission header (first col
    # = id, last col = the predicted target), so we can emit EXACT ready-to-run
    # commands instead of leaving the reviewer to guess argparse args (it burned
    # turns on missing --sample/--metric/--target in testing).
    def _header(path: str):
        try:
            if path.endswith((".parquet", ".pq")):
                import pandas as pd
                return list(pd.read_parquet(path).columns)
            import csv as _csv
            with open(path) as f:
                return next(_csv.reader(f))
        except Exception:
            return []
    sample_cols = _header(sample_csv) if sample_csv else []
    id_col = sample_cols[0] if sample_cols else None
    pred_col = sample_cols[-1] if len(sample_cols) >= 2 else None

    # Discover an OOF-prediction file for cv_reproduce (preds/oof*.{csv,parquet}).
    def _find_oof() -> str:
        for sub in ("preds", "oof", ".", "output", "artifacts"):
            root = Path(base_ws_dir) / sub if sub != "." else Path(base_ws_dir)
            if not root.exists():
                continue
            try:
                cands = [p for p in root.iterdir()
                         if p.is_file() and p.suffix in (".csv", ".parquet", ".pq")
                         and "oof" in p.name.lower()]
                if cands:
                    return str(sorted(cands)[0])
            except OSError:
                pass
        return ""
    oof_path = _find_oof()

    # Try to name the metric from cv_scores.json / task text (best-effort hint).
    metric_hint = ""
    _blob = (cv_txt + " " + (task_desc or "")).lower()
    for m in ("rmsle", "rmse", "mae", "logloss", "auc", "accuracy", "f1", "mape"):
        if m in _blob:
            metric_hint = m
            break

    # Resolved facts the reviewer would otherwise burn turns discovering, so it
    # can write bespoke checks immediately instead of running `find`/guessing cols.
    resolved = []
    resolved.append(f"  id column (from sample header): {id_col or '(unknown — inspect the header)'}")
    resolved.append(f"  prediction column (from sample header): {pred_col or '(unknown — inspect the header)'}")
    resolved.append(f"  competition metric (best-effort hint): {metric_hint or '(unknown — read the task/cv_scores.json)'}")
    resolved.append(f"  OOF predictions file (for cv_reproduce / consistency / residuals): "
                    f"{oof_path or 'oof_predictions.csv expected in the repo root — look ONCE in ./, preds/, oof*.csv/.parquet; WARN OOF-dependent checks if truly absent'}")
    resolved.append("  submission ledger (YOUR cross-attempt memory, read+append): "
                    "/reviewer/submission_ledger.jsonl")
    resolved_block = "\n".join(resolved)

    prompt = _reviewer_system_prompt(static).replace("{k}", str(k))
    return (
        prompt
        + "\n\n===== TASK =====\n" + (task_desc[:2500] if task_desc else "(none)")
        + "\n\n===== RESOLVED FILE PATHS (use these directly — do NOT `find`) =====\n"
        + paths
        + ("\n\n===== RESOLVED COLUMNS / METRIC / OOF (use these when reading the "
           "artifacts — do NOT re-discover) =====\n" if static else
           "\n\n===== RESOLVED COLUMNS / METRIC / OOF (use these when writing your "
           "bespoke checks — do NOT re-discover) =====\n") + resolved_block
        + f"\n\n===== base repo LISTING (read-only) =====\n{_listing(base_ws_dir)}"
        + f"\n\n===== data LISTING (read-only) =====\n{_listing(data_dir)}"
        + (f"\n\n===== reported CV ====={cv_txt}" if cv_txt else "")
        + _best_of_n_block(attempt_index, n_attempts, prior_trajectory_paths)
        + _trajectory_block(history, trajectory_path, pushed_transcript, static)
        + _workflow_block(store_dir, k, static)
    )


def _best_of_n_block(attempt_index: int, n_attempts: int,
                     prior_trajectory_paths: Sequence[str] | None) -> str:
    """Best-of-N cross-attempt context (paper §sec:axes). Tells the reviewer the
    task runs the base agent N independent times and KEEPS THE BEST (max over
    attempts), which attempt this is, and where the ACCUMULATED prior-attempt
    transcripts live — so the cross-attempt rubrics (no_churn_without_pivot,
    over_optimization, single_model_class_hedging, and the cross-attempt-aware
    diversity/HPO/FE rubrics) can reason over the WHOLE beam, not just this
    attempt. Files follow the convention traj_a{attempt}_t{turn}.txt in /reviewer/
    (attempt index 0..N-1, turn = the base agent's turn count at that snapshot);
    the latest snapshot per PRIOR attempt is listed explicitly below."""
    n = max(int(n_attempts), 1)
    a = int(attempt_index)
    lines = [
        "\n\n===== BEST-OF-N CONTEXT (cross-attempt) =====",
        f"This task runs the base agent as BEST-OF-{n}: up to {n} INDEPENDENT "
        f"attempts, and ONLY THE BEST attempt's submission is scored (the final "
        f"result is the MAX over attempts, not the last). This is attempt "
        f"{a} (0-indexed; i.e. attempt {a + 1} of {n}).",
        "IMPLICATION for your scoring: because the beam keeps the MAX, a genuinely "
        "DIFFERENT approach that ends up second-best costs the run NOTHING, while an "
        "attempt that DUPLICATES an earlier one (same model family/features, near-"
        "identical predictions, re-tuning an already-saturated axis) WASTES one of "
        f"the {n} shots. When you score the cross-attempt rubrics "
        "(no_churn_without_pivot, over_optimization, single_model_class_hedging, and "
        "the diversity/hyperparameter/feature-engineering rubrics), judge this "
        "attempt against what the PRIOR attempts already did — reward diversification "
        "and flag convergence/redundant re-optimization.",
    ]
    prior = [p for p in (prior_trajectory_paths or []) if p]
    if prior:
        lines.append(
            "Accumulated PRIOR-attempt transcripts (read-only; naming convention "
            "traj_a{attempt}_t{turn}.txt = that attempt's latest snapshot, turn = "
            "the base agent's turn count). Skim these to see what earlier attempts "
            "tried before judging cross-attempt redundancy:")
        lines.extend(f"  {p}" for p in prior)
    else:
        lines.append(
            "No prior-attempt transcripts are available yet (this is an early "
            "attempt, or earlier attempts produced no snapshot) — assess "
            "cross-attempt rubrics as not-yet-applicable (LOW importance) until "
            "at least a couple of attempts exist to compare.")
    return "\n".join(lines)


def _trajectory_block(history: str, trajectory_path: str | None,
                      pushed_transcript: str | None, static: bool) -> str:
    """The agent's trajectory τ, delivered per the history axis (paper §sec:axes).
    τ is ALWAYS available — this only chooses HOW:

    - "pull" (default): the FULL turn-annotated trajectory is written to a
      read-only file (`trajectory_path`); we give the path and tell the reviewer
      to read it ON DEMAND when a trajectory-grounded rubric needs process
      evidence (how the agent worked over turns), not just the end-state artifacts.
    - "push": only the agent's NEW turns since your last wake are inlined here
      (`pushed_transcript`) — recent behavior under a fixed window, so you see the
      latest process without re-reading the whole history each wake.

    Either way the process is visible, so trajectory-grounded rubrics (churn,
    repeated crashes, no-pivot, compute burned) can be evaluated — not just the
    artifact-grounded ones. In STATIC mode you cannot run code, so reading the
    trajectory is often your ONLY source of process evidence."""
    mode = (history or "pull").lower()
    if mode == "push":
        delta = (pushed_transcript or "").strip()
        if not delta:
            return ("\n\n===== AGENT TRAJECTORY (PUSH — new turns since your last "
                    "review) =====\n(no new turns since your last wake — judge from "
                    "the artifacts + your prior findings in the ledger.)")
        return ("\n\n===== AGENT TRAJECTORY (PUSH — the agent's NEW turns since your "
                "last review, turn-annotated) =====\n"
                "This is the recent process delta (not the whole history). Use it to "
                "evaluate trajectory-grounded rubrics (how the agent worked lately) "
                "alongside the artifacts.\n\n" + delta)
    # pull (default)
    if trajectory_path:
        return ("\n\n===== AGENT TRAJECTORY (PULL — transcript SO FAR, on demand) =====\n"
                f"The agent's turn-annotated trajectory UP TO THE CURRENT TURN is a "
                f"read-only file at:\n"
                f"  {trajectory_path}\n"
                "This is the trajectory SO FAR — the agent has NOT finished and keeps "
                "working after your review; the file ending at the latest turn is the "
                "present moment, NOT a crash or a finish() call.\n"
                "READ it when a rubric needs PROCESS evidence (how the agent worked "
                "over turns — churn, repeated identical crashes, plateau-without-pivot, "
                "compute burned, suspicious-CV investigation), not just the end-state "
                "artifacts. Skim it; you do not have to read every turn. Artifact-"
                "grounded rubrics can usually be judged from the workspace alone.")
    # No trajectory delivered (should not happen once wired) — be explicit.
    return ("\n\n===== AGENT TRAJECTORY =====\n(trajectory not provided this wake — "
            "judge trajectory-grounded rubrics from whatever the workspace reveals.)")


def _workflow_block(store_dir: Path, k: int, static: bool) -> str:
    """The strict per-wake workflow instructions. In STATIC mode the reviewer has
    NO terminal tool, so it cannot write/run check scripts — it reads the files and
    reasons to a verdict (a linter over read-only W). In DYNAMIC mode it writes +
    runs bespoke checks and records each result."""
    if static:
        return (
            f"\n\nWORKFLOW, strictly (STATIC / read-only — you have NO terminal, so "
            f"you CANNOT run code; read files and REASON): read "
            "/reviewer/submission_ledger.jsonl (your prior verdicts this rollout) → "
            "list_rubrics → for each applicable rubric, OPEN the relevant artifacts "
            "(the submission csv, cv_scores.json, oof_predictions.csv, the modeling "
            "code) with the file-reading tool and reason about whether the rubric is "
            "satisfied from what you can SEE — do NOT attempt to execute anything → "
            "call record_check_result with your ASSIGNED importance_score + p_agent_success and the "
            "evidence you read) → repeat. Prioritize submission_coverage and "
            "submission-vs-OOF consistency (distribution/scale/inverse-transform "
            "mismatches you can spot by reading the numbers + the code). WARN + move on "
            "if a needed artifact is absent. Append your line to submission_ledger.jsonl "
            f"before verdicting. Call submit_verdict by turn {max(k-1,1)}. You have {k} "
            "turns; spend them reading + reasoning + recording, not exploring."
        )
    return (
        f"\n\nWORKFLOW, strictly (cwd={store_dir}): read /reviewer/submission_ledger.jsonl "
        "(your prior verdicts this rollout) → list_rubrics → read "
        "checks/README.md → for each applicable spec, REUSE its indexed script "
        "or WRITE a small bespoke one (using the resolved columns/metric above) → "
        "run it → read its `RESULT:` line → IMMEDIATELY call record_check_result "
        "with the numbers AND its two scores (importance_score + p_agent_success grounded "
        "in the result); do NOT run the next check first→ repeat. ORDER by "
        "cost/reliability: submission_coverage first, then the ALWAYS-runnable "
        "modeling checks (submission_distribution_health, cv_selection_reliability "
        "from the reported per-fold CV, train_test_drift from train+test) and the "
        "OOF checks when oof_predictions.csv exists (submission_vs_oof_consistency "
        "— the money check —, cv_reproduce, residual_vs_feature); WARN + move on "
        "for any OOF check if OOF is truly absent. Before you verdict, APPEND your "
        "line to submission_ledger.jsonl and check for a best-submission "
        "regression. When you write a new script, append it to checks/README.md. "
        f"Call submit_verdict by turn {max(k-1,1)}. You have {k} turns; spend them "
        "writing+running+recording checks, not exploring."
    )


def _run_reviewer_conversation(*, llm_kwargs, base_ws_dir, submission_path, data_dir,
                               store_dir, task_desc, command_timeout, max_iterations,
                               time_budget_seconds, convo_persist_dir, convo_id,
                               evolution: str = "evolving",
                               operationalization: str = "dynamic",
                               history: str = "pull",
                               trajectory_path: str | None = None,
                               pushed_transcript: str | None = None,
                               prior_trajectory_paths: Sequence[str] | None = None,
                               n_attempts: int = 5,
                               attempt_index: int = 0) -> dict:
    """Build tools + agent + conversation, run the loop, return {turns_used, cost}.
    Registers the reviewer's own footer via a ReviewerAgent subclass.

    All three unified-PRM axes (paper §sec:axes) are realized here on ONE
    configurable agent:
    - operationalization: "dynamic" (default) keeps the terminal tool so the
      reviewer WRITES + RUNS bespoke check scripts over the workspace; "static"
      DROPS the terminal tool so it can only READ files and reason (a linter over
      read-only W, no code execution).
    - evolution: "evolving" (default) registers the store-mutation tools
      (add/update/retire_rubric) so the reviewer can grow its own rubric set;
      "frozen" withholds them so the store is fixed for the whole run.
    - history: DELIVERY of the trajectory τ (which is ALWAYS available — no config
      is blind to the process, else trajectory-grounded rubrics could never be
      evaluated). "pull" (default) exposes the full trajectory as a read-only file
      (`trajectory_path`) the reviewer reads on demand; "push" inlines only the
      agent's NEW turns since the last wake (`pushed_transcript`) into the first
      message under a fixed window."""
    from openhands.tools.preset.default import register_default_tools, get_default_tools
    from openhands.sdk.tool.spec import Tool as _ToolSpec
    from openhands.sdk.tool import Tool
    from openhands.tools.terminal import TerminalTool as _TerminalTool

    static = (operationalization or "dynamic").lower() == "static"
    frozen = (evolution or "evolving").lower() == "frozen"

    # Default tools (terminal + file-editor) with the no-output timeout patch, so
    # a check that hangs can't wedge the reviewer (same patch as agent_runner).
    register_default_tools(enable_browser=False)
    base_tools = get_default_tools(enable_browser=False)
    base_tools = [
        _ToolSpec(name=_TerminalTool.name,
                  params={"no_change_timeout_seconds": int(command_timeout)})
        if t.name == _TerminalTool.name else t
        for t in base_tools
    ]
    # STATIC operationalization: strip the code-execution (terminal) tool. The
    # reviewer then inspects the read-only workspace with the file tools and
    # reasons to a verdict — no writing/running check scripts (a static linter).
    if static:
        base_tools = [t for t in base_tools if t.name != _TerminalTool.name]

    # Register + append the rubric-store tools + submit_verdict. The evolution
    # (add/update/retire) tools are only offered when evolution != "frozen".
    tool_specs: list[tuple[str, Any]] = [
        ("list_rubrics", ListRubricsTool),
        ("record_check_result", RecordCheckResultTool),
        ("submit_verdict", SubmitVerdictTool),
    ]
    if not frozen:
        tool_specs[1:1] = [
            ("add_rubric", AddRubricTool),
            ("update_rubric", UpdateRubricTool),
            ("retire_rubric", RetireRubricTool),
        ]
    for name, tool_cls in tool_specs:
        register_tool(name, tool_cls)
    tools = base_tools + [Tool(name=n) for n, _ in tool_specs]

    # Tool-calling mode must match the model family (see submission_selector).
    model_name = str(llm_kwargs.get("model", "")).lower()
    is_anthropic = ("claude" in model_name or "anthropic" in model_name
                    or "sonnet" in model_name or "opus" in model_name
                    or "haiku" in model_name)
    resolved = dict(llm_kwargs)
    if "native_tool_calling" not in resolved:
        resolved["native_tool_calling"] = is_anthropic
    llm = LLM(**{**resolved, "usage_id": "submission-reviewer",
                 "disable_stop_word": True})

    # Publish the budget as a module global so the module-level ReviewerAgent
    # (which cannot close over a local) reads it. Single reviewer per process.
    global _ACTIVE_BUDGET
    budget = ReviewerBudgetTracker(max_iterations, time_budget_seconds)
    _ACTIVE_BUDGET = budget

    # The reviewer's full instructions ride in the first message (built by
    # _build_first_message); we reuse the SDK's stock system prompt template.
    agent = ReviewerAgent(llm=llm, tools=tools)

    # Per-attempt message-stream continuity via create-or-resume: a stable
    # persistence_dir + conversation_id makes each wake within an attempt resume
    # the prior history; delete_on_close=False keeps it on disk for the next
    # submission event. A new attempt passes a new convo_id → fresh stream.
    # cwd is the reviewer's writable workspace (the rubric store dir).
    # Capture the reviewer's full message stream into base-agent-style steps[]
    # (turn/role/content/code/observation) so the whole trajectory — not just the
    # verdict summary — can be persisted to outputs/ like a base attempt_N.json.
    _steps: list[dict] = []
    _turn = {"n": 0}

    def _capture(event):
        try:
            kind = type(event).__name__
            if kind == "ActionEvent":
                act = getattr(event, "action", None)
                tool = getattr(event, "tool_name", "") or ""
                code = getattr(act, "command", None) if act is not None else None
                content = ""
                for c in (getattr(event, "llm_message", None).content
                          if getattr(event, "llm_message", None) else []) or []:
                    content += getattr(c, "text", "") or ""
                _turn["n"] += 1
                _steps.append({"turn": _turn["n"], "role": "assistant",
                               "content": content, "tool": tool,
                               "code": code if code is not None
                               else (act.model_dump() if act is not None else None),
                               "observation": None})
            elif kind == "ObservationEvent":
                obs = getattr(event, "observation", None)
                txt = ""
                try:
                    txt = obs.text if hasattr(obs, "text") else ""
                except Exception:
                    txt = ""
                if _steps:
                    _steps[-1]["observation"] = txt
            elif kind in ("MessageEvent", "SystemPromptEvent"):
                content = ""
                lm = getattr(event, "llm_message", None)
                for c in (lm.content if lm else []) or []:
                    content += getattr(c, "text", "") or ""
                _steps.append({"turn": _turn["n"],
                               "role": getattr(event, "source", "") or "message",
                               "content": content, "tool": None,
                               "code": None, "observation": None})
        except Exception:
            pass  # trajectory capture must never break the reviewer

    conv_kwargs = dict(
        agent=agent, workspace=LocalWorkspace(working_dir=str(store_dir)),
        callbacks=[_capture],
        max_iteration_per_run=max_iterations)
    if convo_persist_dir:
        conv_kwargs["persistence_dir"] = str(convo_persist_dir)
        conv_kwargs["delete_on_close"] = False
        if convo_id:
            # ConversationID is a uuid.UUID; derive a STABLE one from the caller's
            # string key (e.g. "{task}-attempt{n}") so every wake in an attempt
            # resumes the same conversation.
            import uuid as _uuid
            conv_kwargs["conversation_id"] = _uuid.uuid5(
                _uuid.NAMESPACE_URL, f"prm-reviewer/{convo_id}")
    else:
        conv_kwargs["delete_on_close"] = True
    conv = LocalConversation(**conv_kwargs)

    first_msg = _build_first_message(base_ws_dir, submission_path, data_dir,
                                     store_dir, task_desc, max_iterations,
                                     static=static, history=history,
                                     trajectory_path=trajectory_path,
                                     pushed_transcript=pushed_transcript,
                                     prior_trajectory_paths=prior_trajectory_paths,
                                     n_attempts=n_attempts,
                                     attempt_index=attempt_index)
    try:
        conv.send_message(first_msg)
        conv.run()
    finally:
        conv.close()
        _ACTIVE_BUDGET = None

    # Reviewer LLM spend. The SDK Metrics object exposes `accumulated_cost`
    # (float $) and `accumulated_token_usage` (a TokenUsage with prompt/
    # completion/cache token counts) — NOT accumulated_prompt_tokens (which does
    # not exist and previously logged None). Read the real fields so every
    # reviewer_call_NN.json carries dollars + token breakdown for the paper's
    # cost accounting.
    cost = {}
    try:
        m = llm.metrics
        tu = getattr(m, "accumulated_token_usage", None)
        cost = {
            "usd": round(float(getattr(m, "accumulated_cost", 0.0) or 0.0), 4),
            "input_tokens": getattr(tu, "prompt_tokens", None) if tu else None,
            "output_tokens": getattr(tu, "completion_tokens", None) if tu else None,
            "cache_read_tokens": getattr(tu, "cache_read_tokens", None) if tu else None,
            "cache_write_tokens": getattr(tu, "cache_write_tokens", None) if tu else None,
            "reasoning_tokens": getattr(tu, "reasoning_tokens", None) if tu else None,
        }
    except Exception:
        pass
    return {"turns_used": budget.turn_count, "cost": cost, "steps": _steps}
