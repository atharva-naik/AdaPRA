"""
Compare two trajectory collection runs by Kaggle percentile.

Uses private_percentile when available, falls back to public percentile.
Prints a markdown table and optionally dumps JSON for downstream use.

Usage:
  python scripts/adapra/compare_runs.py \
    --baseline outputs/adapra/bon/trajectories/dspredict-easy-bon5 \
    --candidate outputs/adapra/trajectories_v2/dspredict-easy-bon5-v2/dspredict-easy \
    [--json results.json] \
    [--all]  # include tasks only in one run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def get_percentile(outcome: dict) -> float | None:
    """Private percentile if available, else public."""
    priv = outcome.get("private_percentile")
    if priv is not None:
        return priv
    return outcome.get("percentile")


def load_run_scores(run_dir: str) -> tuple[dict[str, list[dict]], dict[str, int], dict[str, int], dict[str, int]]:
    """Load all trajectory JSONs from a run directory, return (task -> [scores], task -> total_count, task -> valid_count, task -> error_count)."""
    scores: dict[str, list[dict]] = {}
    totals: dict[str, int] = {}
    valids: dict[str, int] = {}
    errors: dict[str, int] = {}
    pattern = os.path.join(run_dir, "**", "*.json")
    files = glob(pattern, recursive=True)
    if not files:
        pattern = os.path.join(run_dir, "*.json")
        files = glob(pattern)

    # Pre-scan: find .error.json files that have a corresponding successful .json
    successful_bases = set()
    for f in files:
        if ".error.json" not in f:
            successful_bases.add(f.replace(".json", ""))

    for f in sorted(files):
        # Skip bare error.json files (V2 harness error logs)
        if os.path.basename(f) == "error.json":
            continue
        # Count .error.json toward totals only if no successful version exists
        if ".error.json" in f:
            base = f.replace(".error.json", "")
            if base in successful_bases:
                continue  # skip entirely — successful retry exists
            basename = os.path.basename(f)
            parts = basename.split("__")
            task_for_count = parts[0] if len(parts) >= 2 else os.path.basename(os.path.dirname(f))
            if task_for_count:
                totals[task_for_count] = totals.get(task_for_count, 0) + 1
                errors[task_for_count] = errors.get(task_for_count, 0) + 1
            continue
        try:
            with open(f) as fh:
                traj = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        # Skip non-trajectory JSONs that the recursive glob picks up
        # (e.g. csv_history_attempt_*/manifest.json, which are lists, and any
        # summary/oplog files). Only trajectory records are dicts with fields.
        if not isinstance(traj, dict):
            continue

        # Determine task ID early for counting
        task_for_count = traj.get("task_id") or ""
        if not task_for_count:
            task_field = traj.get("task")
            if isinstance(task_field, dict):
                task_for_count = task_field.get("task_id") or task_field.get("challenge_name") or ""
            elif isinstance(task_field, str):
                task_for_count = task_field
        if not task_for_count:
            basename = os.path.basename(f)
            parts = basename.split("__")
            if len(parts) >= 2:
                task_for_count = parts[0]
            else:
                task_for_count = os.path.basename(os.path.dirname(f))
        if task_for_count:
            totals[task_for_count] = totals.get(task_for_count, 0) + 1

        outcome = traj.get("outcome", {})
        if outcome.get("valid_submission", False) and task_for_count:
            valids[task_for_count] = valids.get(task_for_count, 0) + 1
        if not outcome.get("valid_submission", False):
            continue

        pct = get_percentile(outcome)
        if pct is None:
            continue

        # Determine task ID
        task = traj.get("task_id") or ""
        if not task:
            task_field = traj.get("task")
            if isinstance(task_field, dict):
                task = task_field.get("task_id") or task_field.get("challenge_name") or ""
            elif isinstance(task_field, str):
                task = task_field
        if not task:
            # Fall back to filename parsing: task__model__idx.json
            basename = os.path.basename(f)
            parts = basename.split("__")
            if len(parts) >= 2:
                task = parts[0]
            else:
                # Try parent dir name
                task = os.path.basename(os.path.dirname(f))

        if not task:
            continue

        # Determine attempt index
        attempt = traj.get("attempt_index")
        if attempt is None:
            # Try from filename
            basename = os.path.basename(f)
            if "attempt_" in basename:
                try:
                    attempt = int(basename.split("attempt_")[1].split(".")[0])
                except (ValueError, IndexError):
                    attempt = None
            elif "__" in basename:
                try:
                    idx_part = basename.rsplit("__", 1)[1].replace(".json", "")
                    attempt = int(idx_part)
                except (ValueError, IndexError):
                    attempt = None

        scores.setdefault(task, []).append({
            "percentile": pct,
            "attempt": attempt,
            "file": os.path.basename(f),
            "path": f,
            "private": outcome.get("private_percentile") is not None,
        })

    return scores, totals, valids, errors


def main():
    parser = argparse.ArgumentParser(description="Compare two trajectory runs")
    parser.add_argument("--baseline", required=True, help="Path to baseline run dir")
    parser.add_argument("--candidate", required=True, help="Path to candidate run dir")
    parser.add_argument("--json", default=None, help="Output JSON path")
    parser.add_argument("--all", action="store_true", help="Include tasks only in one run")
    parser.add_argument("--baseline-label", default="V1", help="Label for baseline")
    parser.add_argument("--candidate-label", default="V2", help="Label for candidate")
    args = parser.parse_args()

    baseline, bl_totals, bl_valids, bl_errors = load_run_scores(args.baseline)
    candidate, cl_totals, cl_valids, cl_errors = load_run_scores(args.candidate)

    if not baseline:
        print(f"ERROR: No scored trajectories found in {args.baseline}", file=sys.stderr)
        sys.exit(1)
    if not candidate:
        print(f"ERROR: No scored trajectories found in {args.candidate}", file=sys.stderr)
        sys.exit(1)

    if args.all:
        all_tasks = sorted(set(
            list(baseline.keys()) + list(candidate.keys()) +
            list(bl_totals.keys()) + list(cl_totals.keys())
        ))
    else:
        # Tasks attempted by both runs (have any files in either scored or totals)
        bl_all_tasks = set(baseline.keys()) | set(bl_totals.keys())
        cl_all_tasks = set(candidate.keys()) | set(cl_totals.keys())
        all_tasks = sorted(bl_all_tasks & cl_all_tasks)

    if not all_tasks:
        print("ERROR: No common tasks found between runs", file=sys.stderr)
        sys.exit(1)

    bl = args.baseline_label
    cl = args.candidate_label

    # Header
    print(f"| Task | {bl} Rollouts | {bl} Best | {cl} Rollouts | {cl} Best | Delta |")
    print(f"|------|------------|---------|------------|---------|-------|")

    results = []
    bl_bests = []
    cl_bests = []

    for task in all_tasks:
        short = (task
                 .replace("playground-series-", "")
                 .replace("house-prices-advanced-regression-techniques", "house-prices"))

        b_entries = sorted(baseline.get(task, []), key=lambda x: -x["percentile"])
        c_entries = sorted(candidate.get(task, []), key=lambda x: -x["percentile"])

        b_pcts = [e["percentile"] for e in b_entries]
        c_pcts = [e["percentile"] for e in c_entries]

        b_str = ", ".join(f"{x:.1f}" for x in b_pcts) if b_pcts else "-"
        c_str = ", ".join(
            f"a{e['attempt']}:{e['percentile']:.1f}" if e["attempt"] is not None
            else f"{e['percentile']:.1f}"
            for e in c_entries
        ) if c_entries else "-"

        b_best = max(b_pcts) if b_pcts else None
        c_best = max(c_pcts) if c_pcts else None

        if b_best is not None and c_best is not None:
            delta = c_best - b_best
            delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        elif b_best is not None:
            delta = None
            delta_str = "-"
        elif c_best is not None:
            delta = None
            delta_str = "-"
        else:
            delta = None
            delta_str = "-"
        # For W/T/L: treat no score as 0
        bl_bests.append(b_best if b_best is not None else 0)
        cl_bests.append(c_best if c_best is not None else 0)

        b_best_str = f"{b_best:.1f}" if b_best is not None else "-"
        c_best_str = f"{c_best:.1f}" if c_best is not None else "-"

        print(f"| {short} | {b_str} | {b_best_str} | {c_str} | {c_best_str} | {delta_str} |")

        results.append({
            "task": task,
            "task_short": short,
            "baseline_scores": b_pcts,
            "candidate_scores": c_pcts,
            "baseline_best": b_best,
            "candidate_best": c_best,
            "delta": delta,
        })

    # Summary row
    if bl_bests and cl_bests:
        n = len(bl_bests)
        b_mean = sum(bl_bests) / n
        c_mean = sum(cl_bests) / n
        d_mean = c_mean - b_mean
        # Paper-style mean: only over tasks where that run has scores (no 0 penalty)
        bl_nonzero = [b for b in bl_bests if b > 0]
        cl_nonzero = [c for c in cl_bests if c > 0]
        b_mean_scored = sum(bl_nonzero) / len(bl_nonzero) if bl_nonzero else 0
        c_mean_scored = sum(cl_nonzero) / len(cl_nonzero) if cl_nonzero else 0
        wins = sum(1 for a, b in zip(bl_bests, cl_bests) if b > a)
        losses = sum(1 for a, b in zip(bl_bests, cl_bests) if b < a)
        ties = n - wins - losses

        # Above-median stats (all scored submissions)
        all_bl_pcts = [p for t in all_tasks for e in baseline.get(t, []) for p in [e["percentile"]]]
        all_cl_pcts = [p for t in all_tasks for e in candidate.get(t, []) for p in [e["percentile"]]]
        bl_above = sum(1 for p in all_bl_pcts if p > 50)
        cl_above = sum(1 for p in all_cl_pcts if p > 50)

        # Best-of-pool above median
        bl_best_above = sum(1 for b in bl_bests if b > 50)
        cl_best_above = sum(1 for c in cl_bests if c > 50)

        # Valid submission stats (valid_submission=True / total attempts including errors)
        bl_total_all = sum(bl_totals.get(t, 0) for t in all_tasks)
        cl_total_all = sum(cl_totals.get(t, 0) for t in all_tasks)
        bl_valid = sum(bl_valids.get(t, 0) for t in all_tasks)
        cl_valid = sum(cl_valids.get(t, 0) for t in all_tasks)

        # Error stats
        bl_err_all = sum(bl_errors.get(t, 0) for t in all_tasks)
        cl_err_all = sum(cl_errors.get(t, 0) for t in all_tasks)

        # Task-level stats: how many tasks have at least one valid submission
        bl_tasks_with_valid = sum(1 for t in all_tasks if bl_valids.get(t, 0) > 0)
        cl_tasks_with_valid = sum(1 for t in all_tasks if cl_valids.get(t, 0) > 0)
        n_tasks = len(all_tasks)

        # Task-level above median: best-of-pool > 50 (including tasks only in one run)
        bl_tasks_above = sum(1 for t in all_tasks if baseline.get(t) and max(e["percentile"] for e in baseline[t]) > 50)
        cl_tasks_above = sum(1 for t in all_tasks if candidate.get(t) and max(e["percentile"] for e in candidate[t]) > 50)

        print()

        # Compact summary table
        d_scored = c_mean_scored - b_mean_scored
        rows = [
            ("Best-of-pool mean", f"{b_mean:.1f}%", f"{c_mean:.1f}% ({d_mean:+.1f})"),
            ("  (scored only)", f"{b_mean_scored:.1f}% ({len(bl_nonzero)}t)", f"{c_mean_scored:.1f}% ({len(cl_nonzero)}t)"),
            ("Best-of-pool >median", f"{bl_tasks_above}/{n_tasks}", f"{cl_tasks_above}/{n_tasks}"),
            ("W/T/L", "", f"{wins}/{ties}/{losses}"),
            ("Tasks w/ valid sub", f"{bl_tasks_with_valid}/{n_tasks}", f"{cl_tasks_with_valid}/{n_tasks}"),
            ("Above median (all)", f"{bl_above}/{len(all_bl_pcts)} ({100*bl_above/len(all_bl_pcts):.0f}%)" if all_bl_pcts else "0/0", f"{cl_above}/{len(all_cl_pcts)} ({100*cl_above/len(all_cl_pcts):.0f}%)" if all_cl_pcts else "0/0"),
            ("Valid submissions", f"{bl_valid}/{bl_total_all} ({100*bl_valid/bl_total_all:.0f}%)" if bl_total_all else "0/0", f"{cl_valid}/{cl_total_all} ({100*cl_valid/cl_total_all:.0f}%)" if cl_total_all else "0/0"),
            ("Timeouts", f"{bl_err_all}/{bl_total_all} ({100*bl_err_all/bl_total_all:.0f}%)" if bl_total_all else "0/0", f"{cl_err_all}/{cl_total_all} ({100*cl_err_all/cl_total_all:.0f}%)" if cl_total_all else "0/0"),
        ]
        col0_w = max(len(r[0]) for r in rows) + 2
        col1_w = max(len(r[1]) for r in rows) + 2
        col2_w = max(len(r[2]) for r in rows) + 2
        hline = f"  ├{'─'*(col0_w)}┼{'─'*(col1_w)}┼{'─'*(col2_w)}┤"
        print()
        print(f"  ┌{'─'*(col0_w)}┬{'─'*(col1_w)}┬{'─'*(col2_w)}┐")
        print(f"  │{'Metric':^{col0_w}}│{bl:^{col1_w}}│{cl:^{col2_w}}│")
        print(f"  ├{'─'*(col0_w)}┼{'─'*(col1_w)}┼{'─'*(col2_w)}┤")
        for i, (metric, v1_val, v2_val) in enumerate(rows):
            print(f"  │ {metric:<{col0_w-2}} │ {v1_val:<{col1_w-2}} │ {v2_val:<{col2_w-2}} │")
            if i < len(rows) - 1:
                print(hline)
        print(f"  └{'─'*(col0_w)}┴{'─'*(col1_w)}┴{'─'*(col2_w)}┘")
        # Check how many entries used private percentile
        all_entries = [e for t in all_tasks for e in baseline.get(t, []) + candidate.get(t, [])]
        n_private = sum(1 for e in all_entries if e.get("private"))
        n_public_only = len(all_entries) - n_private
        score_source = "private percentile" if n_public_only == 0 else (
            f"private percentile ({n_private}/{len(all_entries)}), public fallback ({n_public_only}/{len(all_entries)})"
        )
        print(f"\n  Tasks: {len(all_tasks)} total, {n} with scores in both runs")
        print(f"  Scores: {score_source}")

        summary = {
            "tasks_compared": len(all_tasks),
            "common_scored_tasks": n,
            "baseline_mean": round(b_mean, 1),
            "candidate_mean": round(c_mean, 1),
            "delta_mean": round(d_mean, 1),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "baseline_valid": bl_valid,
            "baseline_total": bl_total_all,
            "baseline_errors": bl_err_all,
            "candidate_valid": cl_valid,
            "candidate_total": cl_total_all,
            "candidate_errors": cl_err_all,
        }
    else:
        summary = {}

    if args.json:
        output = {"summary": summary, "tasks": results}
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nJSON written to {args.json}")


if __name__ == "__main__":
    main()
