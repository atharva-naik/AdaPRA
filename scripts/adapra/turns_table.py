#!/usr/bin/env python3
"""Agent and reviewer turns per attempt for the paper tables.

For each run, reports mean agent turns and mean reviewer turns per attempt
(macro: mean within task, then mean over tasks).  Reviewer turns are summed
from prm_reviews_attempt_N/reviewer_call_*.json → turns_used.

Usage:
  python turns_table.py --split easy
  python turns_table.py --split hard
  python turns_table.py --split easy --latex
"""
import argparse
import glob
import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

EASY_RUNS = [
    ("No PRM",         "outputs/adapra/trajectories_v4/dspredict-easy-bon5-qwen3-v2/dspredict-easy"),
    ("$-$ adapt. rub", "outputs/adapra/trajectories_v4/dsp-easy-rev-dyn-frozen-pull-xattempt/dspredict-easy"),
    ("AdaPRA",         "outputs/adapra/trajectories_v4/dsp-easy-rev-dyn-evolve-pull-xattempt/dspredict-easy"),
    ("$-$ PRA",        "outputs/adapra/trajectories_v4/dsp-easy-rev-dyn-evolve-nopra-pull-xattempt/dspredict-easy"),
]

HARD_RUNS = [
    ("No PRM",         "/path/to/backup/trajectories_v4/dspredict-hard-noprm/dspredict-hard"),
    ("$-$ adapt. rub", "/path/to/scratch/dsgym_v4_hard_frozen_xattempt_v2_out/trajectories_v4/dspredict-hard-frozen-xattempt-v2/dspredict-hard"),
    ("AdaPRA",         "/path/to/scratch/dsgym_v4_hard_evolve_xattempt_out/trajectories_v4/dspredict-hard-evolve-xattempt/dspredict-hard"),
    ("$-$ PRA",        "/path/to/scratch/dsgym_v4_hard_evolve_nopra_xattempt_out/trajectories_v4/dspredict-hard-evolve-nopra-xattempt/dspredict-hard"),
]


def reviewer_turns_for_attempt(task_dir: Path, attempt_idx: int) -> int:
    """Sum turns_used from reviewer_call_*.json for one attempt."""
    review_dir = task_dir / f"prm_reviews_attempt_{attempt_idx}"
    if not review_dir.is_dir():
        return 0
    total = 0
    for f in sorted(review_dir.glob("reviewer_call_*.json")):
        try:
            total += json.load(open(f)).get("turns_used", 0)
        except (json.JSONDecodeError, OSError):
            pass
    return total


def compute_turns(run_path: str):
    """Return per-task macro means: (agent_turns, reviewer_turns, n_tasks, n_attempts)."""
    base = Path(run_path) if os.path.isabs(run_path) else REPO / run_path
    if not base.is_dir():
        return None, None, 0, 0

    task_agent_means = []
    task_reviewer_means = []
    total_attempts = 0

    for task_dir in sorted(base.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("_"):
            continue
        atts = sorted(task_dir.glob("attempt_*.json"))
        if not atts:
            continue

        agent_turns = []
        rev_turns = []
        for att_path in atts:
            try:
                d = json.load(open(att_path))
            except (json.JSONDecodeError, OSError):
                continue
            nt = d.get("num_turns")
            if nt is None:
                continue
            att_idx = int(att_path.stem.split("_")[1])
            agent_turns.append(nt)
            rev_turns.append(reviewer_turns_for_attempt(task_dir, att_idx))
            total_attempts += 1

        if agent_turns:
            task_agent_means.append(statistics.mean(agent_turns))
            task_reviewer_means.append(statistics.mean(rev_turns))

    if not task_agent_means:
        return None, None, 0, total_attempts

    macro_agent = statistics.mean(task_agent_means)
    macro_reviewer = statistics.mean(task_reviewer_means)
    return macro_agent, macro_reviewer, len(task_agent_means), total_attempts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["easy", "hard"], default="easy")
    parser.add_argument("--latex", action="store_true", help="emit LaTeX table fragment")
    args = parser.parse_args()

    runs = EASY_RUNS if args.split == "easy" else HARD_RUNS
    label = "DSPredict-Easy" if args.split == "easy" else "DSPredict-Hard"

    results = []
    for name, path in runs:
        agent, reviewer, n_tasks, n_att = compute_turns(path)
        results.append((name, agent, reviewer, n_tasks, n_att))

    if not args.latex:
        print(f"\n{'='*70}")
        print(f"Turns per attempt (macro) — {label}")
        print(f"{'='*70}")
        print(f"{'Method':<18} {'Tasks':>5} {'Att':>5} {'Agent':>7} {'Review':>7} {'Total':>7}")
        print("-" * 55)
        for name, agent, reviewer, n_tasks, n_att in results:
            if agent is None:
                print(f"{name:<18} {'—':>5} {'—':>5} {'—':>7} {'—':>7} {'—':>7}")
            else:
                total = agent + reviewer
                print(f"{name:<18} {n_tasks:>5} {n_att:>5} {agent:>7.1f} {reviewer:>7.1f} {total:>7.1f}")
    else:
        for name, agent, reviewer, n_tasks, n_att in results:
            if agent is None:
                print(f"{name} & -- & -- & -- \\\\")
            else:
                total = agent + reviewer
                print(f"{name} & {agent:.1f} & {reviewer:.1f} & {total:.1f} \\\\")


if __name__ == "__main__":
    main()
