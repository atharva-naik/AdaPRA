#!/usr/bin/env python
"""Reproduce the paper's API cost table (tab:costs, appendix "Cost").

Per run (one row of the table):
  Attempts  : number of attempt_*.json files in the run (including attempts that
              crashed before recording a cost, and unreadable/truncated JSONs).
  Agent ($) : sum of metadata.cost over readable attempts (Qwen3-Coder-Next on Bedrock);
              unreadable JSONs contribute $0 and are reported on stderr.
  PRA ($)   : sum of cost.usd over every PRA invocation record
              {task}/prm_reviews_attempt_*/reviewer_call_*.json (Claude Sonnet 4.6).
              Harness-only runs have no such files and print "---".
  Total ($) : Agent + PRA.
  $/attempt : Total / Attempts.

Usage:
  python scripts/adapra/paper_cost_table.py
  python scripts/adapra/paper_cost_table.py --latex
"""
import argparse
import glob
import json
import os
import sys

T = "outputs/adapra/trajectories_v4"
BACKUP = "/path/to/backup/trajectories_v4"

# (benchmark, row label, trajectory dir {task}/attempt_N.json)
RUNS = [
    ("DSPredict-Easy", "Harness only", f"{T}/dspredict-easy-bon5-qwen3-v2/dspredict-easy"),
    ("DSPredict-Easy", "+ PRA + adaptive rubrics (AdaPRA)", f"{T}/dsp-easy-rev-dyn-evolve-pull-xattempt/dspredict-easy"),
    ("DSPredict-Easy", "- adaptive rubrics", f"{T}/dsp-easy-rev-dyn-frozen-pull-xattempt/dspredict-easy"),
    ("DSPredict-Easy", "- PRA", f"{T}/dsp-easy-rev-dyn-evolve-nopra-pull-xattempt/dspredict-easy"),
    ("DSPredict-Hard", "Harness only", f"{BACKUP}/dspredict-hard-noprm/dspredict-hard"),
    ("DSPredict-Hard", "+ PRA + adaptive rubrics (AdaPRA)",
     "/path/to/scratch/dsgym_v4_hard_evolve_xattempt_out/trajectories_v4/dspredict-hard-evolve-xattempt/dspredict-hard"),
    ("DSPredict-Hard", "- adaptive rubrics",
     "/path/to/scratch/dsgym_v4_hard_frozen_xattempt_v2_out/trajectories_v4/dspredict-hard-frozen-xattempt-v2/dspredict-hard"),
    ("DSPredict-Hard", "- PRA",
     "/path/to/scratch/dsgym_v4_hard_evolve_nopra_xattempt_out/trajectories_v4/dspredict-hard-evolve-nopra-xattempt/dspredict-hard"),
]


def run_cost(root):
    attempts, agent, bad = 0, 0.0, []
    for f in sorted(glob.glob(os.path.join(root, "*", "attempt_*.json"))):
        attempts += 1
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            bad.append(f)
            continue
        agent += (d.get("metadata") or {}).get("cost") or 0.0
    calls = glob.glob(os.path.join(root, "*", "prm_reviews_attempt_*", "reviewer_call_*.json"))
    pra = 0.0
    for f in calls:
        try:
            pra += (json.load(open(f)).get("cost") or {}).get("usd") or 0.0
        except (json.JSONDecodeError, OSError):
            bad.append(f)
    return attempts, agent, (pra if calls else None), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="print LaTeX table rows")
    args = ap.parse_args()
    for bench, label, root in RUNS:
        if not os.path.isdir(root):
            print(f"[missing] {bench} {label}: {root}", file=sys.stderr)
            continue
        n, agent, pra, bad = run_cost(root)
        total = agent + (pra or 0.0)
        for f in bad:
            print(f"[unreadable] {f}", file=sys.stderr)
        if args.latex:
            fmt = lambda x: f"{x:,.0f}".replace(",", "{,}")
            p = "---" if pra is None else fmt(pra)
            print(f"{bench:15s} {label:36s} & {n} & {fmt(agent)} & {p} & "
                  f"{fmt(total)} & {total / n:.1f} \\\\")
        else:
            p = "---" if pra is None else f"{pra:8.0f}"
            print(f"{bench:15s} {label:36s} att={n:4d} agent=${agent:6.0f} PRA=${p:>6s} "
                  f"total=${total:6.0f} $/att={total / n:4.1f}")


if __name__ == "__main__":
    main()
