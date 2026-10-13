#!/usr/bin/env python3
"""Frequency of rubric-store operations (add / update / retire) for AdaPRA and
AdaPRA - PRA on DSPredict-Easy and DSPredict-Hard.

Reads each task's reviewer_rubric_store/rubric_oplog.jsonl. Per-task rates use
the tasks that have an op log as the denominator, the same convention as the
adaptation statistics in the results section (Easy AdaPRA 0.8 adds/task, 51% of
tasks with >=1 add; Hard AdaPRA 1.2 adds/task, 79%).

  (a) operations per task, by operation type
  (b) adds per task, by the attempt in which the rubric was added

Usage: python plot_rubric_store_ops.py [--out our_paper/figures/rubric_store_ops.pdf]
"""
import argparse
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
T4 = REPO / "outputs/adapra/trajectories_v4"
RUNS = [
    ("Easy", "AdaPRA", T4 / "dsp-easy-rev-dyn-evolve-pull-xattempt/dspredict-easy"),
    ("Easy", "$-$PRA", T4 / "dsp-easy-rev-dyn-evolve-nopra-pull-xattempt/dspredict-easy"),
    ("Hard", "AdaPRA", Path("/path/to/scratch/dsgym_v4_hard_evolve_xattempt_out/trajectories_v4/"
                            "dspredict-hard-evolve-xattempt/dspredict-hard")),
    ("Hard", "$-$PRA", Path("/path/to/scratch/dsgym_v4_hard_evolve_nopra_xattempt_out/trajectories_v4/"
                            "dspredict-hard-evolve-nopra-xattempt/dspredict-hard")),
]
OPS = ["add", "update", "retire"]
N_ATTEMPTS = 5


def count_ops(run_dir):
    logs = sorted(run_dir.glob("*/reviewer_rubric_store/rubric_oplog.jsonl"))
    by_op = collections.Counter()
    adds_by_attempt = collections.Counter()
    tasks_with_add = set()
    for f in logs:
        for line in open(f):
            o = json.loads(line)
            op = o.get("op")
            if op not in OPS:
                continue
            by_op[op] += 1
            if op == "add":
                adds_by_attempt[int(o.get("attempt") or 0)] += 1
                tasks_with_add.add(f.parent.parent.name)
    return len(logs), by_op, adds_by_attempt, len(tasks_with_add)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "our_paper/figures/rubric_store_ops.pdf"))
    args = ap.parse_args()

    stats = []
    for bench, setting, d in RUNS:
        n, by_op, by_att, n_add = count_ops(d)
        stats.append((f"{bench} {setting}", n, by_op, by_att))
        print(f"{bench:4s} {setting:7s} tasks_with_log={n:2d} "
              + " ".join(f"{op}={by_op[op]}" for op in OPS)
              + f"  adds/task={by_op['add'] / n:.2f}  tasks_with_add={100 * n_add / n:.0f}%"
              + f"  adds_by_attempt={[by_att[a] for a in range(N_ATTEMPTS)]}")

    colors = ["#1f77b4", "#9ecae1", "#d62728", "#fc9272"]
    hatches = ["", "//", "", "//"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    w = 0.2

    ax = axes[0]
    x = np.arange(len(OPS))
    for i, (label, n, by_op, _) in enumerate(stats):
        bars = ax.bar(x + (i - 1.5) * w, [by_op[op] / n for op in OPS], w, label=label,
                      color=colors[i], hatch=hatches[i], edgecolor="black", linewidth=0.4)
        # raw counts on top, so zero-height bars are still readable
        for b, op in zip(bars, OPS):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, str(by_op[op]),
                    ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([op.capitalize() for op in OPS])
    ax.set_ylabel("Operations per task")
    ax.set_title("(a) Operations by type", fontsize=9)

    ax = axes[1]
    x = np.arange(N_ATTEMPTS)
    for i, (label, n, _, by_att) in enumerate(stats):
        ax.bar(x + (i - 1.5) * w, [by_att[a] / n for a in range(N_ATTEMPTS)], w, label=label,
               color=colors[i], hatch=hatches[i], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(a + 1) for a in range(N_ATTEMPTS)])
    ax.set_xlabel("Attempt")
    ax.set_ylabel("Adds per task")
    ax.set_title("(b) Adds by attempt", fontsize=9)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.yaxis.label.set_size(8)
        ax.xaxis.label.set_size(8)
    axes[1].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
