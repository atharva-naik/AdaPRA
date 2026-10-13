#!/usr/bin/env python3
"""Medal rates for every run in the registry, from stored percentiles (no rescore).

For each run we take the best-of-pool submission per task (blended: private
percentile where the comp has a private leaderboard, else public — same rule as
update_results_table.py), invert the stored percentile to a rank via
medal_from_percentile, and tally gold/silver/bronze/any across tasks.

Denominators: `total_tasks` from the registry (38 Easy / 54 Hard) so medal rate is
comparable across runs regardless of how many tasks finished. Also prints the
scored-task denom. A "near-cutoff" count flags tasks whose derived rank sits within
NEAR ranks of a medal threshold — those are the only ones where the tie-inversion
could be off by a medal, so spot-rescore them before quoting an individual claim.

Usage:
  python scripts/adapra/medal_rates.py                # all runs
  python scripts/adapra/medal_rates.py --runs V1 V4-PRM-reviewer-v6
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from dsgym.eval.metrics.dspredict.dspredict_metric import (  # noqa: E402
    medal_from_percentile,
)

REGISTRY_PATH = REPO_ROOT / "configs" / "adapra" / "run_registry.json"
EASY_LB = REPO_ROOT / "data" / "DSPredict_Offline_Leaderboard" / "easy_leaderboard_stats_2026-02-01.json"
HARD_LB = REPO_ROOT / "data" / "DSPredict_Offline_Leaderboard" / "hard_leaderboard_stats_2026-02-01.json"
NEAR = 3  # tasks within this many ranks of a cutoff are flagged for rescore-confirm


def _n_teams(lb_entry, use_private):
    scores = lb_entry.get("private_scores") if use_private else lb_entry.get("public_scores")
    return len(scores) if scores else None


def _cutoffs(total):
    """(gold, silver, bronze) rank cutoffs — mirrors dspredict_metric's ladder.

    Duplicated here (small, stable) so we don't touch the original metric module.
    """
    import math

    def pct_ceil(p):
        return math.ceil(p * total)

    if total <= 99:
        return pct_ceil(0.10), pct_ceil(0.20), pct_ceil(0.40)
    elif 100 <= total <= 249:
        return min(10, total), pct_ceil(0.20), pct_ceil(0.40)
    elif 250 <= total <= 999:
        return (min(total, 10 + math.ceil(0.002 * total)), min(50, total), min(100, total))
    else:
        return (min(total, 10 + math.ceil(0.002 * total)), pct_ceil(0.05), pct_ceil(0.10))


def _near_cutoff(rank, n_teams):
    if rank < 1 or not n_teams:
        return False
    return any(abs(rank - t) <= NEAR for t in _cutoffs(n_teams + 1))


def best_medals_for_run(base: Path, run_format: str, lb: dict, total_tasks: int):
    # task_id -> (best_pct, use_private, n_teams) for the best-of-pool valid attempt
    task_best = {}

    def consider(task, o):
        if not o.get("valid_submission", False):
            return
        priv = o.get("private_percentile")
        use_private = priv is not None
        pct = priv if use_private else o.get("percentile")
        if pct is None:
            return
        lb_entry = lb.get(task)
        if not lb_entry:
            return
        n_teams = _n_teams(lb_entry, use_private)
        if n_teams is None:
            return
        if task not in task_best or pct > task_best[task][0]:
            task_best[task] = (pct, use_private, n_teams)

    if run_format == "flat":
        for f in sorted(base.glob("*.json")):
            if "mock" in f.name or ".error" in f.name or f.name == "error.json":
                continue
            try:
                r = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            task = r.get("task_id") or f.stem.split("__")[0]
            consider(task, r.get("outcome", {}))
    else:
        for task_dir in sorted(base.iterdir()):
            if not task_dir.is_dir():
                continue
            for f in sorted(task_dir.glob("attempt_*.json")):
                try:
                    r = json.loads(f.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                consider(task_dir.name, r.get("outcome", {}))

    tally = {"gold": 0, "silver": 0, "bronze": 0, "none": 0}
    any_medal = 0
    near = 0
    per_task = {}
    for task, (pct, use_private, n_teams) in task_best.items():
        rank, medal = medal_from_percentile(n_teams, pct)
        tally[medal] = tally.get(medal, 0) + 1
        if medal in ("gold", "silver", "bronze"):
            any_medal += 1
        if _near_cutoff(rank, n_teams):
            near += 1
        per_task[task] = {"pct": round(pct, 3), "rank": rank, "medal": medal,
                          "n_teams": n_teams, "private": use_private}

    return {
        "scored_tasks": len(task_best),
        "total_tasks": total_tasks,
        "gold": tally["gold"], "silver": tally["silver"], "bronze": tally["bronze"],
        "any_medal": any_medal,
        "near_cutoff": near,
        "per_task": per_task,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="Only these run IDs (default: all)")
    ap.add_argument("--per-task", action="store_true", help="Print per-task medal for each run")
    ap.add_argument("--json-out", help="Write full results (incl. per-task) to this path")
    args = ap.parse_args()

    registry = json.loads(REGISTRY_PATH.read_text())
    easy_lb = json.loads(EASY_LB.read_text())
    hard_lb = json.loads(HARD_LB.read_text())

    run_ids = args.runs or list(registry["runs"].keys())

    print(f"{'Run':<22}{'scored':>7}{'gold':>6}{'silver':>7}{'bronze':>7}"
          f"{'any':>5}{'medal% (/tot)':>14}{'near':>6}")
    print("-" * 74)

    out = {}
    for rid in run_ids:
        if rid not in registry["runs"]:
            print(f"⚠️  unknown run: {rid}", file=sys.stderr)
            continue
        info = registry["runs"][rid]
        base = REPO_ROOT / info["path"] if not str(info["path"]).startswith("/") else Path(info["path"])
        if not base.exists():
            print(f"{rid:<22}  (path not found)")
            continue
        total_tasks = info.get("total_tasks", 38)
        is_hard = total_tasks == 54 or "hard" in str(info["path"]).lower()
        lb = hard_lb if is_hard else easy_lb
        res = best_medals_for_run(base, info.get("format", "nested"), lb, total_tasks)
        out[rid] = res
        rate = 100 * res["any_medal"] / total_tasks
        print(f"{rid:<22}{res['scored_tasks']:>4}/{total_tasks:<2}"
              f"{res['gold']:>6}{res['silver']:>7}{res['bronze']:>7}"
              f"{res['any_medal']:>5}{rate:>12.1f}%{res['near_cutoff']:>6}")
        if args.per_task:
            for t, d in sorted(res["per_task"].items(), key=lambda kv: -kv[1]["pct"]):
                if d["medal"] != "none":
                    print(f"    {t:<48}{d['medal']:>7}  rank {d['rank']}/{d['n_teams']}  pct={d['pct']}")

    print("\nmedal% = any-medal tasks / total_tasks (38 Easy, 54 Hard).")
    print(f"near = tasks whose derived rank is within {NEAR} of a cutoff → rescore to confirm.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2) + "\n")
        print(f"\n💾 {args.json_out}")


if __name__ == "__main__":
    main()
