"""Compare per-rubric failure incidence between two reviewer runs that share one rubric set.

Used for the easy-to-hard generalization analysis: the frozen-rubric (−adaptive rubrics)
runs on DSPredict-Easy and DSPredict-Hard both score the same 19 seed rubrics every
review, so per-rubric failure rates are directly comparable across benchmarks.

A rubric FAILS in a review iff importance >= 0.5 and p_agent_success < 0.6 (the PRA's
injection rule). Per rubric and run we report:
  att_fail  = share of attempts (with >= 1 review) where the rubric failed at least once
  rev_fail  = share of all reviews where the rubric failed
  mean_imp  = mean importance over all reviews (how relevant the PRA judged it)
plus Spearman rank correlations of each statistic between the two runs.

Usage:
  python scripts/adapra/rubric_failure_transfer.py \
    --run-a outputs/adapra/trajectories_v4/dsp-easy-rev-dyn-frozen-pull-xattempt/dspredict-easy \
    --run-b /path/to/scratch/dsgym_v4_hard_frozen_xattempt_v2_out/trajectories_v4/dspredict-hard-frozen-xattempt-v2/dspredict-hard \
    --names Easy Hard
"""
import argparse
import glob
import json
import os
from collections import defaultdict

from scipy.stats import spearmanr

IMP_T, P_T = 0.5, 0.6


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_run(run_dir):
    att_n = 0
    att_fail = defaultdict(int)
    rev_n = defaultdict(int)
    rev_fail = defaultdict(int)
    imp_sum = defaultdict(float)
    tasks = set()
    for task in sorted(os.listdir(run_dir)):
        for adir in sorted(glob.glob(os.path.join(run_dir, task, "prm_reviews_attempt_*"))):
            calls = sorted(glob.glob(os.path.join(adir, "reviewer_call_*.json")))
            failed = set()
            n_ok = 0
            for c in calls:
                try:
                    scores = json.load(open(c)).get("scores") or {}
                except (json.JSONDecodeError, OSError):
                    continue
                if not scores:
                    continue
                n_ok += 1
                for rid, v in scores.items():
                    i, p = _f(v.get("importance_score")), _f(v.get("p_agent_success"))
                    if i is None or p is None:
                        continue
                    rev_n[rid] += 1
                    imp_sum[rid] += i
                    if i >= IMP_T and p < P_T:
                        rev_fail[rid] += 1
                        failed.add(rid)
            if n_ok:
                att_n += 1
                tasks.add(task)
                for rid in failed:
                    att_fail[rid] += 1
    stats = {}
    for rid in rev_n:
        stats[rid] = {
            "att_fail": att_fail[rid] / att_n,
            "rev_fail": rev_fail[rid] / rev_n[rid],
            "mean_imp": imp_sum[rid] / rev_n[rid],
        }
    return stats, att_n, len(tasks), sum(rev_n.values()) // max(len(rev_n), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    na, nb = args.names
    sa, atta, ta, ra = load_run(args.run_a)
    sb, attb, tb, rb = load_run(args.run_b)
    shared = sorted(set(sa) & set(sb), key=lambda r: -(sa[r]["att_fail"] + sb[r]["att_fail"]))
    print(f"{na}: {ta} tasks, {atta} attempts, ~{ra} reviews/rubric | "
          f"{nb}: {tb} tasks, {attb} attempts, ~{rb} reviews/rubric | shared rubrics {len(shared)}")
    print(f"only in {na}: {sorted(set(sa) - set(sb))}\nonly in {nb}: {sorted(set(sb) - set(sa))}")
    print(f"\n{'rubric':42s} {na+' att%':>9s} {nb+' att%':>9s} {na+' rev%':>9s} {nb+' rev%':>9s} {na+' imp':>8s} {nb+' imp':>8s}")
    for r in shared:
        a, b = sa[r], sb[r]
        print(f"{r:42s} {100*a['att_fail']:9.1f} {100*b['att_fail']:9.1f} {100*a['rev_fail']:9.1f} "
              f"{100*b['rev_fail']:9.1f} {a['mean_imp']:8.2f} {b['mean_imp']:8.2f}")
    out = {"shared": shared, na: sa, nb: sb, "spearman": {}}
    print()
    for k in ("att_fail", "rev_fail", "mean_imp"):
        rho, p = spearmanr([sa[r][k] for r in shared], [sb[r][k] for r in shared])
        out["spearman"][k] = {"rho": rho, "p": p}
        print(f"Spearman {k}: rho={rho:.3f} p={p:.4f}")
    if args.json_out:
        json.dump(out, open(args.json_out, "w"), indent=2)


if __name__ == "__main__":
    main()
