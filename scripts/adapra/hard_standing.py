#!/usr/bin/env python3
"""Best-of-5 standing for DSPredict-Hard runs.

Reuses standing_so_far.py's EXACT machinery — `load_run_scores` (from
compare_runs) + `best_of`/`best_entry` — pointed at a flat Hard run dir (stock
DSGym) or a nested v4 run dir.

best-of-5 per task = max percentile over that task's valid+scored attempts,
using private_percentile-else-public (via load_run_scores' get_percentile),
then mean over the task set. `>median` = pct > 50. Fixed denom = 54 tasks.

Usage:
  # single-run standing (default = stock DSGym baseline)
  python hard_standing.py
  python hard_standing.py --run <run_dir>

  # compare two runs (e.g. Ours v4-noPRM vs stock DSGym) — matched subset
  # (tasks the --run has scored, stock restricted to the same) + fixed-54:
  python hard_standing.py --run <ours_dir> --compare-to <stock_dir>
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

_ATT_RE = re.compile(r"__\d+\.json$")   # stock flat attempt: <task>__<model>__NN.json

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "adapra"))
sys.path.insert(0, str(REPO))
from compare_runs import load_run_scores  # noqa: E402  (same import standing_so_far uses)

STOCK = "outputs/adapra/trajectories/dspredict-hard-dsgym-stock"
OURS = "outputs/adapra/trajectories_v4/dspredict-hard-noprm/dspredict-hard"
TOTAL_TASKS = 54          # all DSPredict-Hard comps (fixed denom)
PRIVATE_TASKS = 44        # informational; private denom is derived from scored data


def best_entry(scores_for_task):
    """Winning entry (max percentile) for best-of-5 — verbatim from standing_so_far."""
    return max(scores_for_task, key=lambda e: e["percentile"])


def best_of(scores_for_task):
    return best_entry(scores_for_task)["percentile"]


def short(t):
    return t.replace("playground-series-", "")


def run_label(run):
    """Row label for a run dir: its run name, skipping a trailing benchmark
    subdir (`<run>/dspredict-hard`) so nested v4 dirs label as `<run>`."""
    parts = Path(str(run).rstrip("/")).parts
    if len(parts) > 1 and parts[-1].startswith("dspredict-"):
        return parts[-2]
    return parts[-1]


def attempts_per_task(run):
    """Attempts *tried* per task (scored or not), layout-agnostic.

    v4 (Ours) is nested `<task>/attempt_*.json`; stock DSGym is flat
    `<task>__<model>__NN.json`. Counts every attempt file, so a task that ran
    all 5 attempts but produced no scoreable submission still shows 5."""
    base = REPO / run
    counts = collections.Counter()
    nested = list(base.glob("*/attempt_*.json"))
    if nested:
        for f in nested:
            counts[f.parent.name] += 1
    else:
        for f in base.glob("*.json"):
            if _ATT_RE.search(f.name):
                counts[f.name.split("__")[0]] += 1
    return counts


def has_medal(entry) -> bool:
    """Winning entry's Kaggle medal, read from its JSON (load_run_scores drops
    it). Private medal iff the pick used the private board, else public."""
    try:
        d = json.load(open(entry["path"])).get("outcome", {})
    except Exception:
        return False
    m = d.get("private_medal") if entry.get("private") else d.get("medal")
    if m is None:
        m = d.get("private_medal") or d.get("medal")
    return bool(m) and str(m).lower() not in ("none", "false", "")


def single_run_standing(run):
    """The original stock-standing report (unchanged behavior)."""
    scores = load_run_scores(str(REPO / run))[0]
    tasks = sorted(scores)
    print("=" * 72)
    print(f"DSPredict-Hard standing — {run}")
    print(f"scored tasks: {len(tasks)} / {TOTAL_TASKS}   "
          "(best-of-5 = max private-else-public pct per task, standing_so_far logic)")
    print("=" * 72)
    print(f"{'task':<44}{'att':>4}{'best5':>9}{'priv?':>7}")
    print("-" * 64)
    bests = []
    for t in tasks:
        b = best_of(scores[t])
        n = len(scores[t])
        is_priv = best_entry(scores[t])["private"]
        bests.append(b)
        print(f"{short(t):<44}{n:>4}{b:>9.1f}{('yes' if is_priv else 'pub'):>7}")
    print("-" * 64)
    n_scored = len(bests)
    above = sum(1 for v in bests if v > 50)
    mean_scored = sum(bests) / n_scored if n_scored else float("nan")
    mean_fixed = sum(bests) / TOTAL_TASKS
    print(f"best-of-5 mean  (scored-task denom, n={n_scored}): {mean_scored:5.1f}%")
    print(f"best-of-5 mean  (fixed denom = {TOTAL_TASKS}):          {mean_fixed:5.1f}%")
    print(f">median (>50)   : {above}/{n_scored} scored ({100*above/n_scored:.1f}%)   "
          f"| {above}/{TOTAL_TASKS} of all ({100*above/TOTAL_TASKS:.1f}%)")


def summarize(scores, valids, totals, task_set, denom):
    bests, medals, aboves, per_attempt = [], 0, 0, []
    n_scored_tasks = 0
    for t in task_set:
        entries = scores.get(t, [])
        if not entries:
            continue
        n_scored_tasks += 1
        be = best_entry(entries)
        bests.append(be["percentile"])
        per_attempt.extend(e["percentile"] for e in entries)
        if be["percentile"] > 50:
            aboves += 1
        if has_medal(be):
            medals += 1
    return {
        "n": n_scored_tasks,
        "best5": (sum(bests) / denom) if denom else float("nan"),
        "above": 100 * aboves / denom if denom else float("nan"),
        "medal": 100 * medals / denom if denom else float("nan"),
        # valid% = SCOREABLE rate: fraction of task_set with >=1 submission that
        # passed Kaggle's format check and scored (n_scored_tasks / denom). Symmetric
        # for both runs, format-checked on both sides — NOT the harness's self-reported
        # valid_submission (which counts wrong-shape CSVs as "valid"). See "valid =
        # Kaggle-scoreable".
        "valid": 100 * n_scored_tasks / denom if denom else float("nan"),
        # per-att = mean percentile over SCORED attempts only (load_run_scores drops
        # invalid/unscored), matching standing_so_far's per-scored-attempt logic.
        "peratt": (sum(per_attempt) / len(per_attempt)) if per_attempt else float("nan"),
    }


def _row(label, m):
    return (f"{label:<26}{m['best5']:>8.1f}{m['above']:>9.1f}{m['medal']:>8.1f}"
            f"{m['valid']:>9.1f}{m['peratt']:>11.1f}{m['n']:>7}")


def compare(ours_run, stock_run, on_attempted=False, kept_partials=None,
            ours_label=None, stock_label=None):
    # rows are labelled from the run dirs (or --label/--compare-label), NOT a fixed
    # "stock DSGym"/"Ours": --compare-to is the reference, --run the candidate.
    ol = (ours_label or run_label(ours_run))[:25]
    sl = (stock_label or run_label(stock_run))[:25]
    o_scores, o_tot, o_val, _ = load_run_scores(str(REPO / ours_run))
    s_scores, s_tot, s_val, _ = load_run_scores(str(REPO / stock_run))
    ours_done = sorted(o_scores)
    matched = sorted(set(ours_done) & set(s_scores))

    print("=" * 84)
    print("DSPredict-Hard standing so far — candidate vs reference  (best-of-5)")
    print(f"  cand = {ol:<25} (--run)        {ours_run}")
    print(f"  ref  = {sl:<25} (--compare-to) {stock_run}")
    print(f"cand has scored attempts for {len(ours_done)}/{TOTAL_TASKS} tasks; "
          f"ref covers {len(matched)} of them.")
    print("=" * 84)
    hdr = (f"{'run / view':<26}{'best5':>8}{'>med%':>9}{'medal%':>8}"
           f"{'valid%':>9}{'per-att':>11}{'nTasks':>7}")

    print(f"\n-- MATCHED: {len(matched)} tasks both runs scored (denom = these) --")
    print(hdr); print("-" * 78)
    dm = len(matched)
    print(_row(sl, summarize(s_scores, s_val, s_tot, matched, dm)))
    print(_row(ol, summarize(o_scores, o_val, o_tot, matched, dm)))

    # SETTLED: every task OURS has FINISHED, i.e. all 5 attempt JSONs exist —
    # regardless of whether any scored. A finished task that produced no scoreable
    # submission is a genuine 0.0 for that run (not missing data), so format-rejects
    # count against us. Stock is a fixed, finished artifact: its best-of-5 uses
    # however many attempts it ran (stock is NOT 5/5 everywhere — 16/54 tasks are
    # 1-4 attempts; a stock crash with no scoreable submission is a 0.0 for stock).
    # summarize() treats a task with no scored entry as 0 in the numerator while
    # counting it in the denominator.
    # --kept-partials: tasks that are final with <5 attempt files also count as settled.
    o_att = attempts_per_task(ours_run)
    kept_partials = set(kept_partials or ())
    unknown = sorted(t for t in kept_partials if t not in o_att)
    if unknown:
        sys.exit(f"--kept-partials: no attempt files in Ours for {unknown}")
    settled = sorted(t for t, v in o_att.items() if v >= 5 or t in kept_partials)
    ds = len(settled)
    kp_note = ""
    if kept_partials:
        kp_note = (f"; incl. {len(kept_partials)} kept partial(s): "
                   + ", ".join(f"{short(t)} ({o_att[t]} att)" for t in sorted(kept_partials)))
    print(f"\n-- SETTLED: {ds} tasks OURS finished (all 5 attempt JSONs exist{kp_note}) "
          "(stock as-is; unscored/format-reject = 0.0 for that run) --")
    print(hdr); print("-" * 78)
    print(_row(sl, summarize(s_scores, s_val, s_tot, settled, ds)))
    print(_row(ol, summarize(o_scores, o_val, o_tot, settled, ds)))
    print(f"\n-- per-task best-of-5 on the {ds} settled tasks (0.0 = tried 5, none scoreable) --")
    print(f"{'task':<44}{'ref':>8}{'cand':>8}{'delta':>8}")
    print("-" * 68)
    for t in settled:
        sb = best_of(s_scores[t]) if t in s_scores else 0.0
        ob = best_of(o_scores[t]) if t in o_scores else 0.0
        mark = f"  * kept partial ({o_att[t]} att)" if t in kept_partials else ""
        print(f"{short(t)[:43]:<44}{sb:>8.1f}{ob:>8.1f}{ob - sb:>+8.1f}{mark}")

    if on_attempted:
        # FIXED-N over EVERY task OURS has attempted (>=1 attempt file, scored or
        # not), denom = that count, unscored/format-reject = 0.0 for BOTH runs.
        # The apples-to-apples fixed-denom read while OURS is mid-run: it neither
        # penalizes OURS for tasks it hasn't started (unlike FIXED-54) nor drops
        # its own format-rejects (unlike MATCHED). "first N attempted" == this set.
        attempted = sorted(o_att.keys())
        da = len(attempted)
        print(f"\n-- FIXED-{da} on OURS-ATTEMPTED set (unscored/format-reject = 0.0 for BOTH runs) --")
        print(hdr); print("-" * 78)
        print(_row(sl, summarize(s_scores, s_val, s_tot, attempted, da)))
        print(_row(ol, summarize(o_scores, o_val, o_tot, attempted, da)))

    print(f"\n-- FIXED-{TOTAL_TASKS} (paper headline; unscored task = 0.0) --")
    print(hdr); print("-" * 78)
    allt = sorted(set(s_scores) | set(o_scores))
    print(_row(sl, summarize(s_scores, s_val, s_tot, allt, TOTAL_TASKS)))
    print(_row(ol, summarize(o_scores, o_val, o_tot, allt, TOTAL_TASKS)))

    print(f"\n-- per-task best-of-5 on the {len(matched)} matched tasks --")
    print(f"{'task':<44}{'ref':>8}{'cand':>8}{'delta':>8}")
    print("-" * 68)
    for t in matched:
        sb, ob = best_of(s_scores[t]), best_of(o_scores[t])
        print(f"{short(t)[:43]:<44}{sb:>8.1f}{ob:>8.1f}{ob - sb:>+8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=STOCK, help="run dir (default: stock DSGym)")
    ap.add_argument("--compare-to", default=None,
                    help="stock/baseline run dir; if set, --run is treated as the "
                         "candidate (Ours) and a side-by-side comparison is printed")
    ap.add_argument("--on-attempted", action="store_true",
                    help="in compare mode, add a FIXED-N block over EVERY task Ours "
                         "has attempted (denom = that count; unscored = 0.0 for both) "
                         "— the apples-to-apples fixed-denom read while Ours is mid-run")
    ap.add_argument("--label", default=None,
                    help="row label for --run (default: run name from the path)")
    ap.add_argument("--compare-label", default=None,
                    help="row label for --compare-to (default: run name from the path)")
    ap.add_argument("--kept-partials", default="",
                    help="comma-separated task IDs to count as SETTLED despite <5 "
                         "attempt files")
    args = ap.parse_args()
    kept = [t.strip() for t in args.kept_partials.split(",") if t.strip()]
    if args.compare_to:
        compare(args.run, args.compare_to, on_attempted=args.on_attempted,
                kept_partials=kept, ours_label=args.label,
                stock_label=args.compare_label)
    else:
        single_run_standing(args.run)


if __name__ == "__main__":
    main()
