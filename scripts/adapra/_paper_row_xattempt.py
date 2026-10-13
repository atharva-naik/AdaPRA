"""Compute a DSPredict-Easy paper table row (default run: - adaptive rubrics) under
the paper's aggregation convention:
  best-of-5  : canonical compute_run_scores (task-first, fixed denom 38) + medal
  per-attempt: Valid = FLAT (valid/total attempts);
               Pct   = task-first MACRO (mean scored-attempt blended pct within
                       task, then mean over tasks with >=1 scored attempt);
               Median= task-first MACRO (per task % of scored attempts pct>50,
                       then mean over tasks).
Blended pct = private_percentile if not None else percentile.
"""
import json, glob, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "urt", str(Path(__file__).resolve().parent / "update_results_table.py"))
urt = importlib.util.module_from_spec(spec); spec.loader.exec_module(urt)

RUN = "outputs/adapra/trajectories_v4/dsp-easy-rev-dyn-frozen-pull-xattempt/dspredict-easy"
# Optional override: `_paper_row_xattempt.py DIR` (nested layout) replaces RUN.
if len(sys.argv) > 1:
    RUN = sys.argv[1]
LB_PATH = "data/DSPredict_Offline_Leaderboard/easy_leaderboard_stats_2026-02-01.json"
TOTAL = 38
lb = json.load(open(urt.REPO_ROOT / LB_PATH))

# ---- canonical best-of-5 + medal + per-attempt Valid (flat) ----
s = urt.compute_run_scores(RUN, "nested", TOTAL, lb)
b5 = s["dspredict_easy_fixed_denom"]
diag = s["dspredict_easy_diagnostics"]
med = s["dspredict_easy_medals"]

# ---- macro per-attempt Pct + Median (task-first) ----
base = urt.REPO_ROOT / RUN
task_means = []      # per-task mean blended pct
task_medfrac = []    # per-task % of scored attempts with pct>50
per_task = {}
for task_dir in sorted(base.iterdir()):
    if not task_dir.is_dir():
        continue
    pcts = []
    for f in sorted(task_dir.glob("attempt_*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        o = r.get("outcome", {})
        if not o.get("valid_submission", False):
            continue
        priv = o.get("private_percentile")
        pct = priv if priv is not None else o.get("percentile")
        if pct is None:
            continue
        pcts.append(pct)
    if pcts:
        tm = sum(pcts) / len(pcts)
        mf = 100 * sum(1 for p in pcts if p > 50) / len(pcts)
        task_means.append(tm)
        task_medfrac.append(mf)
        per_task[task_dir.name] = (len(pcts), round(tm, 1), round(mf, 1))

macro_pct = sum(task_means) / len(task_means)
macro_med = sum(task_medfrac) / len(task_medfrac)

print("=== BEST-OF-5 (task-first, fixed denom 38) ===")
print(f"  Valid  = {b5['valid']}")
print(f"  Pct    = {b5['percentile']}")
print(f"  Median = {b5['median']}")
print(f"  Medal  = {med['medal_rate']}  (gold {med['gold']} silver {med['silver']} bronze {med['bronze']}, any {med['any_medal']}/{TOTAL})")
print(f"  scored_tasks={s['scored_tasks']}  tasks_with_valid={s['tasks_with_valid_sub']}")
print()
print("=== PER-ATTEMPT (mean) ===")
print(f"  Valid  = {diag['attempt_valid_rate']}  (FLAT: {diag['valid_attempts']}/{diag['total_attempts']} attempts)")
print(f"  Pct    = {round(macro_pct,1)}  (MACRO over {len(task_means)} tasks; flat mean_per_attempt={diag['mean_per_attempt']})")
print(f"  Median = {round(macro_med,1)}  (MACRO over {len(task_medfrac)} tasks)")
print()
print("=== LaTeX row cells (Best-of-5 | Per-attempt) ===")
print(f"  {b5['valid']} & {b5['percentile']} & {b5['median']} & {med['medal_rate']} & & {diag['attempt_valid_rate']} & {round(macro_pct,1)} & {round(macro_med,1)}")
print()
print("=== per-task (n_scored, mean_pct, %>50) ===")
for t in sorted(per_task):
    n, tm, mf = per_task[t]
    print(f"  {t:18s} n={n} mean={tm:5.1f} >50={mf:5.1f}")
