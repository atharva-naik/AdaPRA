"""Compute a DSPredict-Easy paper table row, including the per-attempt medal rate.
Defaults to the - adaptive rubrics run; pass a run directory to score another run.
  best-of-5  : canonical compute_run_scores (task-first, fixed denom 38) + medal
  per-attempt: Valid = FLAT (valid/total attempts);
               Pct   = task-first MACRO (mean scored-attempt blended pct within
                       task, then mean over tasks with >=1 scored attempt);
               Median= task-first MACRO (per task % of scored attempts pct>50,
                       then mean over tasks);
               Medal = FLAT (medal attempts / (38 * 5)).
Blended pct = private_percentile if not None else percentile.

Usage:
  python scripts/adapra/paper_easy_peratt_medal.py [RUN_DIR [nested|flat]]
"""
import json, glob, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "urt", str(Path(__file__).resolve().parent / "update_results_table.py"))
urt = importlib.util.module_from_spec(spec); spec.loader.exec_module(urt)

RUN = "outputs/adapra/trajectories_v4/dsp-easy-rev-dyn-frozen-pull-xattempt/dspredict-easy"
# Optional override: `paper_easy_peratt_medal.py DIR [nested|flat]` replaces RUN.
if len(sys.argv) > 1:
    RUN = sys.argv[1]
FMT = sys.argv[2] if len(sys.argv) > 2 else "nested"   # flat = stock DSGym-harness layout
LB_PATH = "data/DSPredict_Offline_Leaderboard/easy_leaderboard_stats_2026-02-01.json"
TOTAL = 38
lb = json.load(open(urt.REPO_ROOT / LB_PATH))

# ---- canonical best-of-5 + medal + per-attempt Valid (flat) ----
s = urt.compute_run_scores(RUN, FMT, TOTAL, lb)
b5 = s["dspredict_easy_fixed_denom"]
diag = s["dspredict_easy_diagnostics"]
med = s["dspredict_easy_medals"]

# ---- macro per-attempt Pct + Median (task-first) ----
base = urt.REPO_ROOT / RUN
task_means = []      # per-task mean blended pct
task_medfrac = []    # per-task % of scored attempts with pct>50
per_task = {}
medal_att = 0   # same rule as _paper_row_hard.py: private_medal when private board exists, else medal
medal_derived = 0   # attempts whose medal comes from the compute_run_scores fallback (no persisted tier)
if FMT == "flat":   # group flat files (<task>__<model>__NN.json) by task, as compute_run_scores does
    groups = {}
    for f in sorted(base.glob("*.json")):
        if "mock" in f.name or ".error" in f.name or f.name == "error.json":
            continue
        try:
            t = json.load(open(f)).get("task_id") or f.stem.split("__")[0]
        except Exception:
            continue
        groups.setdefault(t, []).append(f)
    task_files = sorted(groups.items())
else:
    task_files = [(d.name, sorted(d.glob("attempt_*.json"))) for d in sorted(base.iterdir()) if d.is_dir()]
for task_name, files in task_files:
    pcts = []
    for f in files:
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
        att_med = o.get("private_medal") if priv is not None else o.get("medal")
        if att_med not in ("gold", "silver", "bronze", "none"):
            # same fallback compute_run_scores uses for best-of-5: invert the stored percentile
            _, att_med = urt.medal_from_percentile(urt._n_teams(lb, task_name, priv is not None), pct)
            medal_derived += 1
        if att_med in ("gold", "silver", "bronze"):
            medal_att += 1
    if pcts:
        tm = sum(pcts) / len(pcts)
        mf = 100 * sum(1 for p in pcts if p > 50) / len(pcts)
        task_means.append(tm)
        task_medfrac.append(mf)
        per_task[task_name] = (len(pcts), round(tm, 1), round(mf, 1))

macro_pct = sum(task_means) / len(task_means)
macro_med = sum(task_medfrac) / len(task_medfrac)
pa_medal = round(100 * medal_att / (TOTAL * 5), 1)   # Hard per-att Medal convention (flat, /190)

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
print(f"  Medal  = {pa_medal}  (FLAT: {medal_att}/{TOTAL * 5} attempts; {medal_derived} attempts via percentile fallback)")
print()
print("=== LaTeX row cells (Best-of-5 | Per-attempt) ===")
print(f"  {b5['valid']} & {b5['percentile']} & {b5['median']} & {med['medal_rate']} & & {diag['attempt_valid_rate']} & {round(macro_pct,1)} & {round(macro_med,1)} & {pa_medal}")
print()
print("=== per-task (n_scored, mean_pct, %>50) ===")
for t in sorted(per_task):
    n, tm, mf = per_task[t]
    print(f"  {t:18s} n={n} mean={tm:5.1f} >50={mf:5.1f}")
