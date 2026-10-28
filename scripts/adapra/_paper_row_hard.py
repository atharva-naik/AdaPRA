"""Hard-table row producer — strict Hard analogue of _paper_row_xattempt.py.

Reuses the paper's aggregation convention via
update_results_table.compute_run_scores:
  best-of-5  : task-first, fixed denom 54, blended private-else-public, medal
               (private_medal when comp has a private board, else public medal;
                "none" stays no-medal).
  per-attempt: Valid  = FLAT (valid_submission attempts / total attempts);
               Pct    = task-first MACRO (mean scored-attempt blended pct within
                        task, then mean over tasks with >=1 scored attempt);
               Median = task-first MACRO (per task % of scored attempts pct>50,
                        then mean over tasks).
Handles both nested ({task}/attempt_N.json) and flat ({task}__model__NN.json).
"""
import json, glob, os, sys, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "urt", str(Path(__file__).resolve().parent / "update_results_table.py"))
urt = importlib.util.module_from_spec(spec); spec.loader.exec_module(urt)

LB_PATH = "data/DSPredict_Offline_Leaderboard/hard_leaderboard_stats_2026-02-01.json"
TOTAL = 54
lb = json.load(open(urt.REPO_ROOT / LB_PATH))

RUNS = [
    # ("stock DSGym", "outputs/adapra/trajectories/dspredict-hard-dsgym-stock", "flat"),
    # ("noPRM (Ours)", "/path/to/backup/trajectories_v4/dspredict-hard-noprm/dspredict-hard", "nested"),
    ("- adaptive rubrics", "/path/to/scratch/dsgym_v4_hard_frozen_xattempt_v2_out/trajectories_v4/dspredict-hard-frozen-xattempt-v2/dspredict-hard", "nested"),
    ("AdaPRA (evolve)", "/path/to/scratch/dsgym_v4_hard_evolve_xattempt_out/trajectories_v4/dspredict-hard-evolve-xattempt/dspredict-hard", "nested"),
    ("AdaPRA - PRA", "/path/to/scratch/dsgym_v4_hard_evolve_nopra_xattempt_out/trajectories_v4/dspredict-hard-evolve-nopra-xattempt/dspredict-hard", "nested"),
]
# Optional override: `_paper_row_hard.py NAME DIR [NAME DIR ...]` replaces RUNS; DIR is nested
# ({task}/attempt_N.json) unless suffixed with `:flat` (stock DSGym runs: {task}__model__NN.json).
if len(sys.argv) > 1:
    _a = sys.argv[1:]
    RUNS = [(_a[i], _a[i + 1][:-5], "flat") if _a[i + 1].endswith(":flat") else (_a[i], _a[i + 1], "nested")
            for i in range(0, len(_a), 2)]


def macro_per_attempt(base: Path, layout: str):
    """Mirror _paper_row_xattempt.py lines 36-61 for both layouts."""
    # group blended pcts of valid+scored attempts per task
    per_task = {}
    if layout == "flat":
        groups = {}
        for f in sorted(base.glob("*.json")):
            if "mock" in f.name or ".error" in f.name or f.name == "error.json":
                continue
            m = re.match(r"(.+?)__.+__(\d+)\.json$", f.name)
            task = m.group(1) if m else f.stem.split("__")[0]
            groups.setdefault(task, []).append(f)
        items = groups.items()
    else:
        items = [(d.name, sorted(d.glob("attempt_*.json")))
                 for d in sorted(base.iterdir()) if d.is_dir()]
    task_means, task_medfrac = [], []
    scored_att = 0
    medal_att = 0   # attempts with a medal (private-then-public: private_medal when private board exists)
    for task, files in items:
        pcts = []
        for f in files:
            try:
                o = json.load(open(f)).get("outcome", {})
            except Exception:
                continue
            if not o.get("valid_submission", False):
                continue
            priv = o.get("private_percentile")
            use_private = priv is not None
            pct = priv if use_private else o.get("percentile")
            if pct is None:
                continue
            pcts.append(pct)
            med = o.get("private_medal") if use_private else o.get("medal")
            if med not in (None, "none", "None", ""):
                medal_att += 1
        if pcts:
            scored_att += len(pcts)
            task_means.append(sum(pcts) / len(pcts))
            task_medfrac.append(100 * sum(1 for p in pcts if p > 50) / len(pcts))
            per_task[task] = (len(pcts), round(task_means[-1], 1), round(task_medfrac[-1], 1))
    macro_pct = sum(task_means) / len(task_means) if task_means else 0.0
    macro_med = sum(task_medfrac) / len(task_medfrac) if task_medfrac else 0.0
    return macro_pct, macro_med, len(task_means), scored_att, medal_att, per_task


print(f"{'run':<18} | Best-of-5 Valid  Pct  Median Medal | Per-att Valid  Pct  Median  (macro n_tasks)")
print("-" * 100)
for name, run, layout in RUNS:
    s = urt.compute_run_scores(run, layout, TOTAL, lb)
    b5 = s["dspredict_easy_fixed_denom"]
    diag = s["dspredict_easy_diagnostics"]
    med = s["dspredict_easy_medals"]
    macro_pct, macro_med, ntasks, scored_att, medal_att, _ = macro_per_attempt(
        urt.REPO_ROOT / run if not os.path.isabs(run) else Path(run), layout)
    # HARD convention (differs from Easy: Hard has valid-but-unscoreable comps):
    #   Valid = SCOREABLE (pct-not-None) based, best-of-5 denom 54, per-att denom 54*5=270.
    #   per-att Pct/Median = MACRO over scored attempts (mimics Easy standing_so_far).
    #   per-att Medal = medal-attempts / 270 (private-then-public).
    b5_valid = round(100 * s["scored_tasks"] / TOTAL, 1)
    pa_valid = round(100 * scored_att / (TOTAL * 5), 1)
    pa_medal = round(100 * medal_att / (TOTAL * 5), 1)
    print(f"{name:<18} | {b5_valid:>5} {b5['percentile']:>6} {b5['median']:>6} {med['medal_rate']:>5} "
          f"| {pa_valid:>6} {round(macro_pct,1):>6} {round(macro_med,1):>6} {pa_medal:>5}  (macro n={ntasks})")
    print(f"{'':<18}   scored_tasks={s['scored_tasks']} scored_att={scored_att} medal_att={medal_att} "
          f"best5_medals g{med['gold']}/s{med['silver']}/b{med['bronze']} any{med['any_medal']}")
    print(f"  LaTeX: {b5_valid} & {b5['percentile']} & {b5['median']} & {med['medal_rate']} & & "
          f"{pa_valid} & {round(macro_pct,1)} & {round(macro_med,1)} & {pa_medal}")
    print()
