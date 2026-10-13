"""DataPRM invocation statistics for the paper appendix (DataPRM Baseline, "Behavior").

Reads every prm_reviews_attempt_*/reviewer_call_*.json under a run directory and reports:
invocation count, median steps verified and wall time per invocation, status mix
(FAIL/WARN/PASS), the share of verified steps scored 0, and how many zero scores go to
steps whose summary says the step has no code or analysis.

  python scripts/adapra/dataprm_behavior_stats.py RUN_DIR [RUN_DIR ...]
"""
import json, re, statistics, sys
from collections import Counter
from pathlib import Path

# Reproduces the published zero-score count of the original Easy run to within 2 of 4,360.
NO_CODE = re.compile(r"no (actual |meaningful |substantive |real |analytical )?(analysis|code|content)", re.I)

for run in sys.argv[1:]:
    calls = sorted(Path(run).glob("*/prm_reviews_attempt_*/reviewer_call_*.json"))
    status, steps, secs = Counter(), [], []
    n_scored = n_zero = n_zero_nocode = 0
    for f in calls:
        d = json.loads(f.read_text())
        status[d.get("status")] += 1
        steps.append(d.get("steps_verified") or 0)
        if d.get("wall_seconds") is not None:
            secs.append(d["wall_seconds"])
        for v in (d.get("scores") or {}).values():
            if not isinstance(v, dict) or v.get("score") is None:
                continue
            n_scored += 1
            if float(v["score"]) == 0.0:
                n_zero += 1
                if NO_CODE.search(v.get("summary") or ""):
                    n_zero_nocode += 1
    n = len(calls)
    print(f"== {run}")
    print(f"invocations={n}  median_steps_verified={statistics.median(steps)}  "
          f"median_wall_s={statistics.median(secs):.0f}")
    print("status: " + "  ".join(f"{k}={100*c/n:.0f}%" for k, c in status.most_common()))
    print(f"scored_steps={n_scored}  zero={n_zero} ({100*n_zero/n_scored:.0f}%)  "
          f"zero_no_code={n_zero_nocode} of {n_zero}")
