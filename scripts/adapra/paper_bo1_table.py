"""Single-attempt (best-of-1) rows on DSPredict-Easy: cadence table (tab:cadence),
and alternative base agent (tab:base-agent).

Each run is scored on attempt_0 only, over a fixed denominator of 38 tasks:
  Valid   = % of tasks whose attempt_0 has a Kaggle-scored submission;
  Pct     = mean blended percentile (private if present, else public), 0 if invalid;
  >Median = % of tasks with blended percentile > 50.
For a best-of-5 run (harness only, c=10) attempt_0 serves as its single-attempt point.
Every run after the first is compared with the first, task-paired over all 38 tasks
(missing = 0): W/L/T, one-sided sign test, and one-sided Wilcoxon on nonzero differences.

Usage:
  paper_bo1_table.py                     # cadence table
  paper_bo1_table.py --set base-agent    # DeepSeek-V3.2 base agent
  paper_bo1_table.py NAME=DIR [NAME=DIR ...]   # custom; DIR = .../dspredict-easy
"""
import argparse
import json
from pathlib import Path

from scipy.stats import binomtest, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[2]
T = "outputs/adapra/trajectories_v4/{}/dspredict-easy"
SETS = {
    "cadence": [
        ("Harness only", T.format("dspredict-easy-bon5-qwen3-v2")),
        ("c=5", T.format("dsp-easy-rev-dyn-evolve-pull-xattempt-c5-bo1")),
        ("c=10 (AdaPRA)", T.format("dsp-easy-rev-dyn-evolve-pull-xattempt")),
        ("c=20", T.format("dsp-easy-rev-dyn-evolve-pull-xattempt-c20-bo1")),
    ],
    "base-agent": [
        ("DeepSeek-V3.2 harness only", T.format("dsp-easy-noprm-deepseek-v32-bo1")),
        ("DeepSeek-V3.2 AdaPRA", T.format("dsp-easy-adapra-deepseek-v32-bo1")),
    ],
}
TOTAL = 38


def attempt0(run_dir: Path, tasks: list[str]) -> dict[str, float | None]:
    out = {}
    for t in tasks:
        f = run_dir / t / "attempt_0.json"
        pct = None
        if f.exists():
            o = json.load(open(f)).get("outcome", {})
            if o.get("valid_submission"):
                priv = o.get("private_percentile")
                pct = priv if priv is not None else o.get("percentile")
        out[t] = pct
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), default="cadence")
    ap.add_argument("runs", nargs="*", help="NAME=DIR pairs; overrides --set")
    args = ap.parse_args()
    runs = [tuple(r.split("=", 1)) for r in args.runs] if args.runs else SETS[args.set]

    first = REPO_ROOT / runs[0][1]
    tasks = sorted(p.name for p in first.iterdir() if p.is_dir())
    assert len(tasks) == TOTAL, f"{first}: {len(tasks)} task dirs, expected {TOTAL}"

    scores = [(name, attempt0(REPO_ROOT / d, tasks)) for name, d in runs]
    print(f"{'Run':28s} {'Valid':>6s} {'Pct':>6s} {'>Med':>6s}")
    for name, s in scores:
        v = [p for p in s.values() if p is not None]
        print(f"{name:28s} {100 * len(v) / TOTAL:6.1f} {sum(v) / TOTAL:6.1f} "
              f"{100 * sum(p > 50 for p in v) / TOTAL:6.1f}")

    ref_name, ref = scores[0]
    print(f"\nPaired over {TOTAL} tasks vs {ref_name} (missing = 0):")
    for name, s in scores[1:]:
        d = [(s[t] or 0) - (ref[t] or 0) for t in tasks]
        w, l = sum(x > 0 for x in d), sum(x < 0 for x in d)
        sign_p = binomtest(w, w + l, 0.5, alternative="greater").pvalue if w + l else float("nan")
        nz = [x for x in d if x != 0]
        wil_p = wilcoxon(nz, alternative="greater").pvalue if nz else float("nan")
        print(f"  {name:28s} diff {sum(d) / TOTAL:+5.1f}  W/L/T {w}/{l}/{TOTAL - w - l}  "
              f"sign p={sign_p:.3f}  Wilcoxon p={wil_p:.3f}")


if __name__ == "__main__":
    main()
