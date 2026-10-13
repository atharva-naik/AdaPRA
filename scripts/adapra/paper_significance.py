"""Paired significance tests behind the paper's significance tables (tab:significance, -hard)
and the p-values quoted in the results text.

Convention:
  - task-paired over ALL tasks (38 Easy / 54 Hard); a task with no scored attempt = 0
  - best attempt = max blended percentile (private-else-public) over valid attempts
  - per attempt  = mean blended percentile over a task's valid attempts
  - one-sided sign test (binomtest 'greater', ties dropped), one-sided Wilcoxon on nonzero d
  - one-sided paired permutation test: random sign flips of d_i, statistic = mean(d)

Usage:
  python scripts/adapra/paper_significance.py [--n-perm 200000] [--seed 0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[2]
E = str(REPO_ROOT / "outputs/adapra/trajectories_v4/{}/dspredict-easy")
BACKUP = "/path/to/backup/trajectories_v4"
H_SCRATCH = "/path/to/scratch/dsgym_v4_hard_{0}_out/trajectories_v4/dspredict-hard-{1}/dspredict-hard"

RUNS = {
    "easy": {
        "harness": E.format("dspredict-easy-bon5-qwen3-v2"),
        "adapra": E.format("dsp-easy-rev-dyn-evolve-pull-xattempt"),
        "no_adapt": E.format("dsp-easy-rev-dyn-frozen-pull-xattempt"),
        "no_pra": E.format("dsp-easy-rev-dyn-evolve-nopra-pull-xattempt"),
    },
    "hard": {
        "harness": f"{BACKUP}/dspredict-hard-noprm/dspredict-hard",
        "adapra": H_SCRATCH.format("evolve_xattempt", "evolve-xattempt"),
        "no_adapt": H_SCRATCH.format("frozen_xattempt_v2", "frozen-xattempt-v2"),
        "no_pra": H_SCRATCH.format("evolve_nopra_xattempt", "evolve-nopra-xattempt"),
    },
    "deepseek": {
        "harness": E.format("dsp-easy-noprm-deepseek-v32-bo1"),
        "adapra": E.format("dsp-easy-adapra-deepseek-v32-bo1"),
    },
}
TOTAL = {"easy": 38, "hard": 54, "deepseek": 38}

COMPARISONS = [  # (benchmark, condition, reference)
    ("easy", "adapra", "harness"), ("easy", "no_adapt", "harness"), ("easy", "no_pra", "harness"),
    ("easy", "adapra", "no_adapt"), ("easy", "adapra", "no_pra"),
    ("hard", "adapra", "harness"), ("hard", "no_adapt", "harness"), ("hard", "no_pra", "harness"),
    ("hard", "adapra", "no_adapt"), ("hard", "adapra", "no_pra"),
    ("deepseek", "adapra", "harness"),
]


_DEC = json.JSONDecoder()
_MARK = b'\n  "outcome": '  # top-level key under indent=2; JSON strings hold no raw newlines
CACHE = Path("/path/to/scratch/paper_significance_cache.json")


def read_outcome(f, chunk=4 << 20):
    """Top-level `outcome` dict without parsing the file (Hard trajectories reach ~1 GB; a huge
    `steps` precedes it and a huge `metadata` can follow it). Anchors on the top-level key line,
    which cannot match nested keys or string content, and scans in fixed chunks alternately from
    both ends, so memory stays at a few chunks. A file that does not end in '}' is truncated
    (json.load would fail) and returns None."""
    size = f.stat().st_size
    with open(f, "rb") as fh:
        fh.seek(max(0, size - 4096))
        if not fh.read().rstrip().endswith(b"}"):
            return None
        off, lo, hi, k = None, 0, size, len(_MARK)
        while off is None and lo < hi:
            fh.seek(lo)  # forward chunk [lo, lo+chunk+k)
            buf = fh.read(min(chunk, hi - lo) + k)
            i = buf.find(_MARK)
            if i >= 0:
                off = lo + i
                break
            lo += chunk
            if lo >= hi:
                break
            start = max(lo, hi - chunk)  # backward chunk [start, hi+k)
            fh.seek(start)
            buf = fh.read(hi - start + k)
            i = buf.rfind(_MARK)
            if i >= 0:
                off = start + i
            hi = start
        if off is None:
            return None
        win = 1 << 20
        while True:
            fh.seek(off + k)
            txt = fh.read(win).decode("utf-8", errors="ignore")
            try:
                return _DEC.raw_decode(txt.lstrip())[0]
            except ValueError:
                if off + k + win >= size:
                    return None
                win *= 4


def load(run_dir, cache):
    """task -> list of blended percentiles of valid, scored attempts.
    Per-file outcomes are cached (keyed by path, size, mtime) so a restart skips finished files."""
    out, n_none = {}, 0
    for d in sorted(Path(run_dir).iterdir()):
        if not d.is_dir():
            continue
        pcts = []
        for f in sorted(d.glob("attempt_*.json")):
            st = f.stat()
            key = f"{f}|{st.st_size}|{int(st.st_mtime)}"
            if key not in cache:
                o = read_outcome(f)
                cache[key] = None if o is None else {
                    k: o.get(k) for k in ("valid_submission", "private_percentile", "percentile")}
            o = cache[key]
            if o is None:
                n_none += 1
                continue  # a few truncated JSONs on Hard harness-only
            if not o.get("valid_submission", False):
                continue
            pct = o.get("private_percentile")
            if pct is None:
                pct = o.get("percentile")
            if pct is not None:
                pcts.append(pct)
        out[d.name] = pcts
    tmp = CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache))
    tmp.replace(CACHE)
    print(f"# loaded {run_dir} ({n_none} unreadable/truncated)", file=sys.stderr, flush=True)
    return out


def per_task(scores, tasks, agg):
    return np.array([agg(scores.get(t, [])) if scores.get(t) else 0.0 for t in tasks])


def perm_p(d, n_perm, rng):
    obs = d.mean()
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (flips * np.abs(d)).mean(axis=1)
    return (np.sum(null >= obs - 1e-12) + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    file_cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    cache = {}
    for bench, cond, ref in COMPARISONS:
        for c in (cond, ref):
            if (bench, c) not in cache:
                cache[(bench, c)] = load(RUNS[bench][c], file_cache)
        a, b = cache[(bench, cond)], cache[(bench, ref)]
        tasks = sorted(set(a) | set(b))
        assert len(tasks) == TOTAL[bench], (bench, cond, ref, len(tasks))
        for metric, agg in (("best", max), ("per-att", lambda x: sum(x) / len(x))):
            d = per_task(a, tasks, agg) - per_task(b, tasks, agg)
            w, l = int((d > 0).sum()), int((d < 0).sum())
            sign_p = binomtest(w, w + l, 0.5, alternative="greater").pvalue
            nz = d[d != 0]
            wil_p = wilcoxon(nz, alternative="greater").pvalue
            print(f"{bench:8s} {cond:8s} vs {ref:8s} {metric:7s} "
                  f"delta={d.mean():+5.1f}  W/L={w}/{l}  sign={sign_p:.3f}  "
                  f"wilcoxon={wil_p:.3f}  perm={perm_p(d, args.n_perm, rng):.4f}")


if __name__ == "__main__":
    main()
