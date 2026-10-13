#!/usr/bin/env python3
"""
Update the living results table JSON with current scores from all runs.

Reads trajectory directories listed in configs/adapra/run_registry.json,
computes best-of-pool percentile per task, and writes both DSGym-style
(denom=scored tasks) and fixed-denom (denom=38) scores into
configs/adapra/results_table.json.

Usage:
  python scripts/adapra/update_results_table.py

  # Only update specific runs
  python scripts/adapra/update_results_table.py --runs V2 V3-PRM

  # Print table to stdout without writing
  python scripts/adapra/update_results_table.py --dry-run

  # Also print markdown table
  python scripts/adapra/update_results_table.py --markdown
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from dsgym.eval.metrics.dspredict.dspredict_metric import medal_from_percentile  # noqa: E402

REGISTRY_PATH = REPO_ROOT / "configs" / "adapra" / "run_registry.json"
RESULTS_PATH = REPO_ROOT / "configs" / "adapra" / "results_table.json"
EASY_LB = REPO_ROOT / "data" / "DSPredict_Offline_Leaderboard" / "easy_leaderboard_stats_2026-02-01.json"
HARD_LB = REPO_ROOT / "data" / "DSPredict_Offline_Leaderboard" / "hard_leaderboard_stats_2026-02-01.json"
TOTAL_TASKS = 38


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, FileNotFoundError):
        return None


def _n_teams(leaderboard: dict | None, task: str, use_private: bool) -> int | None:
    """Board size for `task` on the side the winning pct came from (private/public)."""
    if not leaderboard:
        return None
    e = leaderboard.get(task)
    if not e:
        return None
    scores = e.get("private_scores") if use_private else e.get("public_scores")
    return len(scores) if scores else None


def _near_medal_cutoff(rank: int, n_teams: int, near: int = 3) -> bool:
    """True if `rank` is within `near` of a gold/silver/bronze cutoff.

    Mirrors the ladder in dspredict_metric (total = n_teams + 1). Only used to flag
    percentile-inverted medals whose tie-ambiguity could flip a tier → rescore.
    """
    import math

    total = n_teams + 1

    def pct_ceil(p):
        return math.ceil(p * total)

    if total <= 99:
        cuts = (pct_ceil(0.10), pct_ceil(0.20), pct_ceil(0.40))
    elif 100 <= total <= 249:
        cuts = (min(10, total), pct_ceil(0.20), pct_ceil(0.40))
    elif 250 <= total <= 999:
        cuts = (min(total, 10 + math.ceil(0.002 * total)), min(50, total), min(100, total))
    else:
        cuts = (min(total, 10 + math.ceil(0.002 * total)), pct_ceil(0.05), pct_ceil(0.10))
    return any(abs(rank - c) <= near for c in cuts)


def load_registry() -> dict:
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def load_results_table() -> dict:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def compute_run_scores(run_path: str, run_format: str, total_tasks: int = TOTAL_TASKS,
                       leaderboard: dict | None = None) -> dict:
    """Compute best-of-pool scores for a run directory.

    total_tasks is the fixed denominator for the fixed-denom columns (38 for
    DSPredict-Easy; pass 54 for DSPredict-Hard via the registry's total_tasks).

    leaderboard (offline stats keyed by task_id) enables medal-rate: for each
    best-of-pool task we invert the stored percentile to a rank via
    medal_from_percentile (no Kaggle rescore) and tally gold/silver/bronze.
    """
    base = REPO_ROOT / run_path
    if not base.exists():
        return None

    task_bests = {}  # task_id -> best FINAL-submission pct (best-of-pool)
    # task_id -> (use_private, n_teams) for whichever attempt is currently best,
    # so the medal is derived on the same board the winning pct came from.
    task_best_meta = {}
    task_best_ever = {}  # task_id -> best pct across ALL scored CSV versions (history)
    task_valid_count = {}  # task_id -> number of valid submissions
    tasks_attempted = set()  # all task_ids with at least one attempt
    total_attempts = 0
    valid_attempts = 0
    # Diagnostic (beam-quality) accumulator: blended pct of EVERY scored valid
    # attempt (not deduped to best-per-task) → floor-raising / mean-per-attempt.
    attempt_pcts = []

    if run_format == "flat":
        # Count error files toward tasks_attempted
        for f in sorted(base.glob("*.error.json")):
            try:
                r = json.loads(f.read_text())
                task = r.get("task_id") or f.stem.split("__")[0].rsplit(".", 1)[0]
                tasks_attempted.add(task)
            except (json.JSONDecodeError, OSError):
                pass

        for f in sorted(base.glob("*.json")):
            if "mock" in f.name or ".error" in f.name or f.name == "error.json":
                continue
            total_attempts += 1
            try:
                r = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            task = r.get("task_id") or f.stem.split("__")[0]
            tasks_attempted.add(task)
            o = r.get("outcome", {})

            if not o.get("valid_submission", False):
                continue
            valid_attempts += 1

            # Blended: private percentile when the comp has a private leaderboard,
            # else fall back to public. Use an explicit None check (not `or`) so a
            # legitimate 0.0 private percentile isn't treated as falsy and silently
            # replaced by the public score. Public-only comps leave
            # private_percentile as None (the scorer does not fabricate it).
            priv = o.get("private_percentile")
            use_private = priv is not None
            pct = priv if use_private else o.get("percentile")
            if pct is None:
                task_valid_count[task] = task_valid_count.get(task, 0) + 1
                continue

            task_valid_count[task] = task_valid_count.get(task, 0) + 1
            attempt_pcts.append(pct)
            if task not in task_bests or pct > task_bests[task]:
                task_bests[task] = pct
                persisted = o.get("private_medal") if use_private else o.get("medal")
                task_best_meta[task] = (use_private,
                                        _n_teams(leaderboard, task, use_private),
                                        persisted)
    else:
        for task_dir in sorted(base.iterdir()):
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            for f in sorted(task_dir.glob("attempt_*.json")):
                total_attempts += 1
                tasks_attempted.add(task)
                try:
                    r = json.loads(f.read_text())
                except (json.JSONDecodeError, OSError):
                    continue

                o = r.get("outcome", {})
                if not o.get("valid_submission", False):
                    continue
                valid_attempts += 1

                # Blended private-else-public with an explicit None check (see the
                # flat-format branch above) so a legitimate 0.0 private percentile
                # is not treated as falsy and dropped to the public score.
                priv = o.get("private_percentile")
                use_private = priv is not None
                pct = priv if use_private else o.get("percentile")
                if pct is None:
                    task_valid_count[task] = task_valid_count.get(task, 0) + 1
                    continue

                task_valid_count[task] = task_valid_count.get(task, 0) + 1
                attempt_pcts.append(pct)
                if task not in task_bests or pct > task_bests[task]:
                    task_bests[task] = pct
                    persisted = o.get("private_medal") if use_private else o.get("medal")
                    task_best_meta[task] = (use_private,
                                            _n_teams(leaderboard, task, use_private),
                                            persisted)

            # best-ever: best pct across ALL scored CSV versions in the csv_history
            # manifests (only present for V4+; requires score_submissions
            # --score-history to have populated the `kaggle` fields).
            for hist in sorted(task_dir.glob("csv_history_attempt_*")):
                m = _load_json(hist / "manifest.json") or []
                for e in m:
                    kg = e.get("kaggle") or {}
                    _pp = kg.get("private_percentile")
                    p = _pp if _pp is not None else kg.get("percentile")
                    if p is not None and (task not in task_best_ever or p > task_best_ever[task]):
                        task_best_ever[task] = p
            # fold the final-submission bests into best-ever too
            if task in task_bests and (task not in task_best_ever
                                       or task_bests[task] > task_best_ever[task]):
                task_best_ever[task] = task_bests[task]

    scored_tasks = len(task_bests)
    tasks_with_valid = len(task_valid_count)
    n_attempted = len(tasks_attempted)

    # Beam-quality diagnostics: best-of-N
    # (the fixed/dsgym pct below) is blind to a lifted floor and killed invalids,
    # so also report the per-ATTEMPT valid rate and the mean over every scored
    # valid attempt (not deduped to best-per-task).
    diagnostics = {
        "attempt_valid_rate": round(100 * valid_attempts / total_attempts, 1) if total_attempts else 0.0,
        "valid_attempts": valid_attempts,
        "total_attempts": total_attempts,
        "mean_per_attempt": round(sum(attempt_pcts) / len(attempt_pcts), 1) if attempt_pcts else None,
        "n_scored_attempts": len(attempt_pcts),
    }

    # Medal rate over the best-of-pool submission per task. Prefer the medal the
    # canonical scorer already persisted from the raw score (exact even at tie
    # clusters); fall back to inverting the stored percentile via
    # medal_from_percentile when no raw-score medal was saved. medal% denom =
    # total_tasks (comparable across runs). near_cutoff counts fallback-derived
    # medals whose rank sits within 3 of a cutoff — rescore those to be sure.
    tally = {"gold": 0, "silver": 0, "bronze": 0}
    any_medal = 0
    near_cutoff = 0
    derived_from_percentile = 0
    per_task_medal = {}
    for task, pct in task_bests.items():
        use_private, n_teams, persisted = task_best_meta.get(task, (True, None, None))
        if persisted in ("gold", "silver", "bronze", "none"):
            medal, rank, exact = persisted, None, True
        else:
            rank, medal = medal_from_percentile(n_teams, pct)
            exact = False
            derived_from_percentile += 1
            if rank >= 1 and n_teams and _near_medal_cutoff(rank, n_teams):
                near_cutoff += 1
        if medal in tally:
            tally[medal] += 1
            any_medal += 1
        per_task_medal[task] = {"medal": medal, "pct": round(pct, 3),
                                "n_teams": n_teams, "private": use_private,
                                "exact": exact, "rank": rank}
    medals = {
        "gold": tally["gold"], "silver": tally["silver"], "bronze": tally["bronze"],
        "any_medal": any_medal,
        "medal_rate": round(100 * any_medal / total_tasks, 1),
        "scored_tasks": scored_tasks,
        "derived_from_percentile": derived_from_percentile,
        "near_cutoff": near_cutoff,
        "per_task": per_task_medal,
    }

    # DSGym style: only over Kaggle-scored tasks (task_bests)
    # valid = scored_tasks / tasks_attempted (what fraction got a Kaggle score)
    # percentile = mean(best_pct) over scored tasks
    # median = above_50th / scored_tasks
    if scored_tasks == 0:
        dsgym_valid = 0.0
        dsgym_pct = 0.0
        dsgym_median = 0.0
    else:
        dsgym_valid = round((tasks_with_valid / total_tasks) * 100, 1)
        dsgym_pct = round(sum(task_bests.values()) / scored_tasks, 1)
        above_median = sum(1 for p in task_bests.values() if p > 50)
        dsgym_median = round((above_median / scored_tasks) * 100, 1)

    # Fixed denom: everything / total_tasks (38 for Easy, 54 for Hard)
    fixed_valid = round((tasks_with_valid / total_tasks) * 100, 1)
    fixed_pct = round(sum(task_bests.values()) / total_tasks, 1)
    above_median_count = sum(1 for p in task_bests.values() if p > 50)
    fixed_median = round((above_median_count / total_tasks) * 100, 1)

    # best-ever (best across all scored CSV versions incl. mid-run) — only
    # meaningful for runs with scored csv_history manifests (V4+).
    be_scored = len(task_best_ever)
    best_ever_stats = None
    if be_scored:
        be_above = sum(1 for p in task_best_ever.values() if p > 50)
        best_ever_stats = {
            "scored_tasks": be_scored,
            "dsgym_style": {
                "percentile": round(sum(task_best_ever.values()) / be_scored, 1),
                "median": round((be_above / be_scored) * 100, 1),
            },
            "fixed_denom": {
                "percentile": round(sum(task_best_ever.values()) / total_tasks, 1),
                "median": round((be_above / total_tasks) * 100, 1),
            },
        }

    return {
        "scored_tasks": scored_tasks,
        "tasks_with_valid_sub": tasks_with_valid,
        "tasks_attempted": n_attempted,
        "total_attempts": total_attempts,
        "valid_attempts": valid_attempts,
        "dspredict_easy_dsgym_style": {
            "valid": dsgym_valid,
            "percentile": dsgym_pct,
            "median": dsgym_median,
        },
        "dspredict_easy_fixed_denom": {
            "valid": fixed_valid,
            "percentile": fixed_pct,
            "median": fixed_median,
        },
        # best-of-any-CSV-ever (None for runs without scored history)
        "dspredict_easy_best_ever": best_ever_stats,
        # beam-quality diagnostics (per-attempt valid rate + mean-per-attempt)
        "dspredict_easy_diagnostics": diagnostics,
        # medal rate over best-of-pool per task (private-or-public blended)
        "dspredict_easy_medals": medals,
    }


def compute_hedge_comparison(run_path: str, run_format: str = "nested") -> dict | None:
    """Paper metric: LAST (final submission best-of-N) vs HEDGE (max(last,
    selected) best-of-N), from each attempt's outcome.selected_scoring.

    For every task we take the best over its attempts of:
      - last_pct   : the final submission the harness would submit
      - hedge_pct  : max(last_i, selected_i) — the last-vs-selector hedge
    then aggregate over tasks (mean percentile, %-above-median, denom=38 fixed).
    This isolates the gain from adding the agentic submission selector on top of
    just taking the last submission — the row for the paper.
    """
    base = REPO_ROOT / run_path
    if not base.exists():
        return None
    task_last: dict[str, float] = {}
    task_hedge: dict[str, float] = {}
    task_dirs = ([base] if run_format == "flat"
                 else [d for d in sorted(base.iterdir()) if d.is_dir()])
    for td in task_dirs:
        attempts = (sorted(base.glob("*.json")) if run_format == "flat"
                    else sorted(td.glob("attempt_*.json")))
        task = td.name
        for aj in attempts:
            try:
                ss = (json.loads(aj.read_text()).get("outcome") or {}).get("selected_scoring") or {}
            except (json.JSONDecodeError, OSError):
                continue
            lp, hp = ss.get("last_pct"), ss.get("hedge_pct")
            if lp is not None and (task not in task_last or lp > task_last[task]):
                task_last[task] = lp
            if hp is not None and (task not in task_hedge or hp > task_hedge[task]):
                task_hedge[task] = hp

    def _agg(d: dict[str, float]) -> dict:
        vals = list(d.values())
        if not vals:
            return {"n": 0}
        above = sum(1 for v in vals if v > 50)
        return {
            "n_tasks": len(vals),
            "dsgym_percentile": round(sum(vals) / len(vals), 1),
            "dsgym_median": round(100 * above / len(vals), 1),
            "fixed_percentile": round(sum(vals) / TOTAL_TASKS, 1),
            "fixed_median": round(100 * above / TOTAL_TASKS, 1),
        }
    return {"last": _agg(task_last), "hedge": _agg(task_hedge),
            "per_task_last": task_last, "per_task_hedge": task_hedge}


def format_markdown_table(results: dict) -> str:
    """Generate a markdown table from results."""
    lines = []
    lines.append("")
    lines.append("## DSPredict-Easy Results (Living Table)")
    lines.append("")
    lines.append("### DSGym-style (denom = scored tasks)")
    lines.append("")
    lines.append("| Run | Valid | Percentile | Median | Scored Tasks |")
    lines.append("|-----|-------|------------|--------|--------------|")

    # DSGym paper entries
    for entry in results.get("dsgym_paper_results", []):
        e = entry["dspredict_easy"]
        lines.append(
            f"| {entry['model']} | {e['valid']} | {e['percentile']} | {e['median']} | — |"
        )

    # Our runs
    for entry in results.get("our_runs", []):
        d = entry["dspredict_easy_dsgym_style"]
        status = "" if entry.get("status") == "complete" else " *"
        lines.append(
            f"| {entry['label']}{status} | {d['valid']} | {d['percentile']} | {d['median']} | {entry['scored_tasks']}/{entry['total_tasks']} |"
        )

    lines.append("")
    lines.append("### Fixed-denom (denom = 38 tasks)")
    lines.append("")
    lines.append("| Run | Valid | Percentile | Median |")
    lines.append("|-----|-------|------------|--------|")

    # DSGym paper entries rescaled: fixed = dsgym * (valid/100)
    for entry in results.get("dsgym_paper_results", []):
        e = entry["dspredict_easy"]
        v = e["valid"]
        fixed_pct = round(e["percentile"] * v / 100, 1)
        fixed_med = round(e["median"] * v / 100, 1)
        lines.append(
            f"| {entry['model']} | {v} | {fixed_pct} | {fixed_med} |"
        )

    for entry in results.get("our_runs", []):
        d = entry["dspredict_easy_fixed_denom"]
        status = "" if entry.get("status") == "complete" else " *"
        lines.append(
            f"| {entry['label']}{status} | {d['valid']} | {d['percentile']} | {d['median']} |"
        )

    lines.append("")
    lines.append("\\* = in progress (not all tasks scored yet)")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Update living results table")
    parser.add_argument("--runs", nargs="*", help="Only update these run IDs (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Print without writing")
    parser.add_argument("--markdown", action="store_true", help="Print markdown table")
    parser.add_argument("--hedge-comparison", action="store_true",
                        help="Print LAST vs HEDGE (max(last,selected)) rows for the "
                             "given --runs (paper metric; reads selected_scoring).")
    args = parser.parse_args()

    registry = load_registry()
    results = load_results_table()

    # Offline leaderboards for medal-rate (chosen per run: Easy vs Hard).
    global _EASY_LB, _HARD_LB
    _EASY_LB = _load_json(EASY_LB) or {}
    _HARD_LB = _load_json(HARD_LB) or {}

    run_ids = args.runs or list(registry["runs"].keys())

    if args.hedge_comparison:
        print("\n=== LAST vs HEDGE (max(last, selected)) — paper comparison ===")
        print(f"{'Run':<16}{'Metric':<8}{'pct(DSGym)':>11}{'med%(DSGym)':>12}"
              f"{'pct(/38)':>10}{'med%(/38)':>10}{'n':>4}")
        for run_id in run_ids:
            if run_id not in registry["runs"]:
                print(f"⚠️  Unknown run_id: {run_id}", file=sys.stderr); continue
            info = registry["runs"][run_id]
            hc = compute_hedge_comparison(info["path"], info.get("format", "nested"))
            if not hc:
                print(f"⚠️  no data for {run_id}", file=sys.stderr); continue
            for label in ("last", "hedge"):
                a = hc[label]
                if a.get("n_tasks"):
                    print(f"{run_id:<16}{label:<8}{a['dsgym_percentile']:>11}"
                          f"{a['dsgym_median']:>12}{a['fixed_percentile']:>10}"
                          f"{a['fixed_median']:>10}{a['n_tasks']:>4}")
            l, h = hc["last"], hc["hedge"]
            if l.get("n_tasks") and h.get("n_tasks"):
                print(f"{'':16}{'Δ hedge':<8}{h['dsgym_percentile']-l['dsgym_percentile']:>+11.1f}"
                      f"{h['dsgym_median']-l['dsgym_median']:>+12.1f}"
                      f"{h['fixed_percentile']-l['fixed_percentile']:>+10.1f}"
                      f"{h['fixed_median']-l['fixed_median']:>+10.1f}")
        return

    # Build lookup of existing our_runs entries by run_id
    existing_runs = {r["run_id"]: r for r in results.get("our_runs", [])}

    for run_id in run_ids:
        if run_id not in registry["runs"]:
            print(f"⚠️  Unknown run_id: {run_id}", file=sys.stderr)
            continue

        run_info = registry["runs"][run_id]
        run_path = run_info["path"]
        run_format = run_info.get("format", "nested")
        run_total_tasks = run_info.get("total_tasks", TOTAL_TASKS)

        # Hard runs (total_tasks 54 or 'hard' in path) score against the Hard board.
        is_hard = run_total_tasks == 54 or "hard" in str(run_path).lower()
        leaderboard = _HARD_LB if is_hard else _EASY_LB

        scores = compute_run_scores(run_path, run_format, run_total_tasks, leaderboard)
        if scores is None:
            print(f"⚠️  Path not found: {run_path}", file=sys.stderr)
            continue

        # Determine status from registry notes
        notes = run_info.get("notes", "").lower()
        if "in progress" in notes:
            status = "in_progress"
        elif "complete" in notes:
            status = "complete"
        elif scores["total_attempts"] >= TOTAL_TASKS * 4:
            status = "complete"
        else:
            status = "in_progress"

        entry = {
            "run_id": run_id,
            "label": run_info["label"],
            "harness": run_id,
            "status": status,
            "scored_tasks": scores["scored_tasks"],
            "tasks_with_valid_sub": scores["tasks_with_valid_sub"],
            "total_tasks": run_total_tasks,
            "total_attempts": scores["total_attempts"],
            "valid_attempts": scores["valid_attempts"],
            "dspredict_easy_dsgym_style": scores["dspredict_easy_dsgym_style"],
            "dspredict_easy_fixed_denom": scores["dspredict_easy_fixed_denom"],
            "dspredict_easy_best_ever": scores.get("dspredict_easy_best_ever"),
            "dspredict_easy_diagnostics": scores.get("dspredict_easy_diagnostics"),
            "dspredict_easy_medals": scores.get("dspredict_easy_medals"),
        }

        existing_runs[run_id] = entry
        d = scores["dspredict_easy_dsgym_style"]
        f = scores["dspredict_easy_fixed_denom"]
        be = scores.get("dspredict_easy_best_ever")
        be_str = (f"  BestEver(DSGym): pct={be['dsgym_style']['percentile']:5.1f}% "
                  f"med={be['dsgym_style']['median']:5.1f}% ({be['scored_tasks']} tasks)") if be else ""
        dg = scores.get("dspredict_easy_diagnostics") or {}
        mpa = dg.get("mean_per_attempt")
        dg_str = (f"  Beam: valid/att={dg.get('attempt_valid_rate', 0):5.1f}% "
                  f"({dg.get('valid_attempts', 0)}/{dg.get('total_attempts', 0)}) "
                  f"mean/att={mpa if mpa is not None else float('nan'):5.1f}%") if dg else ""
        md = scores.get("dspredict_easy_medals") or {}
        near = md.get("near_cutoff", 0)
        md_str = (f"  Medals: {md.get('medal_rate', 0):4.1f}% "
                  f"(G{md.get('gold', 0)}/S{md.get('silver', 0)}/B{md.get('bronze', 0)}"
                  f"{f' ⚠{near}near' if near else ''})") if md else ""
        print(
            f"{'✅' if status == 'complete' else '🔄'} {run_id:12s} "
            f"scored={scores['scored_tasks']:2d}/{run_total_tasks}  "
            f"DSGym: valid={d['valid']:5.1f}% pct={d['percentile']:5.1f}% med={d['median']:5.1f}%  "
            f"Fixed: pct={f['percentile']:5.1f}% med={f['median']:5.1f}%{be_str}{dg_str}{md_str}"
        )

    # Update results
    results["our_runs"] = list(existing_runs.values())

    from datetime import date
    results.setdefault("_meta", {})["last_updated"] = str(date.today())

    if args.markdown:
        print(format_markdown_table(results))

    if not args.dry_run:
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
            f.write("\n")
        print(f"\n💾 Written to {RESULTS_PATH}")
    else:
        print("\n(dry-run, not written)")


if __name__ == "__main__":
    main()
