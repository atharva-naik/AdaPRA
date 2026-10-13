"""
Submit collected DSPredict submission CSVs to Kaggle and record leaderboard metrics.

Reads trajectory JSON files, finds submission CSVs, submits to Kaggle API,
waits for scores, then writes above_median / percentile / medal / final_score
back into the trajectory files.

Usage:
  python scripts/adapra/score_submissions.py \
    --trajectory-dir outputs/adapra/trajectories/dspredict-easy \
    --leaderboard-stats data/DSPredict_Offline_Leaderboard/easy_leaderboard_stats_2026-02-01.json \
    [--dry-run]   # print what would be submitted, no API calls
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dsgym.adapra.schema import TrajectoryRecord
from dsgym.eval.metrics.dspredict.dspredict_metric import (
    DSPredictSubmissionManager,
    get_leaderboard_rank_medal,
)


def load_leaderboard(stats_path: Path) -> dict:
    with open(stats_path) as f:
        return json.load(f)


class MultiKeyManager:
    """Optional multi-credential support: distribute submissions over several
    Kaggle API keys.

    For each submit we pick a key that hasn't hit `per_key_cap` for
    THIS competition; when all keys are exhausted for a competition, submit()
    returns None (caller treats it like budget-exhausted → picked up next day).

    Note: the KaggleSDK client binds KAGGLE_USERNAME/KEY from env at construction,
    so we build one DSPredictSubmissionManager per key up front (each captured its
    own creds) and just route calls to the chosen one.
    """

    def __init__(self, key_files: list[Path], per_key_cap: int = 8):
        self.managers = []
        self.usernames = []
        for kf in key_files:
            creds = json.loads(Path(kf).read_text())
            self.managers.append(DSPredictSubmissionManager(creds=creds))
            self.usernames.append(creds.get("username", str(kf)))
        self.per_key_cap = per_key_cap
        # counts[comp][key_idx] = submissions issued
        self.counts: dict[str, list[int]] = {}
        print(f"MultiKeyManager: {len(self.managers)} keys "
              f"({', '.join(self.usernames)}), {per_key_cap}/comp/key "
              f"→ up to {per_key_cap*len(self.managers)}/comp/day")

    def _pick(self, comp: str) -> int | None:
        c = self.counts.setdefault(comp, [0] * len(self.managers))
        # least-used key that still has headroom
        best = None
        for i, used in enumerate(c):
            if used < self.per_key_cap and (best is None or used < c[best]):
                best = i
        return best

    def submit(self, competition_name: str, file_path: str, description: str):
        idx = self._pick(competition_name)
        if idx is None:
            return None  # all keys exhausted for this comp today
        self.counts[competition_name][idx] += 1
        mgr = self.managers[idx]
        ref = mgr.submit_file(competition_name=competition_name,
                              file_path=file_path, description=description)
        info = mgr.wait_for_submission(ref, timeout_minutes=20)
        return info, self.usernames[idx]


def resolve_submission_path(raw_path: str, repo_root: Path) -> Path | None:
    """Resolve submission_path relative to repo root."""
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p
    candidate = repo_root / p
    if candidate.exists():
        return candidate
    return None


def _has_private_score(record: TrajectoryRecord) -> bool:
    """Check if trajectory already has private_score in its outcome."""
    return getattr(record.outcome, "private_score", None) is not None


def score_trajectory(
    record: TrajectoryRecord,
    leaderboard: dict,
    manager: DSPredictSubmissionManager,
    repo_root: Path,
    dry_run: bool,
    rescore: bool = False,
) -> dict | None:
    if not record.outcome.valid_submission or not record.submission_path:
        return None
    already_scored = record.outcome.final_score is not None
    already_has_private = _has_private_score(record)
    if already_scored and already_has_private and not rescore:
        print(f"  [skip] already scored: {record.trajectory_id}")
        return None
    if already_scored and not already_has_private and not rescore:
        print(f"  [skip-no-private] already scored but missing private: {record.trajectory_id} (use --rescore to resubmit)")
        return None

    sub_path = resolve_submission_path(record.submission_path, repo_root)
    if sub_path is None:
        print(f"  [warn] submission file not found: {record.submission_path}")
        return None

    competition = record.task.task_id
    lb = leaderboard.get(competition)
    if lb is None:
        print(f"  [warn] no leaderboard stats for {competition}")
        return None

    if dry_run:
        print(f"  [dry-run] would submit {sub_path.name} to {competition}")
        return None

    print(f"  submitting {sub_path.name} → {competition} ...", end=" ", flush=True)
    try:
        ref = manager.submit_file(
            competition_name=competition,
            file_path=str(sub_path),
            description=f"DSGym AdaPRA scoring {record.trajectory_id}",
        )
        score_info = manager.wait_for_submission(ref, timeout_minutes=20)
        status = str(score_info.get("status", "")).upper()
        print(f"status={status}")

        if "ERROR" in status:
            print(f"  [warn] Kaggle returned error status for {competition}")
            return None

        def _safe_float(x):
            try: return float(x)
            except (TypeError, ValueError): return None

        pub = _safe_float(score_info.get("public_score"))
        priv = _safe_float(score_info.get("private_score"))

        pub_rank = pub_medal = pub_pct = pub_above = None
        if pub is not None and lb.get("public_scores"):
            pub_rank, pub_medal, pub_pct, pub_above = get_leaderboard_rank_medal(lb["public_scores"], pub)

        priv_rank = priv_medal = priv_pct = priv_above = None
        if priv is not None and lb.get("private_scores"):
            priv_rank, priv_medal, priv_pct, priv_above = get_leaderboard_rank_medal(lb["private_scores"], priv)

        return {
            "final_score": pub,
            "percentile": pub_pct,
            "above_median": pub_above,
            "medal": pub_medal,
            "rank": pub_rank,
            "private_score": priv,
            "private_percentile": priv_pct,
            "private_above_median": priv_above,
            "private_medal": priv_medal,
            "private_rank": priv_rank,
            "status": status,
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def _score_csv_file(
    csv_path: Path, competition: str, lb: dict,
    manager: DSPredictSubmissionManager, description: str,
) -> dict | None:
    """Submit a single CSV to Kaggle and return public/private pct info.

    `manager` may be a DSPredictSubmissionManager (single key) or a
    MultiKeyManager (round-robin over keys). Returns None if a MultiKeyManager
    has exhausted all keys for this competition today.
    """
    try:
        if isinstance(manager, MultiKeyManager):
            res = manager.submit(competition, str(csv_path), description)
            if res is None:
                return {"budget_exhausted": True}
            info, _user = res
        else:
            ref = manager.submit_file(
                competition_name=competition, file_path=str(csv_path), description=description,
            )
            info = manager.wait_for_submission(ref, timeout_minutes=20)
        status = str(info.get("status", "")).upper()
        if "ERROR" in status:
            return {"status": status, "error": True}

        def _f(x):
            try: return float(x)
            except (TypeError, ValueError): return None

        pub, priv = _f(info.get("public_score")), _f(info.get("private_score"))
        pub_pct = priv_pct = None
        pub_above = priv_above = None
        pub_rank = priv_rank = pub_medal = priv_medal = None
        if pub is not None and lb.get("public_scores"):
            pub_rank, pub_medal, pub_pct, pub_above = get_leaderboard_rank_medal(lb["public_scores"], pub)
        if priv is not None and lb.get("private_scores"):
            priv_rank, priv_medal, priv_pct, priv_above = get_leaderboard_rank_medal(lb["private_scores"], priv)
        return {
            "public_score": pub, "percentile": pub_pct, "above_median": pub_above,
            "rank": pub_rank, "medal": pub_medal,
            "private_score": priv, "private_percentile": priv_pct,
            "private_above_median": priv_above,
            "private_rank": priv_rank, "private_medal": priv_medal, "status": status,
        }
    except Exception as e:
        return {"status": "EXCEPTION", "error": True, "message": str(e)}


def _submission_manifest_entries(csv_hist_dir: Path) -> list[dict]:
    """Load submission-like CSV entries from a csv_history manifest.json.

    Matches any archived CSV whose basename contains 'submission' (covers
    submission.csv plus variants like submission_v2.csv the agent may write).
    """
    manifest = csv_hist_dir / "manifest.json"
    if not manifest.exists():
        return []
    try:
        entries = json.loads(manifest.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    out = []
    for e in entries:
        base = Path(e.get("source_path", "")).name.lower()
        if "submission" in base and base.endswith(".csv"):
            out.append(e)
    return out


def _expected_n_lines(task_id: str, data_path: Path, manifest_entries: list[dict]) -> int | None:
    """Canonical submission line count for a task.

    Prefer the authoritative sample_submission.csv line count; if that's absent
    (some tasks ship none), fall back to the most-frequent n_lines across the
    manifest versions (the correctly-shaped submission is written far more often
    than one-off malformed stubs).
    """
    sample = data_path / task_id / "sample_submission.csv"
    if sample.exists():
        try:
            with open(sample) as f:
                return sum(1 for _ in f)
        except OSError:
            pass
    counts = [e.get("n_lines") for e in manifest_entries if e.get("n_lines")]
    if counts:
        from collections import Counter
        return Counter(counts).most_common(1)[0][0]
    return None


# Tasks where the submission legitimately has more rows than sample_submission.csv
# (e.g. IR ranking tasks with a truncated sample). These bypass the row-count guard
# entirely; Kaggle's own server-side check is the authority.
_SHAPE_GUARD_EXEMPT = frozenset({
    "trec-covid-information-retrieval",   # IR ranking: sample 401, valid sub 4001
})


def _n_lines_ok(n_lines, task_id: str, data_path: Path, expected: int | None) -> bool:
    """Row-count guard. Accepts the sample_submission.csv count, and also
    test.csv's line count (one prediction row per test row): some tasks ship a
    truncated stub sample_submission.csv (e.g. planttraits2024: 5 rows vs a
    6545-row test set), which would otherwise pre-reject correct submissions.
    Tasks in _SHAPE_GUARD_EXEMPT skip the check entirely."""
    if expected is None or n_lines in (None, expected):
        return True
    if task_id in _SHAPE_GUARD_EXEMPT:
        return True
    test = data_path / task_id / "test.csv"
    if test.exists():
        try:
            with open(test, "rb") as f:
                return n_lines == sum(1 for _ in f)
        except OSError:
            pass
    return False


def score_task_history(
    task_dir: Path, leaderboard: dict, manager: DSPredictSubmissionManager,
    delay: float, dry_run: bool, data_path: Path, rescore: bool = False,
    max_per_comp: int = 8,
) -> dict | None:
    """Score EVERY distinct submission CSV across all attempts of one task.

    Manifest-centric + skip-already-scored:
      - Each csv_history_attempt_N/manifest.json submission entry gets a
        `kaggle` sub-dict (percentile / public_score / private_* / status)
        written back IN PLACE. The manifest is the source of truth.
      - On re-run, entries that already have a non-error `kaggle` result are
        SKIPPED (unless --rescore) — so rescoring the whole run isn't wasteful.
      - Dedup within the task by md5 (identical content scored once).
      - Malformed CSVs (row count != canonical) are flagged and skipped before
        submission.
      - Each attempt_N.json outcome is populated as before for the FINAL
        submission, PLUS a new `submission_history` field listing every
        version's turn/md5/percentile.

    Aggregations returned:
      - best_final_of_5: best percentile among attempts' FINAL submissions
      - best_any_csv:    best percentile among ALL CSV versions ever produced
    """
    import hashlib
    task_id = task_dir.name
    lb = leaderboard.get(task_id)
    if lb is None:
        print(f"  [warn] no leaderboard stats for {task_id}")
        return None

    seen_md5: dict[str, dict] = {}   # md5 -> kaggle info (reuse across dup files)
    budget = {"used": 0}             # real submissions issued for THIS competition

    multi = isinstance(manager, MultiKeyManager)

    def _score_dedup(csv_file: Path, md5: str | None, tag: str) -> dict | None:
        # Reusing an identical file (same md5) costs no new submission.
        if md5 and md5 in seen_md5:
            return seen_md5[md5]
        if dry_run:
            print(f"  [dry-run] would submit {tag} {csv_file.name}")
            return None
        # Per-competition daily cap. For a single key, enforce max_per_comp here.
        # For MultiKeyManager, IT owns the per-key caps (up to per_key_cap*n_keys),
        # so skip the local cap and let it return budget_exhausted when spent.
        if not multi and budget["used"] >= max_per_comp:
            return {"budget_exhausted": True}
        print(f"  submit {tag} {csv_file.name} ...", end=" ", flush=True)
        info = _score_csv_file(csv_file, task_id, lb, manager,
                               description=f"history {task_id} {tag}")
        if not (info and info.get("budget_exhausted")):
            budget["used"] += 1
        print(f"pct={info.get('percentile') if info else None} (comp submits: {budget['used']})")
        if md5 and info and not info.get("error"):
            seen_md5[md5] = info
        time.sleep(delay)
        return info

    hist_dirs = sorted(task_dir.glob("csv_history_attempt_*"))
    all_entries: list[dict] = []
    for hist_dir in hist_dirs:
        for e in _submission_manifest_entries(hist_dir):
            all_entries.append(e)
    expected = _expected_n_lines(task_id, data_path, all_entries)

    # ---- 1. Score FINALS first (priority: they feed best_final_of_5 + the
    #         attempt outcome fields). History gets the remaining daily budget. ----
    final_summary: list[dict] = []
    final_info_by_attempt: dict[str, dict] = {}
    for attempt_json in sorted(task_dir.glob("attempt_*.json")):
        ai = attempt_json.stem.rsplit("_", 1)[-1]
        final_csv = task_dir / f"submission_attempt_{ai}.csv"
        final_info = None
        if final_csv.exists():
            try:
                raw = final_csv.read_bytes()
                md5 = hashlib.md5(raw).hexdigest()
                n_lines = raw.count(b"\n") + (0 if raw.endswith(b"\n") else 1)
                if not _n_lines_ok(n_lines, task_id, data_path, expected):
                    print(f"  [skip] final/a{ai}: n_lines={n_lines}!=expected {expected}")
                else:
                    r = _score_dedup(final_csv, md5, f"final/a{ai}")
                    if r is not None and not r.get("budget_exhausted"):
                        final_info = r
            except OSError:
                pass
        final_info_by_attempt[ai] = final_info
        final_summary.append({
            "attempt": ai,
            "percentile": (final_info or {}).get("percentile"),
            "private_percentile": (final_info or {}).get("private_percentile"),
        })

    # ---- 2. Score history versions from each attempt's manifest (persist) ----
    n_versions = n_scored = n_skipped_bad = n_skipped_done = 0
    per_version_summary: list[dict] = []
    for hist_dir in hist_dirs:
        attempt_i = hist_dir.name.rsplit("_", 1)[-1]
        manifest_path = hist_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        changed = False
        for e in manifest:
            base = Path(e.get("source_path", "")).name.lower()
            if not ("submission" in base and base.endswith(".csv")):
                continue
            n_versions += 1
            # skip already-scored (unless rescore)
            prev = e.get("kaggle")
            if prev and not prev.get("error") and not rescore:
                n_skipped_done += 1
            elif not _n_lines_ok(e.get("n_lines"), task_id, data_path, expected):
                e["kaggle"] = {"skipped": f"n_lines={e.get('n_lines')}!=expected {expected}"}
                changed = True
                n_skipped_bad += 1
            else:
                info = _score_dedup(hist_dir / e["archived_as"], e.get("md5"),
                                    f"a{attempt_i}/t{e.get('turn')}")
                if info is not None and not info.get("budget_exhausted"):
                    # Only persist real results; budget-exhausted stays unscored
                    # so a later run (after the daily reset) picks it up.
                    e["kaggle"] = info
                    changed = True
                    if not info.get("error"):
                        n_scored += 1
            kg = e.get("kaggle") or {}
            per_version_summary.append({
                "attempt": attempt_i, "turn": e.get("turn"), "md5": e.get("md5"),
                "percentile": kg.get("percentile"), "public_score": kg.get("public_score"),
                "private_percentile": kg.get("private_percentile"),
                "private_score": kg.get("private_score"),
            })
        if changed and not dry_run:
            manifest_path.write_text(json.dumps(manifest, indent=2))

    # ---- 3. Persist each attempt_N.json outcome (final score + history) ----
    if not dry_run:
        for attempt_json in sorted(task_dir.glob("attempt_*.json")):
            ai = attempt_json.stem.rsplit("_", 1)[-1]
            hist_for_attempt = [v for v in per_version_summary if v["attempt"] == ai]
            _persist_attempt_outcome(attempt_json, final_info_by_attempt.get(ai),
                                     hist_for_attempt)

    def _pcts(items, key):
        return [it[key] for it in items if it.get(key) is not None]

    # Private percentile is the real (held-out) metric — report it as primary.
    # Public is secondary. For perpetually-public comps (e.g. house-prices) no
    # private leaderboard exists, so private lists are empty and we fall back.
    final_priv = _pcts(final_summary, "private_percentile")
    any_priv = _pcts(per_version_summary, "private_percentile") + final_priv
    final_pub = _pcts(final_summary, "percentile")
    any_pub = _pcts(per_version_summary, "percentile") + final_pub

    print(f"[{task_id}] versions={n_versions} scored={n_scored} "
          f"skipped_done={n_skipped_done} skipped_bad={n_skipped_bad} "
          f"(expected_lines={expected})")
    result = {
        "task_id": task_id,
        # primary = private; falls back to public when no private LB (public-only)
        "best_final_of_5": max(final_priv) if final_priv else (max(final_pub) if final_pub else None),
        "best_any_csv": max(any_priv) if any_priv else (max(any_pub) if any_pub else None),
        "metric": "private" if final_priv or any_priv else "public",
        # keep both explicitly for transparency
        "best_final_of_5_private": max(final_priv) if final_priv else None,
        "best_any_csv_private": max(any_priv) if any_priv else None,
        "best_final_of_5_public": max(final_pub) if final_pub else None,
        "best_any_csv_public": max(any_pub) if any_pub else None,
        "n_versions": n_versions, "n_scored": n_scored,
        "per_version": per_version_summary, "finals": final_summary,
    }
    print(f"  → [{result['metric']}] best_final_of_5={result['best_final_of_5']}  "
          f"best_any_csv={result['best_any_csv']}  "
          f"(pub: final={result['best_final_of_5_public']} any={result['best_any_csv_public']})")
    return result


def _persist_attempt_outcome(attempt_json: Path, final_info: dict | None,
                             history: list[dict]) -> None:
    """Write final-submission score fields (as before) + a submission_history
    list into an attempt_N.json's outcome."""
    try:
        data = json.loads(attempt_json.read_text())
    except (json.JSONDecodeError, OSError):
        return
    outcome = data.setdefault("outcome", {})
    if final_info and not final_info.get("error"):
        mapping = {
            "final_score": "public_score", "percentile": "percentile",
            "above_median": "above_median", "rank": "rank", "medal": "medal",
            "private_score": "private_score",
            "private_percentile": "private_percentile",
            "private_above_median": "private_above_median",
            "private_rank": "private_rank", "private_medal": "private_medal",
        }
        for out_key, info_key in mapping.items():
            if final_info.get(info_key) is not None:
                outcome[out_key] = final_info[info_key]
    outcome["submission_history"] = history
    attempt_json.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def run_score_history(traj_dir: Path, leaderboard: dict, delay: float,
                      dry_run: bool, data_path: Path, rescore: bool = False,
                      max_per_comp: int = 8,
                      kaggle_keys: list[Path] | None = None) -> None:
    """Score full submission history for every task under traj_dir.

    Each task is a distinct Kaggle competition with its own ~10/day submission
    cap, so max_per_comp bounds submissions per task per run; unscored versions
    are picked up on a later day via skip-already-scored. If kaggle_keys are
    given, a MultiKeyManager round-robins over them (~max_per_comp per key per
    comp), multiplying the daily cap.
    """
    if dry_run:
        manager = None
    elif kaggle_keys:
        manager = MultiKeyManager(kaggle_keys, per_key_cap=max_per_comp)
    else:
        manager = DSPredictSubmissionManager()
    task_dirs = sorted(p for p in traj_dir.iterdir()
                       if p.is_dir() and any(p.glob("attempt_*.json")))
    print(f"Found {len(task_dirs)} task dirs with trajectories.")
    results = []
    for task_dir in task_dirs:
        r = score_task_history(task_dir, leaderboard, manager, delay, dry_run,
                               data_path, rescore, max_per_comp)
        if r:
            results.append(r)
    if not results:
        return
    ff = [r["best_final_of_5"] for r in results if r["best_final_of_5"] is not None]
    aa = [r["best_any_csv"] for r in results if r["best_any_csv"] is not None]
    n_priv = sum(1 for r in results if r.get("metric") == "private")
    print(f"\n=== History-scoring summary ({len(results)} tasks; "
          f"{n_priv} private-LB, {len(results)-n_priv} public-only) ===")
    print("  (primary metric = private percentile; falls back to public for "
          "public-only comps like house-prices)")
    if ff:
        print(f"best_final_of_5  — mean {sum(ff)/len(ff):.1f}%  above-median {sum(1 for x in ff if x>50)}/{len(ff)}")
    if aa:
        print(f"best_any_csv     — mean {sum(aa)/len(aa):.1f}%  above-median {sum(1 for x in aa if x>50)}/{len(aa)}")
    out_path = traj_dir.parent.parent / "scored_history_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full history results → {out_path}")


def _resolve_csv(task_dir: Path, name: str, artifact_root: Path | None) -> Path:
    """Locate a submission CSV. Bulky submission CSVs may live on the scratch
    artifact mirror (see harness csv_artifact_root) rather than next to the
    attempt JSONs. Prefer the home task_dir; fall back to the scratch mirror
    under the run/split/task tree (tail after 'trajectories_v4')."""
    home = task_dir / name
    if home.exists() or artifact_root is None:
        return home
    parts = task_dir.resolve().parts
    if "trajectories_v4" in parts:
        tail = Path(*parts[parts.index("trajectories_v4") + 1:])
    else:
        tail = Path(*parts[-3:])
    scratch = artifact_root / tail / name
    return scratch if scratch.exists() else home


def score_task_selected(
    task_dir: Path, leaderboard: dict, manager, delay: float, dry_run: bool,
    data_path: Path, rescore: bool = False, max_per_comp: int = 10,
    artifact_root: Path | None = None,
) -> dict | None:
    """Score the LAST (final) and SELECTED submission for each attempt of a task.

    Per attempt i we grade BOTH:
      - submission_attempt_i.csv       (the "last"/final the harness would submit)
      - selected_submission_attempt_i.csv (the offline selector's pick, if present)
    and report max(last_i, selected_i). At most 2 grades/attempt, deduped by md5
    (selected==last collapses to ONE Kaggle call), so a 5-attempt task issues at
    most 10 submissions — matching Kaggle's ~10/comp/day cap. Results are written
    into each attempt_i.json outcome under `selected_scoring`.

    Aggregations returned per task:
      - best_last:     best pct among the 5 final submissions
      - best_selected: best pct among the 5 selector picks
      - best_hedge:    best pct among max(last_i, selected_i) over attempts
    """
    import hashlib
    task_id = task_dir.name
    lb = leaderboard.get(task_id)
    if lb is None:
        print(f"  [warn] no leaderboard stats for {task_id}")
        return None

    # Data-centric tasks (e.g. s3e21) submit a TRAINING DATASET, not predictions,
    # so their row count legitimately differs from sample_submission.csv. The
    # n_lines==expected guard wrongly skips them — the selector already bypasses
    # row-count for these and matches COLUMN SCHEMA instead. Mirror that here.
    data_centric = False
    exp_cols = None
    try:
        from dsgym.datasets.loaders.dspredict import _is_data_centric
        for aj in sorted(task_dir.glob("attempt_*.json")):
            desc = (json.loads(aj.read_text()).get("task") or {}).get("description", "")
            if desc:
                data_centric = _is_data_centric(desc)
                break
        if data_centric:
            sample = data_path / task_id / "sample_submission.csv"
            if sample.exists():
                exp_cols = sample.open().readline().strip().split(",")
    except Exception:
        data_centric = False

    seen_md5: dict[str, dict] = {}
    budget = {"used": 0}
    multi = isinstance(manager, MultiKeyManager)

    # Skip-already-scored cache: md5 -> {percentile, private_percentile} from a
    # prior run, read out of each attempt's persisted selected_scoring. Lets a
    # re-run (e.g. after re-selecting) submit ONLY CSVs it hasn't scored yet —
    # unchanged finals are not re-submitted, saving Kaggle quota. --rescore
    # bypasses this. Seeds seen_md5 so within- and cross-run dedup are unified.
    if not rescore:
        for aj in task_dir.glob("attempt_*.json"):
            try:
                outcome = json.loads(aj.read_text()).get("outcome") or {}
            except (OSError, json.JSONDecodeError):
                continue
            prev = outcome.get("selected_scoring") or {}
            for side in ("last", "selected"):
                m = prev.get(f"{side}_md5")
                if m and m not in seen_md5 and (
                    prev.get(f"{side}_public_pct") is not None
                    or prev.get(f"{side}_private_pct") is not None
                    # Also cache md5s that Kaggle DEFINITIVELY scored as
                    # unscoreable (format-reject → pct=None). Identical bytes
                    # will always reject the same way, so re-submitting them
                    # each run only burns Kaggle quota. Transient failures
                    # (budget-exhausted / network) never set {side}_scored, so
                    # they are still retried on a later run.
                    or prev.get(f"{side}_scored")):
                    seen_md5[m] = {"percentile": prev.get(f"{side}_public_pct"),
                                   "private_percentile": prev.get(f"{side}_private_pct"),
                                   "rank": prev.get(f"{side}_rank"),
                                   "medal": prev.get(f"{side}_medal"),
                                   "private_rank": prev.get(f"{side}_private_rank"),
                                   "private_medal": prev.get(f"{side}_private_medal")}
            # Also seed from DEFAULT-mode scores. Default scoring writes
            # outcome.final_score/percentile/private_percentile for the SAME
            # submission_attempt_i.csv this fn grades as `last`. Without this,
            # a run already default-scored would re-submit every last CSV as a
            # duplicate (same md5 -> same Kaggle result). Key by the md5 of the
            # resolved last CSV so the `last/aI` grade is skipped.
            if outcome.get("final_score") is not None:
                ai = aj.stem.rsplit("_", 1)[-1]
                last_csv = _resolve_csv(task_dir, f"submission_attempt_{ai}.csv", artifact_root)
                try:
                    m = hashlib.md5(last_csv.read_bytes()).hexdigest()
                except OSError:
                    m = None
                if m and m not in seen_md5:
                    seen_md5[m] = {"percentile": outcome.get("percentile"),
                                   "private_percentile": outcome.get("private_percentile"),
                                   "rank": outcome.get("rank"),
                                   "medal": outcome.get("medal"),
                                   "private_rank": outcome.get("private_rank"),
                                   "private_medal": outcome.get("private_medal")}

    def _pcts(info: dict | None) -> tuple[float | None, float | None]:
        """Return (public_pct, private_pct) WITHOUT conflating them. Public-only
        comps (e.g. house-prices) have private=None — we must NOT stuff the
        public value into a private field (that fabricates a private score)."""
        if not info or info.get("error") or info.get("budget_exhausted"):
            return None, None
        return info.get("percentile"), info.get("private_percentile")

    def _primary(pub: float | None, priv: float | None) -> float | None:
        """Primary metric for aggregation = private where it exists, else public."""
        return priv if priv is not None else pub

    def _medals(info: dict | None) -> tuple:
        """(pub_rank, pub_medal, priv_rank, priv_medal) from a kaggle_info dict,
        guarded like _pcts. Returns Nones for an errored/exhausted grade or a
        cache entry that only carried percentiles."""
        if not info or info.get("error") or info.get("budget_exhausted"):
            return None, None, None, None
        return (info.get("rank"), info.get("medal"),
                info.get("private_rank"), info.get("private_medal"))

    def _scored(info: dict | None) -> bool:
        """True iff a real submission got a DEFINITIVE Kaggle verdict — a
        percentile, OR an unscoreable reject Kaggle itself returned (status
        contains ERROR: bad format/columns/values, permanent for identical
        bytes). False for a client-side EXCEPTION (network/timeout — never got
        a verdict), budget_exhausted, or a pre-submit shape skip, so those are
        retried on a later run rather than cached as done. Note EXCEPTION has
        no 'ERROR' substring, so it stays retriable."""
        if not info or info.get("budget_exhausted") or info.get("skipped"):
            return False
        if info.get("error"):
            return "ERROR" in str(info.get("status", "")).upper()
        return True

    def _grade(csv_file: Path, tag: str) -> tuple[dict | None, str | None]:
        """Returns (kaggle_info, md5). md5 lets the caller persist last_md5/
        selected_md5 so a later run can skip-already-scored."""
        if not csv_file.exists():
            return None, None
        try:
            raw = csv_file.read_bytes()
        except OSError:
            return None, None
        md5 = hashlib.md5(raw).hexdigest()
        if md5 in seen_md5:            # already graded (this run OR a prior run)
            return seen_md5[md5], md5
        if dry_run:
            print(f"  [dry-run] would submit {tag} {csv_file.name}")
            return None, md5
        if not multi and budget["used"] >= max_per_comp:
            return {"budget_exhausted": True}, md5
        if data_centric:
            # Column-schema match, NOT row count (submission is a training set).
            if exp_cols is not None:
                try:
                    cols = raw.split(b"\n", 1)[0].decode(errors="replace").strip().split(",")
                except Exception:
                    cols = None
                if cols is not None and cols != exp_cols:
                    print(f"  [skip] {tag}: columns {cols}!=expected {exp_cols}")
                    return {"skipped": f"cols_mismatch"}, md5
        else:
            n_lines = raw.count(b"\n") + (0 if raw.endswith(b"\n") else 1)
            exp = _expected_n_lines(task_id, data_path, [])
            if not _n_lines_ok(n_lines, task_id, data_path, exp):
                print(f"  [skip] {tag}: n_lines={n_lines}!=expected {exp}")
                return {"skipped": f"n_lines={n_lines}!={exp}"}, md5
        print(f"  submit {tag} {csv_file.name} ...", end=" ", flush=True)
        info = _score_csv_file(csv_file, task_id, lb, manager,
                               description=f"selected {task_id} {tag}")
        if not (info and info.get("budget_exhausted")):
            budget["used"] += 1
        pub, priv = _pcts(info)
        print(f"pub={pub} priv={priv} (comp submits: {budget['used']})")
        if info and not info.get("error"):
            seen_md5[md5] = info
        time.sleep(delay)
        return info, md5

    per_attempt = []
    for attempt_json in sorted(task_dir.glob("attempt_*.json")):
        ai = attempt_json.stem.rsplit("_", 1)[-1]
        # Read the attempt JSON up-front so we can recover a prior score when its
        # CSV has since vanished, and so we can skip rewriting an unchanged file.
        try:
            data = json.loads(attempt_json.read_text())
        except (OSError, json.JSONDecodeError):
            data = None
        prev = ((data or {}).get("outcome") or {}).get("selected_scoring") or {}

        last_csv = _resolve_csv(task_dir, f"submission_attempt_{ai}.csv", artifact_root)
        sel_csv = _resolve_csv(task_dir, f"selected_submission_attempt_{ai}.csv", artifact_root)
        last_info, last_md5 = _grade(last_csv, f"last/a{ai}")
        sel_info, sel_md5 = _grade(sel_csv, f"sel/a{ai}")
        last_pub, last_priv = _pcts(last_info)
        sel_pub, sel_priv = _pcts(sel_info)
        last_rank, last_medal, last_priv_rank, last_priv_medal = _medals(last_info)
        sel_rank, sel_medal, sel_priv_rank, sel_priv_medal = _medals(sel_info)
        last_scored, sel_scored = _scored(last_info), _scored(sel_info)

        # PRESERVATION BIAS — scoped strictly to a CSV that was scored before and
        # is NO LONGER ON DISK this run (e.g. the scratch artifact was purged).
        # In that one case, keep the prior recorded score instead of nulling it.
        # A CSV that is still present and grades to a NEW value updates normally
        # (this guard never fires when last_csv/sel_csv exists), so legitimately
        # improved selected scores are not blocked.
        if (not last_csv.exists() and last_pub is None and last_priv is None
                and (prev.get("last_public_pct") is not None
                     or prev.get("last_private_pct") is not None)):
            last_md5 = prev.get("last_md5")
            last_pub, last_priv = prev.get("last_public_pct"), prev.get("last_private_pct")
            last_rank, last_medal = prev.get("last_rank"), prev.get("last_medal")
            last_priv_rank, last_priv_medal = prev.get("last_private_rank"), prev.get("last_private_medal")
            last_scored = bool(prev.get("last_scored") or last_pub is not None or last_priv is not None)
        if (not sel_csv.exists() and sel_pub is None and sel_priv is None
                and (prev.get("selected_public_pct") is not None
                     or prev.get("selected_private_pct") is not None)):
            sel_md5 = prev.get("selected_md5")
            sel_pub, sel_priv = prev.get("selected_public_pct"), prev.get("selected_private_pct")
            sel_rank, sel_medal = prev.get("selected_rank"), prev.get("selected_medal")
            sel_priv_rank, sel_priv_medal = prev.get("selected_private_rank"), prev.get("selected_private_medal")
            sel_scored = bool(prev.get("selected_scored") or sel_pub is not None or sel_priv is not None)

        # primary metric = private-or-public, per side; hedge = max over sides
        last_p, sel_p = _primary(last_pub, last_priv), _primary(sel_pub, sel_priv)
        hedge = max([p for p in (last_p, sel_p) if p is not None], default=None)
        rec = {"attempt": ai,
               "last_md5": last_md5, "selected_md5": sel_md5,
               "last_public_pct": last_pub, "last_private_pct": last_priv,
               "selected_public_pct": sel_pub, "selected_private_pct": sel_priv,
               "last_pct": last_p, "selected_pct": sel_p, "hedge_pct": hedge,
               # rank/medal per side so downstream medal-rate reads the SELECTOR's
               # medal too, not just the final sub's (seeded back into seen_md5 on
               # re-runs via the {side}_rank/_medal keys).
               "last_rank": last_rank, "last_medal": last_medal,
               "last_private_rank": last_priv_rank, "last_private_medal": last_priv_medal,
               "selected_rank": sel_rank, "selected_medal": sel_medal,
               "selected_private_rank": sel_priv_rank, "selected_private_medal": sel_priv_medal,
               # Kaggle gave a definitive verdict for this md5 (a pct OR a
               # permanent format-reject). Lets a later run skip-cache even the
               # None-pct rejects; NOT set for budget-exhausted/transient fails.
               "last_scored": last_scored, "selected_scored": sel_scored,
               "metric": "private" if (last_priv is not None or sel_priv is not None)
                         else "public"}
        per_attempt.append(rec)
        # Only rewrite when the scoring record actually changed. An already-scored
        # attempt whose CSV vanished merges back to exactly its prior record above
        # (rec == prev), so it is skipped — the scorer never dirties a file it has
        # no new information for.
        if not dry_run and data is not None and rec != prev:
            try:
                o = data.setdefault("outcome", {})
                o["selected_scoring"] = rec
                # Write into the CORRECT canonical outcome fields (never conflate):
                # public -> percentile, private -> private_percentile. Public-only
                # comps leave private_percentile as None (do NOT fabricate it).
                # Also persist the RAW score + canonical rank/medal from the last
                # submission so downstream medal-rate needs no percentile inversion
                # (which is ambiguous at tie clusters). last_info carries these from
                # _score_csv_file; None-guarded so a missing side doesn't clobber.
                if last_pub is not None:
                    o["percentile"] = last_pub
                if last_priv is not None:
                    o["private_percentile"] = last_priv
                if isinstance(last_info, dict):
                    for k in ("public_score", "rank", "medal", "above_median",
                              "private_score", "private_rank", "private_medal",
                              "private_above_median"):
                        if last_info.get(k) is not None:
                            o[k] = last_info[k]
                attempt_json.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            except (OSError, json.JSONDecodeError):
                pass

    def _best(key):
        vals = [a[key] for a in per_attempt if a.get(key) is not None]
        return max(vals) if vals else None

    result = {"task_id": task_id, "per_attempt": per_attempt,
              "best_last": _best("last_pct"),
              "best_selected": _best("selected_pct"),
              "best_hedge": _best("hedge_pct")}
    print(f"[{task_id}] best_last={result['best_last']} "
          f"best_selected={result['best_selected']} "
          f"best_hedge={result['best_hedge']}")
    return result


def run_score_selected(traj_dir: Path, leaderboard: dict, delay: float,
                       dry_run: bool, data_path: Path, rescore: bool = False,
                       max_per_comp: int = 10,
                       kaggle_keys: list[Path] | None = None,
                       artifact_root: Path | None = None,
                       only_task: str | None = None) -> None:
    """Score last vs selected (max-hedge) for every task under traj_dir."""
    if dry_run:
        manager = None
    elif kaggle_keys:
        manager = MultiKeyManager(kaggle_keys, per_key_cap=max_per_comp)
    else:
        manager = DSPredictSubmissionManager()
    task_dirs = sorted(p for p in traj_dir.iterdir()
                       if p.is_dir() and any(p.glob("attempt_*.json"))
                       and (only_task is None or p.name == only_task))
    print(f"Found {len(task_dirs)} task dirs.")
    results = []
    for task_dir in task_dirs:
        r = score_task_selected(task_dir, leaderboard, manager, delay, dry_run,
                                data_path, rescore, max_per_comp, artifact_root)
        if r:
            results.append(r)
    if not results:
        return

    def _mean(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    n = len(results)
    print(f"\n=== Selected-vs-last summary ({n} tasks; primary=private pct) ===")
    for key, label in (("best_last", "last (final)      "),
                       ("best_selected", "selected (picker) "),
                       ("best_hedge", "HEDGE max(l,sel)  ")):
        m = _mean(key)
        vals = [r[key] for r in results if r.get(key) is not None]
        am = sum(1 for x in vals if x > 50)
        if m is not None:
            print(f"  {label} — mean {m:.1f}%  above-median {am}/{len(vals)}")
    out_path = traj_dir.parent.parent / "scored_selected_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Selected-scoring results → {out_path}")


def update_trajectory_file(path: Path, metrics: dict) -> None:
    with open(path) as f:
        data = json.load(f)
    outcome = data.setdefault("outcome", {})
    for key in ("final_score", "percentile", "above_median", "medal", "rank",
                "private_score", "private_percentile", "private_above_median", "private_medal"):
        if key in metrics:
            outcome[key] = metrics[key]
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--leaderboard-stats", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rescore", action="store_true",
                        help="Resubmit and overwrite already-scored trajectories (use to backfill private scores)")
    parser.add_argument("--delay", type=float, default=5.0,
                        help="Seconds to wait between submissions (avoid rate limiting)")
    parser.add_argument("--score-history", action="store_true",
                        help="Score EVERY distinct submission CSV across all attempts "
                             "(from csv_history manifests), and report best_final_of_5 "
                             "and best_any_csv per task. Many more Kaggle submissions.")
    parser.add_argument("--score-selected", action="store_true",
                        help="Score the FINAL (submission_attempt_i.csv) and the "
                             "SELECTOR's pick (selected_submission_attempt_i.csv) for "
                             "each attempt, report per-attempt max(last, selected) and "
                             "per-task best_last / best_selected / best_hedge. Dedups by "
                             "md5 (selected==last -> 1 call), <=10 submits/comp. Run the "
                             "offline selector first to create the selected_* CSVs.")
    parser.add_argument("--data-path", default="data/data/dspredict-easy",
                        help="Path to task data (for sample_submission.csv row-count check)")
    parser.add_argument("--max-per-competition", type=int, default=None,
                        help="Max Kaggle submissions per competition per run. Default "
                             "8 for --score-history (leave headroom under the ~10/day "
                             "cap), 10 for --score-selected (last+selected per attempt "
                             "needs the full budget so no pick is blocked). With "
                             "--kaggle-keys this is PER KEY. Unscored versions are "
                             "picked up on a later run via skip-already-scored.")
    parser.add_argument("--kaggle-keys", default=None,
                        help="Comma-separated kaggle.json paths (or a dir of them) to "
                             "distribute submissions over. "
                             "score-history only.")
    parser.add_argument("--csv-artifact-root", default=None,
                        help="Root where bulky submission CSVs live when persisted to "
                             "scratch instead of next to the attempt JSONs (harness "
                             "csv_artifact_root). Falls back to it when a CSV is not in "
                             "the trajectory dir. score-selected only.")
    parser.add_argument("--task", default=None,
                        help="Only score this task_id (e.g. playground-series-s3e13).")
    args = parser.parse_args()

    traj_dir = Path(args.trajectory_dir)
    lb = load_leaderboard(Path(args.leaderboard_stats))

    # Per-mode default cap: score-selected needs the full 10 (last+selected per
    # attempt = up to 10 distinct submits/comp) so no pick is blocked; history
    # leaves headroom at 8.
    if args.max_per_competition is None:
        args.max_per_competition = 10 if args.score_selected else 8

    if args.score_history:
        keys = None
        if args.kaggle_keys:
            kp = Path(args.kaggle_keys)
            if kp.is_dir():
                keys = sorted(kp.glob("*.json"))
            else:
                keys = [Path(x) for x in args.kaggle_keys.split(",") if x.strip()]
        run_score_history(traj_dir, lb, args.delay, args.dry_run,
                          Path(args.data_path), args.rescore,
                          args.max_per_competition, keys)
        return

    if args.score_selected:
        keys = None
        if args.kaggle_keys:
            kp = Path(args.kaggle_keys)
            keys = (sorted(kp.glob("*.json")) if kp.is_dir()
                    else [Path(x) for x in args.kaggle_keys.split(",") if x.strip()])
        run_score_selected(traj_dir, lb, args.delay, args.dry_run,
                           Path(args.data_path), args.rescore,
                           args.max_per_competition, keys,
                           artifact_root=Path(args.csv_artifact_root) if args.csv_artifact_root else None,
                           only_task=args.task)
        return
    files = sorted(f for f in traj_dir.rglob("*.json")
                   if not f.name.endswith(".error.json") and not f.name.startswith("scored_")
                   and not f.name == "error.json"
                   # Skip reviewer-agent sub-JSONs (rubric stores, reviewer call
                   # logs under prm_reviews_attempt_*/ and reviewer_rubric_store/)
                   # — these are not trajectory records and have no `task` key.
                   and not any(p == "reviewer_rubric_store" or p.startswith("prm_reviews_attempt_")
                               for p in f.parts))

    manager = None if args.dry_run else DSPredictSubmissionManager()

    def _is_traj(f: Path) -> bool:
        try:
            return "task" in json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            return False
    files = [f for f in files if _is_traj(f)]

    valid_files = [f for f in files if TrajectoryRecord.from_json_file(str(f)).outcome.valid_submission]
    print(f"Found {len(files)} trajectories, {len(valid_files)} with valid submissions.")

    results = []
    for path in valid_files:
        record = TrajectoryRecord.from_json_file(str(path))
        print(f"[{record.task.task_id}] {record.trajectory_id}")
        metrics = score_trajectory(record, lb, manager, REPO_ROOT, args.dry_run, args.rescore)
        if metrics:
            update_trajectory_file(path, metrics)
            results.append({"trajectory_id": record.trajectory_id, "task_id": record.task.task_id, **metrics})
            def _fmt(v, fmt=".4f"): return format(v, fmt) if v is not None else "None"
            print(f"    pub={_fmt(metrics['final_score'])} ({_fmt(metrics['percentile'],'.1f')}% / above={metrics['above_median']})"
                  f"  priv={_fmt(metrics['private_score'])} ({_fmt(metrics['private_percentile'],'.1f')}% / above={metrics['private_above_median']})")
        if not args.dry_run and metrics:
            time.sleep(args.delay)

    if results:
        pub_above = sum(1 for r in results if r.get("above_median"))
        priv_above = sum(1 for r in results if r.get("private_above_median"))
        priv_scored = sum(1 for r in results if r.get("private_score") is not None)
        pub_pcts = [r["percentile"] for r in results if r.get("percentile") is not None]
        priv_pcts = [r["private_percentile"] for r in results if r.get("private_percentile") is not None]
        print(f"\n=== Summary ===")
        print(f"Scored: {len(results)}/{len(valid_files)}")
        print(f"Public  — above median: {pub_above}/{len(results)} ({100*pub_above/len(results):.1f}%)  "
              f"mean percentile: {sum(pub_pcts)/len(pub_pcts):.1f}%" if pub_pcts else "")
        if priv_scored:
            print(f"Private — above median: {priv_above}/{priv_scored} ({100*priv_above/priv_scored:.1f}%)  "
                  f"mean percentile: {sum(priv_pcts)/len(priv_pcts):.1f}%" if priv_pcts else "")

        out_path = traj_dir.parent.parent / "scored_summary.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Full results → {out_path}")


if __name__ == "__main__":
    main()
