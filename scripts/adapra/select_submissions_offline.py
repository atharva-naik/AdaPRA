"""
Offline (post-hoc) submission selection.

Runs the agentic submission-selector sub-agent over already-collected V4
trajectories — no need to run it inline during collection (safer: a selector
crash can't affect the run, and it's re-runnable). For each task/attempt it:
  1. Reconstructs the event stream from the saved trajectory steps (the events
     JSONL isn't persisted to the output dir).
  2. Runs select_submission_subagent over that attempt's submission candidates.
  3. Writes the pick to <csv_history_attempt_i>/final_selection.json AND copies
     the chosen CSV to a selected_submission_attempt_i.csv in the task dir
     (does NOT overwrite the original submission_attempt_i.csv).

Idempotent: attempts that already have a final_selection.json are skipped
(unless --rerun).

Usage:
  PYTHONPATH=. python scripts/adapra/select_submissions_offline.py \
    --run outputs/adapra/trajectories_v4/dspredict-easy-bon5-qwen3/dspredict-easy \
    [--task playground-series-s3e14] [--rerun] [--min-candidates 2]
"""
from __future__ import annotations
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def build_events(steps: list[dict]) -> list[dict]:
    """Reconstruct the event stream the selector's transcript renderer expects
    from saved TrajectoryRecord steps."""
    ev = []
    for s in steps:
        role, tt = s.get("role"), s.get("tool_type")
        if role == "assistant" and tt in ("bash", "file_write", "file_read", "think", "finish"):
            ev.append({"kind": "ActionEvent",
                       "action": {"kind": tt, "command": s.get("code", "")},
                       "thought": s.get("content", "")})
        elif role == "user" and s.get("observation"):
            ev.append({"kind": "ObservationEvent",
                       "observation": {"content": [{"type": "text", "text": s["observation"]}]}})
    return ev


def _expected_lines(task_id: str, data_path: Path, subs: list[dict]) -> int | None:
    sample = data_path / task_id / "sample_submission.csv"
    if sample.exists():
        try:
            with open(sample) as f:
                return sum(1 for _ in f)
        except OSError:
            pass
    counts = [e.get("n_lines") for e in subs if e.get("n_lines")]
    return Counter(counts).most_common(1)[0][0] if counts else None


def _sample_columns(task_id: str, data_path: Path) -> list[str] | None:
    """Header columns of sample_submission.csv (for data-centric schema match)."""
    sample = data_path / task_id / "sample_submission.csv"
    if sample.exists():
        try:
            return sample.open().readline().strip().split(",")
        except OSError:
            pass
    return None


def _is_data_centric(task_desc: str) -> bool:
    """Data-centric competition: submission is a TRAINING DATASET, not
    predictions, and may have fewer rows than sample_submission — so the
    selector must match COLUMNS, not row count. Same signals as the loader."""
    d = (task_desc or "").lower()
    return any(s in d for s in (
        "instead of submitting predictions", "submit a dataset",
        "submission will be used as a training dataset",
        "submit a dataset that will be used to train", "data-centric"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="task-container dir (…/dspredict-easy)")
    ap.add_argument("--task", default=None, help="only this task_id")
    ap.add_argument("--data-path", default="data/data/dspredict-easy",
                    help="for sample_submission.csv row-count check")
    ap.add_argument("--model", default="bedrock/us.anthropic.claude-sonnet-4-6",
                    help="Selector model. Sonnet 4.6 by default — the selector is a "
                         "quality-critical, few-call reasoning task and the coder model "
                         "(qwen) mis-reasoned gap-vs-std. Cheap to run on Sonnet.")
    ap.add_argument("--native-tool-calling", type=lambda s: s.lower() == "true",
                    default=None,
                    help="Force native tool calling on/off. Default: auto per model "
                         "(Anthropic->True, Qwen->False). Only set to override.")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--rerun", action="store_true",
                    help="re-select even if final_selection.json exists")
    ap.add_argument("--min-candidates", type=int, default=2,
                    help="skip attempts with fewer than this many distinct candidates")
    ap.add_argument("--csv-artifact-root", default=None,
                    help="Root where bulky csv_history_attempt_i/ dirs live when they "
                         "were persisted to scratch instead of next to the attempt "
                         "JSONs (see harness csv_artifact_root). The run/split/task "
                         "tree is mirrored under it; selected_submission_*.csv is still "
                         "written back into the JSON task dir. Default: colocated.")
    args = ap.parse_args()

    from dsgym.harness.submission_selector import select_submission_subagent
    # native_tool_calling is auto-selected per model family inside
    # select_submission_subagent (Anthropic -> native True, Qwen -> False).
    # Only pass it here to force a specific mode.
    llm_kwargs = {"model": args.model, "aws_region_name": args.region}
    if args.native_tool_calling is not None:
        llm_kwargs["native_tool_calling"] = args.native_tool_calling
    base = Path(args.run)
    data_path = Path(args.data_path)
    artifact_root = Path(args.csv_artifact_root) if args.csv_artifact_root else None

    def _scratch_task_dir(task_dir: Path) -> Path:
        """Mirror of the task dir under csv_artifact_root (scratch). When no
        artifact_root is set, falls back to the home task dir."""
        if artifact_root is None:
            return task_dir
        parts = task_dir.resolve().parts
        if "trajectories_v4" in parts:
            tail = Path(*parts[parts.index("trajectories_v4") + 1:])
        else:
            tail = Path(*parts[-3:])
        return artifact_root / tail

    def _hist_dir(task_dir: Path, ai: str) -> Path:
        """csv_history dir for this attempt. When --csv-artifact-root is set the
        bulky history lives on scratch under the mirrored run/split/task tree
        (tail after 'trajectories_v4'); otherwise it sits next to the JSON."""
        return _scratch_task_dir(task_dir) / f"csv_history_attempt_{ai}"

    import tempfile
    n_done = n_skip = n_run = 0
    for td in sorted(p for p in base.iterdir() if p.is_dir()):
        if args.task and td.name != args.task:
            continue
        for aj in sorted(td.glob("attempt_*.json")):
            ai = aj.stem.rsplit("_", 1)[-1]
            hist = _hist_dir(td, ai)
            if not hist.exists() or not (hist / "manifest.json").exists():
                continue
            sel_path = hist / "final_selection.json"
            if sel_path.exists() and not args.rerun:
                n_skip += 1
                continue
            man = json.loads((hist / "manifest.json").read_text())
            subs = [e for e in man
                    if "submission" in Path(e.get("source_path", "")).name.lower()
                    and e.get("n_lines")]
            steps = json.loads(aj.read_text()).get("steps", [])
            # Data-centric tasks (e.g. s3e21): submission is a TRAINING dataset
            # that may have FEWER rows than sample — filter candidates by COLUMN
            # schema, not row count.
            task_desc = (json.loads(aj.read_text()).get("task", {}) or {}).get("description", "")
            data_centric = _is_data_centric(task_desc)
            exp_cols = _sample_columns(td.name, data_path) if data_centric else None
            exp = None if data_centric else _expected_lines(td.name, data_path, subs)
            if data_centric and exp_cols:
                def _cols(e):
                    p = hist / e["archived_as"]
                    try: return p.open().readline().strip().split(",")
                    except OSError: return []
                cands = [e for e in subs if _cols(e) == exp_cols]
            else:
                cands = [e for e in subs if exp is None or e["n_lines"] == exp]
            if len(cands) < args.min_candidates:
                n_skip += 1
                continue

            # write the chosen CSV to a NON-destructive path (don't clobber the
            # original final submission); harness/scoring can prefer it if wanted.
            # Bulky CSVs go to the scratch mirror (csv_artifact_root), NOT home
            # NFS — the scorer's _resolve_csv falls back there. Keeps home lean.
            scratch_td = _scratch_task_dir(td)
            scratch_td.mkdir(parents=True, exist_ok=True)
            out_csv = str(scratch_td / f"selected_submission_attempt_{ai}.csv")
            try:
                res = select_submission_subagent(
                    llm_kwargs=llm_kwargs, events=build_events(steps),
                    csv_history_dir=hist, submission_path=out_csv,
                    expected_lines=exp, workspace_dir=tempfile.mkdtemp(),
                    expected_columns=exp_cols)
                n_run += 1
                print(f"{td.name} a{ai}: {res.get('status', 'chose')} "
                      f"turn={res.get('turn')} script={res.get('script')}", flush=True)
            except Exception as e:
                print(f"{td.name} a{ai}: ERROR {str(e)[:100]}", flush=True)
                continue
            n_done += 1
    print(f"\nSelected {n_run}, skipped {n_skip} (already done / too few candidates).")


if __name__ == "__main__":
    main()
