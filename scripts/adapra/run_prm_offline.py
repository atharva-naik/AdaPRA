"""
Offline (post-hoc) agentic-PRM scoring over collected V4 trajectories.

For each task/attempt, fires the PRM every N action-steps (the calling strategy
we ablate on) and writes one JSON per call into
  <task_dir>/prm_calls_attempt_<i>/prm_call_00.json, prm_call_01.json, ...
Each JSON is self-contained: rubric scores, composite, and THAT call's cost.
The attempt's total PRM cost is the sum across its per-call JSONs (derived; no
rollup file). Safe/re-runnable: skips attempts whose folder already has calls
unless --rerun.

Usage:
  PYTHONPATH=. python scripts/adapra/run_prm_offline.py \
    --run outputs/adapra/trajectories_v4/dspredict-easy-bon5-qwen3-v2/dspredict-easy \
    --every-n-steps 10 [--task playground-series-s3e14] [--rerun]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def build_events(steps: list[dict]) -> list[dict]:
    """Reconstruct the event stream the transcript renderer expects from saved
    TrajectoryRecord steps (events JSONL isn't persisted to the output dir)."""
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


def _n_action_steps(events: list[dict]) -> int:
    return sum(1 for e in events if (e.get("kind") or e.get("type")) == "ActionEvent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="task-container dir (…/dspredict-easy)")
    ap.add_argument("--task", default=None, help="only this task_id")
    ap.add_argument("--every-n-steps", type=int, default=10,
                    help="fire the PRM every N action-steps (default 10)")
    ap.add_argument("--model", default="bedrock/qwen.qwen3-coder-next")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--rerun", action="store_true",
                    help="re-score even if prm_calls_attempt_i/ already has calls")
    args = ap.parse_args()

    from dsgym.harness.prm_agent import score_trajectory_at_step
    llm_kwargs = {"model": args.model, "aws_region_name": args.region}
    base = Path(args.run)

    n_attempts = n_calls = 0
    for td in sorted(p for p in base.iterdir() if p.is_dir()):
        if args.task and td.name != args.task:
            continue
        for aj in sorted(td.glob("attempt_*.json")):
            ai = aj.stem.rsplit("_", 1)[-1]
            out_dir = td / f"prm_calls_attempt_{ai}"
            if out_dir.exists() and any(out_dir.glob("prm_call_*.json")) and not args.rerun:
                continue
            try:
                data = json.loads(aj.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            events = build_events(data.get("steps", []))
            total_steps = _n_action_steps(events)
            if total_steps == 0:
                continue
            task_desc = (data.get("task", {}) or {}).get("description", "")

            # Fire at N, 2N, ... and always a final call at the last step.
            cutoffs = list(range(args.every_n_steps, total_steps + 1, args.every_n_steps))
            if not cutoffs or cutoffs[-1] != total_steps:
                cutoffs.append(total_steps)

            if args.rerun:  # clear stale calls
                for f in out_dir.glob("prm_call_*.json"):
                    f.unlink()
            for idx, cut in enumerate(cutoffs):
                rec = score_trajectory_at_step(
                    llm_kwargs=llm_kwargs, events=events, task_desc=task_desc,
                    up_to_step=cut, trigger="every_n_steps",
                    attempt_index=int(ai),
                    out_path=out_dir / f"prm_call_{idx:02d}.json")
                n_calls += 1
                print(f"{td.name} a{ai} call {idx} @step{rec['step']}: "
                      f"composite={rec['composite_score']:.3f} "
                      f"cost={rec['cost']['total_tokens']}tok", flush=True)
            n_attempts += 1
    print(f"\nScored {n_attempts} attempts, {n_calls} PRM calls total.")


if __name__ == "__main__":
    main()
