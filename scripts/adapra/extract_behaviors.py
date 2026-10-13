"""
Behavior-extraction sweep — step 1 of PRM rubric discovery.

Motivation:
the current PRM rubrics measure hygiene the agent already satisfies (47% pure
praise) and MISS the process failures that actually decide Kaggle outcomes
(leaky/overfit CV trusted, worse-than-best model shipped, incomplete submission,
wasted-motion churn). To fix the rubric set empirically rather than by intuition,
we sweep a frontier judge (Sonnet) over EVERY attempt in a run and elicit the
concrete, generalizable behaviors that drove each outcome — good behaviors from
strong attempts, issues from weak/invalid ones. Step 2 (cluster_to_rubrics.py)
clusters these into MECE rubric proposals.

This borrows the "distinguish among diverse high/low outcomes" idea from Chasing
the Tail (Zhang et al. 2026, Algorithm 1) but is specialized to OUR domain: the
judge is prompted as a data-science process reward model and told which
competition failure modes matter here. It sees THREE views of each attempt:
  1. the turn-annotated transcript (the process tau; prm_agent.truncated_transcript),
  2. the end-of-attempt workspace snapshot (the artifacts W; metadata.workspace_snapshot),
  3. the agent's own end-of-attempt summary (metadata.attempt_summary, the "abstract").
Each mined behavior is tagged grounding = "artifact" (checkable from W alone) or
"trajectory" (checkable only from the process), matching the paper's rubric
taxonomy so discovery yields both artifact- and trajectory-grounded rubrics.
Use --no-artifacts / --no-abstract to ablate back to the transcript-only sweep.

Labeling (per attempt):
  - invalid submission            -> verdict "below"  (what-not-to-do signal)
  - valid, private/public pct      -> banded WITHIN-TASK (top/bottom/mid) AND by
                                      ABSOLUTE percentile; both stored so the
                                      clustering step chooses the cutoff, not this
                                      sweep. verdict = above if within-task top band.
  - public-only comps (house-prices, titanic) use `percentile` when
    `private_percentile` is absent — they are NOT dropped.
  - empty transcript (num_turns==0) -> skipped (nothing to extract).

Output: one JSON per attempt under --out-dir + aggregated behaviors.jsonl.

Usage:
  PYTHONPATH=. python scripts/adapra/extract_behaviors.py \
    --run outputs/adapra/trajectories_v4/dspredict-easy-bon5-qwen3-v2/dspredict-easy \
    --out-dir outputs/adapra/behavior_sweep/bon5-qwen3-v2 \
    --model bedrock/us.anthropic.claude-sonnet-4-6 --region us-east-1 --workers 6
"""
from __future__ import annotations
import argparse, json, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# PRM-style system prompt: same framing/inputs as dsgym/harness/prm_agent.py, but
# the TASK is behavior extraction for rubric discovery (not scoring). Domain-
# specific: it names the Kaggle failure/success modes we care about so the judge
# surfaces the signal a process reward model for DS agents actually needs.
EXTRACT_SYSTEM = """You are an expert data-science process reward model (PRM) analyst.

Below is the record of an AGENT solving a Kaggle-style data-science competition,
in up to THREE parts:
  1. TRAJECTORY (the process): the turn-annotated transcript — every action it
     took ([Turn N] ACTION ...) and the (truncated) tool output each produced
     ([Turn N] OUTPUT ...). The initial [Turn 0] USER message is the competition
     task. This is the process tau.
  2. WORKSPACE (the artifacts): a snapshot of the persistent files the agent left
     at the end of this attempt — submission CSVs, code modules, notes, score
     registries, cached predictions. This is the workspace W: what the process
     actually produced, inspectable directly.
  3. ABSTRACT: the agent's own end-of-attempt summary (validation score, approach,
     preprocessing, model). A quick orientation, not ground truth.
Some parts may be absent (empty transcript, empty workspace). The TRAJECTORY plus
WORKSPACE together are what a live process supervisor would see.

We are BUILDING the rubric set for that PRM. Rather than score this trajectory,
your job is to mine it for the CONCRETE, GENERALIZABLE behaviors that drove its
outcome — the raw material from which we will later distill process-reward
rubrics. You are told whether this attempt did WELL or POORLY relative to sibling
attempts on the SAME competition (same data, same time budget), so isolate
PROCESS, not task difficulty or luck.

{mode_instruction}

In THIS domain, the behaviors that actually move Kaggle outcomes cluster around
(use these as lenses, not a checklist — surface whatever the transcript shows):
  - Validation discipline: is the CV scheme consistent and honest, or is a CV gain
    a leak/overfit artifact (calibration/meta-model fit on eval data, an outlier
    CV vs sibling models, tuning noise on a tiny dataset) that the agent TRUSTS?
  - Model selection for submission: does the FINAL submission come from the
    best-CV model, or did a later/worse experiment overwrite it (no revert)?
  - Submission integrity: does the final file cover ALL test rows with real,
    in-range predictions — or is it partial / default-filled / corrupted /
    invalid, despite looking "valid"?
  - Exploration efficiency: productive iteration vs wasted motion — repeated
    identical crashes, re-EDA loops, timed-out searches, plateaued churn.
  - Modeling strength: diverse model families, sensible ensembling/tuning, target
    handling appropriate to the metric.
  - Decisiveness & recovery: catching its own mistakes, reverting regressions,
    finishing with a verified submission vs meandering to the deadline.

Rules for every item you output (so items distill cleanly into rubrics):
- ATOMIC: one distinct behavior per item. Split anything joined by "and".
- SPECIFIC & OBSERVABLE: name the exact action — model family, CV scheme, target
  transform, file op, submission check, bug, re-exploration loop. NOT vague
  hygiene ("did good EDA", "followed best practices").
- GENERALIZABLE: phrase so it could apply to OTHER competitions (e.g. "shipped a
  model with worse CV than one it had already trained" — NOT "shipped v5 over v4").
- EVIDENCE-GROUNDED: cite concrete evidence — a [Turn N] snippet from the
  trajectory AND/OR a workspace path (e.g. "preds/submission.csv", "cv_scores.json").
- GROUNDING (required): tag where this behavior is CHECKABLE:
    "artifact"  = verifiable from the WORKSPACE files alone, without watching the
                  process — e.g. the submission covers all test rows / has in-range
                  values, a CV-score registry file exists, OOF predictions are
                  cached to disk, a code module leaks test data into a fit call.
    "trajectory" = verifiable ONLY from the PROCESS, because it is about HOW the
                  agent worked and leaves no decisive trace in the final files —
                  e.g. burned many turns without pivoting, kept re-running the same
                  failing cell, never reverted after a CV regression, abandoned a
                  better model mid-run. When in doubt, ask: "could I confirm this
                  from the end-state files only?" If yes -> artifact; if I'd need to
                  replay the turns -> trajectory.
- Prefer behaviors that plausibly MOVED the outcome; skip incidental ones.

Respond with ONLY valid JSON:
{{
  "summary": "one sentence: the single biggest process driver of this attempt's outcome",
  "items": [
    {{
      "kind": "good_behavior" | "issue",
      "behavior": "atomic, generalizable description of what the agent did or failed to do",
      "lens": "validation-discipline | model-selection | submission-integrity | exploration-efficiency | modeling-strength | decisiveness-recovery | other",
      "grounding": "artifact" | "trajectory",
      "impact": "how this plausibly helped or hurt the Kaggle outcome, and why a PRM should reward/penalize it",
      "evidence": ["[Turn N] <brief verbatim snippet>", "or workspace path: preds/submission.csv"]
    }}
  ]
}}
Output 2-6 items. If the transcript is empty or too short to judge, return
{{"summary": "insufficient transcript", "items": []}}."""

MODE_ABOVE = (
    "VERDICT: this attempt did WELL — it is in the TOP band of valid sibling "
    "attempts on this competition. Extract mainly GOOD_BEHAVIORS that plausibly "
    "drove the strong result (behaviors a PRM should REWARD and reinforce mid-run). "
    "You may also note an issue that kept it from being even better.")
MODE_BELOW = (
    "VERDICT: this attempt did POORLY — it is in the BOTTOM band of sibling "
    "attempts, OR it produced an INVALID/broken submission. Extract mainly ISSUES "
    "that plausibly caused the weak result (behaviors a PRM should PENALIZE and "
    "steer the agent away from mid-run). You may also note a good behavior it "
    "still exhibited.")


# Workspace-snapshot rendering. The persistent workspace W is where artifact-
# grounded behaviors are checkable (submission integrity, CV registries, cached
# OOF preds, leaky code). We filter harness/runtime noise, rank signal files
# first so they survive the file cap, and include truncated contents for small
# text files (path+size only for the rest).
_WS_SKIP_PREFIX = (".home/", ".openhands", ".cache/", ".config/", ".local/")
_WS_SKIP_SUBSTR = ("__pycache__/", "/.git/", ".ipynb_checkpoints/", "catboost_info/",
                   "wandb/", "lightning_logs/", ".venv/")
_WS_BINARY_SUFFIX = (".pyc", ".pkl", ".parquet", ".h5", ".hdf5", ".joblib", ".model",
                     ".bin", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".so", ".zip",
                     ".gz", ".png", ".jpg", ".jpeg", ".pdf", ".tfevents", ".log")
_CSV_SUFFIX = (".csv", ".tsv")


def _ws_rank(path: str) -> int:
    """Lower = more relevant to process-reward behavior. Keeps submissions, code,
    notes, and score/registry/oof files ahead of incidental data dumps."""
    p = path.lower()
    if "submission" in p and p.endswith(_CSV_SUFFIX):
        return 0
    if p.endswith((".md", ".txt")) or "note" in p or "readme" in p:
        return 1
    if p.endswith(".py"):
        return 2
    if any(k in p for k in ("registry", "cv_score", "cv_result", "scores", "oof",
                            "leaderboard", "ledger", "log")) and p.endswith((".json", ".csv", ".jsonl")):
        return 3
    if p.endswith((".json", ".yaml", ".yml")):
        return 4
    if p.endswith(_CSV_SUFFIX):
        return 5
    return 6


def _render_workspace(snapshot: dict, max_files: int = 40, max_content: int = 1800,
                      csv_head_lines: int = 12) -> str:
    """Render metadata.workspace_snapshot (the end-of-attempt file state) into a
    compact, judge-readable listing: filtered, ranked, with truncated contents."""
    files = (snapshot or {}).get("files") or []
    kept = []
    for f in files:
        path = f.get("path", "")
        if not path or path.startswith(_WS_SKIP_PREFIX) or any(s in path for s in _WS_SKIP_SUBSTR):
            continue
        if path.endswith(_WS_BINARY_SUFFIX):
            continue
        kept.append(f)
    if not kept:
        return ""
    kept.sort(key=lambda f: (_ws_rank(f.get("path", "")), f.get("path", "")))
    shown, extra = kept[:max_files], len(kept) - max_files
    lines = []
    for f in shown:
        path, size = f.get("path", ""), f.get("size", 0)
        content = f.get("content")
        if content is None:
            lines.append(f"### {path}  ({size} B, contents omitted)")
            continue
        if path.lower().endswith(_CSV_SUFFIX):
            rows = content.splitlines()
            body = "\n".join(rows[:csv_head_lines])
            more = f"\n... (+{len(rows) - csv_head_lines} more rows)" if len(rows) > csv_head_lines else ""
            lines.append(f"### {path}  ({size} B, {len(rows)} rows; head)\n{body}{more}")
        else:
            body = content[:max_content]
            more = f"\n... (+{len(content) - max_content} more chars)" if len(content) > max_content else ""
            lines.append(f"### {path}  ({size} B)\n{body}{more}")
    if extra > 0:
        lines.append(f"### (+{extra} more files not shown)")
    return "\n\n".join(lines)


def _load_outcome_score(o: dict):
    """Return (score, valid). Uses private pct, falling back to public pct for
    public-only comps (house-prices, titanic)."""
    valid = bool(o.get("valid_submission"))
    score = o.get("private_percentile")
    if score is None:
        score = o.get("percentile")
    return score, valid


def _band(score, lo, hi):
    if score is None:
        return "unscored"
    if score >= hi:
        return "top"
    if score <= lo:
        return "bottom"
    return "mid"


def _extract_one(aj: Path, verdict: str, meta: dict, llm_kwargs: dict, out_dir: Path, opts: dict):
    from dsgym.harness.prm_agent import truncated_transcript
    from dsgym.adapra.llm_judge_v2 import LLMJudgeV2, _extract_json
    from scripts.adapra.run_prm_offline import build_events

    data = json.loads(aj.read_text())
    steps = data.get("steps") or []
    events = build_events(steps)
    task = data.get("task")
    task_desc = task.get("description", "") if isinstance(task, dict) else ""
    if task_desc:
        events = [{"kind": "MessageEvent", "source": "user",
                   "llm_message": {"content": task_desc[:3000]}}] + events
    transcript, n = truncated_transcript(events, up_to_step=None)

    md = data.get("metadata") or {}
    ws_text = _render_workspace(md.get("workspace_snapshot")) if opts.get("with_artifacts", True) else ""
    abstract = (md.get("attempt_summary") or "").strip() if opts.get("with_abstract", True) else ""

    rec = {"task": aj.parent.name, "attempt": aj.stem, "verdict": verdict, "n_turns": n,
           "has_workspace": bool(ws_text), "has_abstract": bool(abstract), **meta}
    out_f = out_dir / f"{aj.parent.name}__{aj.stem}.json"
    if n == 0:
        rec.update(summary="insufficient transcript", items=[])
        out_f.write_text(json.dumps(rec, indent=2))
        return rec

    mode = MODE_ABOVE if verdict == "above" else MODE_BELOW
    prompt = EXTRACT_SYSTEM.format(mode_instruction=mode)
    prompt += "\n\n===== TRAJECTORY (the process tau; turn-annotated, as the agent saw it) =====\n" + transcript
    if ws_text:
        prompt += ("\n\n===== WORKSPACE (the artifacts W; files left at the end of this attempt) =====\n"
                   + ws_text)
    if abstract:
        prompt += "\n\n===== ABSTRACT (agent's own end-of-attempt summary) =====\n" + abstract[:4000]
    prompt += "\n\nRemember: respond with ONLY the JSON object."
    judge = LLMJudgeV2(**{"workers": 1, **llm_kwargs})
    raw = judge._call(prompt)
    parsed = _extract_json(raw) or {}
    rec.update(
        summary=parsed.get("summary", ""),
        items=parsed.get("items", []),
        cost={"input_tokens": judge.total_prompt_tokens,
              "output_tokens": judge.total_completion_tokens},
    )
    out_f.write_text(json.dumps(rec, indent=2))
    return rec


def build_jobs(base: Path, out_dir: Path, only_task, rerun, abs_lo, abs_hi):
    jobs = []
    for td in sorted(p for p in base.iterdir() if p.is_dir()):
        if only_task and td.name != only_task:
            continue
        atts = []
        for aj in sorted(td.glob("attempt_*.json")):
            try:
                o = json.loads(aj.read_text()).get("outcome") or {}
            except Exception:
                continue
            score, valid = _load_outcome_score(o)
            atts.append((aj, score, valid))
        valid_scores = [s for _, s, v in atts if v and s is not None]
        # within-task quartile cutoffs (fall back to median split for small n)
        if len(valid_scores) >= 4:
            q = statistics.quantiles(valid_scores, n=4)
            wt_lo, wt_hi = q[0], q[2]
        elif valid_scores:
            m = statistics.median(valid_scores)
            wt_lo = wt_hi = m
        else:
            wt_lo = wt_hi = None

        for aj, score, valid in atts:
            wt = "bottom" if not valid else _band(score, wt_lo, wt_hi)
            ab = "bottom" if not valid else _band(score, abs_lo, abs_hi)
            # verdict drives which extraction mode: invalid or within-task bottom
            # -> below; within-task top -> above; mid -> below-leaning (still mine
            # issues, but mid attempts rarely define the tail — treat as below).
            verdict = "above" if (valid and wt == "top") else "below"
            meta = {"score": score, "valid": valid,
                    "within_task_band": wt, "absolute_band": ab,
                    "wt_cut": [wt_lo, wt_hi], "abs_cut": [abs_lo, abs_hi]}
            out_f = out_dir / f"{td.name}__{aj.stem}.json"
            if out_f.exists() and not rerun:
                continue
            jobs.append((aj, verdict, meta))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="bedrock/us.anthropic.claude-sonnet-4-6")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--abs-lo", type=float, default=25.0, help="absolute-band bottom cutoff")
    ap.add_argument("--abs-hi", type=float, default=75.0, help="absolute-band top cutoff")
    ap.add_argument("--task", default=None, help="only this task_id (debug)")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--no-artifacts", action="store_true",
                    help="do NOT feed the workspace snapshot W (transcript-only, legacy behavior)")
    ap.add_argument("--no-abstract", action="store_true",
                    help="do NOT feed the agent's end-of-attempt summary")
    args = ap.parse_args()

    base = Path(args.run)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    llm_kwargs = {"model": args.model, "aws_region_name": args.region}
    opts = {"with_artifacts": not args.no_artifacts, "with_abstract": not args.no_abstract}

    jobs = build_jobs(base, out_dir, args.task, args.rerun, args.abs_lo, args.abs_hi)
    n_above = sum(1 for _, v, _ in jobs if v == "above")
    print(f"[extract] {len(jobs)} attempts to process "
          f"(above={n_above}, below={len(jobs)-n_above}, workers={args.workers}) "
          f"artifacts={opts['with_artifacts']} abstract={opts['with_abstract']}")

    recs = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_extract_one, aj, v, m, llm_kwargs, out_dir, opts): aj
                for aj, v, m in jobs}
        for f in as_completed(futs):
            aj = futs[f]
            try:
                r = f.result()
                recs.append(r)
                print(f"  {r['task']}/{r['attempt']} [{r['verdict']}|wt={r.get('within_task_band')}] "
                      f"{len(r['items'])} items")
            except Exception as e:
                print(f"  ERROR {aj}: {e}")

    # Aggregate ALL per-attempt files (incl. pre-existing) into one JSONL.
    agg = out_dir / "behaviors.jsonl"
    n = 0
    with agg.open("w") as fh:
        for jf in sorted(out_dir.glob("*__attempt_*.json")):
            fh.write(json.dumps(json.loads(jf.read_text())) + "\n")
            n += 1
    print(f"[extract] wrote {agg} ({n} attempts)")


if __name__ == "__main__":
    main()
