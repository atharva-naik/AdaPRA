"""
Cluster extracted behaviors into MECE rubric proposals — step 2 of PRM rubric
discovery (pairs with extract_behaviors.py).

This is a FROM-SCRATCH, UNIFIED discovery: no pre-existing rubric set is shown to
the proposer, so the entire rubric set is discovered bottom-up from the mined
behaviors. Stage 1 (extract_behaviors.py) is already rubric-agnostic — it mines
behaviors purely from each trajectory/workspace/abstract using generic
data-science process lenses — so the whole pipeline discovers rubrics from the
ground up, which is the claim the paper makes.

Reads behaviors.jsonl (per-attempt good_behaviors + issues mined by the sweep),
selects the tail the paper argues matters (default: within-task TOP-band
good_behaviors + within-task BOTTOM-band / invalid issues), pools them into ONE
set, and asks a proposer LLM to cluster them into a small MECE set of
process-reward rubrics. A GOOD_BEHAVIOR and an ISSUE are treated as positive and
negative evidence for the SAME underlying good-practice rubric, so clustering both
sides together yields a single non-overlapping rubric set (the "unified PRM"),
rather than two lists that would duplicate axes across sides.

Each proposal is emitted in the dsgym RubricDef shape (rubric_id, category A-I,
4-label natural_language_instruction, display_name, display_description,
suggested_actions_hint) plus a grounding tag (artifact vs trajectory, carried from
extract_behaviors.py), so it drops into dsgym/adapra/rubrics.py after human
review.

Usage:
  PYTHONPATH=. python scripts/adapra/cluster_to_rubrics.py \
    --behaviors outputs/adapra/behavior_sweep/bon5-qwen3-v2-grounded/behaviors.jsonl \
    --out outputs/adapra/behavior_sweep/bon5-qwen3-v2-grounded/proposed_rubrics_scratch.json \
    --model bedrock/us.anthropic.claude-sonnet-4-6 --region us-east-1 \
    --band within_task   # or: absolute | all
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path


CLUSTER_SYSTEM = """You are designing the COMPLETE rubric set, from scratch, for a
PROCESS REWARD MODEL (PRM) that scores data-science agents on Kaggle-style
competitions, turn by turn, from a turn-annotated transcript of their actions and
tool outputs together with the persistent workspace they build.

You are given a large pool of BEHAVIORS mined from many real agent trajectories.
Each behavior is either a GOOD_BEHAVIOR (observed in a top-outcome attempt) or an
ISSUE (observed in a bottom-outcome / invalid attempt). Treat these as positive and
negative EVIDENCE FOR THE SAME UNDERLYING RUBRICS: a good behavior is an instance of
an agent satisfying some good-practice criterion, and an issue is an instance of an
agent violating one. Your rubric set must cover both. Each behavior is atomic,
generalizable, and evidence-grounded, and carries a GROUNDING tag in parentheses
(lens|grounding):
  - "artifact"   = checkable from the agent's persistent WORKSPACE files alone
                   (submission integrity, a CV-score registry file, cached OOF
                   predictions, leaky code in a module).
  - "trajectory" = checkable only from the PROCESS — HOW the agent worked, leaving
                   no decisive trace in the final files (churn without pivoting,
                   repeated identical crashes, a never-reverted CV regression).

Your job: discover, from the ground up, the SMALL MECE set of process-reward
rubrics implied by this pool. Assume NO pre-existing rubrics — derive every rubric
bottom-up from the behaviors. Together the rubrics should account for the behaviors
that separate strong from weak outcomes in the pool.

Follow these rubric-writing principles:
- MECE: mutually exclusive (no two rubrics penalize the same underlying behavior)
  and collectively exhaustive (together they cover the outcome-separating behaviors
  in the pool). Fold a good behavior and its mirror-image issue into ONE rubric.
- ATOMIC: one distinct behavior per rubric. Break up anything with "and".
- SPECIFIC & OBSERVABLE: scoreable from a transcript and/or workspace — name
  concrete signals (a leaked CV metric, a submission that overwrote a better model,
  a partial submission file, a repeated identical crash). Avoid vague ("good
  practices").
- COVERAGE-WEIGHTED: prioritize clusters that appear FREQUENTLY across the pool and
  clearly track outcome; drop rare/incidental one-offs.
- DOMAIN-GROUNDED: every rubric must be about data-science competition PROCESS.
- GROUNDING COVERAGE: surface BOTH artifact-grounded and trajectory-grounded
  rubrics — do not collapse everything into one type. Assign each rubric a
  "grounding" equal to where it is decisively checkable. A rubric is "artifact"
  only if a supervisor could confirm it from the end-state workspace files alone;
  if confirming it requires inspecting HOW the agent worked over turns, it is
  "trajectory". Keep a rubric's grounding consistent with the behaviors in it.

For each rubric cluster, output a proposal in this schema (the dsgym RubricDef
shape). Phrase everything as a GOOD PRACTICE (yes = agent did the good thing):
{
  "rubric_id": "snake_case_id",
  "category": "A|B|C|D|E|F|G|H|I",   // A=validation/metric, B=exploration/model-diversity, C=preprocessing/target, E=efficiency/robustness/submission, etc — pick the closest
  "grounding": "artifact" | "trajectory",   // where the rubric is decisively checkable (see GROUNDING COVERAGE)
  "display_name": "Neutral Topic Label (Title Case)",
  "description": "one failure-framed sentence: 'Agent fails to ...'",
  "display_description": "one neutral sentence: 'Whether the agent ...'",
  "natural_language_instruction": "4-label scoring instruction with EXACTLY these labels, each on its own line: 'Label \\'agent_failure\\' if: ...', 'Label \\'agent_success\\' if: ...', 'Label \\'not_applicable\\' if: ...', 'Label \\'insufficient_evidence\\' if: ...'. Ground each label in concrete, transcript- or workspace-observable signals.",
  "suggested_actions_hint": "a concrete imperative the PRM should suggest when the agent is FAILING this rubric (names the exact fix, not generic advice)",
  "supporting_behavior_count": <int: how many pool behaviors fall in this cluster>,
  "example_evidence": ["1-3 [Turn N] snippets or workspace paths drawn from the pool"]
}

Aim for a complete but compact set — typically 12-20 rubrics total, no redundant or
overlapping rubrics, both groundings represented. Respond with ONLY valid JSON:
{"rubrics": [ ...proposals... ], "notes": "brief rationale for the set"}."""


def load_behaviors(path: Path, band: str):
    """Return a single unified pool of behaviors filtered to the requested tail
    band. Each entry keeps its kind (good_behavior|issue) so the proposer can see
    positive and negative evidence for the same rubric."""
    pool = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        wt = rec.get("within_task_band")
        ab = rec.get("absolute_band")
        valid = rec.get("valid")
        for it in rec.get("items", []):
            kind = it.get("kind")
            # Tail selection: which attempts count toward each kind.
            if band == "within_task":
                is_top = valid and wt == "top"
                is_bot = (not valid) or wt == "bottom"
            elif band == "absolute":
                is_top = valid and ab == "top"
                is_bot = (not valid) or ab == "bottom"
            else:  # all
                is_top = is_bot = True
            keep = (kind == "good_behavior" and is_top) or (kind == "issue" and is_bot)
            if not keep:
                continue
            pool.append({
                "kind": kind,
                "behavior": it.get("behavior", ""),
                "lens": it.get("lens", ""),
                "grounding": it.get("grounding", ""),
                "impact": it.get("impact", ""),
                "evidence": it.get("evidence", []),
                "task": rec.get("task"), "attempt": rec.get("attempt"),
            })
    return pool


def _pool_text(items):
    lines = []
    for i, it in enumerate(items, 1):
        ev = " | ".join(it["evidence"][:2])
        g = it.get("grounding") or "?"
        k = "GOOD" if it.get("kind") == "good_behavior" else "ISSUE"
        lines.append(f'{i}. [{k}] ({it["lens"]}|{g}) {it["behavior"]}  '
                     f'[impact: {it["impact"]}]  [{it["task"]}/{it["attempt"]}] {ev}')
    return "\n".join(lines)


def propose(pool, llm_kwargs: dict):
    from dsgym.adapra.llm_judge_v2 import LLMJudgeV2, _extract_json
    prompt = (
        CLUSTER_SYSTEM
        + f"\n\n===== BEHAVIOR POOL ({len(pool)} behaviors; GOOD = reward, ISSUE = penalize) =====\n"
        + _pool_text(pool)
        + "\n\nDiscover the complete MECE process-reward rubric set implied by the "
          "pool above, from scratch. Respond with ONLY the JSON object."
    )
    judge = LLMJudgeV2(**{"workers": 1, "max_completion_tokens": 16000, **llm_kwargs})
    raw = judge._call(prompt)
    parsed = _extract_json(raw) or {}
    return parsed, {"input_tokens": judge.total_prompt_tokens,
                    "output_tokens": judge.total_completion_tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="bedrock/us.anthropic.claude-sonnet-4-6")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--band", choices=["within_task", "absolute", "all"],
                    default="within_task", help="which outcome tail to cluster from")
    args = ap.parse_args()

    pool = load_behaviors(Path(args.behaviors), args.band)
    n_good = sum(1 for it in pool if it["kind"] == "good_behavior")
    print(f"[cluster] band={args.band}  pool={len(pool)}  "
          f"(good={n_good}, issue={len(pool) - n_good})")
    print(f"[cluster] lenses: {Counter(it['lens'] for it in pool)}")
    print(f"[cluster] grounding (pool): {Counter(it['grounding'] for it in pool)}")

    llm_kwargs = {"model": args.model, "aws_region_name": args.region}
    props, cost = propose(pool, llm_kwargs)
    rubrics = props.get("rubrics", [])

    out = {
        "mode": "from_scratch_unified",
        "band": args.band,
        "counts": {"pool": len(pool), "good": n_good, "issue": len(pool) - n_good},
        "rubrics": rubrics,
        "notes": props.get("notes", ""),
        "cost": cost,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[cluster] proposed {len(rubrics)} rubrics (from scratch) -> {args.out}")
    print(f"[cluster] proposed grounding: {Counter(r.get('grounding', '?') for r in rubrics)}")
    print(f"[cluster] proposed categories: {Counter(r.get('category', '?') for r in rubrics)}")


if __name__ == "__main__":
    main()
