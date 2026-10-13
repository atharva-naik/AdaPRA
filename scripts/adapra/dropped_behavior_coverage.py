"""Read-only robustness check: do behaviors dropped by stage-two tail filtering
fit within the 17 discovered rubrics? Judge = Sonnet 4.6 (Bedrock, us-east-1).

Bedrock credentials come from the standard AWS environment variables or ~/.aws.

Usage:
  python scripts/adapra/dropped_behavior_coverage.py [--dry]
"""
import json, random, re, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import litellm

ROOT = Path(__file__).resolve().parents[2] / "outputs/adapra/behavior_sweep/bon5-qwen3-v2-grounded"
OUT = ROOT / "dropped_behavior_coverage.json"
MODEL, REGION = "bedrock/us.anthropic.claude-sonnet-4-6", "us-east-1"
BATCH, WORKERS = 30, 4
DRY = "--dry" in sys.argv

# ---------- reconstruct kept / dropped ----------
kept, dropped = [], []
for line in (ROOT / "behaviors.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    wt, valid = rec.get("within_task_band"), rec.get("valid")
    is_top = bool(valid and wt == "top")
    is_bot = bool((not valid) or wt == "bottom")
    for it in rec.get("items", []):
        kind = it.get("kind")
        keep = (kind == "good_behavior" and is_top) or (kind == "issue" and is_bot)
        row = {"task": rec.get("task"), "attempt": rec.get("attempt"), "band": wt,
               "valid": valid, "kind": kind, "behavior": it.get("behavior", ""),
               "lens": it.get("lens", ""), "grounding": it.get("grounding", ""),
               "evidence": it.get("evidence", [])}
        if keep:
            kept.append(row); continue
        if (kind == "good_behavior" and is_bot) or (kind == "issue" and is_top):
            row["reason_dropped"] = "against_outcome"
        elif wt == "mid":
            row["reason_dropped"] = "middle"
        elif wt == "unscored":
            row["reason_dropped"] = "unscored"
        else:
            row["reason_dropped"] = f"other:{wt}"
        dropped.append(row)

rc = Counter(d["reason_dropped"] for d in dropped)
print(f"kept={len(kept)} dropped={len(dropped)} reasons={dict(rc)}")
assert len(kept) + len(dropped) == 906, len(kept) + len(dropped)
assert len(dropped) == 387, len(dropped)
assert len(kept) == 519, len(kept)
assert rc == {"against_outcome": 112, "middle": 259, "unscored": 16}, rc

# ---------- rubrics ----------
rubrics = [r for r in json.load(open(ROOT / "final_rubric_set.json"))["rubrics"]
           if r.get("keep") is not False and r["rubric_id"] != "workspace_competition_match"]
assert len(rubrics) == 17, len(rubrics)
RIDS = {r["rubric_id"] for r in rubrics}
rubric_txt = "\n".join(f"- {r['rubric_id']} ({r['display_name']}): {r['description']}" for r in rubrics)

PROMPT = """You are auditing coverage of a rubric set for data-science agent trajectories.
Below are 17 process-reward rubrics, then a numbered list of behaviors observed in agent
attempts. Each behavior is tagged GOOD (a positive behavior) or ISSUE (a problem).

For EACH behavior, decide:
- the single best-matching rubric_id, if the behavior is clearly an instance (positive or
  negative) of that rubric, i.e. a grader applying that rubric would reward or penalize it; or
- "none", if no rubric clearly covers it.
Do not stretch: if the match is only loose or tangential, answer "none".

===== RUBRICS =====
{rubrics}

===== BEHAVIORS =====
{items}

Respond with ONLY a JSON object, no prose, of the form:
{{"results": [{{"idx": 1, "rubric_id": "<rubric_id or none>", "reason": "<one line>"}}, ...]}}
with exactly one entry per behavior idx 1..{n}."""


def fmt(batch):
    out = []
    for i, it in enumerate(batch, 1):
        k = "GOOD" if it["kind"] == "good_behavior" else "ISSUE"
        ev = " | ".join(str(e)[:160] for e in it["evidence"][:2])
        out.append(f"{i}. [{k}] ({it['lens']}) {it['behavior']}  [evidence: {ev}]")
    return "\n".join(out)


usage = {"prompt": 0, "completion": 0, "calls": 0}


def call(batch):
    prompt = PROMPT.format(rubrics=rubric_txt, items=fmt(batch), n=len(batch))
    if DRY:
        print(prompt[:3000]); sys.exit(0)
    for attempt in range(6):
        try:
            resp = litellm.completion(model=MODEL, aws_region_name=REGION, temperature=0,
                                      max_tokens=6000,
                                      messages=[{"role": "user", "content": prompt}])
            u = resp.usage
            usage["prompt"] += u.prompt_tokens or 0
            usage["completion"] += u.completion_tokens or 0
            usage["calls"] += 1
            txt = resp.choices[0].message.content
            m = re.search(r"\{.*\}", txt, re.S)
            res = json.loads(m.group(0))["results"]
            by = {int(r["idx"]): r for r in res}
            assert set(by) == set(range(1, len(batch) + 1)), "idx mismatch"
            out = []
            for i, it in enumerate(batch, 1):
                rid = str(by[i]["rubric_id"]).strip()
                if rid not in RIDS and rid != "none":
                    rid = "INVALID:" + rid
                out.append({**it, "judge_rubric": rid, "judge_reason": by[i].get("reason", "")})
            return out
        except Exception as e:
            print(f"  retry {attempt}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            time.sleep(min(2 ** attempt * 5, 60))
    raise RuntimeError("batch failed")


def run(items):
    batches = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
    with ThreadPoolExecutor(WORKERS) as ex:
        res = list(ex.map(call, batches))
    return [x for b in res for x in b]


random.seed(0)
kept_sample = random.sample(kept, 60)
dropped_res = run(dropped)
kept_res = run(kept_sample)


def none_rate(rows):
    return (sum(r["judge_rubric"] == "none" for r in rows) / len(rows)) if rows else float("nan")


def summarize(rows, key):
    out = {}
    for v in sorted({r[key] for r in rows}):
        sub = [r for r in rows if r[key] == v]
        out[v] = {"n": len(sub), "none_rate": round(none_rate(sub), 4)}
    return out


cost = usage["prompt"] / 1e6 * 3 + usage["completion"] / 1e6 * 15
summary = {
    "n_dropped": len(dropped_res), "dropped_none_rate": round(none_rate(dropped_res), 4),
    "by_reason": summarize(dropped_res, "reason_dropped"),
    "by_kind": summarize(dropped_res, "kind"),
    "kept_sample_n": len(kept_res), "kept_sample_none_rate": round(none_rate(kept_res), 4),
    "dropped_top_rubrics": Counter(r["judge_rubric"] for r in dropped_res).most_common(),
    "kept_sample_top_rubrics": Counter(r["judge_rubric"] for r in kept_res).most_common(),
    "invalid_labels": sum(r["judge_rubric"].startswith("INVALID") for r in dropped_res + kept_res),
    "usage": usage, "est_cost_usd_at_3_15": round(cost, 3),
}
OUT.write_text(json.dumps({"summary": summary, "dropped": dropped_res,
                           "kept_sample": kept_res}, indent=1))
print(json.dumps(summary, indent=1))
print("\n=== NONE examples (dropped) ===")
for r in [r for r in dropped_res if r["judge_rubric"] == "none"]:
    print(f"- [{r['kind']}/{r['reason_dropped']}] {r['behavior']}  || {r['judge_reason']}")
