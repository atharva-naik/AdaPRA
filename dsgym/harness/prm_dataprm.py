"""DataPRM baseline reviewer (reviewer ``impl: dataprm``).

Plugs DataMind's DataPRM (Qiu et al., 2026; github.com/zjunlp/DataMind,
checkpoint zjunlp/DataPRM-ScienceAgentBench) into the SAME reviewer slot the
PRA occupies: same cadence wakes, same finish gate, same injection channel.
Only the supervisor changes, so the row isolates "generic step-level PRM" vs
"rubric PRA".

Faithful to the released evaluation code (dataprm/evaluate):
  * JUDGE_SYS_PROMPT / JUDGE_USR_PROMPT verbatim (incl. the query_image tool
    signature the ScienceAgentBench checkpoint was trained with);
  * one paragraph = one agent step (action + its output), verified in order,
    each with the prior paragraphs AND their previous verifications as history;
  * multi-round <reasoning>/<code> loop, stop on "</code>", code executed with
    the cumulative-successful-code Interpreter semantics, output wrapped in
    <interpreter>, max_rounds=10, max_tokens=8192, obs_max_len=8192, 30 s exec
    timeout, sampling temperature 0.7 / top_p 0.9 / top_k 20;
  * <score> in {1, 0.5, 0} + <summary> parsed exactly as _parse_prediction.

Adaptations required by the online setting (documented, not tuned):
  * the paragraphs verified at a wake are the agent turns since the previous
    wake (the reviewer is woken every ``cadence_turns``), rather than a whole
    finished trajectory;
  * the agent's prior code is NOT re-executed to rebuild state (hours of
    training); the live workspace already holds every file the agent produced,
    and it is exposed to DataPRM's code via symlinks in a writable exec dir,
    with the agent's ``/workspace`` paths remapped to that dir;
  * query_image has no backing vision model here and returns an
    "unavailable" string (DSPredict-Easy is tabular);
  * history paragraphs are clipped (oldest first) to fit the 24K-token
    training context.

Status for the harness gate: FAIL if any new step scored 0, WARN if any 0.5,
else PASS. verdict_text lists the flagged steps with DataPRM's summaries.
"""
from __future__ import annotations

import ast
import gc
import json
import os
import re
import subprocess
import tempfile
import time as _time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Verbatim from dataprm/evaluate/prompt.py
# ---------------------------------------------------------------------------
JUDGE_SYS_PROMPT = """
You are a Data Analysis Expert that can accurately verify the correctness and reasonableness of data analysis steps. The following is the data analysis problem and a solution trajectory (split into paragraphs, enclosed with [Paragraph] tags and indexed from 1):

# Task
Your task is to verify the correctness and reasonableness of the analysis and code in the designated paragraph, and score it according to the following criteria :
- 1 indicates that the paragraph is correct and reasonable. The analysis and code in the paragraph are accurate and helpful for solving the problem.
- 0.5 indicates that the paragraph is generally correct, but with some details omitted or minor errors, which can be easily fixedd in the subsequent steps and do not significantly affect the overall solution.
- 0 indicates that the paragraph contain fatal analysis errors, code errors or has severe omissions, which lead to incorrect or misleading results, and significantly affect the overall solution.

# Verification Format
1. Your verification for each paragraph should be constructed by 2 parts, wrapped by '<reasoning></reasoning>' and '<code></code>' separately. You should first analyze the correctness of the paragraph in '<reasoning></reasoning>' part, then write code to verify your analysis in '<code></code>' part.
2. In '<reasoning></reasoning>' part, you are required to present in detail the key steps of the paragraph or the steps for which you had doubts regarding their correctness, and explicitly analyze whether each step is accurate: for correct steps, explain why you initially doubted their correctness and why they are indeed correct; for erroneous steps, explain the reason for the error and the impact of that error on the solution.
3. In '<code></code>' part, you must write **Python code** in the form of ```python\\n{{CODE}}\\n```. You can use code to interact with the data files and verify the correctness of the paragraph. Make sure to use **print** function to show the results in the code.
4. System will execute the code and return results in <interpreter>...</interpreter>. Every time you get the code execution result, you must conduct reasoning and analyze these results carefully between <reasoning> and </reasoning>, following this process:
    - If the code execution results cannot support you in obtaining a verification outcome, or if an execution error occurs, you must continue the analysis more deeply and write the corresponding code.
    - If you believe that you can already derive a verification outcome, you can first analyze the code execution results, then give the final verification outcome in '<score></score>' part. Note that the verification outcome should be one of {{1, 0.5, 0}}. After giving the score, you need to provide a brief summary in '<summary></summary>' part in one or two sentences, explaining why the paragraph is given that score.

# Tools
You can use the following tool functions to assist with validation. Each tool function has already been implemented, and you can call it using Python code. Remember to use the **print** function to display the results when using these tools.
{tools}

# Important Notes
1. Pay attention that you must follow the format of ```python\\n{{CODE}}\\n``` when you write the code, otherwise the code will not be executed.
2. The code execution directory is consistent with the paragraph code execution directory, and you can access any files generated up to the current paragraph.
3. Whenever you're confident about the verification, you can directly provide your verification inside <score>...</score> and <summary>...</summary>, even if it's just the first step.
4. If the Paragraph Index is 10 (final step), but the model still not generate the answer (enclosed with <Answer>...</Answer>), you should score it 0 directly, because the model has tried too many times but still cannot generate the answer.
5. When reviewing answers, ignore the formatting and focus solely on correctness.
"""

JUDGE_USR_PROMPT = """
# Problem
{problem}

# Problem Files
{file_list}

# Hsitory Trajectory
{paragraph_list}

# Current Paragraph
{current_paragraph}
"""

# Verbatim from dataprm/evaluate/tools/tool_prompt.py
QUERY_IMAGE_PROMPT = """
def query_image(query: str, image_path: str) -> str:
    '''
    Ask a question about the image

    Parameters:
      - query: The question to ask (string)
      - image_path: The path to the image file (supports .png, .jpg, etc.)

    Returns: The query result as a string

    Example:
      result = query_image("Describe the image in detail.", "diagram.png")
    '''
"""

# Stub implementation (no vision model is served for this baseline).
QUERY_IMAGE_STUB = '''
def query_image(query: str, image_path: str) -> str:
    return "query_image is unavailable in this environment."
'''

# Native generation / execution settings (run_eval.py, llm.py, interpreter.py).
MAX_ROUNDS = 10
MAX_TOKENS = 8192
OBS_MAX_LEN = 8192
EXEC_TIMEOUT = 30
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 20

_UNVERIFIED = "Score: -1.0\nSummary: [not verified]"

# Prompt-size caps (chars) so a wake fits the 24K-token training context.
CURRENT_PARA_MAX = 12_000
HISTORY_PARA_MAX = 2_000
HISTORY_TOTAL_MAX = 40_000
PROBLEM_MAX = 12_000


# ---------------------------------------------------------------------------
# Interpreter (port of dataprm/evaluate/interpreter.py)
# ---------------------------------------------------------------------------
class _StatementRemover(ast.NodeTransformer):
    def __init__(self, functions_to_remove):
        self.functions_to_remove = set(functions_to_remove)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id in self.functions_to_remove:
                return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)


def _run_task(code_lines, workspace, timeout, tool_lines, python):
    running_code = ("\n".join(tool_lines) + "\n\n" if tool_lines else "") + "\n".join(code_lines)

    def _sanitize(text, script_path=None):
        if script_path:
            text = text.replace(script_path, "<string>")
        return text.replace(os.path.realpath(workspace), "<WORKSPACE>")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=workspace) as f:
        f.write(running_code)
        path = f.name
    try:
        r = subprocess.run([python, path], cwd=workspace, capture_output=True,
                           text=True, timeout=timeout)
        if r.returncode == 0:
            return True, _sanitize(r.stdout.strip(), path)
        err = "\n".join(l for l in (r.stdout + r.stderr).strip().splitlines()
                        if not l.strip().startswith("ERROR conda.cli.main_run"))
        return False, f"[Error]:\n{_sanitize(err, path)}"
    except subprocess.TimeoutExpired as e:
        return False, f"[Error]:\nCode execution timed out after {timeout} seconds. {e}"
    except Exception as e:
        return False, f"[Error]:\nAn unexpected error occurred during code execution: {_sanitize(str(e))}"
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


class _Interpreter:
    """Cumulative code: successful snippets are kept (prints stripped) and
    re-run as a prefix of the next snippet, as in the released interpreter."""

    def __init__(self, python, path_map=None, timeout_length=EXEC_TIMEOUT):
        self.python = python
        self.path_map = path_map or []
        self.timeout_length = timeout_length
        self.codes: list[str] = []

    @staticmethod
    def _extract(code):
        m = re.search(r"```(?:python)?(.*?)```", code, re.DOTALL)
        return m.group(1).strip() if m else code

    def _remap(self, code):
        for src, dst in self.path_map:
            code = code.replace(src, dst)
        return code

    def run_code(self, code, workspace, tool_functions=None):
        snippet = self._remap(self._extract(code)).split("\n")
        all_code = self.codes.copy() + snippet
        tool_lines = self._extract(tool_functions).split("\n") if tool_functions else None
        try:
            succ, out = _run_task(all_code, workspace, self.timeout_length, tool_lines, self.python)
        except Exception as e:
            succ, out = False, f"[Error]: {e}"
        out = str(out).strip()
        if succ:
            self._update(all_code)
        gc.collect()
        return out

    def _update(self, code):
        self.codes = []
        try:
            tree = _StatementRemover(["print", "query_document"]).visit(ast.parse("\n".join(code)))
            ast.fix_missing_locations(tree)
            self.codes = [c for c in ast.unparse(tree).splitlines() if c.strip()]
        except Exception:
            for c in code:
                if c.strip().startswith("print") or "print(" in c:
                    self.codes.append(re.match(r"^\s*", c).group(0) + "pass")
                else:
                    self.codes.append(c)


# ---------------------------------------------------------------------------
# LLM loop (port of dataprm/evaluate/llm.py LLM.generate)
# ---------------------------------------------------------------------------
def _chat(base_url, api_key, payload, timeout=600):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _generate(messages, *, base_url, api_key, model_name, workspace, interpreter,
              deadline):
    response_message: list[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    rounds = 0
    try:
        for _ in range(MAX_ROUNDS):
            if _time.time() > deadline:
                response_message.append("[Error]: reviewer time budget exhausted")
                break
            rounds += 1
            resp = _chat(base_url, api_key, {
                "model": model_name, "messages": messages,
                "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
                "stop": ["</code>"], "top_p": TOP_P, "top_k": TOP_K,
            })
            choice = resp["choices"][0]
            ans = choice["message"].get("content") or ""
            u = resp.get("usage") or {}
            usage["prompt_tokens"] += u.get("prompt_tokens", 0)
            usage["completion_tokens"] += u.get("completion_tokens", 0)
            if choice.get("finish_reason") == "stop" and "<code>" in ans and "</code>" not in ans:
                ans += "</code>"
            response_message.append(ans)
            m = re.search(r"<code>(.*?)</code>", ans, re.DOTALL)
            if not m or "<score>" in ans:
                messages.append({"role": "assistant", "content": ans})
                break
            out = interpreter.run_code(m.group(1).strip(), workspace,
                                       tool_functions=QUERY_IMAGE_STUB)
            if len(out) > OBS_MAX_LEN:
                h = OBS_MAX_LEN // 2
                out = (f"First {h} characters:\n{out[:h]}\n......\nLast {h} characters:\n"
                       f"{out[-h:]}\n\nThe observation is too long, truncated to the first "
                       f"and last {h} characters. If you need more information, please try "
                       f"to write new code to get the information.")
            out = f"<interpreter>\n{out}\n</interpreter>"
            response_message.append(out)
            messages.append({"role": "assistant", "content": ans})
            messages.append({"role": "user", "content": out})
    except Exception as e:
        response_message.append(f"[Error]: {e}")
    return "\n".join(response_message), rounds, usage


def _parse_prediction(reasoning):
    """Verbatim logic of run_eval._parse_prediction."""
    if "<score>" in reasoning and "</score>" in reasoning:
        raw = reasoning.split("<score>")[-1].split("</score>")[0].strip()
        summary = "No summary found."
        if "<summary>" in reasoning and "</summary>" in reasoning:
            summary = reasoning.split("<summary>")[-1].split("</summary>")[0].strip()
        try:
            score = float(raw)
            if score not in (0.0, 0.5, 1.0):
                score = -1.0
        except Exception:
            score, summary = -1.0, "No score found."
    else:
        score, summary = -1.0, "No score found."
    return score, summary


# ---------------------------------------------------------------------------
# Trajectory → paragraphs
# ---------------------------------------------------------------------------
_TURN_RE = re.compile(r"^\[Turn (\d+)\] ", re.MULTILINE)


def _paragraphs(transcript: str) -> dict[int, str]:
    """{turn: text} from the harness's turn-annotated transcript. Turn 0 is the
    task message; each later turn is one agent step (action + output)."""
    out: dict[int, list[str]] = {}
    starts = [(m.start(), int(m.group(1))) for m in _TURN_RE.finditer(transcript)]
    for i, (s, t) in enumerate(starts):
        e = starts[i + 1][0] if i + 1 < len(starts) else len(transcript)
        out.setdefault(t, []).append(transcript[s:e].strip())
    return {t: "\n".join(v) for t, v in out.items()}


def _clip(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    h = n // 2
    return text[:h] + "\n......[truncated]......\n" + text[-h:]


def _file_list(data_dir: str) -> str:
    try:
        names = sorted(p.name for p in Path(data_dir).iterdir())
    except OSError:
        return "(none)"
    return "\n".join(names) if names else "(none)"


def _prepare_exec_dir(exec_dir: Path, base_ws_dir: str, data_dir: str):
    """Writable dir whose entries symlink the agent workspace and task data, so
    DataPRM's code sees the files the agent produced (link_files analogue)."""
    exec_dir.mkdir(parents=True, exist_ok=True)
    # The agent works in /workspace/<task_id>; mirror that so remapped
    # /workspace/<task_id>/... paths resolve to the live task workspace.
    task_link = exec_dir / Path(data_dir).name
    if not task_link.exists() and not task_link.is_symlink():
        try:
            os.symlink(base_ws_dir, task_link)
        except OSError:
            pass
    for root in (data_dir, base_ws_dir):
        try:
            for p in Path(root).iterdir():
                link = exec_dir / p.name
                if not link.exists() and not link.is_symlink():
                    os.symlink(p, link)
        except OSError:
            pass


def _render(flagged: list[dict]) -> str:
    if not flagged:
        return ""
    lines = [
        "## Process Reward Model Feedback (mid-run)",
        "A step-level verifier checked your recent turns. It scores each step 1 "
        "(correct), 0.5 (minor issues), or 0 (fatal error). Steps it flagged:",
        "",
    ]
    for f in flagged:
        lines.append(f"- Turn {f['turn']} (score {f['score']:g}): {f['summary']}")
    lines += ["", "Address these issues if they affect your solution, then continue."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point (same contract as prm_reviewer.run_submission_reviewer)
# ---------------------------------------------------------------------------
def run_dataprm_reviewer(*, llm_kwargs: dict, base_ws_dir: str, data_dir: str,
                         prm_reviews_dir: str | None, task_desc: str,
                         trajectory_path: str | None, convo_persist_dir: str | None,
                         submission_index: int = 0, attempt_index: int = 0,
                         time_budget_seconds: float = 1200.0, **_ignored) -> dict:
    t0 = _time.time()
    deadline = t0 + max(60.0, float(time_budget_seconds) - 30.0)
    base_url = llm_kwargs.get("base_url") or "http://localhost:19002/v1"
    api_key = llm_kwargs.get("api_key") or "dummy"
    model_name = llm_kwargs.get("model") or "dataprm"

    if not trajectory_path or not Path(trajectory_path).is_file():
        return {"status": "ERROR", "error": "dataprm needs the pull trajectory file"}
    paras = _paragraphs(Path(trajectory_path).read_text(errors="replace"))
    turns = sorted(t for t in paras if t >= 1)

    state_dir = Path(convo_persist_dir or "/reviewer/convos/dataprm")
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "dataprm_state.json"
    try:
        preds: dict[str, str] = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        preds = {}

    exec_dir = state_dir / "exec_ws"
    _prepare_exec_dir(exec_dir, base_ws_dir, data_dir)
    path_map = [("/workspace/", str(exec_dir) + "/"), ("/workspace", str(exec_dir))]
    python = llm_kwargs.get("python") or "python"

    problem = _clip(task_desc or paras.get(0, ""), PROBLEM_MAX)
    file_list = _file_list(data_dir)
    sys_prompt = JUDGE_SYS_PROMPT.format(tools=QUERY_IMAGE_PROMPT.strip())

    new_turns = [t for t in turns if str(t) not in preds]
    steps, flagged = [], []
    usage_tot = {"prompt_tokens": 0, "completion_tokens": 0}
    rounds_tot = 0
    for t in new_turns:
        if _time.time() > deadline:
            break
        prior = [u for u in turns if u < t]
        # History = prior paragraphs + their verifications, newest kept first.
        hist_blocks, used = [], 0
        for k, u in reversed(list(enumerate(prior, start=1))):
            blk = (f"[Paragraph_{k}]\n{_clip(paras[u], HISTORY_PARA_MAX)}\n[/Paragraph_{k}]\n"
                   f"[Verification_{k}]\n{preds.get(str(u), _UNVERIFIED)}\n"
                   f"[/Verification_{k}]\n\n")
            if used + len(blk) > HISTORY_TOTAL_MAX:
                break
            hist_blocks.append(blk)
            used += len(blk)
        idx = len(prior) + 1
        current = f"[Paragraph_{idx}]\n{_clip(paras[t], CURRENT_PARA_MAX)}\n[/Paragraph_{idx}]\n"
        user = JUDGE_USR_PROMPT.format(problem=problem, file_list=file_list,
                                       paragraph_list="".join(reversed(hist_blocks)),
                                       current_paragraph=current)
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user}]
        interp = _Interpreter(python=python, path_map=path_map)
        st = _time.time()
        reasoning, rounds, usage = _generate(
            messages, base_url=base_url, api_key=api_key, model_name=model_name,
            workspace=str(exec_dir), interpreter=interp, deadline=deadline)
        score, summary = _parse_prediction(reasoning)
        preds[str(t)] = f"Score: {score}\nSummary: {summary}"
        rounds_tot += rounds
        for k in usage_tot:
            usage_tot[k] += usage[k]
        steps.append({"turn": t, "score": score, "summary": summary,
                      "rounds": rounds, "elapsed_seconds": round(_time.time() - st, 1),
                      "token_usage": usage, "reasoning": reasoning})
        if score in (0.0, 0.5):
            flagged.append({"turn": t, "score": score, "summary": summary})

    try:
        state_path.write_text(json.dumps(preds))
    except OSError:
        pass

    scored = [s["score"] for s in steps if s["score"] >= 0]
    if any(s == 0.0 for s in scored):
        status = "FAIL"
    elif any(s == 0.5 for s in scored):
        status = "WARN"
    else:
        status = "PASS"
    out = {
        "status": status,
        "verdict_text": _render(flagged),
        "scores": {f"turn_{s['turn']}": {"score": s["score"], "summary": s["summary"]}
                   for s in steps},
        "findings": flagged,
        "forced": False,
        "turns_used": rounds_tot,
        "steps_verified": len(steps),
        "steps_skipped": len(new_turns) - len(steps),
        "wall_seconds": round(_time.time() - t0, 1),
        "submission_index": submission_index,
        "attempt_index": attempt_index,
        "cost": {"token_usage": usage_tot},
    }
    if prm_reviews_dir:
        try:
            d = Path(prm_reviews_dir)
            d.mkdir(parents=True, exist_ok=True)
            n = len(list(d.glob("reviewer_call_*.json")))
            (d / f"reviewer_call_{n:02d}.json").write_text(json.dumps(out, indent=2))
            (d / f"reviewer_trajectory_{n:02d}.json").write_text(json.dumps(
                {"role": "dataprm", "attempt_index": attempt_index,
                 "submission_index": submission_index, "steps": steps}, indent=2))
        except OSError:
            pass
    return out
