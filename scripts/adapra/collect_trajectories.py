"""
Collect trajectories for AdaPRA using DSGym agents.

Supported datasets:
  dspredict-easy / dspredict-hard  — DSPredictReActAgent + Docker executor
  daeval / dabstep / discoverybench / qrdata / dsbio  — ReActDSAgent + Docker executor

Real mode requires the DSGym Docker executor stack running (see docs/adapra.md).
Mock mode works without Docker for any dataset.

Usage:
  # DSPredict (Kaggle competitions)
  python scripts/adapra/collect_trajectories.py \
    --dataset dspredict-easy \
    --agent-config configs/adapra/agent_bedrock.yaml

  # Other datasets
  python scripts/adapra/collect_trajectories.py \
    --dataset daeval \
    --agent-config configs/adapra/agent_bedrock.yaml

  # Mock (no API key or Docker)
  python scripts/adapra/collect_trajectories.py \
    --dataset daeval --mock --task-limit 3
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass


def _import_rubric_module(name: str):
    """Import dsgym.adapra submodule directly, bypassing dsgym/__init__.py."""
    parts = name.split(".")
    path = REPO_ROOT / "dsgym" / "adapra" / Path(*parts)
    file_path = (path / "__init__.py") if (path / "__init__.py").exists() else Path(str(path) + ".py")
    full_name = f"dsgym.adapra.{name}"
    spec = importlib.util.spec_from_file_location(full_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


_import_rubric_module("schema")
_import_rubric_module("rubrics")

from dsgym.adapra.schema import (  # noqa: E402
    TaskInfo, TrajectoryRecord, TrajectoryStep, OutcomeLabels
)

# ---------------------------------------------------------------------------
# Dataset registry — maps dataset name → loader class + kwargs
# ---------------------------------------------------------------------------

# Registry: dataset_name → (module, class, init_kwargs, load_kwargs)
# init_kwargs go to DatasetClass(**init_kwargs), load_kwargs go to .load(**load_kwargs)
DATASET_REGISTRY = {
    "dspredict-easy":  ("dsgym.datasets.loaders.dspredict",     "DSPredictDataset",     {"virtual_data_root": "/data"}, {"split": "easy"}),
    "dspredict-hard":  ("dsgym.datasets.loaders.dspredict",     "DSPredictDataset",     {"virtual_data_root": "/data"}, {"split": "hard"}),
    "daeval":          ("dsgym.datasets.loaders.daeval",         "DAEvalDataset",        {"virtual_data_root": "/data"}, {}),
    "dabstep":         ("dsgym.datasets.loaders.dabstep",        "DABStepDataset",       {"virtual_data_root": "/data"}, {"split": "dev"}),
    "discoverybench":  ("dsgym.datasets.loaders.discovery_bench","DiscoveryBenchDataset",{"virtual_data_root": "/data"}, {"split": "test"}),
    "qrdata":          ("dsgym.datasets.loaders.qrdata",         "QRDataDataset",        {"virtual_data_root": "/data"}, {}),
    "dsbio":           ("dsgym.datasets.loaders.dsbio",          "DSBioDataset",         {"virtual_data_root": "/data"}, {"split": "hard"}),
}


def load_agent_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _sample_to_task_info(sample: dict, dataset: str) -> TaskInfo:
    """Normalize a DSGym sample dict → TaskInfo regardless of dataset."""
    ei = sample.get("extra_info", {})
    is_dspredict = dataset.startswith("dspredict")

    # task_id: prefer explicit id fields, fall back to index
    task_id = (
        ei.get("id") or ei.get("challenge_name") or ei.get("task_id")
        or ei.get("metadata_id") or ei.get("query_id")
        or str(ei.get("index", "unknown"))
    )
    challenge_name = ei.get("challenge_name") or task_id

    # description: either the full prompt content or the question field
    prompt = sample.get("prompt", [])
    description = ""
    for msg in prompt:
        if isinstance(msg, dict) and msg.get("role") == "user":
            description = msg.get("content", "")
            break

    return TaskInfo(
        task_id=str(task_id),
        challenge_name=challenge_name,
        split=ei.get("split", dataset),
        description=description,
        docker_challenge_path=ei.get("docker_challenge_path", ""),
        data_files=(
            list(ei.get("data_files", {}).values())
            if isinstance(ei.get("data_files"), dict)
            else ei.get("data_files", [])
        ),
        source=dataset,
        metadata={k: v for k, v in ei.items()
                  if k not in ("id", "challenge_name", "task_id", "question", "split")},
    )


def _conversation_to_steps(conversation: list[dict], trajectory_list: list[dict] | None) -> list[TrajectoryStep]:
    """Convert DSGym conversation + trajectory list → TrajectoryStep list."""
    # Prefer trajectory list if available (DSPredictReActAgent provides it)
    if trajectory_list:
        steps = []
        for step in trajectory_list:
            content = step.get("content", "")
            role = step.get("role", "user")
            code = "\n\n".join(re.findall(r"<(?:code|python)>(.*?)</(?:code|python)>", content, re.DOTALL)).strip() if role == "assistant" else ""
            obs = "\n\n".join(re.findall(r"<information>(.*?)</information>", content, re.DOTALL)).strip() if role == "user" else ""
            if not obs and role == "user":
                obs = content.strip()
            has_err = bool(re.search(r"(Traceback|Error:|Exception:)", obs, re.IGNORECASE))
            err_type_m = re.search(r"(AttributeError|TypeError|ValueError|KeyError|ImportError|RuntimeError|Error)", obs, re.IGNORECASE) if has_err else None
            steps.append(TrajectoryStep(
                turn=step.get("turn", 0),
                role=role,
                content=content,
                code=code,
                observation=obs,
                done=step.get("done", False),
                step_time=step.get("step_time", 0.0),
                has_error=has_err,
                error_type=err_type_m.group(0) if err_type_m else None,
                reasoning_content=step.get("reasoning_content", ""),
            ))
        return steps

    # Fall back to flat conversation list (ReActDSAgent)
    steps = []
    for i, msg in enumerate(conversation):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        turn = i // 2
        code = "\n\n".join(re.findall(r"<(?:code|python)>(.*?)</(?:code|python)>", content, re.DOTALL)).strip() if role == "assistant" else ""
        obs = "\n\n".join(re.findall(r"<information>(.*?)</information>", content, re.DOTALL)).strip() if role == "user" else ""
        if not obs and role == "user":
            obs = content.strip()
        has_err = bool(re.search(r"(Traceback|Error:|Exception:)", obs, re.IGNORECASE))
        err_type_m = re.search(r"(AttributeError|TypeError|ValueError|KeyError|ImportError|RuntimeError|Error)", obs, re.IGNORECASE) if has_err else None
        steps.append(TrajectoryStep(
            turn=turn,
            role=role,
            content=content,
            code=code,
            observation=obs,
            has_error=has_err,
            error_type=err_type_m.group(0) if err_type_m else None,
        ))
    return steps


def _result_to_record(
    result: dict,
    task_info: TaskInfo,
    model: str,
    backend: str,
    run_id: str,
    run_config: dict,
    elapsed: float,
    dataset: str,
) -> dict:
    """Convert DSGym agent result dict → TrajectoryRecord dict."""
    is_dspredict = dataset.startswith("dspredict")
    steps = _conversation_to_steps(
        result.get("conversation", []),
        result.get("trajectory") if is_dspredict else None,
    )
    num_errors = sum(1 for s in steps if s.has_error)
    submission_path = result.get("solution") if is_dspredict and result.get("success") else None
    final_answer = result.get("solution", "") if not is_dspredict else ""
    if is_dspredict and not final_answer:
        for step in reversed(steps):
            m = re.search(r"<answer>(.*?)</answer>", step.content, re.DOTALL)
            if m:
                final_answer = m.group(1).strip()
                break

    record = TrajectoryRecord(
        trajectory_id=run_id,
        task=task_info,
        agent_model=model,
        agent_backend=f"dsgym_{backend}",
        run_config=run_config,
        steps=steps,
        final_answer=final_answer,
        submission_path=submission_path,
        outcome=OutcomeLabels(
            valid_submission=result.get("success", False) if is_dspredict else None,
            num_execution_errors=num_errors,
            completed=result.get("success", False),
        ),
        num_turns=result.get("turns", len(steps) // 2),
        total_tokens=result.get("metadata", {}).get("total_tokens", 0),
        execution_time=elapsed,
        errors=[result["error"]] if result.get("error") else [],
        metadata=result.get("metadata", {}),
    )
    return record.to_dict()


def _mock_record(task_info: TaskInfo, model: str, run_id: str, dataset: str) -> dict:
    """Minimal synthetic record for smoke-testing without Docker."""
    is_dspredict = dataset.startswith("dspredict")
    steps = [
        TrajectoryStep(turn=0, role="assistant", content="<python>\nimport pandas as pd\nprint('hello')\n</python>",
                       code="import pandas as pd\nprint('hello')"),
        TrajectoryStep(turn=0, role="user", content="hello", observation="hello"),
        TrajectoryStep(turn=1, role="assistant",
                       content="<answer>Mock answer for testing</answer>"),
    ]
    record = TrajectoryRecord(
        trajectory_id=run_id, task=task_info,
        agent_model=model, agent_backend="mock",
        steps=steps, final_answer="Mock answer for testing",
        submission_path="/submission/submission.csv" if is_dspredict else None,
        outcome=OutcomeLabels(
            valid_submission=True if is_dspredict else None,
            num_execution_errors=0, completed=True,
        ),
        num_turns=2, execution_time=1.0,
    )
    return record.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect AdaPRA trajectories for any DSGym dataset")
    parser.add_argument("--dataset", default="dspredict-easy",
                        choices=list(DATASET_REGISTRY.keys()),
                        help="Dataset to collect from")
    parser.add_argument("--split", default=None,
                        help="Override default split (e.g. dev/all for dabstep)")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--agent-config", default="configs/adapra/agent_bedrock.yaml")
    parser.add_argument("--output-dir", default=None,
                        help="Output dir (default: outputs/adapra/trajectories/<dataset>)")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--runs-per-task", type=int, default=1)
    parser.add_argument("--task-timeout", type=int, default=3600,
                        help="Wall-clock timeout per task in seconds (default: 3600). 0 = no limit.")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel task workers (default: 1). Set to number of executor containers.")
    parser.add_argument("--manager-url", default=None)
    parser.add_argument("--submission-dir", default="./submissions")
    parser.add_argument("--skip-errors", action="store_true",
                        help="Also skip tasks that have a .error.json (timeout/crash) from a prior run")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"outputs/adapra/trajectories/{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.agent_config)
    cfg = load_agent_config(str(cfg_path)) if cfg_path.exists() else {}
    if not cfg_path.exists():
        print(f"[warn] Config not found at {cfg_path}, using defaults")

    agent_cfg = cfg.get("agent", {})
    model       = agent_cfg.get("model", "bedrock/qwen.qwen3-coder-next")
    max_turns   = agent_cfg.get("max_turns", 200)
    time_out    = agent_cfg.get("time_out", 600)
    temperature = agent_cfg.get("temperature", 0.0)
    top_p       = agent_cfg.get("top_p")
    max_tokens  = agent_cfg.get("max_tokens")
    timeout     = agent_cfg.get("timeout")
    max_retries = agent_cfg.get("max_retries")
    backend     = agent_cfg.get("backend", "litellm")
    api_key     = agent_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    base_url    = agent_cfg.get("base_url")
    aws_region  = (agent_cfg.get("aws_region_name") or os.environ.get("AWS_REGION_NAME")
                   or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1")
    # Fallback regions rotated on retry when the primary region 503s (shared
    # Bedrock preview-model serving capacity). Defaults to ap-southeast-2, the
    # other region that serves qwen.qwen3-coder-next. Accept a YAML list or a
    # comma-separated string; env override via AWS_REGION_FALLBACKS.
    aws_region_fallbacks = agent_cfg.get("aws_region_fallbacks")
    if aws_region_fallbacks is None:
        _env_fb = os.environ.get("AWS_REGION_FALLBACKS")
        aws_region_fallbacks = _env_fb if _env_fb is not None else ["ap-southeast-2"]
    if isinstance(aws_region_fallbacks, str):
        aws_region_fallbacks = [r.strip() for r in aws_region_fallbacks.split(",") if r.strip()]
    provider    = agent_cfg.get("provider") or os.environ.get("AZURE_LLMAPI_PROVIDER") or os.environ.get("LLMAPI_PROVIDER")
    client_type = agent_cfg.get("client_type")
    max_requests = agent_cfg.get("max_requests")
    auth_mode = agent_cfg.get("auth_mode")
    scenario_guid = (
        agent_cfg.get("scenario_guid")
        or os.environ.get("SCENARIO_GUID_3f")
        or os.environ.get("AZURE_LLMAPI_SCENARIO_GUID")
        or os.environ.get("LLMAPI_SCENARIO_GUID")
    )
    use_managed_identity = agent_cfg.get("use_managed_identity")
    manager_url = args.manager_url or agent_cfg.get("manager_url", "http://localhost:5001")
    submission_dir = args.submission_dir or agent_cfg.get("submission_dir", "./submissions")
    mock        = args.mock or agent_cfg.get("mock", False)

    is_dspredict = args.dataset.startswith("dspredict")

    # Enforce: code cell timeout <= task_timeout / 3
    if args.task_timeout > 0 and time_out > args.task_timeout // 3:
        capped = args.task_timeout // 3
        print(f"[info] Capping code cell timeout: {time_out}s -> {capped}s "
              f"(must be <= task_timeout/3 = {args.task_timeout}//3)")
        time_out = capped

    # Load dataset (skip in mock mode to avoid heavyweight deps like kagglesdk)
    if mock:
        samples = [
            {
                "extra_info": {"id": f"{args.dataset}_mock_{i:03d}", "split": args.dataset},
                "prompt": [{"role": "user", "content": f"Mock task {i} for {args.dataset}"}],
            }
            for i in range(args.task_limit or 3)
        ][args.start_index:]
    else:
        module_path, class_name, init_kwargs, default_load_kwargs = DATASET_REGISTRY[args.dataset]
        mod = importlib.import_module(module_path)
        DatasetClass = getattr(mod, class_name)
        load_kwargs = dict(default_load_kwargs)
        if args.split:
            load_kwargs["split"] = args.split
        load_kwargs["limit"] = args.task_limit
        # qrdata.load() has no start_index parameter
        if args.dataset != "qrdata":
            load_kwargs["start_index"] = args.start_index

        print(f"[info] Loading {args.dataset}...")
        dataset_obj = DatasetClass(**init_kwargs)
        samples = dataset_obj.load(**load_kwargs)
        print(f"[info] Loaded {len(samples)} tasks")

    # Inject time constraint info into system prompts
    cell_timeout_min = time_out // 60
    task_timeout_min = args.task_timeout // 60 if args.task_timeout > 0 else 0
    if task_timeout_min > 0:
        time_constraint_suffix = (
            f"\n\nTime constraints:\n"
            f"- Each code cell has a maximum execution time of {cell_timeout_min} minutes. "
            f"If your code times out, simplify your approach (fewer iterations, smaller grid search, simpler model).\n"
            f"- You have a total wall-clock budget of {task_timeout_min} minutes for the entire task. "
            f"Do not spend excessive time on hyperparameter tuning — a well-chosen default often beats an exhaustive search that times out.\n"
            f"- If you encounter consecutive execution timeouts, immediately switch to a simpler and faster approach "
            f"rather than retrying similar expensive computations."
        )
        for sample in samples:
            prompt = sample.get("prompt", [])
            for msg in prompt:
                if msg.get("role") == "system":
                    msg["content"] += time_constraint_suffix
                    break

    # Agent factory — each worker thread gets its own instance to avoid shared state
    def _make_agent():
        if is_dspredict:
            from dsgym.agents import DSPredictReActAgent as AgentClass
        else:
            from dsgym.agents import ReActDSAgent as AgentClass
        agent_kwargs: dict = {
            "manager_url": manager_url,
            "max_turns": max_turns,
            "temperature": temperature,
            "output_dir": submission_dir,
        }
        if is_dspredict:
            agent_kwargs["submission_dir"] = submission_dir
            agent_kwargs["time_out"] = time_out
        if api_key:
            agent_kwargs["api_key"] = api_key
        if base_url:
            agent_kwargs["base_url"] = base_url
        if aws_region:
            agent_kwargs["aws_region_name"] = aws_region
        if aws_region_fallbacks:
            agent_kwargs["aws_region_fallbacks"] = aws_region_fallbacks
        optional_agent_keys = {
            "top_p": top_p,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "max_retries": max_retries,
            "provider": provider,
            "client_type": client_type,
            "max_requests": max_requests,
            "auth_mode": auth_mode,
            "scenario_guid": scenario_guid,
            "use_managed_identity": use_managed_identity,
        }
        agent_kwargs.update(
            {key: value for key, value in optional_agent_keys.items() if value is not None}
        )
        return AgentClass(backend=backend, model=model, **agent_kwargs)

    run_config = {
        "model": model,
        "backend": backend,
        "max_turns": max_turns,
        "manager_url": manager_url,
        "dataset": args.dataset,
    }
    if provider:
        run_config["provider"] = provider
    if scenario_guid:
        run_config["scenario_guid_configured"] = True
    if max_tokens:
        run_config["max_tokens"] = max_tokens

    # Build work items (sample, run_idx) pairs, skipping already-done ones
    import threading
    safe_model = model.replace("/", "-").replace(":", "-")
    counters = {"collected": 0, "skipped": 0, "errors": 0}
    lock = threading.Lock()

    work_items = []
    for i, sample in enumerate(samples):
        task_info = _sample_to_task_info(sample, args.dataset)
        for run_idx in range(args.runs_per_task):
            run_id = f"{task_info.task_id}__{safe_model}__{run_idx:02d}"
            out_path = output_dir / f"{run_id}.json"
            error_path = output_dir / f"{run_id}.error.json"
            partial_path = output_dir / f"{run_id}.partial.json"
            if out_path.exists() or partial_path.exists() or (args.skip_errors and error_path.exists()):
                with lock:
                    counters["skipped"] += 1
                print(f"[skip] {out_path.name}")
            else:
                work_items.append((i + 1, len(samples), sample, task_info, run_idx, run_id, out_path))

    print(f"[info] {counters['skipped']} already done, {len(work_items)} to collect "
          f"(workers={args.num_workers}, timeout={args.task_timeout}s)")

    def _run_one(item):
        seq, total, sample, task_info, run_idx, run_id, out_path = item
        print(f"\n[{seq}/{total}] {task_info.challenge_name}  (run {run_idx})")
        try:
            if mock:
                record_dict = _mock_record(task_info, model, run_id, args.dataset)
            else:
                agent = _make_agent()
                t0 = time.time()
                if args.task_timeout > 0:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                        fut = ex.submit(agent.solve_task, sample)
                        try:
                            result = fut.result(timeout=args.task_timeout)
                        except concurrent.futures.TimeoutError:
                            elapsed = time.time() - t0
                            print(f"  [timeout] {run_id}  exceeded {args.task_timeout}s")
                            # Try to save partial trajectory from agent state
                            partial_traj = getattr(agent, '_partial_trajectory', [])
                            partial_conv = getattr(agent, '_partial_conversation', [])
                            partial_turns = getattr(agent, '_partial_turns', 0)
                            if partial_traj:
                                partial_result = {
                                    "solution": "",
                                    "success": False,
                                    "turns": partial_turns,
                                    "error": f"task_timeout after {elapsed:.0f}s",
                                    "metadata": {
                                        "model": model,
                                        "backend": backend,
                                        "dspredict": True,
                                        "max_turns": max_turns,
                                        "execution_time": elapsed,
                                        "conversation_length": len(partial_conv),
                                        "status": "timeout",
                                    },
                                    "conversation": partial_conv,
                                    "trajectory": partial_traj,
                                    "raw_result": None,
                                }
                                record_dict = _result_to_record(
                                    result=partial_result, task_info=task_info,
                                    model=model, backend=backend,
                                    run_id=run_id, run_config=run_config,
                                    elapsed=elapsed, dataset=args.dataset,
                                )
                                record_dict["outcome"]["completed"] = False
                                record_dict["status"] = "timeout"
                                partial_path = output_dir / f"{run_id}.partial.json"
                                partial_path.write_text(json.dumps(record_dict, indent=2, default=str))
                                print(f"  [partial] {partial_path.name}  turns={partial_turns}")
                            else:
                                (output_dir / f"{run_id}.error.json").write_text(
                                    json.dumps({"trajectory_id": run_id, "task_id": task_info.task_id,
                                                "error": f"task_timeout after {elapsed:.0f}s",
                                                "status": "timeout"}, indent=2)
                                )
                            with lock:
                                counters["errors"] += 1
                            return
                else:
                    result = agent.solve_task(sample)
                elapsed = time.time() - t0
                record_dict = _result_to_record(
                    result=result, task_info=task_info,
                    model=model, backend=backend,
                    run_id=run_id, run_config=run_config,
                    elapsed=elapsed, dataset=args.dataset,
                )

            out_path.write_text(json.dumps(record_dict, indent=2, default=str))
            turns = record_dict.get("num_turns", "?")
            valid = record_dict.get("outcome", {}).get("valid_submission")
            completed = record_dict.get("outcome", {}).get("completed")
            print(f"  [ok] {out_path.name}  turns={turns}  valid={valid}  completed={completed}")
            with lock:
                counters["collected"] += 1

        except Exception as exc:
            import traceback as tb
            print(f"  [error] {run_id}: {exc}")
            tb.print_exc()
            (output_dir / f"{run_id}.error.json").write_text(
                json.dumps({"trajectory_id": run_id, "task_id": task_info.task_id,
                            "error": str(exc), "status": "failed"}, indent=2)
            )
            with lock:
                counters["errors"] += 1

    if work_items:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            list(pool.map(_run_one, work_items))

    print(f"\n[done] collected={counters['collected']}  skipped={counters['skipped']}  errors={counters['errors']}")


if __name__ == "__main__":
    main()
