"""
Collect trajectories using the V4 harness (pure OpenHands, Docker exec).

Usage:
  python scripts/adapra/collect_trajectories_v4.py \
    --dataset dspredict-easy \
    --agent-config configs/adapra/harness_v4.yaml \
    --output-dir outputs/adapra/trajectories_v4/dspredict-easy

  # Quick test (1 task, 2 attempts, single worker)
  python scripts/adapra/collect_trajectories_v4.py \
    --dataset dspredict-easy \
    --task-limit 1 --max-attempts 2 --workers 1

  # Full run with PRM feedback
  python scripts/adapra/collect_trajectories_v4.py \
    --dataset dspredict-easy \
    --agent-config configs/adapra/harness_v4.yaml \
    --prm-feedback --workers 4
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    """Import dsgym.adapra submodule directly."""
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
from dsgym.adapra.schema import TaskInfo  # noqa: E402


DATASET_REGISTRY = {
    "dspredict-easy": ("dsgym.datasets.loaders.dspredict", "DSPredictDataset", {"virtual_data_root": "/data"}, {"split": "easy"}),
    "dspredict-hard": ("dsgym.datasets.loaders.dspredict", "DSPredictDataset", {"virtual_data_root": "/data"}, {"split": "hard"}),
}


def load_dataset(dataset_name: str, task_limit: int | None = None):
    """Load dataset and return list of TaskInfo objects."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")

    module_path, class_name, init_kwargs, load_kwargs = DATASET_REGISTRY[dataset_name]
    module = importlib.import_module(module_path)
    dataset_cls = getattr(module, class_name)
    dataset = dataset_cls(**init_kwargs)
    samples = dataset.load(**load_kwargs)

    tasks = []
    for i, sample in enumerate(samples):
        if task_limit and i >= task_limit:
            break

        ei = sample.get("extra_info", {})
        if isinstance(ei, str):
            import ast
            ei = ast.literal_eval(ei)

        task_id = (
            ei.get("id") or ei.get("challenge_name") or ei.get("task_id")
            or f"{dataset_name}_{i}"
        )
        challenge_name = ei.get("challenge_name") or str(task_id)

        prompt = sample.get("prompt", [])
        description = ""
        for msg in prompt:
            if msg.get("role") == "user":
                description = msg.get("content", "")
                break

        task_info = TaskInfo(
            task_id=str(task_id),
            challenge_name=challenge_name,
            split=load_kwargs.get("split", "unknown"),
            description=description,
            docker_challenge_path=ei.get("docker_challenge_path", ""),
            data_files=ei.get("data_files", []),
            source=dataset_name,
            metadata={k: v for k, v in ei.items()
                      if k not in ("id", "challenge_name", "task_id", "docker_challenge_path", "data_files")},
        )
        tasks.append(task_info)

    return tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect trajectories with V4 harness (pure OpenHands, Docker exec)"
    )
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--agent-config", type=str, help="YAML config for agent/harness settings")
    parser.add_argument("--output-dir", type=str, default="outputs/adapra/trajectories_v4")
    parser.add_argument("--task-limit", type=int, default=None, help="Max tasks to process")
    parser.add_argument("--task-id", type=str, default=None,
                        help="Comma-separated task_id(s) to run EXACTLY (filters the "
                             "loaded dataset to these ids; useful for a targeted "
                             "single-task run). Applied before --task-limit.")
    parser.add_argument("--max-attempts", type=int, default=None, help="Attempts per task")
    parser.add_argument("--max-iterations-per-attempt", type=int, default=None)
    parser.add_argument("--command-timeout", type=int, default=None)
    parser.add_argument("--task-timeout", type=int, default=None)
    parser.add_argument("--model", type=str, default=None, help="Override model from config")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.add_argument("--log-completions", type=str, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu", action="store_true", default=False,
                        help="Enable GPU: apptainer runtime adds --nv and pins "
                             "CUDA_VISIBLE_DEVICES to the SLURM-allocated GPUs "
                             "(by UUID). Needed for CV/audio/NLP tasks (e.g. "
                             "DSPredict-Hard); leave off for CPU-only tabular runs.")

    # V4-specific
    parser.add_argument("--backend", type=str, default=None, choices=["docker", "apptainer"],
                        help="Container backend (default: docker, or from config)")
    parser.add_argument("--docker-image", type=str, default=None)
    parser.add_argument("--sif-image", type=str, default=None,
                        help="Path to .sif image (required for --backend apptainer)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Host path to competition data (mounted as /data:ro)")
    parser.add_argument("--workspace-root", type=str, default=None,
                        help="Host root for workspace volumes")
    # Workspace is wiped at run start by DEFAULT (worker_*/ and .submissions/
    # under workspace-root) so stale scripts from a prior run can't leak into a
    # fresh experiment. Within-run cross-attempt memory is unaffected — the wipe
    # fires once before any task starts. Harness infra dirs (.aws_creds,
    # .empty_mask) are preserved. Pass --no-clean-workspace to disable (e.g. an
    # intentional resume that should keep cross-run workspace memory).
    parser.add_argument("--no-clean-workspace", action="store_false",
                        dest="clean_workspace", default=True,
                        help="Disable the default run-start workspace wipe.")
    parser.add_argument("--final-submission-select", action="store_true", default=False,
                        help="After each attempt's loop, run a flag-gated selection "
                             "phase (outside the time budget): one structured LLM call "
                             "picks the best of the CSVs already produced and copies it "
                             "to /submission/submission.csv. No code execution.")

    # PRM feedback
    parser.add_argument("--prm-feedback", action="store_true", default=False)
    parser.add_argument("--prm-judge-model", type=str, default=None)
    parser.add_argument("--prm-rubric-ids", type=str, default=None)
    parser.add_argument("--prm-min-importance", type=float, default=0.5)
    parser.add_argument("--prm-failure-threshold", type=float, default=0.5)
    return parser.parse_args()


# Per-PROCESS stable worker slot. CONTAMINATION FIX: worker_id must NOT be
# derived from the task's submit index (idx % workers). ProcessPoolExecutor does
# NOT pin a slot to a process — when a process frees up it grabs the NEXT queued
# task, so two tasks that happen to share `idx % workers` (e.g. idx 0 and idx 4
# with 4 workers) can run CONCURRENTLY in two different pool processes. Both then
# write the SAME per-worker host tree (`worker_{id}/`, its non-task-keyed
# `events_attempt_0.jsonl`, config, csv_history, mask dir) and stomp each other —
# the observed bug where s5e2 read s4e3's event stream. A per-process slot,
# assigned once at pool-init from a shared counter, guarantees each of the N live
# processes owns a DISTINCT `worker_{slot}/`; two tasks in one process are
# sequential (safe). Single-worker path sets it directly.
_WORKER_SLOT: int | None = None


class _WorkerLinePrefixer:
    """Wrap a stream so every emitted line is prefixed with the worker slot.

    Makes the interleaved multi-worker driver log readable: instead of a single
    undifferentiated stream where you can't tell which of N workers printed what
    (and where a busy worker drowns out the others), every line is tagged
    ``[w{slot}]``. Line-buffered + flushed so output appears promptly even
    without ``python -u`` — the block-buffering that made a healthy run look
    dead in the log.
    """

    def __init__(self, stream, slot):
        self._stream = stream
        self._prefix = f"[w{slot}] "
        self._at_line_start = True

    def write(self, text):
        if not text:
            return
        # Prefix at the start of each line; preserve embedded newlines.
        out = []
        for ch in text:
            if self._at_line_start and ch != "\n":
                out.append(self._prefix)
                self._at_line_start = False
            out.append(ch)
            if ch == "\n":
                self._at_line_start = True
        self._stream.write("".join(out))
        self._stream.flush()

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _init_worker_slot(counter, lock):
    """ProcessPoolExecutor initializer: claim one unique slot for this process."""
    global _WORKER_SLOT
    with lock:
        _WORKER_SLOT = counter.value
        counter.value += 1
    # Tag + line-flush this process's stdout/stderr so the shared driver log is
    # readable per-worker and never block-buffered.
    sys.stdout = _WorkerLinePrefixer(sys.stdout, _WORKER_SLOT)
    sys.stderr = _WorkerLinePrefixer(sys.stderr, _WORKER_SLOT)


def _process_single_task(
    i: int,
    task_info: TaskInfo,
    output_dir: Path,
    total_tasks: int,
    harness_kwargs: dict,
    worker_id: int | None = None,
):
    """Process a single task with its own container.

    worker_id is the per-PROCESS slot (from _init_worker_slot), NOT the task
    index. When None (parallel path), it is read from the process-local
    _WORKER_SLOT so concurrent tasks never share a `worker_{id}/` host tree.
    """
    from dsgym.harness.openhands_harness_v4 import OpenHandsHarnessV4

    if worker_id is None:
        worker_id = _WORKER_SLOT if _WORKER_SLOT is not None else 0

    task_output_dir = output_dir / task_info.task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)

    harness = OpenHandsHarnessV4(**harness_kwargs)

    print(f"\n{'='*60}")
    print(f"[{i+1}/{total_tasks}] Task: {task_info.task_id} (worker {worker_id})")
    print(f"{'='*60}")

    container_name = None
    try:
        container_name = harness.start_container(worker_id, task_info=task_info)
        records = harness.run_task(
            task_info,
            container_name=container_name,
            worker_id=worker_id,
            output_dir=str(task_output_dir),
        )
        return task_info.task_id, True, None

    except Exception as e:
        print(f"   Error on {task_info.task_id}: {e}")
        traceback.print_exc()

        error_path = task_output_dir / "error.json"
        with open(error_path, "w") as f:
            json.dump({"error": str(e), "task_id": task_info.task_id}, f, indent=2)

        return task_info.task_id, False, str(e)

    finally:
        if container_name:
            harness.stop_container(container_name)


def main():
    args = parse_args()

    config = {}
    if args.agent_config:
        with open(args.agent_config) as f:
            config = yaml.safe_load(f) or {}

    model = args.model or config.get("model", "bedrock/qwen.qwen3-coder-next")
    max_attempts = args.max_attempts or config.get("max_attempts", 5)
    max_iterations = args.max_iterations_per_attempt or config.get("max_iterations_per_attempt", 200)
    command_timeout = args.command_timeout or config.get("command_timeout", 600)
    task_timeout = args.task_timeout or config.get("task_timeout", 7200)
    # Hard per-command timeout cap (clamps the agent-supplied action.timeout so a
    # single blocking runaway command can't bypass the no-output guard and burn the
    # whole attempt). Default = task_timeout//3, mirroring the v2 collector.
    max_command_timeout = config.get("max_command_timeout", task_timeout // 3)
    # Prompt-variant flag: which host prompt dir to bind-mount into the container.
    # "prompts_dir_v4" (default) or "prompts_dir_v4_twrap" (timeout-safe variant that
    # steers the agent away from huge bash-call timeouts toward a shell `timeout N`).
    prompts_dir = config.get("prompts_dir", "prompts_dir_v4")
    temperature = config.get("temperature", 0.0)
    docker_image = args.docker_image or config.get("docker_image", "dsgym-v4-runner:latest")
    backend = args.backend or config.get("backend", "docker")
    sif_image = args.sif_image or config.get("sif_image")
    data_path = args.data_path or config.get("data_path", "../data/data")
    workspace_root = args.workspace_root or config.get("workspace_root", "/tmp/dsgym_v4")

    if backend == "apptainer" and not sif_image:
        raise SystemExit("--backend apptainer requires --sif-image (or sif_image in config)")

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Configuration (V4 -- pure OpenHands, {backend}):")
    print(f"   Dataset: {args.dataset}")
    print(f"   Model: {model}")
    print(f"   Backend: {backend}")
    if backend == "apptainer":
        print(f"   SIF image: {sif_image}")
    else:
        print(f"   Docker image: {docker_image}")
    print(f"   Data path: {data_path}")
    print(f"   Workspace root: {workspace_root}")
    print(f"   Max attempts: {max_attempts}")
    print(f"   Max iterations/attempt: {max_iterations}")
    print(f"   Command timeout: {command_timeout}s")
    print(f"   Max command timeout (hard clamp): {max_command_timeout}s")
    print(f"   Task timeout: {task_timeout}s")
    print(f"   Prompt dir: {prompts_dir}")
    print(f"   Workers: {args.workers}")
    print(f"   Output dir: {output_dir}")
    print()

    # When filtering to explicit --task-id(s), load the FULL dataset first (the
    # targets may sit past --task-limit's first-N cut), filter, then apply limit.
    if args.task_id:
        wanted = {t.strip() for t in args.task_id.split(",") if t.strip()}
        tasks = load_dataset(args.dataset, task_limit=None)
        tasks = [t for t in tasks if t.task_id in wanted]
        missing = wanted - {t.task_id for t in tasks}
        if missing:
            raise SystemExit(f"--task-id: no such task(s) in {args.dataset}: {sorted(missing)}")
        if args.task_limit:
            tasks = tasks[:args.task_limit]
        print(f"Filtered to {len(tasks)} task(s) via --task-id: {[t.task_id for t in tasks]}")
    else:
        tasks = load_dataset(args.dataset, task_limit=args.task_limit)
        print(f"Loaded {len(tasks)} tasks from {args.dataset}")

    # Build PRM scorer if enabled
    from dsgym.harness.prm_feedback import PRMScorer

    prm_config = config.get("prm_feedback", {})
    prm_enabled = args.prm_feedback or prm_config.get("enabled", False)
    prm_scorer = None
    if prm_enabled:
        prm_judge_model = (
            args.prm_judge_model
            or prm_config.get("judge_model", "bedrock/us.anthropic.claude-sonnet-4-6-v1:0")
        )
        prm_rubric_ids = (
            args.prm_rubric_ids.split(",") if args.prm_rubric_ids
            else prm_config.get("rubric_ids")
        )
        prm_scorer = PRMScorer(
            judge_model=prm_judge_model,
            rubric_ids=prm_rubric_ids,
            exclude_rubric_ids=prm_config.get("exclude_rubric_ids"),
            min_importance=args.prm_min_importance or prm_config.get("min_importance", 0.5),
            failure_threshold=args.prm_failure_threshold or prm_config.get("failure_threshold", 0.5),
            # Judge region is SEPARATE from the agent's — the judge is usually a
            # different model (e.g. claude-sonnet on Bedrock) served in different
            # regions than the qwen agent. Prefer prm_feedback.aws_region_name,
            # fall back to the top-level region. Same for the fallback list.
            aws_region_name=prm_config.get("aws_region_name") or config.get("aws_region_name"),
            aws_region_fallbacks=prm_config.get("aws_region_fallbacks"),
            workers=prm_config.get("workers", 8),
        )
        print(f"   PRM feedback: ENABLED (judge={prm_judge_model})")

    # GPU gate: apptainer runtime only adds --nv (and pins CUDA to the allocated
    # GPUs) when gpu_id is non-None. Off by default (tabular Easy runs are
    # CPU-only); pass --gpu for CV/audio/NLP-heavy sets like DSPredict-Hard.
    gpu_id = 0 if args.gpu else config.get("gpu_id")

    harness_kwargs = dict(
        model=model,
        backend=backend,
        docker_image=docker_image,
        sif_image=sif_image,
        gpu_id=gpu_id,
        data_path=data_path,
        workspace_root=workspace_root,
        max_attempts=max_attempts,
        max_iterations_per_attempt=max_iterations,
        command_timeout=command_timeout,
        max_command_timeout=max_command_timeout,
        task_timeout=task_timeout,
        prompts_dir=prompts_dir,
        temperature=temperature,
        aws_region_name=config.get("aws_region_name"),
        aws_access_key_id=config.get("aws_access_key_id"),
        aws_secret_access_key=config.get("aws_secret_access_key"),
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
        log_completions=args.log_completions,
        prm_scorer=prm_scorer,
        condenser_max_size=config.get("condenser_max_size", 80),
        final_submission_select=args.final_submission_select
            or config.get("final_submission_select", False),
        # Inline PRM (fires DURING each attempt, writes prm_call_NN.json to the
        # host-visible .harness dir, injects feedback into chat). Config block:
        #   prm_inline: {enabled, every_n_steps, on_new_submission,
        #                inject_feedback, llm_kwargs}
        prm_inline=config.get("prm_inline", {}),
        # Agentic submission REVIEWER (Option B: own :ro-isolated apptainer
        # instance, EXECUTES code-based rubric checks on each new submission,
        # maintains a per-task rubric store, injects a verdict, pauses the base
        # clock). Config block:
        #   prm_reviewer: {enabled, max_turns, inject_feedback, inject_on_pass,
        #                  llm_kwargs, command_timeout, time_budget_seconds}
        prm_reviewer=config.get("prm_reviewer", {}),
        # Bulky CSV artifacts (csv_history, submission CSVs, fs_snapshots) go here
        # (roomy scratch), NOT the home/git output_dir. JSON stays in output_dir.
        csv_artifact_root=config.get("csv_artifact_root"),
        # Resilience against transient Bedrock ServiceUnavailable bursts. Widen the
        # SDK in-call retry window; resume the (state-persisting) conversation after
        # a ConversationRunError instead of losing the attempt.
        llm_num_retries=config.get("llm_num_retries", 10),
        llm_retry_max_wait=config.get("llm_retry_max_wait", 120),
        run_resume_retries=config.get("run_resume_retries", 3),
        run_resume_wait=config.get("run_resume_wait", 30),
    )

    # Optionally wipe stale per-worker workspaces from a prior run so old-prompt
    # scripts can't leak into a fresh experiment. Removes worker_*/, .submissions/,
    # and .reviewer_memory/ (the agentic-reviewer rubric store); harness infra dirs
    # (.aws_creds, .empty_mask) are preserved. SYMMETRY: the reviewer's per-task
    # rubric store must reset run-to-run exactly like the base agent's workspace —
    # both persist cross-attempt WITHIN a run and reset BETWEEN runs. Leaving
    # .reviewer_memory behind would let a prior run's discovered rubrics leak into a
    # fresh experiment (cross-run contamination), so it is wiped here too.
    if args.clean_workspace and workspace_root:
        import shutil
        ws_root = Path(workspace_root)
        removed = 0
        for child in (list(ws_root.glob("worker_*"))
                      + [ws_root / ".submissions", ws_root / ".reviewer_memory"]):
            if child.exists():
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        print(f"   Cleaned workspace root {ws_root} ({removed} dirs removed)")

    # Filter to pending tasks
    pending_tasks = []
    for i, task_info in enumerate(tasks):
        task_output_dir = output_dir / task_info.task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)

        if args.skip_existing:
            existing = list(task_output_dir.glob("attempt_*.json"))
            if len(existing) >= 1:
                print(f"  Skipping {task_info.task_id} ({len(existing)}/{max_attempts} attempts exist)")
                continue

        pending_tasks.append((i, task_info))

    print(f"\n{len(pending_tasks)} tasks to process, {args.workers} workers")

    if args.workers <= 1:
        for idx, (i, task_info) in enumerate(pending_tasks):
            _process_single_task(i, task_info, output_dir, len(tasks), harness_kwargs, worker_id=0)
    else:
        # Each of the args.workers processes claims a UNIQUE, STABLE slot at init
        # (shared counter) so concurrent tasks never collide on a `worker_{id}/`
        # host tree. worker_id is intentionally NOT passed per-task — it is the
        # per-process slot read inside _process_single_task. See _init_worker_slot.
        import multiprocessing as _mp
        _ctx = _mp.get_context()
        _counter = _ctx.Value("i", 0)
        _lock = _ctx.Lock()
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker_slot,
            initargs=(_counter, _lock),
        ) as executor:
            futures = {}
            for idx, (i, task_info) in enumerate(pending_tasks):
                fut = executor.submit(
                    _process_single_task, i, task_info, output_dir, len(tasks),
                    harness_kwargs,
                )
                futures[fut] = task_info
            for future in as_completed(futures):
                task_id, success, error = future.result()
                status = "OK" if success else "FAIL"
                print(f"   [{status}] {task_id}")

    print(f"\nDone! Results in {output_dir}")


if __name__ == "__main__":
    main()
