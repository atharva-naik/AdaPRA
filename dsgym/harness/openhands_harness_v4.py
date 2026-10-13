"""
OpenHands-based harness v4: pure OpenHands in Docker containers.

Same logic as V3 (multi-attempt, inject-context preamble, PRM feedback
between attempts, forced summary turn, TrajectoryRecord output) but
implemented on pure OpenHands infrastructure:
- No DSGym executor manager
- No Jupyter kernel
- TerminalTool (tmux) + FileEditorTool (native filesystem)
- Host reads submission directly from Docker volume mount

Each worker owns a long-lived Docker container. Per attempt, the host
writes an attempt_config.json to the shared volume and runs agent_runner.py
inside the container via docker exec.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from dsgym.adapra.schema import (
    OutcomeLabels,
    TaskInfo,
    TrajectoryRecord,
    TrajectoryStep,
)
from .event_converter_v4 import events_jsonl_to_trajectory_record
from .prm_feedback import PRMScorer


AGENT_RUNNER_PATH = "/app/agent_runner.py"
HARNESS_DIR = ".harness"


def _events_has_conversation_error(events_path: Path) -> bool:
    """True if the events JSONL contains a ConversationErrorEvent.

    Signature of a crashed attempt: the SDK emits ConversationErrorEvent when a
    run fails (e.g. the Bedrock BadRequestError "missing field `messages`"). Used
    by the phantom-crash guard to distinguish a real 0-turn attempt from one that
    died before taking any action. Cheap line scan; missing file → False.
    """
    try:
        with open(events_path) as f:
            for line in f:
                if "ConversationErrorEvent" in line:
                    return True
    except OSError:
        return False
    return False


# Host-side source locations for the runner + prompts, bind-mounted into the
# container at /app so the image doesn't need rebuilding when they change.
_HARNESS_SRC_DIR = Path(__file__).resolve().parent
_AGENT_RUNNER_SRC = _HARNESS_SRC_DIR / "agent_runner.py"
_PROMPTS_SRC = _HARNESS_SRC_DIR / "prompts_dir_v4"
# The dsgym package root — mounted read-only at /app/dsgym (with /app on
# PYTHONPATH) so the in-container agent_runner can import the inline-PRM code
# (dsgym.harness.prm_agent / prm_feedback / adapra.rubrics). Mounted OUTSIDE
# /workspace so the agent never sees it, and :ro so it can't be modified.
_DSGYM_PKG_SRC = _HARNESS_SRC_DIR.parent
_DSGYM_PKG_PATH = "/app/dsgym"

SUMMARY_REQUEST_PROMPT = """\
Your attempt is now complete. Please provide a detailed summary of this attempt. \
Use the finish tool with the following format:

## Validation Score
<your best CV/validation score as a single number with the metric name, e.g. "0.1239 (RMSE with log-transformed target)">

## Approach
<bullet points describing your full pipeline: data loading, preprocessing, feature engineering, model choice, hyperparameters, evaluation method>

## What Worked
<bullet points of specific things that improved the score — be concrete with numbers where possible>

## What Didn't Work
<bullet points of things you tried that didn't help or made things worse>

## Suggestions for Next Attempt
<bullet points of what the next attempt should try differently>

Be specific and quantitative. Include model names, hyperparameter values, and score comparisons where relevant."""


class OpenHandsHarnessV4:
    """
    V4 harness: pure OpenHands inside Docker containers via docker exec.

    Same logic as V3 (multi-attempt, inject-context, PRM feedback, forced
    summary turn), but execution is pure OpenHands with standard tools.
    The host orchestrates; the runner inside the container just executes.
    """

    def __init__(
        self,
        model: str,
        docker_image: str = "dsgym-v4-runner:latest",
        data_path: str = "../data/data",
        workspace_root: str = "/tmp/dsgym_v4",
        max_attempts: int = 5,
        max_iterations_per_attempt: int = 200,
        command_timeout: int = 600,
        max_command_timeout: int = 3600,
        task_timeout: int = 7200,
        prompts_dir: str = "prompts_dir_v4",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        condenser_max_size: int = 80,
        log_completions: Optional[str] = None,
        prm_scorer: Optional[PRMScorer] = None,
        memory_limit: str = "24g",
        cpus: int = 8,
        gpu_id: Optional[int] = None,
        backend: str = "docker",
        sif_image: Optional[str] = None,
        final_submission_select: bool = False,
        prm_inline: Optional[dict] = None,
        prm_reviewer: Optional[dict] = None,
        csv_artifact_root: Optional[str] = None,
        llm_num_retries: int = 10,
        llm_retry_max_wait: int = 120,
        run_resume_retries: int = 3,
        run_resume_wait: int = 30,
    ):
        self.model = model
        self.docker_image = docker_image
        # Bulky CSV artifacts (per-turn csv_history, submission CSVs, fs_snapshots)
        # go HERE (a roomy scratch dir) instead of output_dir, which is on the
        # small home quota + git-tracked. Only JSON (attempt_*.json,
        # prm_calls_*, error.json) stays in output_dir. Defaults to a scratch
        # sibling of workspace_root. See _artifact_dir().
        self.csv_artifact_root = (
            Path(csv_artifact_root) if csv_artifact_root
            else Path(str(workspace_root).rstrip("/") + "_artifacts"))
        if backend not in ("docker", "apptainer"):
            raise ValueError(f"backend must be 'docker' or 'apptainer', got {backend!r}")
        self.backend = backend
        self.final_submission_select = final_submission_select
        # Inline-PRM config dict passed through to agent_runner (or None/{} off):
        #   {enabled, every_n_steps, on_new_submission, inject_feedback, llm_kwargs}
        self.prm_inline = prm_inline or {}
        # Agentic submission REVIEWER (flag-gated, Option B — own apptainer
        # instance with the base workspace :ro). Dict passed through to the base
        # agent_runner (enable + cadence) AND consumed host-side to launch the
        # reviewer instance + run the watcher:
        #   {enabled, max_turns, inject_feedback, inject_on_pass, llm_kwargs, ...}
        self.prm_reviewer = prm_reviewer or {}
        # Image reference used for the active backend: a Docker tag for docker,
        # a path to a .sif file for apptainer.
        if backend == "apptainer":
            if not sif_image:
                raise ValueError("backend='apptainer' requires sif_image (path to .sif)")
            self.container_image = str(Path(sif_image).resolve())
        else:
            self.container_image = docker_image
        self.sif_image = sif_image
        self.data_path = str(Path(data_path).resolve())
        self.workspace_root = Path(workspace_root)
        # Stable per-RUN tag derived from the (absolute) workspace_root. Node-local
        # scratch that is keyed ONLY by worker_id — the /tmp mask dirs (base +
        # reviewer) and the apptainer instance names — is SHARED across every run
        # on the same node, because each run's ProcessPoolExecutor independently
        # hands out the SAME slots worker_0..N. Two concurrent same-node runs then
        # collide on those paths: run B's `instance start` rmtree's the shared
        # /tmp/dsgym_v4_mask/worker_0 that run A's LIVE instance depends on, so
        # run A's per-task /data bind-point stub is destroyed and /data mounts
        # EMPTY (observed: a B run corrupted the main run's s3e15/s3e19/s3e22).
        # Namespacing these by a workspace_root-derived tag makes them disjoint
        # per run, so parallel same-node runs (different workspace_root) can't
        # stomp each other. Same run reusing the same root across restarts reuses
        # the same tag (fine — the mask dir is wiped fresh each start regardless).
        import hashlib as _hashlib
        self._run_tag = _hashlib.md5(
            str(self.workspace_root.resolve()).encode()).hexdigest()[:8]
        self.max_attempts = max_attempts
        self.max_iterations_per_attempt = max_iterations_per_attempt
        self.command_timeout = command_timeout
        self.max_command_timeout = max_command_timeout
        self.task_timeout = task_timeout
        # Which host prompt dir gets bind-mounted to /app/prompts. Default is the
        # standard v4 prompt; "prompts_dir_v4_twrap" is the timeout-safe variant
        # (steers the agent away from huge bash-call timeouts toward a shell
        # `timeout N` wrapper — see the per-command clamp / churn analysis).
        self._prompts_src = _HARNESS_SRC_DIR / prompts_dir
        if not self._prompts_src.is_dir():
            raise ValueError(f"prompts_dir not found: {self._prompts_src}")
        self.temperature = temperature
        self.condenser_max_size = condenser_max_size
        self.log_completions = log_completions
        self.prm_scorer = prm_scorer
        self.memory_limit = memory_limit
        self.cpus = cpus
        self.gpu_id = gpu_id

        # Resume-on-error knobs (threaded through the attempt config to the
        # base runner). One transient Bedrock ServiceUnavailable burst that
        # exhausts the SDK's in-call retries otherwise raises ConversationRunError
        # and PERMANENTLY ends the attempt — even one that already wrote a valid
        # submission (observed: qwen endpoint saturated by a co-tenant run, base
        # attempt died with final_answer=None). llm_num_retries/retry_max_wait
        # widen the in-call retry window; run_resume_* let the runner re-run() the
        # (state-persisting) conversation to resume mid-work after such a raise.
        self.run_resume_retries = run_resume_retries
        self.run_resume_wait = run_resume_wait
        self.llm_kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "api_key": api_key,
            "base_url": base_url,
            "aws_region_name": aws_region_name,
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "native_tool_calling": False,
            "disable_stop_word": True,
            "max_message_chars": 30000,
            "num_retries": llm_num_retries,
            "retry_max_wait": llm_retry_max_wait,
        }
        if log_completions:
            self.llm_kwargs["log_completions"] = True
            self.llm_kwargs["log_completions_folder"] = log_completions
        self.llm_kwargs = {k: v for k, v in self.llm_kwargs.items() if v is not None}

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    def _detect_foreign_challenge(self, record, task_id: str) -> Optional[str]:
        """Backstop for the per-worker events-file collision. Scan the agent-
        delivered user messages (the converted event stream, i.e. NOT the step-0
        task_message the harness itself prepends) for a "CHALLENGE NAME: <id>"
        header. If any names a challenge OTHER than this task, return that
        foreign id; else None. A collided read (s5e2 getting s4e3's stream) then
        surfaces as a hard signal instead of a silently mis-scored trajectory."""
        import re as _re
        # NOTE: the char class MUST include underscore — challenge ids like
        # `understanding_cloud_organization` contain `_`, and omitting it made the
        # match stop at the first underscore ("understanding"), which then never
        # equals the full task_id and falsely tripped the guard, discarding EVERY
        # attempt of that task (0/5). Keep `_` in the class.
        pat = _re.compile(r"CHALLENGE NAME:\s*([A-Za-z0-9_\-]+)", _re.IGNORECASE)
        for step in record.steps:
            if getattr(step, "role", None) != "user":
                continue
            m = pat.search(step.content or "")
            if m and m.group(1) != task_id:
                return m.group(1)
        return None

    def _submission_host_dir(self, worker_id: int, task_id: str) -> Path:
        """Host dir backing the /submission bind, kept OUTSIDE the workspace
        tree so per-task `rm -rf /workspace/{task_id}` can't dangle the mount."""
        return self.workspace_root / ".submissions" / f"worker_{worker_id}" / task_id

    def _reviewer_memory_host_dir(self, worker_id: int, task_id: str) -> Path:
        """Host dir backing the reviewer's writable /reviewer memory. Kept
        OUTSIDE the workspace tree (like .submissions) so per-task workspace
        wipes can't delete it, and per (worker, task) so the rubric store —
        which persists across all attempts of a task — never bleeds between
        different competitions run on the same worker."""
        return self.workspace_root / ".reviewer_memory" / f"worker_{worker_id}" / task_id

    def _instance_name(self, worker_id: int, task_info: Optional[TaskInfo]) -> str:
        """UNIQUE-per-task instance/container name.

        Historically this was just `dsgym-v4-worker-{id}`, reused across every
        task a worker ran. That triggered the instance-start RACE: when a worker
        finished a task and started the next, the previous instance's teardown
        (systemd cgroup removal) could still be in progress, so `instance start`
        collided with the still-registered name -> FATAL "already exists" (86
        hits in one run, ~8 tasks lost). Appending a short task-derived suffix
        makes each task's instance name distinct, so a lagging teardown of the
        PREVIOUS instance can never collide with the NEXT start. Falls back to
        the bare worker name when no task_info (legacy).

        Also namespaced by self._run_tag (workspace_root-derived): apptainer
        instance names are node/user-GLOBAL, and two concurrent same-node runs
        both assign slots worker_0..N. With disjoint task sets the task-hash
        suffix already keeps names distinct, but if the two runs ever touch the
        SAME task_id the names would collide — and one run's _ensure_instance_gone
        would tear down the OTHER run's live instance. The run tag makes the two
        runs' instance namespaces disjoint regardless of task-set overlap."""
        base = f"dsgym-v4-worker-{self._run_tag}-{worker_id}"
        if task_info is None:
            return base
        import hashlib
        h = hashlib.md5(task_info.task_id.encode()).hexdigest()[:8]
        return f"{base}-{h}"

    def _instance_exists(self, instance_name: str) -> bool:
        r = subprocess.run(["apptainer", "instance", "list"],
                           capture_output=True, text=True, timeout=30)
        # match the name as a whole word (avoid worker-1 matching worker-10)
        for line in r.stdout.splitlines():
            if line.split()[:1] == [instance_name]:
                return True
        return False

    def _instance_healthy(self, container_name: str) -> bool:
        """Real liveness probe: exec a trivial command INSIDE the instance.

        `instance list` showing the name is NOT sufficient — an OOM (code -9)
        can kill the instance's processes and destroy its cgroup while the name
        lingers, so a following `exec` fails with "couldn't create cgroup
        manager" / "no instance found". We actually run `true` and check it
        succeeds; only then is the instance usable for the next attempt.
        """
        if self.backend != "apptainer":
            # Docker: treat "container is running" as healthy.
            r = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
                capture_output=True, text=True, timeout=15,
            )
            return r.returncode == 0 and r.stdout.strip() == "true"
        try:
            r = subprocess.run(
                self._exec_prefix(container_name) + ["true"],
                capture_output=True, text=True, timeout=30,
                env=self._apptainer_env(),
            )
            return r.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def _ensure_instance_running(
        self, worker_id: int, task_info: Optional[TaskInfo], container_name: str
    ) -> None:
        """Guarantee a healthy instance before an attempt; restart if it died.

        The per-worker instance is started once and reused across all attempts.
        If an attempt OOMs (code -9) it can take the instance down, and every
        subsequent attempt would FATAL ("no instance found") — cascading the
        whole task to failure. Here we synchronously health-check and, if the
        instance is dead, tear it down cleanly and start a fresh one (blocking
        until it's usable).

        We deliberately do NOT re-run _setup_task_workspace here: that would
        `rm -rf /workspace/{task_id}` and destroy the scripts, cached models,
        cv_scores.json, and README the agent built in earlier attempts — the
        whole point of the persistent-workspace-across-attempts design. The
        workspace is a HOST bind mount, so it survives the instance death
        untouched; restarting just re-mounts the same host dir. The task dirs
        already exist from the first start_container, so no setup is needed.
        """
        if self._instance_healthy(container_name):
            return
        print(f"  ⚠️ instance {container_name} unhealthy — restarting "
              f"(workspace preserved on host mount)")
        if self.backend == "apptainer":
            self._ensure_instance_gone(container_name)
        else:
            self.stop_container(container_name)
        # start_container rebuilds all binds and blocks until start returns 0.
        # The workspace bind points at the same host dir, so prior-attempt state
        # is intact — we intentionally skip the destructive workspace wipe.
        self.start_container(worker_id, task_info=task_info)

    def _ensure_instance_gone(self, instance_name: str, timeout_s: float = 60.0) -> None:
        """Stop an instance and BLOCK until it's actually gone.

        `apptainer instance stop` is async — it returns before teardown, so a
        following `instance start` can hit "instance already exists". We issue
        the stop and poll `instance list` until the name disappears, escalating
        to a signal-based stop, so the subsequent start is safe.
        """
        if not self._instance_exists(instance_name):
            return
        # First a graceful stop, then escalate.
        for attempt, extra in enumerate(([], ["-s", "SIGKILL"], ["-F"])):
            subprocess.run(["apptainer", "instance", "stop"] + extra + [instance_name],
                           capture_output=True, timeout=60)
            deadline = time.time() + timeout_s / 3
            while time.time() < deadline:
                if not self._instance_exists(instance_name):
                    return
                time.sleep(2)
        # Final check; if still present the start will surface the error.
        if self._instance_exists(instance_name):
            print(f"  ⚠️ instance {instance_name} still present after stop attempts")

    def start_container(self, worker_id: int, task_info: Optional[TaskInfo] = None) -> str:
        """Start a long-lived container for a worker (docker or apptainer).

        If ``task_info`` is given, ONLY that task's data folder is mounted (at
        the path its description references, ``/data/<source>/<task_id>``),
        instead of the whole dataset dir. This prevents cross-task data leakage
        — otherwise ``ls /data`` exposes every competition's data, which can
        confuse or contaminate weaker agents.

        The task's submission dir is also bound to ``/submission`` so the
        canonical ``/submission/submission.csv`` path (used by the DSPredict
        instructions and the V4 system prompt) resolves. The host reads the
        submission back from the same dir under the workspace mount.
        """
        worker_workspace = self.workspace_root / f"worker_{worker_id}"
        worker_workspace.mkdir(parents=True, exist_ok=True)
        data_bind = self._task_data_bind(task_info)
        # Pre-create the submission dir so it can be bound to /submission.
        # It lives OUTSIDE the workspace tree (a sibling under .submissions/)
        # so that `rm -rf /workspace/{task_id}` during per-task setup cannot
        # delete the dir backing the /submission bind — deleting it would make
        # the mount dangle (st_nlink=0, unwritable) and the agent could not save.
        submission_bind = None
        if task_info is not None:
            host_sub_dir = self._submission_host_dir(worker_id, task_info.task_id)
            host_sub_dir.mkdir(parents=True, exist_ok=True)
            submission_bind = (str(host_sub_dir), "/submission")
        # Unique-per-task instance name (kills the "already exists" start race).
        instance_name = self._instance_name(worker_id, task_info)
        if self.backend == "apptainer":
            return self._start_apptainer(worker_id, worker_workspace, data_bind,
                                         submission_bind, instance_name)
        return self._start_docker(worker_id, worker_workspace, data_bind,
                                  submission_bind, instance_name)

    def _task_data_bind(self, task_info: Optional[TaskInfo]) -> tuple[str, str]:
        """Return (host_src, container_dest) for the data bind.

        Per-task when task_info is provided (only that task's folder, at the
        description's path); otherwise the whole dataset dir at /data (legacy).

        The whole-/data fallback is DANGEROUS: it exposes every competition's
        data to the agent, which can then read and accidentally solve a DIFFERENT
        task (observed: s4e12 read s3e26). It should only happen for a legacy
        no-task-info call — never silently for a real task whose dir is missing.
        We warn loudly if the fallback triggers with a task_info set.
        """
        if task_info is not None:
            task_src = Path(self.data_path) / task_info.task_id
            if task_src.is_dir():
                dest = f"/data/{task_info.source}/{task_info.task_id}"
                return str(task_src), dest
            print(f"  🚩 DATA BIND FALLBACK: task {task_info.task_id!r} dir not "
                  f"found at {task_src} — mounting WHOLE /data (cross-task leak "
                  f"risk!). Check data_path/task_id.")
        return self.data_path, "/data"

    def _start_docker(self, worker_id: int, worker_workspace: Path,
                      data_bind: tuple[str, str],
                      submission_bind: Optional[tuple[str, str]] = None,
                      container_name: Optional[str] = None) -> str:
        # BASE-WORKER docker start (works today). NOTE for the docker port:
        # the dynamic REVIEWER has no equivalent
        # here — add a `_start_reviewer_docker` MODELED ON THIS METHOD but with
        # the reviewer's extra mounts (base workspace ro at /base_ws, reviewer
        # memory rw at /reviewer, submission ro at /submission). Reuse this
        # method's mount recipe verbatim (data ro, agent_runner ro, prompts ro,
        # dsgym pkg ro, ~/.aws, PYTHONPATH=/app) — the reviewer runs the SAME
        # image and is woken by `docker exec ... agent_runner.py --reviewer-config`.
        # Do this as an ADDITIVE new method; do not fold reviewer-specific logic
        # into this base-worker path.
        container_name = container_name or f"dsgym-v4-worker-{worker_id}"

        # Remove any stale container with the same name before starting.
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True, timeout=10,
        )

        data_src, data_dest = data_bind
        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "-v", f"{data_src}:{data_dest}:ro",
            "-v", f"{worker_workspace}:/workspace",
            "-v", f"{_AGENT_RUNNER_SRC}:{AGENT_RUNNER_PATH}:ro",
            "-v", f"{self._prompts_src}:/app/prompts:ro",
            # dsgym package (read-only) for inline-PRM imports; /app on PYTHONPATH.
            "-v", f"{_DSGYM_PKG_SRC}:{_DSGYM_PKG_PATH}:ro",
            "-e", "PYTHONPATH=/app",
            "--memory", self.memory_limit,
            "--cpus", str(self.cpus),
        ]
        if submission_bind is not None:
            cmd += ["-v", f"{submission_bind[0]}:{submission_bind[1]}"]

        # AWS credentials (read-only, non-standard path)
        aws_dir = Path.home() / ".aws"
        if aws_dir.exists():
            cmd += [
                "-v", f"{aws_dir}:/etc/aws:ro",
                "-e", "AWS_SHARED_CREDENTIALS_FILE=/etc/aws/credentials",
                "-e", "AWS_CONFIG_FILE=/etc/aws/config",
            ]
        if self.llm_kwargs.get("aws_region_name"):
            cmd += ["-e", f"AWS_REGION={self.llm_kwargs['aws_region_name']}"]

        # GPU
        if self.gpu_id is not None:
            cmd += ["--gpus", f"device={self.gpu_id}"]

        cmd += [self.docker_image]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start container {container_name}: {result.stderr}"
            )

        print(f"  Started container: {container_name}")
        return container_name

    def _start_apptainer(self, worker_id: int, worker_workspace: Path,
                         data_bind: tuple[str, str],
                         submission_bind: Optional[tuple[str, str]] = None,
                         instance_name: Optional[str] = None) -> str:
        """Start a long-lived Apptainer instance for a worker.

        Uses instance start (analogous to `docker run -d`). The workspace and
        data are bind-mounted; AWS creds are bound read-only at a non-standard
        path.

        IMPORTANT — filesystem isolation: `--containall` alone does NOT isolate
        the agent on clusters where `/home` is an automount and `/scratch` is a
        system-wide `bind path` in apptainer.conf; both leak
        into the container regardless of `--containall`/`--no-mount`. Without
        masking, LLM-generated code could read the user's source, ~/.aws
        credentials, and other users' /scratch — and even write to the repo.
        We therefore bind an empty directory over /home and /scratch to mask
        them. AWS creds are re-exposed only at the non-standard /etc/aws path.

        /data is masked the SAME way. The SIF bakes in shared /data/...
        directories as mountpoints, and those are host autofs NFS paths that
        propagate into the container under --containall — so the WHOLE /data tree
        (every competition, all users' data) was visible regardless of our
        per-task bind. Since our dspredict data lives at
        /data/.../dspredict-easy/<all comps>, the agent could `ls` a sibling
        competition and accidentally solve the wrong task (the s4e12->s3e26 leak).
        We bind an empty dir over /data FIRST, then bind ONLY this task's dir on
        top at /data/<source>/<task_id>, so the container sees just that one task.
        """
        instance_name = instance_name or f"dsgym-v4-worker-{worker_id}"

        # Ensure any stale instance with this name is FULLY gone before starting.
        # `apptainer instance stop` returns BEFORE teardown completes, so a naive
        # stop-then-start races and start fails with "instance already exists"
        # (observed ~37 lost tasks). Stop, then poll until the name disappears.
        self._ensure_instance_gone(instance_name)

        # Empty directory used to mask sensitive host paths (/home, /scratch, /data).
        # MUST be NODE-LOCAL (/tmp), not under workspace_root: workspace_root is on
        # /scratch (NFS/autofs), and an empty mask dir on NFS does NOT form a clean
        # opaque overlay on /data — the subsequent per-task /data bind punches back
        # through it and re-exposes the whole autofs /data tree (all competitions).
        # A node-local mask overlays cleanly. Per-WORKER so concurrent workers don't
        # race, and wiped FRESH each start: apptainer creates the per-task bind-point
        # stub (/data/<source>/<task_id>) INSIDE the mask dir on the host, so reusing
        # it would accumulate prior tasks' stubs and re-leak sibling task names.
        # Also per-RUN (self._run_tag): two concurrent same-node runs both use
        # worker_0..N, so a bare worker-keyed /tmp path is SHARED between them and
        # run B's fresh-wipe rmtree destroys run A's live per-task bind stub → run A
        # /data mounts empty. The run tag keeps the two runs' mask dirs disjoint.
        mask_dir = Path("/tmp") / "dsgym_v4_mask" / self._run_tag / f"worker_{worker_id}"
        if mask_dir.exists():
            shutil.rmtree(mask_dir, ignore_errors=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        data_src, data_dest = data_bind
        # Read-only root (NO --writable-tmpfs): the agent can ONLY write to the
        # host bind mounts (/workspace, /submission). Any other write — /scratch,
        # /, a hallucinated path — fails LOUDLY with a read-only/permission error
        # instead of silently vanishing into a tmpfs overlay (the bug that lost
        # s3e11's submission to /scratch). This also guarantees every CSV the
        # agent produces is on a host-visible mount, so nothing is unfindable.
        # /home and /scratch are masked read-only so agents can't write there.
        cmd = [
            "apptainer", "instance", "start",
            "--containall",
            # Mask leaky host paths read-only before adding our own binds. /data
            # is masked too (SIF-baked autofs mountpoints leak the whole tree),
            # then the per-task data bind below re-exposes ONLY this task's dir.
            "--bind", f"{mask_dir}:/home:ro",
            "--bind", f"{mask_dir}:/scratch:ro",
            "--bind", f"{mask_dir}:/data:ro",
            "--bind", f"{data_src}:{data_dest}:ro",
            "--bind", f"{worker_workspace}:/workspace",
            "--bind", f"{_AGENT_RUNNER_SRC}:{AGENT_RUNNER_PATH}:ro",
            "--bind", f"{self._prompts_src}:/app/prompts:ro",
            # dsgym package (read-only) so agent_runner can import the inline-PRM
            # code; /app is on PYTHONPATH (see _apptainer_env). Outside /workspace.
            "--bind", f"{_DSGYM_PKG_SRC}:{_DSGYM_PKG_PATH}:ro",
        ]
        if submission_bind is not None:
            cmd += ["--bind", f"{submission_bind[0]}:{submission_bind[1]}"]

        # AWS credentials at the non-standard /etc/aws path. Bind from a staged
        # copy OUTSIDE $HOME: binding directly from ~/.aws makes apptainer
        # resolve the source under /home, which triggers the autofs mount of the
        # real home inside the container. Staging under workspace_root avoids
        # that (it doesn't fully hide /home — autofs still leaks on access, see
        # the home-access red flag — but it stops OUR bind from being the cause).
        staged_aws = self._stage_aws_creds()
        if staged_aws is not None:
            cmd += ["--bind", f"{staged_aws}:/etc/aws:ro"]

        if self.gpu_id is not None:
            cmd += ["--nv"]

        cmd += [self.container_image, instance_name]

        env = self._apptainer_env()
        # Run from a neutral cwd: the launching process's cwd is typically under
        # /home, which is masked inside the container and would trigger a
        # harmless-but-noisy chdir warning.
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, env=env, cwd="/tmp"
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start apptainer instance {instance_name}: {result.stderr}"
            )

        print(f"  Started apptainer instance: {instance_name}")
        return instance_name

    def _reviewer_instance_name(self, worker_id: int, task_info: TaskInfo) -> str:
        """Reviewer instance name — the base name with a `-rev` suffix so it
        never collides with the base instance in `apptainer instance list`."""
        return self._instance_name(worker_id, task_info) + "-rev"

    def _start_reviewer_apptainer(
        self, worker_id: int, task_info: TaskInfo, host_task_ws: Path,
        data_bind: tuple[str, str], reviewer_mem: Path, instance_name: str,
        submission_host_dir: Path,
    ) -> str:
        """Start the reviewer's OWN apptainer instance (Option B).

        Mounts the base agent's task workspace READ-ONLY at /base_ws (kernel-level
        isolation — the reviewer cannot modify the repo it audits), the per-task
        data READ-ONLY at /data/<source>/<task_id> (same bind the base agent
        sees), and the reviewer's writable memory at /reviewer. The rubric store
        + check scripts + conversation persistence + reviewer_call logs all live
        under /reviewer, which persists across all attempts of THIS task.

        Sibling of _start_apptainer; reuses the same masking + creds + PYTHONPATH
        plumbing. A distinct per-worker mask dir avoids racing the base instance.

        DOCKER PORT : this method is the template for
        the missing `_start_reviewer_docker`. Copy the MOUNT SEMANTICS below —
        /base_ws:ro, /reviewer (rw), /submission:ro, /data/<src>/<task>:ro,
        agent_runner + prompts + dsgym pkg ro, ~/.aws — into the docker-run form
        used by `_start_docker`. DROP the /home //scratch //data empty-dir masks
        (they are the apptainer-only autofs mitigation; docker isolates mounts by
        default). Command should keep the container alive (`sleep infinity`) since
        the wake is a separate `docker exec`. Keep THIS apptainer method untouched.
        """
        self._ensure_instance_gone(instance_name)

        # Per-RUN (self._run_tag) as well as per-worker: without the tag two
        # concurrent same-node runs share /tmp/dsgym_v4_mask_rev/worker_0 and the
        # reviewer-instance start of one run rmtree's the other's live reviewer
        # bind stub — the SAME collision as the base mask dir, one layer down.
        mask_dir = Path("/tmp") / "dsgym_v4_mask_rev" / self._run_tag / f"worker_{worker_id}"
        if mask_dir.exists():
            shutil.rmtree(mask_dir, ignore_errors=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        reviewer_mem.mkdir(parents=True, exist_ok=True)
        data_src, data_dest = data_bind

        cmd = [
            "apptainer", "instance", "start",
            "--containall",
            "--bind", f"{mask_dir}:/home:ro",
            "--bind", f"{mask_dir}:/scratch:ro",
            "--bind", f"{mask_dir}:/data:ro",
            "--bind", f"{data_src}:{data_dest}:ro",
            # The base repo, READ-ONLY. This is the whole point of Option B.
            "--bind", f"{host_task_ws}:/base_ws:ro",
            # The reviewer's OWN writable memory (rubric store, checks, convos).
            "--bind", f"{reviewer_mem}:/reviewer",
            # The submission under review, READ-ONLY at the canonical path.
            "--bind", f"{submission_host_dir}:/submission:ro",
            "--bind", f"{_AGENT_RUNNER_SRC}:{AGENT_RUNNER_PATH}:ro",
            "--bind", f"{self._prompts_src}:/app/prompts:ro",
            "--bind", f"{_DSGYM_PKG_SRC}:{_DSGYM_PKG_PATH}:ro",
        ]
        staged_aws = self._stage_aws_creds()
        if staged_aws is not None:
            cmd += ["--bind", f"{staged_aws}:/etc/aws:ro"]
        if self.gpu_id is not None:
            cmd += ["--nv"]
        cmd += [self.container_image, instance_name]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            env=self._apptainer_env(), cwd="/tmp",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start reviewer instance {instance_name}: {result.stderr}")
        print(f"  Started reviewer apptainer instance: {instance_name}")
        return instance_name

    def _start_reviewer_docker(
        self, worker_id: int, task_info: TaskInfo, host_task_ws: Path,
        data_bind: tuple[str, str], reviewer_mem: Path, instance_name: str,
        submission_host_dir: Path,
    ) -> str:
        """Start the reviewer's OWN Docker container (Option B, Docker backend).

        Docker twin of _start_reviewer_apptainer. Mounts the base agent's task
        workspace READ-ONLY at /base_ws, the reviewer's writable memory at
        /reviewer, and the submission READ-ONLY at /submission — plus the same
        data, agent_runner, prompts, dsgym-pkg, and AWS-creds binds used by
        _start_docker. No /home /scratch /data empty-dir masks needed (those are
        an Apptainer-only autofs mitigation; Docker isolates mounts by default).
        The container runs `sleep infinity` and is woken per-wake by
        `docker exec ... agent_runner.py --reviewer-config`.
        """
        # Remove any stale container with the same name before starting.
        subprocess.run(
            ["docker", "rm", "-f", instance_name],
            capture_output=True, timeout=10,
        )

        reviewer_mem.mkdir(parents=True, exist_ok=True)
        data_src, data_dest = data_bind

        cmd = [
            "docker", "run", "-d",
            "--name", instance_name,
            "-v", f"{data_src}:{data_dest}:ro",
            # Base agent's workspace, read-only (the whole point of Option B).
            "-v", f"{host_task_ws}:/base_ws:ro",
            # Reviewer's own writable memory (rubric store, checks, convos).
            "-v", f"{reviewer_mem}:/reviewer",
            # Submission under review, read-only.
            "-v", f"{submission_host_dir}:/submission:ro",
            "-v", f"{_AGENT_RUNNER_SRC}:{AGENT_RUNNER_PATH}:ro",
            "-v", f"{self._prompts_src}:/app/prompts:ro",
            "-v", f"{_DSGYM_PKG_SRC}:{_DSGYM_PKG_PATH}:ro",
            "-e", "PYTHONPATH=/app",
            "--memory", self.memory_limit,
            "--cpus", str(self.cpus),
        ]

        # AWS credentials (read-only, non-standard path — same as _start_docker).
        aws_dir = Path.home() / ".aws"
        if aws_dir.exists():
            cmd += [
                "-v", f"{aws_dir}:/etc/aws:ro",
                "-e", "AWS_SHARED_CREDENTIALS_FILE=/etc/aws/credentials",
                "-e", "AWS_CONFIG_FILE=/etc/aws/config",
            ]
        if self.llm_kwargs.get("aws_region_name"):
            cmd += ["-e", f"AWS_REGION={self.llm_kwargs['aws_region_name']}"]

        if self.gpu_id is not None:
            cmd += ["--gpus", f"device={self.gpu_id}"]

        cmd += [self.docker_image, "sleep", "infinity"]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start reviewer container {instance_name}: {result.stderr}"
            )
        print(f"  Started reviewer container: {instance_name}")
        return instance_name

    def _run_reviewer_watcher(self, *, stop_event, worker_id: int,
                              task_info: TaskInfo, host_task_ws: Path,
                              data_bind: tuple[str, str], reviewer_mem: Path,
                              submission_host_dir: Path, requests_dir: Path,
                              attempt_index: int, reviewer_add=None):
        """Host-side watcher (runs in a thread while the base agent attempt runs).

        Crosses the process/mount boundary of Option B: the base agent, inside
        its OWN instance, writes review-request sentinels into `requests_dir`
        (visible on the host via the workspace mount) and block-polls for a
        result file there. This watcher picks up each request, runs ONE reviewer
        wake in the reviewer's OWN apptainer instance (base workspace :ro), and
        writes the result back where the base agent is polling.

        Fully defensive: any failure on a single request writes an ERROR result
        so the base agent's poll is released, and the watcher keeps going.
        """
        rev_cfg = self.prm_reviewer or {}
        rev_llm_kwargs = rev_cfg.get("llm_kwargs") or self.llm_kwargs
        rev_name = self._reviewer_instance_name(worker_id, task_info)
        _data_dest = data_bind[1]  # /data/<source>/<task_id> inside the instance
        seen: set[str] = set()
        rev_started = {"ok": False}

        def _handle(req_path: Path):
            try:
                req = json.loads(req_path.read_text())
            except (OSError, json.JSONDecodeError):
                return
            turn = req.get("turn")
            # Where the base agent polls (its own workspace-mount path). We write
            # the result to the SAME host file it's watching.
            host_result = requests_dir / f"review_result_{turn}.json"

            def _write_result(obj):
                try:
                    tmp = host_result.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(obj, indent=2))
                    tmp.replace(host_result)
                except OSError:
                    pass

            # Defense-in-depth cross-task guard. The requests_dir is now keyed by
            # task_id so a stale sentinel from a prior task can no longer land
            # here — but if one ever did (path drift, manual copy), the reviewer
            # would be fed the WRONG competition's task_desc while every path
            # points at the CURRENT task, producing exactly the corrupted-mix
            # verdicts that motivated the per-task keying. Every DSPredict task
            # description starts with "**CHALLENGE NAME: <name>**"; if the
            # sentinel's embedded name doesn't match this task, reject it (release
            # the base agent's poll with a harmless ERROR, inject nothing).
            _td = req.get("task_desc", "") or ""
            _want = (task_info.challenge_name or "").strip()
            if _want and "CHALLENGE NAME:" in _td:
                _got = _td.split("CHALLENGE NAME:", 1)[1]
                _got = _got.split("*", 1)[0].split("\n", 1)[0].strip()
                if _got and _got != _want:
                    print(f"  [reviewer-watcher] REJECTED stale sentinel "
                          f"{req_path.name}: task_desc='{_got}' != "
                          f"current='{_want}'")
                    _write_result({"status": "ERROR",
                                   "error": f"task mismatch: sentinel={_got!r} "
                                            f"current={_want!r}"})
                    return

            try:
                # Lazily start the reviewer instance on the first request of the
                # attempt (avoids paying the start cost when no submission fires).
                if not rev_started["ok"]:
                    if self.backend == "apptainer":
                        self._start_reviewer_apptainer(
                            worker_id, task_info, host_task_ws, data_bind,
                            reviewer_mem, rev_name, submission_host_dir)
                    else:
                        self._start_reviewer_docker(
                            worker_id, task_info, host_task_ws, data_bind,
                            reviewer_mem, rev_name, submission_host_dir)
                    rev_started["ok"] = True

                # History-delivery axis (paper §sec:axes): the trajectory τ is
                # ALWAYS made available to the reviewer. "pull" = the base agent
                # wrote the FULL transcript to a host-visible file next to its
                # sentinel (req["traj_file"], a basename in requests_dir); copy it
                # into the reviewer's writable memory so it appears read-only at
                # /reviewer/<name>, and hand the reviewer that container path.
                # "push" = the delta text rides inline in req["pushed_transcript"].
                rev_history = req.get("history", "pull")
                trajectory_path = None
                traj_file = req.get("traj_file")
                if traj_file:
                    try:
                        src = requests_dir / traj_file
                        if src.is_file():
                            # Attempt-INDEXED destination so snapshots ACCUMULATE
                            # across attempts in the persistent reviewer memory.
                            # traj_file itself is turn-keyed (review_traj_<turn>.txt)
                            # and would collide across attempts (both turn counters
                            # start at 0); rename to traj_a{attempt}_t{turn}.txt.
                            _dest = f"traj_a{attempt_index}_t{turn}.txt"
                            (reviewer_mem / _dest).write_text(src.read_text())
                            trajectory_path = f"/reviewer/{_dest}"
                    except OSError as e:
                        print(f"  [reviewer-watcher] traj copy failed: {e}")

                # Best-of-N cross-attempt context: the latest transcript snapshot
                # from each PRIOR attempt already sits in the persistent reviewer
                # memory (accumulated by the attempt-indexed copies above). Collect
                # the newest per prior attempt so the reviewer can reason over the
                # whole beam (diversity / over-optimization / plateau rubrics).
                prior_trajectory_paths: list[str] = []
                try:
                    _latest: dict[int, tuple[int, str]] = {}
                    for _p in reviewer_mem.glob("traj_a*_t*.txt"):
                        _m = re.match(r"traj_a(\d+)_t(\d+)\.txt$", _p.name)
                        if not _m:
                            continue
                        _ai, _ti = int(_m.group(1)), int(_m.group(2))
                        if _ai >= attempt_index:
                            continue  # current attempt is trajectory_path
                        if _ai not in _latest or _ti > _latest[_ai][0]:
                            _latest[_ai] = (_ti, _p.name)
                    prior_trajectory_paths = [
                        f"/reviewer/{_latest[_ai][1]}" for _ai in sorted(_latest)
                    ]
                except OSError as e:
                    print(f"  [reviewer-watcher] prior-traj scan failed: {e}")

                # The reviewer-config the in-instance agent_runner --reviewer-config
                # reads. All paths are the reviewer INSTANCE's container paths.
                reviewer_config = {
                    # PRA implementation: "adapra" (default) or "dataprm".
                    # Selected in agent_runner._run_reviewer_mode.
                    "impl": rev_cfg.get("impl", "adapra"),
                    "llm_kwargs": rev_llm_kwargs,
                    "history": rev_history,
                    "trajectory_path": trajectory_path,
                    "pushed_transcript": req.get("pushed_transcript"),
                    # Best-of-N cross-attempt context: N (max_attempts) + the
                    # accumulated prior-attempt transcript paths in the reviewer's
                    # persistent memory (see prm_reviewer._best_of_n_block).
                    "prior_trajectory_paths": prior_trajectory_paths,
                    "n_attempts": self.max_attempts,
                    "base_ws_dir": "/base_ws",
                    "submission_path": "/submission/submission.csv",
                    "data_dir": _data_dest,
                    "rubric_store_dir": "/reviewer/rubric_store",
                    "prm_reviews_dir": f"/reviewer/reviews_attempt_{attempt_index}",
                    "task_desc": req.get("task_desc", ""),
                    "submission_index": req.get("submission_index", 0),
                    "attempt_index": attempt_index,
                    "command_timeout": rev_cfg.get("command_timeout", self.command_timeout),
                    "max_iterations": req.get("max_iterations",
                                              rev_cfg.get("max_turns", 10)),
                    "time_budget_seconds": rev_cfg.get("time_budget_seconds", 1200.0),
                    # Per-attempt message-stream continuity (create-or-resume).
                    "convo_persist_dir": f"/reviewer/convos/{req.get('convo_id','a')}",
                    "convo_id": req.get("convo_id", f"attempt{attempt_index}"),
                    "home_dir": "/reviewer/.home",
                    # The in-instance runner writes here; we copy it back after.
                    "result_path": f"/reviewer/_result_turn_{turn}.json",
                    # Unified-PRM axes (paper §sec:axes). rubric_set: launch-selected
                    # rubric JSON (path or symbolic name, e.g. "discovered") that
                    # seeds the reviewer's store instead of the built-in 9 seeds.
                    # rubric_set_exclude_ids is a launch-fixed subset filter for
                    # controlled rubric-content ablations.
                    # evolution/operationalization toggle the store-mutation tools
                    # and the code-execution tool respectively (see
                    # prm_reviewer._run_reviewer_conversation). Defaults preserve the
                    # legacy dynamic+evolving reviewer.
                    "rubric_set": rev_cfg.get("rubric_set"),
                    "rubric_set_exclude_ids": rev_cfg.get("rubric_set_exclude_ids"),
                    "evolution": rev_cfg.get("evolution", "evolving"),
                    "operationalization": rev_cfg.get("operationalization", "dynamic"),
                    # Unified scoring thresholds (importance-gate + p_success cut);
                    # None → prm_reviewer module defaults (0.5 / 0.6).
                    "min_importance": rev_cfg.get("min_importance"),
                    "failure_threshold": rev_cfg.get("failure_threshold"),
                }
                # Write the reviewer-config to the reviewer's writable memory
                # (host side = reviewer_mem, container side = /reviewer).
                cfg_name = f"_review_request_turn_{turn}.json"
                (reviewer_mem / cfg_name).write_text(json.dumps(reviewer_config))
                host_rev_result = reviewer_mem / f"_result_turn_{turn}.json"
                if host_rev_result.exists():
                    host_rev_result.unlink()

                # Exec the reviewer wake inside its instance. Its own budget caps
                # runtime; we add a host watchdog on top. CRITICAL: the watchdog
                # MUST be SHORTER than the base agent's block-poll
                # (poll_timeout_seconds, default 1800s) — otherwise a wedged-but-
                # alive reviewer outlives the base agent's patience: the base
                # times out and resumes WITHOUT the verdict while this exec keeps
                # eating node resources for the remainder of its budget (the
                # "timeout inversion" bug). So cap the watchdog at
                # min(time_budget, poll_timeout - 120) so it always fires first.
                exec_cmd = self._exec_prefix(rev_name) + [
                    "python", "-u", AGENT_RUNNER_PATH,
                    "--reviewer-config", f"/reviewer/{cfg_name}",
                ]
                _budget = float(rev_cfg.get("time_budget_seconds", 1200.0))
                _base_poll = float(rev_cfg.get("poll_timeout_seconds", 1800.0))
                wd = min(_budget, max(60.0, _base_poll - 120.0))
                _rev_t0 = time.monotonic()
                try:
                    proc = subprocess.run(
                        exec_cmd, capture_output=True, text=True, timeout=wd,
                        env=self._apptainer_env(),
                    )
                finally:
                    # Credit the base agent's clock: this reviewer wake ran on the
                    # host watcher thread while the base agent block-polled, so its
                    # wall-clock must NOT count against the agent's task_timeout.
                    # Runs even on TimeoutExpired (the reviewer still burned that
                    # wall-clock) before the exception propagates to the handler.
                    if reviewer_add is not None:
                        reviewer_add(time.monotonic() - _rev_t0)
                # Persist the reviewer exec's stdout/stderr to a per-wake log so a
                # genuine reviewer crash/OOM/503 is DIAGNOSABLE after the fact
                # (previously capture_output discarded both streams, making every
                # "the reviewer died" post-mortem pure guesswork).
                try:
                    (reviewer_mem / f"_exec_turn_{turn}.log").write_text(
                        f"# reviewer exec turn {turn} rc={proc.returncode} "
                        f"watchdog={wd:.0f}s\n"
                        f"===== STDOUT =====\n{proc.stdout or ''}\n"
                        f"===== STDERR =====\n{proc.stderr or ''}\n")
                except OSError:
                    pass
                # Copy the reviewer's result back to where the base agent polls.
                if host_rev_result.exists():
                    _write_result(json.loads(host_rev_result.read_text()))
                else:
                    _write_result({"status": "ERROR",
                                   "error": "reviewer produced no result file"})
            except Exception as e:  # never leave the base agent block-polling
                _write_result({"status": "ERROR", "error": str(e)})

        try:
            while not stop_event.is_set():
                try:
                    reqs = sorted(requests_dir.glob("review_request_*.json")) \
                        if requests_dir.is_dir() else []
                except OSError:
                    reqs = []
                for rp in reqs:
                    key = rp.name
                    if key in seen:
                        continue
                    seen.add(key)
                    _handle(rp)
                stop_event.wait(1.0)
        finally:
            if rev_started["ok"]:
                if self.backend == "apptainer":
                    self._ensure_instance_gone(rev_name)
                else:
                    self.stop_container(rev_name)

    def _stage_aws_creds(self) -> Optional[str]:
        """Copy ~/.aws into workspace_root/.aws_creds so it can be bound without
        resolving a path under $HOME (which triggers the autofs home mount).
        Returns the staged dir path, or None if no ~/.aws exists."""
        aws_dir = Path.home() / ".aws"
        if not aws_dir.exists():
            return None
        staged = self.workspace_root / ".aws_creds"
        try:
            staged.mkdir(parents=True, exist_ok=True)
            for name in ("credentials", "config"):
                src = aws_dir / name
                if src.exists():
                    dst = staged / name
                    dst.write_bytes(src.read_bytes())
                    os.chmod(dst, 0o600)
        except OSError:
            return None
        return str(staged)

    def _apptainer_env(self) -> dict:
        """Environment for apptainer exec so the containerized process picks up
        AWS credentials at the non-standard mount path. APPTAINERENV_* vars are
        forwarded into the instance by apptainer."""
        env = os.environ.copy()
        aws_dir = Path.home() / ".aws"
        if aws_dir.exists():
            env["APPTAINERENV_AWS_SHARED_CREDENTIALS_FILE"] = "/etc/aws/credentials"
            env["APPTAINERENV_AWS_CONFIG_FILE"] = "/etc/aws/config"
        region = self.llm_kwargs.get("aws_region_name")
        if region:
            env["APPTAINERENV_AWS_REGION"] = region
        # Put /app on the container's PYTHONPATH so agent_runner can import the
        # dsgym package bind-mounted at /app/dsgym (inline PRM). /app also holds
        # agent_runner.py itself.
        env["APPTAINERENV_PYTHONPATH"] = "/app"
        # GPU pinning (only when a gpu_id is requested; --nv is added to the
        # instance-start cmd in the same condition). --containall ESCAPES the
        # SLURM device cgroup, so without an explicit pin the container sees ALL
        # physical GPUs on the node, and the host CUDA_VISIBLE_DEVICES can leak
        # in via --nv and point at another job's GPUs. nvidia-smi run here (in
        # the SLURM shell) RESPECTS the cgroup and returns exactly our allocated
        # GPUs; forward their UUIDs via the higher-precedence APPTAINERENV_ prefix
        # and strip the host var so it cannot override. Mirrors the stock
        # executor launch_instances.sh pin.
        if self.gpu_id is not None:
            env.pop("CUDA_VISIBLE_DEVICES", None)
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=15,
                )
                uuids = [u.strip() for u in out.stdout.splitlines() if u.strip()]
            except Exception:
                uuids = []
            if uuids:
                env["APPTAINERENV_CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
        return env

    def stop_container(self, container_name: str):
        """Stop and remove a container / instance, waiting until it's gone.

        For apptainer we block until teardown completes (stop is async), so the
        next task reusing this worker name can start cleanly.
        """
        if self.backend == "apptainer":
            self._ensure_instance_gone(container_name)
        else:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True, timeout=30,
            )

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    def run_task(
        self,
        task_info: TaskInfo,
        container_name: str,
        worker_id: int,
        output_dir: Optional[str] = None,
    ) -> list[TrajectoryRecord]:
        """Run multi-attempt loop for a single task."""
        host_workspace = self.workspace_root / f"worker_{worker_id}"
        task_id = task_info.task_id
        # Host dir backing /submission (sibling of the workspace tree).
        submission_host_dir = self._submission_host_dir(worker_id, task_id)

        # Setup task workspace inside container
        self._setup_task_workspace(container_name, task_id)

        all_records: list[TrajectoryRecord] = []
        attempt_summaries: list[str] = []
        prm_feedbacks: list[Optional[str]] = []
        best_score: Optional[float] = None

        task_workspace = host_workspace / task_id
        # Directory for on-disk filesystem snapshots taken at attempt boundaries
        # (before attempt_0 = fresh; before attempt_i = state carried over from
        # attempt_{i-1}; and a final snapshot after the last attempt).
        fs_snap_dir = host_workspace / HARNESS_DIR / "fs_snapshots"
        fs_snap_dir.mkdir(parents=True, exist_ok=True)

        for attempt_i in range(self.max_attempts):
            # Guarantee a healthy instance before the attempt. If a prior attempt
            # OOM'd (code -9) it may have killed the instance; without this, every
            # remaining attempt would FATAL ("no instance found") and the task
            # would cascade to failure with phantom stale submissions. This
            # blocks until the instance is usable again (restarting if needed).
            self._ensure_instance_running(worker_id, task_info, container_name)

            # Snapshot the workspace as it stands BEFORE this attempt (fresh for
            # attempt 0; the carried-over state from the prior attempt otherwise).
            self._save_fs_snapshot(
                task_workspace, fs_snap_dir / f"snapshot_before_attempt_{attempt_i}.json"
            )
            # Snapshot what /data exposes INSIDE the container (paths+sizes only),
            # to catch cross-task data-mount leaks the workspace snapshot can't
            # see (e.g. s4e12 reading s3e26's data). Flags >1 task dir visible.
            self._snapshot_data_mount(
                container_name, fs_snap_dir / f"data_mount_attempt_{attempt_i}.json"
            )

            # Integrity check: ONLY this task's data must be visible under
            # /data/<source>/, AND this task's own data must be present. If a
            # foreign task leaked in (s4e12 -> s3e26) OR this task's own data
            # went empty/missing (reviewer-v6 house-prices silent corruption),
            # restart the instance once to redo the per-task bind; if still
            # broken, abort the task rather than collect a trajectory that solves
            # the WRONG competition or trains on NO data. The abort fires before
            # the agent runs, so no garbage-but-"valid" record is written.
            if not self._assert_task_isolation(worker_id, task_info, container_name):
                print(f"   Aborting {task_id} after attempt {attempt_i}: "
                      f"task-data mount could not be restored (wrong/empty data).")
                break

            # Delete previous submission (workspace files persist). The
            # submission dir is the /submission bind's backing store, outside
            # the workspace tree.
            sub_path = submission_host_dir / "submission.csv"
            if sub_path.exists():
                sub_path.unlink()

            attempt_id = (
                f"{task_info.task_id}__{self.model.split('/')[-1]}"
                f"__{uuid.uuid4().hex[:4]}"
            )
            print(f"\n{'='*60}")
            print(f"  Attempt {attempt_i + 1}/{self.max_attempts} -- {attempt_id}")
            print(f"{'='*60}")

            attempt_start = time.time()

            # Build preamble
            preamble = self._build_inject_preamble(
                attempt_index=attempt_i,
                attempt_summaries=attempt_summaries,
                prm_feedbacks=prm_feedbacks,
                best_score=best_score,
            )

            # Compose task message (preamble + task description)
            task_message = task_info.description
            if preamble:
                task_message = preamble + "\n\n---\n\n" + task_message

            # Reviewer-condition ONLY: a neutral artifact-hygiene nudge so the base
            # agent persists its out-of-fold predictions. The reviewer needs OOF on
            # disk to recompute the honest CV / run submission-vs-OOF consistency /
            # residual diagnostics; no run currently saves it. Phrased as generic
            # hygiene (NOT "be careful about your CV") so it does not bias modeling
            # behavior — this keeps the already-complete no-PRM / static-PRM
            # baselines comparable WITHOUT a rerun (the nudge is absent there).
            if bool((self.prm_reviewer or {}).get("enabled", False)):
                task_message = task_message + (
                    "\n\n---\n\n"
                    "ARTIFACT HYGIENE: whenever you run cross-validation, also save "
                    "your out-of-fold (OOF) predictions to `oof_predictions.csv` in "
                    "your working directory (one row per training example: the id "
                    "column plus your OOF prediction, matching the target column). "
                    "Overwrite it with the OOF predictions of your final chosen model."
                )

            # Write attempt config to volume
            config = self._build_attempt_config(
                task_id=task_id,
                attempt_index=attempt_i,
                task_message=task_message,
            )
            harness_dir = host_workspace / HARNESS_DIR
            harness_dir.mkdir(parents=True, exist_ok=True)
            config_path = harness_dir / f"attempt_{attempt_i}_config.json"
            config_path.write_text(json.dumps(config, indent=2))

            # Clear THIS attempt-index's stale per-worker output artifacts before
            # the agent runs. host_workspace/.harness is per-WORKER and REUSED
            # across every task a worker takes, and these files are keyed only by
            # attempt index (NOT by task_id). If an attempt is force-killed before
            # writing fresh output (e.g. the 2h task_timeout SIGKILLs s5e2/s5e6
            # every attempt), the reader below would otherwise pick up the PRIOR
            # task's events_attempt_{i}.jsonl -> _detect_foreign_challenge fires
            # and the attempt is discarded as "contamination" (s5e2 inherited
            # s4e8's event stream). csv_history_attempt_{i} is likewise not
            # task-keyed, so a stale checkpoint could be salvaged as the wrong
            # task's submission by the timeout recovery path. Wiping them here
            # makes a killed attempt read EMPTY events (num_turns=0, no foreign
            # challenge) and fall through to its own csv-history recovery instead.
            for _stale in (
                harness_dir / f"events_attempt_{attempt_i}.jsonl",
                harness_dir / f"error_attempt_{attempt_i}.json",
                harness_dir / f"meta_attempt_{attempt_i}.json",
                harness_dir / f"home_flags_attempt_{attempt_i}.json",
            ):
                try:
                    _stale.unlink()
                except FileNotFoundError:
                    pass
            _stale_csv_hist = harness_dir / f"csv_history_attempt_{attempt_i}"
            if _stale_csv_hist.is_dir():
                shutil.rmtree(_stale_csv_hist, ignore_errors=True)

            # Agentic submission reviewer (Option B): while the base agent runs,
            # a host-side watcher thread picks up review-request sentinels the
            # base runner writes on each new submission, runs the reviewer in its
            # OWN :ro-isolated instance, and writes verdicts back. The watcher is
            # torn down after the attempt (and tears down its reviewer instance).
            rev_enabled = bool((self.prm_reviewer or {}).get("enabled", False))
            rev_stop = rev_thread = None
            # Base-agent-clock accounting. The host watchdog below must fire on the
            # base agent's OWN wall-clock (task_timeout), NOT on agent+reviewer
            # wall-clock. The reviewer runs on the host watcher thread while the
            # base agent block-polls; that reviewer time is already excluded from
            # the in-container agent budget (BudgetTracker.add_paused). We mirror
            # that here: the watcher accumulates each reviewer wake's duration into
            # this thread-safe counter, and _run_agent_in_container's watchdog
            # subtracts it so the deadline tracks agent-only time.
            _rev_get = None
            if rev_enabled:
                import threading as _threading
                _rev_lock = _threading.Lock()
                _rev_secs = [0.0]

                def _rev_add(seconds, _lock=_rev_lock, _acc=_rev_secs):
                    with _lock:
                        _acc[0] += max(0.0, float(seconds))

                def _rev_get(_lock=_rev_lock, _acc=_rev_secs):
                    with _lock:
                        return _acc[0]
                # Per-TASK request dir. host_workspace is per-WORKER and reused
                # across every task the worker runs, so a dir keyed only by
                # attempt index (the old `reviewer_requests_attempt_{i}`) let a
                # prior task's leftover review_request_*.json sentinels sit in the
                # same folder — the next task's watcher re-processed them and fed
                # the reviewer a STALE `task_desc` from the wrong competition
                # (bug: reviewer TASK block = earlier task, paths = current task).
                # Keying by task_id (like _reviewer_memory_host_dir) gives each
                # task its own sentinel namespace so cross-task bleed is impossible.
                requests_dir = (host_workspace / HARNESS_DIR
                                / f"reviewer_requests_{task_id}_attempt_{attempt_i}")
                requests_dir.mkdir(parents=True, exist_ok=True)
                reviewer_mem = self._reviewer_memory_host_dir(worker_id, task_id)
                reviewer_mem.mkdir(parents=True, exist_ok=True)
                rev_stop = _threading.Event()
                rev_thread = _threading.Thread(
                    target=self._run_reviewer_watcher, kwargs=dict(
                        stop_event=rev_stop, worker_id=worker_id,
                        task_info=task_info, host_task_ws=task_workspace,
                        data_bind=self._task_data_bind(task_info),
                        reviewer_mem=reviewer_mem,
                        submission_host_dir=submission_host_dir,
                        requests_dir=requests_dir, attempt_index=attempt_i,
                        reviewer_add=_rev_add),
                    daemon=True)
                rev_thread.start()

            # Run agent inside container
            try:
                timed_out, exit_code = self._run_agent_in_container(
                    container_name=container_name,
                    config_path_in_container=f"/workspace/{HARNESS_DIR}/attempt_{attempt_i}_config.json",
                    reviewer_seconds=_rev_get,
                )
            finally:
                if rev_stop is not None:
                    rev_stop.set()
                    rev_thread.join(timeout=60)
            # An attempt that never ran to completion (apptainer FATAL 255, OOM
            # SIGKILL -9, etc. — but NOT a timeout, which is -1 and may have
            # produced real work) leaves no fresh submission. Guard against
            # recording a stale leftover CSV as a valid "phantom" submission.
            attempt_ran = timed_out or exit_code == 0

            # Read events from volume
            events_path = host_workspace / HARNESS_DIR / f"events_attempt_{attempt_i}.jsonl"
            record = events_jsonl_to_trajectory_record(
                events_path=events_path,
                task_info=task_info,
                agent_model=self.model,
                attempt_index=attempt_i,
            )
            record.trajectory_id = attempt_id
            record.execution_time = time.time() - attempt_start

            # PHANTOM-CRASH GUARD. exit_code==0 is NOT sufficient proof the agent
            # did real work this attempt: an attempt can crash mid-run on a
            # Bedrock BadRequestError ("missing field `messages`") that the SDK
            # retries 4× then swallows, exiting 0 with an events file that holds
            # only SystemPrompt + a MessageEvent + N ConversationErrorEvent and
            # ZERO ActionEvents (num_turns==0). Because the v4 workspace persists
            # across attempts, the PRIOR attempt's submission.csv is still on the
            # mount and would be recorded as this (crashed) attempt's valid
            # submission — the turns=0/valid=True phantom. Treat an attempt with
            # zero real turns AND a recorded conversation error as not-run so the
            # stale-CSV read below is skipped. (A genuine 0-turn attempt with no
            # error is vanishingly rare and equally shouldn't inherit a stale CSV.)
            _err_file = (host_workspace / HARNESS_DIR
                         / f"error_attempt_{attempt_i}.json")
            _crashed = _err_file.exists() or _events_has_conversation_error(events_path)
            if record.num_turns == 0 and _crashed:
                if attempt_ran:
                    print(f"   ⚠️ attempt {attempt_i}: 0 real turns + conversation "
                          f"error (crashed mid-run, exit {exit_code}); treating as "
                          f"NOT run so a stale prior-attempt CSV isn't recorded as "
                          f"a phantom valid submission.")
                attempt_ran = False

            # CONTAMINATION GUARD (defense-in-depth). events_attempt_N.jsonl lives
            # under the per-WORKER (`worker_{id}/`) tree, NOT keyed by task_id. The
            # root fix is the per-process worker slot in collect_trajectories_v4.py
            # (concurrent tasks can no longer share a worker tree). This is the
            # backstop: if a foreign task's event stream is EVER read here, its
            # first agent-delivered user message names a DIFFERENT challenge than
            # this task (observed: s5e2 read s4e3's stream). Detect that mismatch
            # and refuse to record the contaminated attempt rather than silently
            # scoring the wrong competition. Matches on the "CHALLENGE NAME: <id>"
            # header the loader puts at the top of every task_message.
            _leaked = self._detect_foreign_challenge(record, task_id)
            if _leaked:
                print(f"   ⚠️ CONTAMINATION: {task_id} attempt {attempt_i} event "
                      f"stream references foreign challenge '{_leaked}' — refusing "
                      f"to record (per-worker events file collision).")
                # Skip this attempt entirely; do NOT write a wrong-competition record.
                continue

            # Prepend the full task message as step 0
            record.steps.insert(0, TrajectoryStep(
                turn=0,
                role="user",
                content=task_message,
            ))

            # Read submission directly from volume (no kernel!). Skip entirely if
            # the attempt didn't run — otherwise a stale CSV left on the mount by
            # a PRIOR attempt would be copied out and recorded as a valid
            # submission for THIS (crashed) attempt (the phantom-submission bug).
            if attempt_ran:
                submission_content, has_submission = self._read_submission(
                    submission_host_dir, host_workspace / task_id
                )
            else:
                submission_content, has_submission = "", False
                print(f"   ⚠️ attempt did not run (exit {exit_code}); "
                      f"not recording a submission")
            # MID-ROLLOUT MOUNT GUARD. `_assert_task_isolation` above only proves
            # the data was present at attempt START. The other observed failure
            # (house-prices, stacked + reviewer-v6 runs) is the task's own data
            # mount vanishing PART-WAY through an attempt: the agent trains early
            # while data is live, the mount drops (reviewer saw "train.csv
            # unavailable for 4+ consecutive wakes"), and the agent then re-emits
            # a STALE cached CSV from a prior attempt — recorded as a "valid"
            # submission that in fact trained on no/partial data (4 of 5
            # house-prices attempts were byte-identical). Re-probe the mount at
            # attempt END: if it was healthy at start but is empty/absent now, the
            # submission is untrustworthy. Demote it to invalid rather than score a
            # phantom. Cheap `ls -A` probe; a probe failure returns True (don't
            # block on a flaky probe), so this only fires on a confirmed drop.
            mount_dropped = attempt_ran and has_submission and \
                not self._own_data_present(container_name, task_info)
            if mount_dropped:
                print(f"   🚩 DATA MOUNT DROPPED MID-ATTEMPT: /data/"
                      f"{task_info.source}/{task_info.task_id} was present at "
                      f"start but is empty/absent at attempt {attempt_i} end; the "
                      f"submission likely reuses a stale CSV (trained on no data). "
                      f"Marking attempt INVALID — re-run this task.")
                has_submission = False

            # TIMEOUT CSV-HISTORY RECOVERY. On a timeout the watchdog SIGKILLs the
            # whole process group at task_timeout+120, mid-rollout. The container's
            # own final-select (_select_best_submission) never runs, so the
            # /submission mount holds no fresh CSV even though the agent wrote many
            # valid submissions during the run — those attempts were being recorded
            # as valid=False (0-turn phantoms) and thrown away. Recover the last
            # SHAPE-VALID checkpoint from csv_history here, host-side (the container
            # can't be trusted to clean up after a SIGKILL). Only on genuine
            # timeouts, and only when we don't already have a live submission; never
            # overrides a mount-drop demotion (recovery re-validates shape but a
            # dropped mount means the whole run is suspect).
            if timed_out and not has_submission and not mount_dropped:
                _ch_dir = (host_workspace / HARNESS_DIR
                           / f"csv_history_attempt_{attempt_i}")
                rec_content, recovered, rec_turn = \
                    self._recover_submission_from_csv_history(_ch_dir, task_info)
                if recovered:
                    submission_content, has_submission = rec_content, True
                    record.metadata["submission_recovered_from_csv_history"] = True
                    record.metadata["recovered_submission_turn"] = rec_turn
                    print(f"   ♻️  TIMEOUT RECOVERY: attempt {attempt_i} was "
                          f"SIGKILL'd with no live submission; recovered the last "
                          f"shape-valid csv_history checkpoint (turn {rec_turn}) "
                          f"— recording as a valid submission.")

            if has_submission:
                record.outcome.valid_submission = True
                if output_dir:
                    # Submission CSV -> scratch artifact dir (not home/git output_dir).
                    sub_out = self._artifact_dir(output_dir) / f"submission_attempt_{attempt_i}.csv"
                    sub_out.parent.mkdir(parents=True, exist_ok=True)
                    sub_out.write_text(submission_content)
                    record.submission_path = str(sub_out)
            else:
                record.outcome.valid_submission = False

            # NOTE: submission-write turns come from the CSV-history manifest
            # (csv_history_attempt_{i}/manifest.json) — the ground-truth record
            # of which turns actually changed submission.csv (dedup by md5). The
            # earlier text-scan heuristic (_find_submission_write_turns) is NOT
            # used: it counted code *mentions* of to_csv and disagreed with the
            # real writes (e.g. reported turns [10,16,23,28,33,38] when the disk
            # actually changed at [13,21,24,29,39]). Use the manifest instead.

            # Workspace snapshot
            task_workspace = host_workspace / task_id
            if task_workspace.exists():
                record.metadata["workspace_snapshot"] = self._capture_workspace_snapshot(
                    task_workspace
                )

            # Home-access red flag: if the agent read/mounted the host home or
            # credentials this attempt, surface it so the trajectory can be
            # quarantined/discarded (autofs /home can't be fully masked here).
            home_flag_file = harness_dir / f"home_flags_attempt_{attempt_i}.json"
            if home_flag_file.exists():
                try:
                    flags = json.loads(home_flag_file.read_text())
                    if flags:
                        record.metadata["home_access_flagged"] = True
                        record.metadata["home_access_events"] = flags
                        print(f"   🚩 {len(flags)} home-access red flag(s) recorded")
                except (json.JSONDecodeError, OSError):
                    pass

            # Read metadata (cost/tokens) from runner. Prefer the attempt-keyed
            # file: the un-suffixed meta.json is overwritten every attempt, so a
            # crashed attempt that dies before rewriting it would otherwise leave
            # the PRIOR attempt's tokens/cost to be mis-attributed here (the
            # identical-token duplication bug). Fall back to meta.json for
            # trajectories collected before the attempt-keyed write landed.
            meta_path = harness_dir / f"meta_attempt_{attempt_i}.json"
            if not meta_path.exists():
                meta_path = harness_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    record.metadata["cost"] = meta.get("cost", 0)
                    token_usage = meta.get("token_usage", {})
                    if isinstance(token_usage, dict):
                        record.total_tokens = token_usage.get("total_tokens", 0)
                        record.metadata["input_tokens"] = token_usage.get("prompt_tokens", 0)
                        record.metadata["output_tokens"] = token_usage.get("completion_tokens", 0)
                except (json.JSONDecodeError, OSError):
                    pass

            if timed_out:
                record.metadata["timed_out"] = True

            # Extract summary from finish tool
            summary_text = record.final_answer or "Summary not available."
            attempt_summaries.append(summary_text)

            # Parse the agent's SELF-REPORTED validation score from its summary
            # (the "## Validation Score" line it writes via the finish tool).
            # Not inference — just extracting what the agent stated. Useful for
            # the inject-context preamble and for diagnosing bad attempts (e.g.
            # a None here flagged a run where the agent never reported a score).
            val_score = self._parse_val_score(summary_text)
            if val_score is not None and (best_score is None or val_score > best_score):
                best_score = val_score

            # PRM scoring
            feedback = None
            if self.prm_scorer and attempt_i < self.max_attempts - 1:
                try:
                    print(f"   Running PRM scoring...")
                    feedback = self.prm_scorer.generate_feedback(record)
                    if feedback:
                        print(f"   PRM feedback generated ({len(feedback)} chars)")
                    else:
                        print(f"   PRM: no actionable feedback")
                    print(f"   {self.prm_scorer.cost_summary()}")
                except Exception as e:
                    print(f"   PRM scoring failed: {e}")
            prm_feedbacks.append(feedback)

            # Store in record metadata
            record.metadata["attempt_summary"] = summary_text
            if feedback:
                record.metadata["prm_feedback"] = feedback

            record.metadata["harness_version"] = "v4"
            record.metadata["attempt_index"] = attempt_i

            all_records.append(record)

            # Save immediately
            if output_dir:
                out_path = Path(output_dir) / f"attempt_{attempt_i}.json"
                record.save(str(out_path))
                print(f"   Saved: {out_path}")
                # Per-turn CSV history is BULKY (218MB submission snapshots seen)
                # and gitignored — persist to the roomy scratch artifact dir, NOT
                # the home/git output_dir (which may be on a small quota).
                self._persist_csv_history(
                    host_workspace / HARNESS_DIR / f"csv_history_attempt_{attempt_i}",
                    self._artifact_dir(output_dir) / f"csv_history_attempt_{attempt_i}",
                )
                # inline-PRM call records (prm_call_NN.json) are small JSON + the
                # core analysis artifact -> keep in output_dir (home/git).
                self._persist_csv_history(
                    host_workspace / HARNESS_DIR / f"prm_calls_attempt_{attempt_i}",
                    Path(output_dir) / f"prm_calls_attempt_{attempt_i}",
                )
                # Agentic-reviewer call records (reviewer_call_NN.json) live in the
                # reviewer's writable memory (/reviewer, host = reviewer_memory).
                # Small JSON + core analysis artifact -> keep in output_dir.
                if (self.prm_reviewer or {}).get("enabled", False):
                    self._persist_csv_history(
                        self._reviewer_memory_host_dir(worker_id, task_id)
                        / f"reviews_attempt_{attempt_i}",
                        Path(output_dir) / f"prm_reviews_attempt_{attempt_i}",
                    )

            print(
                f"   Attempt {attempt_i + 1} complete: val_score={val_score}, "
                f"turns={record.num_turns}, time={record.execution_time:.0f}s, "
                f"submission={'valid' if has_submission else 'NONE'}"
            )

        # Final snapshot after the last attempt completes.
        self._save_fs_snapshot(
            task_workspace, fs_snap_dir / "snapshot_after_final_attempt.json"
        )
        # fs snapshots (workspace file dumps w/ contents) are bulky -> scratch.
        if output_dir:
            self._persist_csv_history(fs_snap_dir, self._artifact_dir(output_dir) / "fs_snapshots")
            # The reviewer's rubric store + op-log persist across all attempts of
            # this task — the durable rubric-dynamics signal for the paper. Copy
            # them (+ check scripts) once, after the last attempt, to output_dir.
            if (self.prm_reviewer or {}).get("enabled", False):
                rev_mem = self._reviewer_memory_host_dir(worker_id, task_id)
                store_dir = rev_mem / "rubric_store"
                if store_dir.is_dir():
                    dst = Path(output_dir) / "reviewer_rubric_store"
                    self._persist_csv_history(store_dir, dst)
                    checks = store_dir / "checks"
                    if checks.is_dir():
                        self._persist_csv_history(checks, dst / "checks")

        return all_records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _exec_prefix(self, container_name: str) -> list[str]:
        """Command prefix to execute inside the running container/instance."""
        if self.backend == "apptainer":
            return ["apptainer", "exec", f"instance://{container_name}"]
        return ["docker", "exec", container_name]

    def _setup_task_workspace(self, container_name: str, task_id: str):
        """Clean and create task workspace dirs inside the container.

        The workspace bind is per-WORKER (worker_{id}:/workspace), reused across
        every task that worker runs sequentially, and it PERSISTS on the host
        across instance restarts (by design, so a task's multi-attempt state
        survives an OOM restart). The old wipe only removed `/workspace/{task_id}`
        — the CURRENT task's subdir — leaving PRIOR tasks' subdirs sitting in the
        shared worker dir. A later task could then `ls /workspace/` and read a
        sibling competition's files (observed: s4e8 seeing s5e6, s4e12 seeing
        s4e9). Nothing downstream needs those leftovers: each task's snapshot /
        csv-history / submission are already persisted to the NFS output dir
        before the task ends. So we wipe EVERYTHING under /workspace except this
        task's dir and the harness dir. Safe against concurrency because each
        worker owns its own workspace dir and runs tasks strictly sequentially.
        The `rm -rf` is safe for /submission too — that is a separate bind
        OUTSIDE the workspace tree (see _submission_host_dir).
        """
        cmd = (
            # Remove sibling task dirs left by prior tasks on this worker; keep
            # only the current task and the harness plumbing dir.
            f"find /workspace -mindepth 1 -maxdepth 1 "
            f"! -name {task_id} ! -name {HARNESS_DIR} -exec rm -rf {{}} + ; "
            f"rm -rf /workspace/{task_id} && "
            f"mkdir -p /workspace/{task_id} /workspace/{HARNESS_DIR}"
        )
        subprocess.run(
            self._exec_prefix(container_name) + ["bash", "-c", cmd],
            capture_output=True, timeout=30,
            env=self._apptainer_env() if self.backend == "apptainer" else None,
        )

    def _build_attempt_config(
        self,
        task_id: str,
        attempt_index: int,
        task_message: str,
    ) -> dict:
        """Build the config JSON that agent_runner.py reads.

        Canonical submission path is ``/submission/submission.csv`` — that dir
        is bound to the host so the DSPredict instructions and the V4 system
        prompt agree, and the host reads it back from the workspace mount.
        """
        # Data-centric competitions (e.g. s3e21) ask the agent to SUBMIT A
        # TRAINING DATASET, not predictions. The system prompt is otherwise
        # saturated with prediction-framing ("write your predictions to
        # submission.csv", "copy your best model's test predictions"), which
        # conflicts and confuses the agent. Flag it so the prompt can override.
        _dc = task_message.lower()
        is_data_centric = any(s in _dc for s in (
            "instead of submitting predictions", "submit a dataset",
            "submission will be used as a training dataset",
            "submit a dataset that will be used to train", "data-centric"))
        return {
            "workspace_dir": f"/workspace/{task_id}",
            "submission_path": "/submission/submission.csv",
            "events_output_path": f"/workspace/{HARNESS_DIR}/events_attempt_{attempt_index}.jsonl",
            "task_message": task_message,
            "summary_request": SUMMARY_REQUEST_PROMPT,
            "llm_kwargs": self.llm_kwargs,
            "system_prompt_kwargs": {
                "attempt_index": attempt_index,
                "total_attempts": self.max_attempts,
                "command_timeout": self.command_timeout,
                "max_command_timeout": self.max_command_timeout,
                "turn_budget": self.max_iterations_per_attempt,
                "time_budget": max(1, self.task_timeout // 60),
                "workspace_dir": f"/workspace/{task_id}",
                "submission_path": "/submission/submission.csv",
                "is_data_centric": is_data_centric,
            },
            "max_iterations": self.max_iterations_per_attempt,
            "time_budget_seconds": self.task_timeout,
            "condenser_max_size": self.condenser_max_size,
            # Resume the (state-persisting) conversation after a ConversationRunError
            # (e.g. exhausted-retry Bedrock ServiceUnavailable) instead of losing the
            # attempt. Bounded so a hard outage can't loop forever.
            "run_resume_retries": self.run_resume_retries,
            "run_resume_wait": self.run_resume_wait,
            # Per-turn CSV history archived under the harness dir (host-visible).
            "csv_history_dir": f"/workspace/{HARNESS_DIR}/csv_history_attempt_{attempt_index}",
            "csv_scan_roots": [f"/workspace/{task_id}", "/submission"],
            # Red-flag report for any turn that reads/mounts the host home/creds.
            "home_flag_path": f"/workspace/{HARNESS_DIR}/home_flags_attempt_{attempt_index}.json",
            # Final-submission SELECTION phase (flag-gated): after the loop, one
            # structured LLM call picks the best of the CSVs already produced and
            # copies it to submission_path. Runs outside the time/turn budget.
            "final_submission_select": self.final_submission_select,
            "submission_path": "/submission/submission.csv",
            # Inline PRM (flag-gated): fires the agentic PRM DURING the attempt.
            # prm_call_NN.json files land under the harness dir (host-visible via
            # the workspace mount); persisted to the output dir like csv_history.
            "prm_calls_dir": f"/workspace/{HARNESS_DIR}/prm_calls_attempt_{attempt_index}",
            "prm_inline": self.prm_inline,
            # Agentic submission reviewer (Option B). The base runner writes
            # review-request sentinels here and block-polls for results; the
            # host watcher (run_task) reads the SAME dir (via the workspace mount)
            # and runs the reviewer in its own instance. All paths are inside the
            # workspace mount so both processes see them.
            "prm_reviewer": self.prm_reviewer,
            # NOTE: this container-side sentinel dir MUST stay byte-for-byte in
            # sync with the host-watcher's `requests_dir` (run_task, ~:899) — the
            # base agent writes review_request_*.json here (via the /workspace
            # mount) and the host watcher reads the SAME host path. Both are keyed
            # by task_id so a prior task's leftover sentinels can't bleed into the
            # next task the (per-WORKER, reused) workspace runs (cross-task bug).
            "reviewer_requests_dir": f"/workspace/{HARNESS_DIR}/reviewer_requests_{task_id}_attempt_{attempt_index}",
            "prm_reviews_dir": f"/workspace/{HARNESS_DIR}/prm_reviews_attempt_{attempt_index}",
        }

    def _run_agent_in_container(
        self,
        container_name: str,
        config_path_in_container: str,
        reviewer_seconds=None,
    ) -> tuple[bool, int]:
        """Run agent_runner.py inside the container.

        Returns (timed_out, exit_code). exit_code is the agent process's return
        code (0 = clean); a nonzero code — especially 255 (apptainer FATAL: no
        instance / cgroup error) or -9 (SIGKILL/OOM) — means the attempt did NOT
        run to completion, so its "submission" on the mount is stale and must not
        be recorded as valid. exit_code is -1 on timeout.

        reviewer_seconds: optional zero-arg callable returning accumulated
        reviewer wall-clock (seconds). The reviewer runs in its own container
        while the base agent block-polls, so raw wall-clock overstates the base
        agent's own elapsed time. The watchdog subtracts reviewer_seconds() so
        the kill fires on BASE-AGENT clock only, matching the in-container
        BudgetTracker (which excludes reviewer time via add_paused).
        """
        import os as _os
        import signal as _signal
        import threading as _threading

        cmd = self._exec_prefix(container_name) + [
            "python", "-u", AGENT_RUNNER_PATH,
            "--config", config_path_in_container,
        ]
        # Hard wall-clock deadline. The previous bug: the `for line in
        # proc.stdout` read loop blocks with NO timeout, so if the agent wedges
        # (stuck in an LLM retry, pipe held open but never exiting) the loop
        # never returns and proc.wait(timeout=) is never even reached — attempts
        # ran 3.6h past the 2h limit with timed_out=False, wedging whole workers.
        # Fix: a watchdog timer that force-kills the ENTIRE process group
        # (start_new_session=True) when the deadline passes, so a killed exec
        # also tears down the in-container agent, not just the host wrapper.
        # Base-agent deadline. The watchdog fires on the base agent's OWN clock:
        #   agent_elapsed = raw_wall_clock - reviewer_seconds()
        # so reviewer-heavy attempts are no longer force-killed prematurely (the
        # old fixed Timer(task_timeout+120) counted agent+reviewer wall-clock and
        # killed reviewer-heavy attempts mid-command at ~7320s with turns=0).
        # An ABSOLUTE backstop caps raw wall-clock at 4x the deadline so a stuck
        # reviewer accumulator can never let an attempt run unbounded.
        deadline = self.task_timeout + 120
        abs_backstop = 4 * deadline
        killed = {"fired": False, "reason": ""}

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=self._apptainer_env() if self.backend == "apptainer" else None,
            start_new_session=True,   # own process group -> killpg tears down children
        )

        _wd_start = time.monotonic()
        _wd_stop = _threading.Event()

        def _killpg():
            for sig in (_signal.SIGTERM, _signal.SIGKILL):
                try:
                    _os.killpg(_os.getpgid(proc.pid), sig)
                except (ProcessLookupError, OSError):
                    break
                time.sleep(3)

        def _watchdog():
            while not _wd_stop.wait(5.0):
                raw = time.monotonic() - _wd_start
                rev = 0.0
                if reviewer_seconds is not None:
                    try:
                        rev = float(reviewer_seconds())
                    except Exception:
                        rev = 0.0
                agent_elapsed = raw - rev
                if agent_elapsed >= deadline:
                    killed["fired"] = True
                    killed["reason"] = (
                        f"base-agent clock {agent_elapsed:.0f}s "
                        f"(raw {raw:.0f}s − reviewer {rev:.0f}s) exceeded {deadline}s")
                    break
                if raw >= abs_backstop:
                    killed["fired"] = True
                    killed["reason"] = (
                        f"raw wall-clock {raw:.0f}s exceeded absolute backstop "
                        f"{abs_backstop}s")
                    break
            if killed["fired"]:
                print(f"   ⏰ Agent exceeded deadline ({killed['reason']}) — "
                      f"force-killing process group (was wedged past task_timeout)")
                _killpg()

        timer = _threading.Thread(target=_watchdog, daemon=True)
        timer.start()
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line:
                    print(f"   {line}", flush=True)
            proc.wait()
        finally:
            _wd_stop.set()

        if killed["fired"]:
            print(f"   Agent timed out ({killed['reason']})")
            return True, -1
        if proc.returncode != 0:
            print(f"   Agent exited with code {proc.returncode}")
        return False, proc.returncode

    def _read_submission(
        self, submission_host_dir: Path, task_workspace: Path
    ) -> tuple[str, bool]:
        """Read the submission from the host mounts. No kernel, no hang risk.

        Prefers the canonical location (the /submission bind's backing dir). If
        that's missing — because a path-confused agent wrote elsewhere under the
        writable workspace — falls back to searching for any submission.csv, then
        any *.csv, under the task workspace. Read-only-root guarantees every
        write the agent made is on a host-visible mount, so nothing is
        unfindable.
        """
        canonical = submission_host_dir / "submission.csv"
        candidates: list[Path] = []
        if canonical.exists():
            candidates.append(canonical)
        task_ws = task_workspace
        if task_ws.is_dir():
            # Any file literally named submission.csv anywhere under the task dir.
            candidates += [p for p in task_ws.rglob("submission.csv")
                           if p != canonical]
            # Last resort: any other CSV under the task dir.
            candidates += [p for p in sorted(task_ws.rglob("*.csv"))
                           if p.name != "submission.csv"]
        for path in candidates:
            try:
                content = path.read_text()
            except OSError:
                continue
            if content.strip():
                if path != canonical:
                    print(f"   ⚠️  submission found at non-canonical path: {path}")
                return content, True
        return "", False

    def _recover_submission_from_csv_history(
        self, csv_history_dir: Path, task_info: Optional[TaskInfo]
    ) -> tuple[str, bool, Optional[int]]:
        """Salvage the last VALID submission checkpoint from csv_history.

        Used when an attempt is SIGKILL'd on timeout: the process-group kill at
        `task_timeout+120` fires mid-rollout, so the container's own final-select
        (agent_runner `_select_best_submission`) never runs and no fresh CSV is
        on the /submission mount — yet the agent may have written many valid
        submissions during the run. csv_history_attempt_{i}/manifest.json is the
        host-side ground-truth archive of every turn that changed submission.csv
        (dedup by md5). We pick the HIGHEST-turn checkpoint whose shape matches
        the task's sample_submission.csv exactly.

        Validity gate (settled empirically): header columns must EQUAL the
        sample_submission header AND row count must EQUAL it. Columns are the
        hard gate — the naive "highest turn" is unsafe: a late checkpoint can be
        malformed (observed s3e15 turn-154 had a stray duplicate `x_e_out`
        column → 3 cols; the last clean 2-col checkpoint was turn 79), or a
        degenerate all-zeros placeholder with the wrong row count (s3e19 turn 40).

        Returns (content, recovered, turn). recovered=False if no checkpoint
        passes or the sample_submission ground-truth can't be read (fail closed:
        never fabricate a submission we can't validate).
        """
        manifest = csv_history_dir / "manifest.json"
        if not manifest.is_file():
            return "", False, None
        # Ground-truth shape from the task's sample_submission.csv on the mount.
        if task_info is None:
            return "", False, None
        sample = Path(self.data_path) / task_info.task_id / "sample_submission.csv"
        if not sample.is_file():
            print(f"   ⚠️  csv-history recovery: no sample_submission at {sample} "
                  f"— cannot validate shape, skipping recovery")
            return "", False, None
        try:
            srdr = list(csv.reader(open(sample, newline="", errors="replace")))
        except OSError:
            return "", False, None
        if not srdr:
            return "", False, None
        exp_header = srdr[0]
        exp_rows = len([r for r in srdr[1:] if r])
        try:
            entries = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            return "", False, None
        subs = [e for e in entries if "submission" in e.get("archived_as", "")]
        # Highest turn first — we want the LAST valid checkpoint.
        subs.sort(key=lambda e: e.get("turn", -1), reverse=True)
        for e in subs:
            f = csv_history_dir / e.get("archived_as", "")
            if not f.is_file():
                continue
            try:
                txt = f.read_text(errors="replace")
            except OSError:
                continue
            rdr = list(csv.reader(io.StringIO(txt)))
            if not rdr:
                continue
            header = rdr[0]
            nrows = len([r for r in rdr[1:] if r])
            if header == exp_header and nrows == exp_rows:
                return txt, True, e.get("turn")
        return "", False, None

    def _capture_workspace_snapshot(
        self,
        task_workspace: Path,
        max_file_size: int = 50_000,
        max_total_inline_bytes: int = 20_000_000,
        max_files_listed: int = 20_000,
    ) -> dict:
        """Capture file tree + small file contents for debugging.

        Size-gated on three axes so a workspace with tens of thousands of small
        files (e.g. tensorflow-speech-recognition: 65k+ ~32KB WAV clips, each
        under the per-file cap) can't inline gigabytes of raw bytes as "text"
        and OOM the worker (a 37GB attempt_0.json / ~101GB in-memory dict took
        the Hard-v2 driver down): (1) per-file content cap ``max_file_size``,
        (2) a total inline-content budget across all files
        ``max_total_inline_bytes``, and (3) a cap on the number of files listed
        ``max_files_listed``. Binary / media / array / archive suffixes are
        skipped from content entirely (their bytes are meaningless as text and
        are exactly what explodes audio/vision workspaces)."""
        snapshot: dict[str, Any] = {"files": []}
        skip_suffixes = {
            ".pkl", ".parquet", ".h5", ".joblib", ".model", ".bin",
            # audio / video
            ".wav", ".flac", ".mp3", ".ogg", ".m4a", ".mp4", ".avi", ".mov",
            # image
            ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp",
            # arrays / tensors / checkpoints
            ".npy", ".npz", ".pt", ".pth", ".ckpt", ".safetensors",
            ".tfrecord", ".pb", ".onnx", ".arrow", ".feather",
            # archives
            ".zip", ".gz", ".tar", ".tgz", ".bz2", ".7z", ".xz", ".zst",
        }
        # Bound the path list first so a workspace with millions of files can't
        # blow up memory just materializing the listing (rglob is a generator;
        # sorting the whole tree would defeat the purpose).
        files: list[Path] = []
        listing_truncated = False
        try:
            for f in task_workspace.rglob("*"):
                if not f.is_file():
                    continue
                files.append(f)
                if len(files) >= max_files_listed:
                    listing_truncated = True
                    break
        except OSError:
            pass
        if listing_truncated:
            snapshot["listing_truncated"] = True
        inline_bytes = 0
        try:
            for f in sorted(files):
                rel = str(f.relative_to(task_workspace))
                size = f.stat().st_size
                entry: dict[str, Any] = {"path": rel, "size": size}
                if (
                    size <= max_file_size
                    and f.suffix.lower() not in skip_suffixes
                    and inline_bytes + size <= max_total_inline_bytes
                ):
                    try:
                        entry["content"] = f.read_text(errors="replace")
                        inline_bytes += size
                    except OSError:
                        pass
                snapshot["files"].append(entry)
        except OSError:
            pass
        return snapshot

    def _artifact_dir(self, output_dir: str) -> Path:
        """Scratch-side mirror of output_dir for BULKY artifacts (CSV history,
        submission CSVs, fs_snapshots). Keeps the home/NFS/git output_dir to
        small JSON only. Mirrors the tail of output_dir under csv_artifact_root so
        the two trees line up (…/csv_artifact_root/<run>/<split>/<task>/…)."""
        out = Path(output_dir)
        # Mirror everything after the first "trajectories_v4" (or the last 3
        # components as a fallback) so runs/splits/tasks map 1:1.
        parts = out.parts
        if "trajectories_v4" in parts:
            tail = Path(*parts[parts.index("trajectories_v4") + 1:])
        else:
            tail = Path(*parts[-3:]) if len(parts) >= 3 else out.name
        d = self.csv_artifact_root / tail
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _persist_csv_history(self, src_dir: Path, dst_dir: Path) -> None:
        """Copy a scratch-side artifact dir (CSV history / fs snapshots) into the
        NFS-backed, git-tracked output dir so it survives node loss or a scratch
        wipe. Overwrites dst each time (cheap, small text/CSV files)."""
        if not src_dir.is_dir():
            return
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in src_dir.iterdir():
                if f.is_file():
                    shutil.copyfile(f, dst_dir / f.name)
        except OSError as e:
            print(f"   ⚠️  failed to persist {src_dir.name}: {e}")

    def _save_fs_snapshot(self, task_workspace: Path, out_path: Path) -> None:
        """Write a filesystem snapshot (file tree + small file contents) to disk.

        Called at attempt boundaries (before each attempt, and after the last)
        so the workspace's evolution across attempts is recorded independently
        of the trajectory JSONs. Reuses _capture_workspace_snapshot's size-gated
        capture (small code files inline, large/binary files listed by size).
        """
        snap = (
            self._capture_workspace_snapshot(task_workspace)
            if task_workspace.exists()
            else {"files": []}
        )
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(snap, indent=2))
        except OSError:
            pass

    def _snapshot_data_mount(self, container_name: str, out_path: Path) -> None:
        """Record what is visible under /data INSIDE the container (paths + sizes,
        NO file contents), so we can detect cross-task data-mount leaks.

        The per-task bind should expose ONLY the current task's folder at
        /data/<source>/<task_id>. If the whole dataset leaks in (e.g. the
        legacy whole-/data fallback), the agent can read — and accidentally
        solve — a DIFFERENT competition (observed: s4e12 read s3e26). This
        snapshot is the ground-truth record of the mount; fs_snapshots only
        cover /workspace, so they cannot catch this class of bug.
        """
        # `find /data -maxdepth 3` gives the tree; `-printf` adds sizes cheaply.
        cmd = "find /data -maxdepth 3 -printf '%y %s %p\\n' 2>/dev/null | head -2000"
        try:
            r = subprocess.run(
                self._exec_prefix(container_name) + ["bash", "-c", cmd],
                capture_output=True, text=True, timeout=30,
                env=self._apptainer_env() if self.backend == "apptainer" else None,
            )
            lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
            # Top-level task dirs directly under /data/<source>/ — the key signal:
            # more than one task dir here == a cross-task leak.
            task_dirs = sorted({
                ln.split(" ", 2)[2] for ln in lines
                if ln.startswith("d ") and ln.split(" ", 2)[2].count("/") == 3
            })
            snap = {"entries": lines, "task_dirs_visible": task_dirs,
                    "n_task_dirs": len(task_dirs)}
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(snap, indent=2))
            if len(task_dirs) > 1:
                print(f"  🚩 DATA LEAK: {len(task_dirs)} task dirs visible under "
                      f"/data (expected 1): {task_dirs[:5]}")
        except (subprocess.TimeoutExpired, OSError) as e:
            print(f"   ⚠️  data-mount snapshot failed: {e}")

    def _foreign_tasks_visible(
        self, container_name: str, task_info: Optional[TaskInfo]
    ) -> list[str]:
        """Return other tasks' dirs visible under /data/<source>/ (should be []).

        The per-task bind must expose ONLY this task at
        /data/<source>/<task_id>. If any OTHER task dir is listable there, the
        agent can read and accidentally solve the wrong competition (the s4e12 ->
        s3e26 bug). We list the parent /data/<source>/ INSIDE the container and
        flag every entry that isn't this task. Empty list == clean.
        """
        if task_info is None:
            return []
        parent = f"/data/{task_info.source}"
        cmd = f"ls -1 {parent} 2>/dev/null"
        try:
            r = subprocess.run(
                self._exec_prefix(container_name) + ["bash", "-c", cmd],
                capture_output=True, text=True, timeout=30,
                env=self._apptainer_env() if self.backend == "apptainer" else None,
            )
        except (subprocess.TimeoutExpired, OSError):
            return []  # can't check — don't block on a probe failure
        names = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        return [n for n in names if n != task_info.task_id]

    def _own_data_present(
        self, container_name: str, task_info: Optional[TaskInfo]
    ) -> bool:
        """Return True iff THIS task's own data dir is present and non-empty
        inside the container.

        Complements `_foreign_tasks_visible` (which catches a WRONG task leaking
        in). The other failure mode is the task's own /data/<source>/<task_id>/
        going EMPTY or missing mid-run — observed in the reviewer-v6 house-prices
        run (attempts saw n_task_dirs=0). The agent then reports "no training
        data", writes a constant fallback submission, and it is still marked
        valid — a SILENT corruption. Root cause of the mount vanishing is still
        unsolved , so this is
        a detection guard, not a fix. We don't hardcode train.csv (some tasks
        ship data.csv, e.g. s3e15); "dir has >=1 entry" is the robust signal.
        """
        if task_info is None:
            return True
        own = f"/data/{task_info.source}/{task_info.task_id}"
        # `ls -A` lists all entries incl. dotfiles; count non-blank lines.
        cmd = f"ls -A {own} 2>/dev/null | head -5"
        try:
            r = subprocess.run(
                self._exec_prefix(container_name) + ["bash", "-c", cmd],
                capture_output=True, text=True, timeout=30,
                env=self._apptainer_env() if self.backend == "apptainer" else None,
            )
        except (subprocess.TimeoutExpired, OSError):
            return True  # can't probe — don't block on a probe failure
        return any(ln.strip() for ln in r.stdout.splitlines())

    def _assert_task_isolation(
        self, worker_id: int, task_info: Optional[TaskInfo], container_name: str
    ) -> bool:
        """Integrity check + self-heal: ensure ONLY this task's data is visible,
        AND that this task's own data is actually present.

        Called before each attempt (after the instance is confirmed healthy).
        Two failure modes, both healed the same way:
          1. FOREIGN task dirs visible under /data/<source>/ — the bind leaked
             another competition in (the s4e12 -> s3e26 wrong-task bug).
          2. OWN data dir empty/missing — the task's own /data/<source>/<task_id>
             vanished (the reviewer-v6 house-prices silent-corruption mode).
        Either way we tear down and restart the instance once (redoing the
        per-task bind) and re-check. If it's STILL broken we return False so the
        caller aborts the task rather than collecting a trajectory that solves
        the wrong competition or trains on no data. Returns True when clean.
        """
        if task_info is None:
            return True
        foreign = self._foreign_tasks_visible(container_name, task_info)
        own_ok = self._own_data_present(container_name, task_info)
        if not foreign and own_ok:
            return True
        if foreign:
            print(f"  🚩 TASK-ISOLATION VIOLATION: {len(foreign)} foreign task "
                  f"dirs visible for {task_info.task_id!r} (e.g. {foreign[:5]}); "
                  f"restarting instance to redo the per-task bind")
        if not own_ok:
            print(f"  🚩 OWN-DATA MISSING: /data/{task_info.source}/"
                  f"{task_info.task_id} is empty/absent inside the container; "
                  f"restarting instance to redo the per-task bind")
        if self.backend == "apptainer":
            self._ensure_instance_gone(container_name)
        else:
            self.stop_container(container_name)
        self.start_container(worker_id, task_info=task_info)
        foreign2 = self._foreign_tasks_visible(container_name, task_info)
        own_ok2 = self._own_data_present(container_name, task_info)
        if foreign2 or not own_ok2:
            reason = (f"foreign dirs {foreign2[:5]}" if foreign2
                      else "own data still empty/absent")
            print(f"  ❌ DATA MOUNT STILL BROKEN after restart ({reason}); "
                  f"aborting task {task_info.task_id!r} rather than running on "
                  f"wrong/empty data")
            return False
        print(f"  ✅ data mount healthy after restart for {task_info.task_id!r}")
        return True

    def _build_inject_preamble(
        self,
        attempt_index: int,
        attempt_summaries: list[str],
        prm_feedbacks: list[Optional[str]],
        best_score: Optional[float],
    ) -> str:
        """Build preamble with all previous context injected inline.

        Identical logic to V3._build_inject_preamble.
        """
        if attempt_index == 0:
            return ""

        parts = []
        parts.append(f"# This is attempt {attempt_index + 1} of {self.max_attempts}")
        parts.append("")

        score_str = (
            f"{best_score:.4f}"
            if best_score is not None
            else "None (no valid submission yet)"
        )
        parts.append(f"**Best validation score so far: {score_str}**")
        parts.append("")

        parts.append("## Previous Attempts")
        parts.append("")

        for i, summary in enumerate(attempt_summaries):
            parts.append(f"### Attempt {i + 1} Summary")
            parts.append(summary)
            parts.append("")

            if i < len(prm_feedbacks) and prm_feedbacks[i]:
                parts.append(f"### Attempt {i + 1} — Expert PRM Feedback")
                parts.append(prm_feedbacks[i])
                parts.append("")

        parts.append("## Instructions for this attempt")
        parts.append(
            f"- This is a BEST-OF-{self.max_attempts} run: only your SINGLE BEST "
            f"attempt is scored (the final result is the MAX over all "
            f"{self.max_attempts} attempts, not the last one)."
        )
        parts.append(
            "- Because the best attempt wins, a genuinely DIFFERENT approach that "
            "ends up slightly worse costs you NOTHING — but an attempt that just "
            "reproduces a previous one (same model family/features, reusing the same "
            "cached predictions, re-tuning an already-saturated axis) WASTES one of "
            "your limited attempts."
        )
        parts.append(
            "- So DIVERSIFY: prefer a distinct modeling class or approach this "
            "attempt (a different model family, a different feature representation, "
            "or a different problem framing) — even a second-best idea is worth "
            "trying, since the beam keeps the best. Do NOT converge to the same "
            "solution as prior attempts."
        )
        parts.append(
            "- Carry forward the LESSONS from previous attempts (reuse what worked, "
            "avoid repeating mistakes and crashes, address PRM feedback) — but do "
            "that in service of a NEW, different attempt, not a rerun of the last one."
        )
        parts.append(
            "- The workspace retains scripts and cached data from previous attempts."
        )
        parts.append("")

        return "\n".join(parts)

    def _parse_val_score(self, summary_text: str) -> Optional[float]:
        """Parse validation score from a summary."""
        match = re.search(
            r"##\s*Validation Score\s*\n+\s*([0-9]+\.?[0-9]*)",
            summary_text,
        )
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None
