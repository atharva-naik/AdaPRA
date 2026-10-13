"""
V4 Event Converter: reads events JSONL from agent_runner.py and produces TrajectoryRecord.

Maps standard OpenHands SDK events (TerminalAction, FileEditorAction, FinishAction, etc.)
to the existing TrajectoryStep schema used by PRM scoring and labeling pipelines.
"""

import json
import re
from pathlib import Path
from typing import Any, Optional

from dsgym.adapra.schema import (
    OutcomeLabels,
    TaskInfo,
    TrajectoryRecord,
    TrajectoryStep,
)

_ERROR_RE = re.compile(
    r"(Traceback \(most recent call last\)|Error:|Exception:|"
    r"FileNotFoundError|KeyError|ValueError|TypeError|IndexError|"
    r"ModuleNotFoundError|ImportError|RuntimeError|MemoryError|"
    r"OSError|PermissionError)"
)


def events_jsonl_to_trajectory_record(
    events_path: Path,
    task_info: TaskInfo,
    agent_model: str,
    attempt_index: int = 0,
    max_observation_chars: int = 30_000,
) -> TrajectoryRecord:
    """Convert a V4 events JSONL file into a TrajectoryRecord."""
    steps: list[TrajectoryStep] = []
    num_turns = 0
    num_errors = 0
    final_answer: Optional[str] = None

    events = _load_events(events_path)

    for event in events:
        # SDK serializes the discriminator as "kind"; older/other paths use
        # "type". Accept both so events are never silently dropped.
        event_type = event.get("kind") or event.get("type", "")

        if event_type == "ActionEvent":
            step = _convert_action_event(event)
            if step:
                num_turns += 1
                step.turn = num_turns
                steps.append(step)

                # Check for finish action
                if step.tool_type == "finish" and step.code:
                    final_answer = step.code

        elif event_type == "ObservationEvent":
            step = _convert_observation_event(event, max_observation_chars)
            if step:
                step.turn = num_turns
                steps.append(step)
                if step.has_error:
                    num_errors += 1

        elif event_type == "MessageEvent":
            step = _convert_message_event(event)
            if step:
                step.turn = num_turns
                steps.append(step)

    record = TrajectoryRecord(
        trajectory_id="",  # set by harness
        task=task_info,
        agent_model=agent_model,
        agent_backend="openhands-v4",
        steps=steps,
        final_answer=final_answer,
        outcome=OutcomeLabels(num_execution_errors=num_errors),
        num_turns=num_turns,
        metadata={"attempt_index": attempt_index},
    )

    return record


def _load_events(events_path: Path) -> list[dict]:
    """Load events from JSONL file."""
    events = []
    if not events_path.exists():
        return events
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def _extract_text(value: Any) -> str:
    """Extract plain text from an SDK content field.

    Content may be a plain string, or a list of TextContent dicts like
    ``[{"type": "text", "text": "...", "cache_prompt": false}]``. Returns the
    joined text so trajectories don't leak raw dict reprs / cache_prompt noise.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p)
    return ""


def _convert_action_event(event: dict) -> Optional[TrajectoryStep]:
    """Convert an ActionEvent to a TrajectoryStep."""
    tool_name = event.get("tool_name", "")
    action = event.get("action") or {}  # may be present-but-null; coerce to {}
    thought = _extract_text(event.get("thought", ""))

    if tool_name == "terminal":
        command = action.get("command", "")
        code = command
        if thought:
            code = f"# {thought}\n{command}" if command else thought
        return TrajectoryStep(
            turn=0,
            role="assistant",
            content=thought or "",
            code=code,
            tool_type="bash",
        )

    elif tool_name == "file_editor":
        cmd = action.get("command", "")
        path = action.get("path", "")

        if cmd == "view":
            code = f"[file_read] {path}"
            if action.get("view_range"):
                code += f" (lines {action['view_range']})"
            return TrajectoryStep(
                turn=0,
                role="assistant",
                content=thought or "",
                code=code,
                tool_type="file_read",
            )
        elif cmd == "create":
            file_text = action.get("file_text", "")
            code = f"[file_create] {path}\n{file_text}"
            return TrajectoryStep(
                turn=0,
                role="assistant",
                content=thought or "",
                code=code,
                tool_type="file_write",
            )
        elif cmd == "str_replace":
            old_str = action.get("old_str", "")
            new_str = action.get("new_str", "")
            code = f"[str_replace] {path}\n-{old_str}\n+{new_str}"
            return TrajectoryStep(
                turn=0,
                role="assistant",
                content=thought or "",
                code=code,
                tool_type="file_write",
            )
        elif cmd == "insert":
            new_str = action.get("new_str", "")
            line_num = action.get("insert_line", 0)
            code = f"[insert] {path} after line {line_num}\n{new_str}"
            return TrajectoryStep(
                turn=0,
                role="assistant",
                content=thought or "",
                code=code,
                tool_type="file_write",
            )
        elif cmd == "undo_edit":
            code = f"[undo_edit] {path}"
            return TrajectoryStep(
                turn=0,
                role="assistant",
                content=thought or "",
                code=code,
                tool_type="file_write",
            )

    elif tool_name == "finish":
        message = action.get("message", "")
        code = f"<answer>{message}</answer>"
        return TrajectoryStep(
            turn=0,
            role="assistant",
            content=thought or "",
            code=code,
            tool_type="finish",
            done=True,
        )

    elif tool_name == "think":
        thought_text = action.get("thought", "") or thought
        return TrajectoryStep(
            turn=0,
            role="assistant",
            content=thought_text,
            code=thought_text,
            tool_type="think",
        )

    # Unknown tool
    if tool_name:
        return TrajectoryStep(
            turn=0,
            role="assistant",
            content=thought or "",
            code=f"[{tool_name}] {json.dumps(action)[:500]}",
            tool_type="",
        )

    return None


def _convert_observation_event(
    event: dict, max_chars: int = 30_000
) -> Optional[TrajectoryStep]:
    """Convert an ObservationEvent to a TrajectoryStep."""
    observation = event.get("observation", {})

    # Extract text content
    obs_text = _extract_text(observation.get("content", []))

    # Truncate if necessary
    if len(obs_text) > max_chars:
        obs_text = obs_text[:max_chars] + f"\n... [truncated, {len(obs_text)} chars total]"

    # Detect errors
    has_error = bool(_ERROR_RE.search(obs_text))
    error_type = None
    if has_error:
        match = _ERROR_RE.search(obs_text)
        if match:
            error_type = match.group(0)

    return TrajectoryStep(
        turn=0,  # set by caller
        role="user",
        content=obs_text,
        observation=obs_text,
        has_error=has_error,
        error_type=error_type,
    )


def _convert_message_event(event: dict) -> Optional[TrajectoryStep]:
    """Convert a MessageEvent to a TrajectoryStep.

    The SDK nests message text under ``llm_message.content`` (a list of
    TextContent dicts); older paths put it at top-level ``content``.
    """
    source = event.get("source", "")
    content = event.get("llm_message", {}).get("content")
    if content is None:
        content = event.get("content", "")

    text = _extract_text(content)

    if source == "agent":
        return TrajectoryStep(turn=0, role="assistant", content=text)
    # "user" is real user/preamble messages; "environment" is the injected
    # budget footer — keep both as user-role context.
    if source in ("user", "environment"):
        return TrajectoryStep(turn=0, role="user", content=text)

    return None
