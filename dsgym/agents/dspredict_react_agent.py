"""
DSPredictReActAgent - specialized ReAct agent for DSPredict-style competitions.

Extends ReActDSAgent to explicitly capture and return detailed per-turn
trajectories suitable for downstream analysis of DSPredict runs.
"""

import copy
import time
import traceback
from typing import Dict, Any, List
import os
from datetime import datetime
import shutil

from .react_ds_agent import ReActDSAgent
from .environment import AllocatedCodeEnv


class DSPredictReActAgent(ReActDSAgent):
    """ReAct agent tuned for DSPredict challenges, returning rich trajectories."""
    def __init__(self, backend: str, model: str, **kwargs):
        super().__init__(backend, model, **kwargs)

        self.submission_dir = kwargs.get('submission_dir', '')
        # Per-cell execution timeout in seconds. Default 300s to allow model training.
        self.time_out = kwargs.get('time_out', 300)

    def solve_task(self, sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve a DSPredict task and return detailed trajectories.

        Returns a dict with keys including:
        - solution: final answer string (may include <answer> tags content)
        - success: bool
        - turns: number of turns executed
        - error: optional error message
        - metadata: run metadata (model, backend, timings, etc.)
        - conversation: full conversation history
        - trajectory: List[Dict] with per-turn details
        - raw_result: compact summary
        """
        start_time = time.time()

        trajectory: List[Dict[str, Any]] = []

        try:
            # Extract conversation and metadata from sample
            conversation = copy.deepcopy(sample.get("prompt", []))
            if not conversation:
                raise ValueError("Sample must contain 'prompt' field with conversation")

            extras = {
                "reward_spec": sample.get("reward_spec", {"ground_truth": ""}),
                "extra_info": sample.get("extra_info", {}),
                "max_turns": self.max_turns,
            }

            # Create environment for this task (local to this call)
            env = AllocatedCodeEnv(
                manager_url=self.manager_url,
                max_turns=self.max_turns,
                output_dir=self.output_dir,
                time_out=self.time_out,
            )

            # Initialize environment
            conversation, _ = env.init(conversation, **extras)
            container_id = env.tool_group.allocated_container

            # Guard: allocation must have produced a container id. With the
            # retry+propagate fix in env.py this should only be reachable if the
            # pool was genuinely exhausted for the whole retry budget; raise a
            # clear error instead of crashing on f"container_{None:03d}".
            if container_id is None:
                raise RuntimeError(
                    "Container allocation failed (manager returned no container "
                    "after retries) — pool exhausted or manager unreachable."
                )

            # Move any stale submission.csv out of the way before the run
            container_dir = os.path.join(self.submission_dir, f"container_{container_id:03d}")
            submission_file = os.path.join(container_dir, "submission.csv")
            if os.path.exists(submission_file):
                stale_path = os.path.join(container_dir, "_stale_submission.csv")
                os.rename(submission_file, stale_path)

            total_tokens = 0
            final_answer = ""
            actual_turns = 0
            # Reset backend usage counters for this trajectory
            if hasattr(self.backend_instance, 'reset_usage'):
                self.backend_instance.reset_usage()
            # Store partial state for timeout recovery
            self._partial_trajectory = trajectory
            self._partial_conversation = conversation
            self._partial_turns = 0

            repeat_count = 0
            last_obs = None

            for turn in range(self.max_turns):
                try:
                    # Generate response
                    response = self.backend_instance.generate(conversation)

                    # Step environment
                    step_start = time.time()
                    step_output = env.step(response)
                    step_time = time.time() - step_start

                    # Count tokens (approximate)
                    total_tokens += len(response.split())
                    actual_turns = turn + 1
                    self._partial_turns = actual_turns
                    self._partial_trajectory = trajectory
                    self._partial_conversation = conversation

                    # Update conversation with assistant response and new observations
                    conversation.append({
                        "role": "assistant",
                        "content": step_output.get("postprocessed_action", response),
                    })
                    reasoning = getattr(self.backend_instance, '_last_reasoning_content', '') or ''
                    trajectory = self.append_traj(trajectory, turn, "assistant", response, step_output.get("done", False), step_output.get("reward", 0.0), step_time, reasoning_content=reasoning)

                    if step_output["observations"]:
                        obs_content = step_output["observations"][0]["content"]
                        conversation.extend(step_output["observations"])
                        trajectory = self.append_traj(trajectory, turn, "user", obs_content, step_output.get("done", False), step_output.get("reward", 0.0), step_time)
                        # Early-stop on repeated identical observations (format loop)
                        if obs_content == last_obs:
                            repeat_count += 1
                            if repeat_count >= 20:
                                print(f"  [early-stop] Same observation repeated {repeat_count} times, aborting")
                                break
                        else:
                            repeat_count = 0
                            last_obs = obs_content
                    else:
                        # no observation means that there is no new code produced, we are reaching the end.
                        step_output["done"] = True
                    # Check if task is complete
                    if step_output["done"]:
                        final_answer = step_output["metadata"].get("final_answer", response)
                        break

                except Exception as step_err:
                    error_msg = f"Turn {turn + 1} failed: {step_err}"
                    # Add error to conversation for recovery
                    conversation.append({
                        "role": "user",
                        "content": f"Error: {error_msg}. Please try a different approach.",
                    })
                    # Record error step in trajectory
                    trajectory = self.append_traj(trajectory, turn, "user", error_msg, False, 0.0, 0.0)
                    continue

            # Save prediction if we have an answer
            if final_answer:
                prefix = sample.get("extra_info", {}).get("id", "temp")
                env.save_prediction(final_answer, filename_prefix=prefix)

            execution_time = time.time() - start_time
            submission_path = ""
            print(f"{submission_file=}")
            success = False
            if os.path.exists(submission_file):
                success = True
                challenge_name = sample.get("extra_info", {}).get("challenge_name", "")
                run_id = sample.get("extra_info", {}).get("id", "unknown")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_name = f"{run_id}_{container_id}_{timestamp}_submission.csv"
                unique_path = os.path.join(container_dir, unique_name)
                os.rename(submission_file, unique_path)
                print(f"Submission saved: {unique_path}")
                submission_path = unique_path


            # Deallocate container (restarts it, clears workspace + submission files)
            env.close()

            # Get real usage from backend if available
            usage = {}
            if hasattr(self.backend_instance, 'get_usage'):
                usage = self.backend_instance.get_usage()

            return {
                "solution": submission_path,
                "success": success,
                "turns": actual_turns,
                "error": None,
                "metadata": {
                    "model": self.model,
                    "backend": self.backend,
                    "dspredict": True,
                    "max_turns": self.max_turns,
                    "total_tokens": usage.get("total_tokens", total_tokens),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "cost_usd": usage.get("total_cost_usd", 0.0),
                    "llm_calls": usage.get("total_calls", 0),
                    "execution_time": execution_time,
                    "conversation_length": len(conversation),
                },
                "conversation": conversation,
                "trajectory": trajectory,
                "raw_result": {
                    "prediction": submission_path,
                    "turns": actual_turns,
                    "total_tokens": usage.get("total_tokens", total_tokens),
                },
            }

        except Exception as e:
            try:
                env.close()
            except Exception:
                pass
            execution_time = time.time() - start_time
            error_trace = traceback.format_exc()
            print(f"Error in agent: {error_trace}")

            usage = {}
            if hasattr(self.backend_instance, 'get_usage'):
                usage = self.backend_instance.get_usage()

            return {
                "solution": "",
                "success": False,
                "turns": getattr(self, '_partial_turns', 0),
                "error": str(e),
                "metadata": {
                    "model": self.model,
                    "backend": self.backend,
                    "dspredict": True,
                    "max_turns": self.max_turns,
                    "total_tokens": usage.get("total_tokens", 0),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "cost_usd": usage.get("total_cost_usd", 0.0),
                    "llm_calls": usage.get("total_calls", 0),
                    "execution_time": execution_time,
                    "error_trace": error_trace,
                },
                "conversation": getattr(self, '_partial_conversation', []),
                "trajectory": trajectory,
                "raw_result": None,
            }
        finally:
            if "env" in locals():
                env.close()

    def append_traj(self, trajectory, turn, role, content, done, reward, step_time, reasoning_content=""):
        step = {
            "turn": turn,
            "role": role,
            "content": content,
            "done": done,
            "reward": reward,
            "step_time": step_time,
        }
        if reasoning_content:
            step["reasoning_content"] = reasoning_content
        trajectory.append(step)
        return trajectory
