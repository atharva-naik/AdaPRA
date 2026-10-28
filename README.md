# AdaPRA: Process Reward Agents with Adaptive Rubrics for Data Science

**Atharva Naik, Vivek Chari, Pol Llado, Daniel Fried, Carolyn Rosé, Benjamin Van Durme, Sam Thomson, Justin Svegliato**
Carnegie Mellon University · Microsoft

[Paper (arXiv:2610.33245)](https://arxiv.org/abs/2610.33245) · [Project page](https://atharva-naik.github.io/AdaPRA/) · Run outputs: [AdaPRA-runs on Hugging Face](https://huggingface.co/datasets/atharva-naik/AdaPRA-runs) (being uploaded)

AdaPRA supervises a data science agent with a *process reward agent* (PRA). A persistent harness keeps the agent's workspace across repeated attempts on a task. Every few turns, the PRA reviews the trajectory and workspace against rubrics, writing and running its own checks against the agent's files, and reports only the relevant failing rubrics back to the agent. The seed rubrics are discovered offline from past agent runs, and the PRA adapts them to each task. On DSGym's DSPredict-Easy and DSPredict-Hard, AdaPRA raises the mean Kaggle percentile of Qwen3-Coder-Next (80B) by 24.5 and 14.8 points.

This repository contains the code behind the paper's reported results:

- the persistent multi-attempt harness;
- the PRA with adaptive rubrics (AdaPRA) and the DataPRM baseline;
- the offline rubric-discovery pipeline and the seed rubrics it produced;
- the run configurations for every reported row;
- the scoring and paper-table scripts.

It builds on [DSGym](https://github.com/fannie1208/DSGym). Run outputs (trajectories, PRA reviews, and per-attempt predictions) are released separately on Hugging Face as [AdaPRA-runs](https://huggingface.co/datasets/atharva-naik/AdaPRA-runs).

## Results

Main results from the paper (Tables 1 and 2), with Qwen3-Coder-Next (80B) as the base agent, Claude Sonnet 4.6 as the PRA, and five attempts per task. **Pct.** is the mean Kaggle leaderboard percentile (private leaderboard where one exists, else public); **Valid**, **>Median**, and **Medal** are the shares of tasks with a scored submission, a score above the leaderboard median, and a medal. All rates use a fixed denominator of every task in the benchmark. See the paper for the definitions of the best-attempt and per-attempt aggregates and for the external references.

**DSPredict-Easy (38 tasks).** Columns 2–5: best attempt; columns 6–9: per attempt (mean).

| Method | Pct. | Valid | >Median | Medal | Pct. | Valid | >Median | Medal |
|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-Next + DSGym | 44.5 | 76.3 | 50.0 | 2.6 | 47.5 | 41.0 | 46.6 | 1.1 |
| Qwen3-Coder-Next + Ours | 60.8 | 100.0 | 71.1 | **7.9** | 51.3 | 84.5 | 53.9 | 2.6 |
| *Process supervision on top of Our Harness* | | | | | | | | |
| + DataPRM | 54.6 | 100.0 | 65.8 | 5.3 | 44.8 | 100.0 | 44.6 | 2.1 |
| + PRA + adaptive rubrics (AdaPRA) | **69.0** | 100.0 | **86.8** | **7.9** | **61.0** | 100.0 | 73.2 | 3.7 |
| &nbsp;&nbsp;− adaptive rubrics | 67.3 | 100.0 | 86.8 | **7.9** | 58.8 | 99.5 | 72.5 | **4.2** |
| &nbsp;&nbsp;− PRA | 63.0 | 100.0 | 86.8 | 2.6 | 57.8 | 100.0 | 73.7 | 2.1 |
| *Single-attempt external references (reported by DSGym)* | | | | | | | | |
| Qwen3-Coder-480B + DSGym | – | – | – | – | 37.1 | 86.5 | 33.3 | – |
| GPT-5.1 (high) + DSGym | – | – | – | – | 60.4 | 100.0 | **75.0** | – |

**DSPredict-Hard (54 tasks).** Columns 2–5: best attempt; columns 6–9: per attempt (mean).

| Method | Pct. | Valid | >Median | Medal | Pct. | Valid | >Median | Medal |
|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-Next + DSGym | 21.3 | 68.5 | 11.1 | 3.7 | 25.2 | 36.3 | 11.8 | 3.0 |
| Qwen3-Coder-Next + Ours | 25.5 | 81.5 | 18.5 | 5.6 | 26.2 | 64.8 | 14.4 | 3.7 |
| *Process supervision on top of Our Harness* | | | | | | | | |
| + DataPRM | 23.1 | 88.9 | 14.8 | 3.7 | 21.3 | 79.3 | 12.8 | 3.7 |
| + PRA + adaptive rubrics (AdaPRA) | **36.1** | **90.7** | **33.3** | 5.6 | **32.1** | 86.3 | **24.1** | 4.1 |
| &nbsp;&nbsp;− adaptive rubrics | 31.8 | 88.9 | 22.2 | 5.6 | 30.2 | **87.8** | 17.5 | 4.4 |
| &nbsp;&nbsp;− PRA | 31.8 | **90.7** | 25.9 | **9.3** | 30.9 | 83.0 | 19.8 | **8.1** |
| *Single-attempt external references (reported by DSGym)* | | | | | | | | |
| Qwen3-Coder-480B + DSGym | – | – | – | – | – | 66.7 | 5.9 | 2.4 |
| GPT-5.1 (high) + DSGym | – | – | – | – | – | 85.7 | 14.3 | 4.8 |

## Layout

| Path | Contents |
|---|---|
| `dsgym/harness/openhands_harness_v4.py` | Persistent harness: one long-lived container per worker, workspace kept across a task's attempts |
| `dsgym/harness/agent_runner.py` | In-container entry point for the base agent and the PRA (OpenHands SDK) |
| `dsgym/harness/prm_reviewer.py` | The PRA: rubric scoring with code execution (`operationalization: dynamic`) or reading only (`static`), and per-task rubric adaptation (`evolve`) |
| `dsgym/harness/prm_dataprm.py`, `prm_agent.py` | DataPRM baseline integration |
| `dsgym/harness/prompts_dir_v4*/` | Base-agent system prompts (`_twrap` is used by the supervised DSPredict-Hard runs) |
| `dsgym/adapra/rubric_sets/` | Seed rubrics: `dspredict_discovered.json` (17 discovered rubrics) and `dspredict_discovered_xattempt.json` (the 19-rubric set used in all supervised runs) |
| `scripts/adapra/extract_behaviors.py`, `cluster_to_rubrics.py`, `dropped_behavior_coverage.py` | Offline rubric discovery (stage 1: behavior extraction; stage 2: clustering into rubrics; coverage check of the behaviors dropped in stage 2) |
| `scripts/adapra/collect_trajectories_v4.py` | Launches a run in our harness |
| `scripts/adapra/collect_trajectories.py`, `dsgym/agents/`, `executors/manager/`, `executors/container_images/kaggle_image/`, `executors/generate_apptainer.py`, `executors/generate_compose.py` | Stock DSGym harness baseline: ReAct agent, executor manager, and Jupyter executor containers |
| `scripts/adapra/score_submissions.py` | Kaggle scoring of the final submission of each attempt |
| `scripts/adapra/_paper_row_xattempt.py`, `_paper_row_hard.py`, `paper_easy_peratt_medal.py`, `paper_bo1_table.py`, `paper_cost_table.py`, `paper_significance.py`, `turns_table.py`, `rubric_failure_transfer.py`, `dataprm_behavior_stats.py`, `plot_rubric_store_ops*.py` | Paper tables and figures |
| `configs/adapra/` | One config per reported condition (below) |
| `executors/container_images/v4_runner/` | Runner image (Docker or Apptainer) |
| `data/task/dspredict/`, `data/DSPredict_Offline_Leaderboard/` | DSPredict task metadata and offline leaderboards (from DSGym) |

## Setup

```bash
pip install -e .                       # Python 3.12
bash executors/container_images/v4_runner/build.sh       # Docker image, or
bash executors/container_images/v4_runner/build_sif.sh   # Apptainer SIF (SIF_OUT=<path>)
```

- **Data.** Competition data go under `data/data/dspredict-easy` and `data/data/dspredict-hard`, following DSGym. `scripts/adapra/stage_dspredict_hard.py` downloads the Hard competitions with the Kaggle API.
- **Models.** The base agent and the PRA are called through LiteLLM. Supply AWS Bedrock credentials through the standard AWS environment variables or `~/.aws`. The harness exposes them to the container only at `/etc/aws`.
- **Scoring.** Kaggle scoring needs `~/.kaggle/kaggle.json`.
- **Paths.** Paths of the form `/path/to/scratch/...` in the configs and scripts are placeholders for local storage. Set `sif_image`, `workspace_root`, and `csv_artifact_root` before running.

## Configurations

All runs use Qwen3-Coder-Next as the base agent and Claude Sonnet 4.6 as the PRA unless noted.

| Condition | DSPredict-Easy | DSPredict-Hard |
|---|---|---|
| DSGym harness (stock baseline) | `agent_dsgym_stock_easy.yaml` (see [Stock DSGym baseline](#stock-dsgym-baseline)) | `agent_dsgym_stock_hard.yaml` |
| Harness only | `harness_v4_apptainer.yaml` | `harness_v4_apptainer.yaml` with `--data-path data/data/dspredict-hard --gpu --task-timeout 14400 --max-iterations-per-attempt 300` and the CUDA runner image |
| + DataPRM | `harness_v4_dataprm_pull_xattempt_fulltau.yaml` | `harness_v4_dataprm_xattempt_hard_{a,b}_fulltau.yaml` (two task halves, `task_splits/`) |
| AdaPRA | `harness_v4_reviewer_dyn_evolve_pull_xattempt.yaml` | `harness_v4_reviewer_dyn_evolve_xattempt_hard.yaml` |
| − adaptive rubrics | `harness_v4_reviewer_dyn_frozen_xattempt.yaml` | `harness_v4_reviewer_dyn_frozen_xattempt_hard.yaml` |
| − PRA | `harness_v4_reviewer_dyn_evolve_nopra_pull_xattempt.yaml` | `harness_v4_reviewer_dyn_evolve_nopra_xattempt_hard.yaml` |
| Cadence c=5 / c=20 (one attempt) | `..._pull_xattempt_c5_bo1.yaml`, `..._c20_bo1.yaml` | |
| DeepSeek-V3.2 base agent (one attempt) | `harness_v4_noprm_deepseek_v32_bo1.yaml`, `harness_v4_reviewer_dyn_evolve_pull_xattempt_deepseek_v32_bo1.yaml` | |

## Running

```bash
PYTHONPATH=. python -u scripts/adapra/collect_trajectories_v4.py \
  --dataset dspredict-easy \
  --agent-config configs/adapra/harness_v4_reviewer_dyn_evolve_pull_xattempt.yaml \
  --output-dir outputs/adapra/trajectories_v4/dsp-easy-rev-dyn-evolve-pull-xattempt \
  --workers 4
```

For DSPredict-Hard, add `--dataset dspredict-hard --gpu --sif-image <CUDA runner SIF>`. For a task subset, add `--task-id <comma-separated ids>`. Each attempt is written to `<output-dir>/<split>/<task>/attempt_<i>.json`. PRA calls are saved under `prm_reviews_attempt_<i>/`, and the per-task rubric store under `reviewer_rubric_store/`.

## Stock DSGym baseline

The "Qwen3-Coder-Next + DSGym" rows run DSGym's own harness: a ReAct agent (`DSPredictReActAgent`, with DSGym's DSPredict system prompt) that executes code in Jupyter kernels served by executor containers behind a manager. The workspace is reset between attempts.

```bash
# 1. Build the executor image (Apptainer; a Dockerfile is also provided)
cd executors/container_images/kaggle_image && apptainer build kaggle_executor.sif kaggle_executor.def && cd -

# 2. Generate and launch 8 executor instances with the benchmark data mounted at /data
python executors/generate_apptainer.py --num-containers 8 \
  --sif-image executors/container_images/kaggle_image/kaggle_executor.sif \
  --mountable-volumes data/data/dspredict-easy --mountable-volume-saveto <submissions-dir> \
  --output-dir executors/apptainer_config
bash executors/apptainer_config/launch_instances.sh

# 3. Start the manager
pip install -r executors/manager/requirements.txt
PORT=5001 CONTAINER_CONFIG_PATH=executors/apptainer_config/container_config.json python executors/manager/main.py &

# 4. Collect five attempts per task
python scripts/adapra/collect_trajectories.py --dataset dspredict-easy \
  --agent-config configs/adapra/agent_dsgym_stock_easy.yaml \
  --output-dir outputs/adapra/bon/trajectories/dspredict-easy-bon5 \
  --manager-url http://localhost:5001 --submission-dir <submissions-dir> \
  --runs-per-task 5 --num-workers 8
```

For DSPredict-Hard, mount `data/data/dspredict-hard`, pass `--gpu-ids` to `generate_apptainer.py` for GPU executors and `--execution-timeout 7200`, and run the collector with `--dataset dspredict-hard --agent-config configs/adapra/agent_dsgym_stock_hard.yaml --task-timeout 14400`. `--runs-per-task` must be passed on the command line; the collector does not read it from the config. These runs write flat trajectory files (`<task>__<model>__<k>.json`); score them with `score_submissions.py`, then compute the Easy row with `paper_easy_peratt_medal.py <dir> flat` and the Hard row with `_paper_row_hard.py NAME <dir>:flat`.

## Scoring and tables

```bash
python scripts/adapra/score_submissions.py --score-selected \
  --trajectory-dir <output-dir>/dspredict-easy \
  --leaderboard-stats data/DSPredict_Offline_Leaderboard/easy_leaderboard_stats_2026-02-01.json \
  --data-path data/data/dspredict-easy --csv-artifact-root <csv_artifact_root> --delay 5

python scripts/adapra/_paper_row_xattempt.py <output-dir>/dspredict-easy   # Easy row (Table 1)
python scripts/adapra/paper_easy_peratt_medal.py <output-dir>/dspredict-easy  # Easy row with per-attempt medal rate
python scripts/adapra/_paper_row_hard.py NAME <output-dir>/dspredict-hard  # Hard row (Table 2); append :flat for stock DSGym runs
python scripts/adapra/paper_bo1_table.py [--set cadence|base-agent]  # single-attempt tables
python scripts/adapra/paper_cost_table.py --latex                           # API cost table
python scripts/adapra/paper_significance.py                                 # paired sign, Wilcoxon and permutation tests
python scripts/adapra/dataprm_behavior_stats.py <DataPRM output-dir>/dspredict-easy
```

`score_submissions.py` checks the shape of each CSV before submitting it, and it skips CSVs it has already scored, matched by MD5. Each row script prints the best-of-5 and per-attempt metrics over a fixed denominator of 38 (Easy) or 54 (Hard) tasks. The row scripts print Valid before Pct.

## Offline rubric discovery

```bash
python scripts/adapra/extract_behaviors.py --help    # stage 1: behaviors from trajectory, workspace and summary views
python scripts/adapra/cluster_to_rubrics.py --help   # stage 2: cluster behaviors into rubrics
python scripts/adapra/dropped_behavior_coverage.py   # check: rubric coverage of the behaviors dropped in stage 2
```

Discovery runs over the harness-only DSPredict-Easy trajectories. The resulting rubric set is `dsgym/adapra/rubric_sets/dspredict_discovered.json`. The supervised runs use its cross-attempt variant, `dspredict_discovered_xattempt.json`, which reframes four rubrics over all of a task's attempts and adds two hand-written cross-attempt rubrics (`over_optimization`, `single_model_class_hedging`).

## Citation

```bibtex
@misc{naik2026adapra,
  title         = {AdaPRA: Process Reward Agents with Adaptive Rubrics for Data Science},
  author        = {Naik, Atharva and Chari, Vivek and Llado, Pol and Fried, Daniel and
                   Ros{\'e}, Carolyn and Van Durme, Benjamin and Thomson, Sam and Svegliato, Justin},
  year          = {2026},
  eprint        = {2610.33245},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2610.33245}
}
```
