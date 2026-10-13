"""
DSPredict dataset loader.
"""

import os
import json
from typing import List, Dict, Any, Optional
from collections import Counter
from statistics import mean

from ..base import BaseDataset
from ..registry import register_dataset
from ..utils import apply_limit_and_start, validate_file_exists, create_standard_task
from ..prompts import SYSTEM_PROMPT_DSPREDICT
from dsgym.datasets.config import get_task_path, RAW_DATA_DIR
from .kaggle_downloader import KaggleChallengeDownloader

DSPREDICT_INSTRUCTIONS = """**INSTRUCTIONS:**
1. Load and explore the **training** and **test** datasets using Python (use the dataset folder location provided).
2. Perform **data preprocessing** (handling missing values, encoding, scaling, feature engineering) and **exploratory analysis** to understand distributions, correlations, and relationships between variables.
3. Where simple preprocessing and baseline models are insufficient, attempt more advanced approaches such as:
   * Model selection (e.g., tree-based models, linear models, neural networks)
   * Cross-validation and hyperparameter tuning
   * Dimensionality reduction, feature selection, or ensembling
   * Robustness checks or combining datasets if useful
4. Use the training data to build a model, evaluate it with proper validation, and then generate **predictions for the test data**.
5. Do one step at a time. Explore and validate thoroughly before moving on to model training and submission.
6. When doing exploration and data analysis, print the results in a clear and concise way.
7. Do not use plotting libraries (assume you cannot view plots). Use text-based summaries and statistics instead.
8. When workflow tags or competition-specific guidelines are provided, you should follow them closely.
9. Only produce the **final submission and answer** when you have enough evidence and validation to support your approach.

When you finished training the best model, you should generate the final submission:

1. Use the best model to generate predictions for the test data located at the path shown above.
2. Save predictions in the required **`submission.csv` format** for the competition at /submission/submission.csv.
3. Provide a concise summary of your approach in the format: <answer>your final summary</answer>
"""


DSPREDICT_DATACENTRIC_INSTRUCTIONS = """**INSTRUCTIONS (DATA-CENTRIC competition — read carefully):**
This is a DATA-CENTRIC competition. Your submission is NOT predictions — it is a
TRAINING DATASET. The competition organizers will train a FIXED model (see the
model code in the description) on the rows you submit, then use it to predict a
HIDDEN test set; your score is the error of those predictions.

1. Read the competition description and the provided `sample_submission.csv`
   CAREFULLY to learn the EXACT required submission schema: the same COLUMN NAMES
   and types as the sample (these must match the hidden test set's features — do
   NOT invent or rename feature columns), plus the `target` column.
2. Explore any data you are given. Your job is to CONSTRUCT/curate the best
   possible training dataset (rows of features + target) so the fixed model
   generalizes to the hidden test set — e.g. clean/augment/select rows, engineer
   values, but keep the schema identical to the sample.
3. Do NOT train your own model and submit its predictions — that is the wrong
   format and will error. Submit the DATASET itself.
4. Validate locally by replicating the organizers' pipeline: train the specified
   fixed model on your dataset, evaluate on a held-out split, iterate.
5. Do one step at a time. No plotting libraries; use text summaries.

When ready, write your final TRAINING DATASET (matching the sample_submission
columns exactly) to the submission path, and provide a concise summary in the
format: <answer>your final summary</answer>
"""


def _is_data_centric(description: str) -> bool:
    """A data-centric competition asks you to SUBMIT A DATASET (used to train a
    fixed model), not predictions. Detect from the description wording."""
    d = (description or "").lower()
    signals = (
        "instead of submitting predictions",
        "submit a dataset",
        "your submission will be used as a training dataset",
        "submit a dataset that will be used to train",
        "data-centric",
    )
    return any(s in d for s in signals)


def create_dspredict_prompt(challenge_name: str, description: str, data_paths: Dict[str, List[str]]) -> str:
    dataset_path = data_paths['virtual'][0]

    # Data-centric competitions (e.g. playground-series-s3e21) require submitting
    # a TRAINING DATASET, not predictions. The default prediction-framed
    # instructions directly contradict that and confuse the agent (observed: it
    # correctly identified the data-centric format, then drifted back to
    # train/test + predictions). Use format-appropriate instructions.
    instructions = (DSPREDICT_DATACENTRIC_INSTRUCTIONS
                    if _is_data_centric(description)
                    else DSPREDICT_INSTRUCTIONS)

    user_content = f"""**CHALLENGE NAME: {challenge_name}**

{description}

**DATASET LOCATIONS (use full paths):**
{dataset_path}

{instructions}
"""
    return user_content


@register_dataset("dspredict")
class DSPredictDataset(BaseDataset):
    """DSPredict dataset loader."""
    
    def __init__(self, virtual_data_root: Optional[str] = None, **kwargs):
        """
        Initialize DSPredict dataset.
        
        Args:
            virtual_data_root: Root path for virtual/docker paths (default: "/data")
            **kwargs: Additional configuration
        """
        super().__init__(virtual_data_root=virtual_data_root, **kwargs)
    
    def load(
        self, 
        limit: Optional[int] = None, 
        split: str = "easy", 
        start_index: int = 0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Load DSPredict dataset.
        
        Args:
            limit: Maximum number of samples to load
            split: Dataset split (easy/hard)
            start_index: Starting index for data selection
            **kwargs: Additional loading parameters
            
        Returns:
            List of dataset samples
        """
        if split == "easy":
            dataset_path = str(get_task_path("dspredict") / "easy.json")
        elif split == "hard":
            dataset_path = str(get_task_path("dspredict") / "hard.json")
        elif split == "lite":
            dataset_path = str(get_task_path("dspredict") / "lite.json")
        else:
            dataset_path = str(get_task_path("dspredict") / "hard.json")
        
        validate_file_exists(dataset_path, f"DSPredict {split} dataset")
        
        # Load JSON file (not JSONL)
        with open(dataset_path, 'r', encoding='utf-8') as f:
            items = json.load(f)

        # Apply start_index and limit
        items = apply_limit_and_start(
            items, limit, start_index, random_sample=False,
            random_seed=self.config.get('random_seed', 42)
        )

        samples = []
        downloader = None
        kaggle_data_dir = RAW_DATA_DIR / f"dspredict-{split}"

        for idx, item in enumerate(items):
            docker_path = item['docker_challenge_path']
            
            if docker_path.startswith('/data/'):
                challenge_dir = docker_path.replace('/data/', '')
            else:
                challenge_dir = docker_path.split('/')[-1]
            
            competition_path = kaggle_data_dir / challenge_dir
            # Count files recursively: some competitions (e.g. mens-march-mania-2022)
            # ship their data inside subdirectories, so a top-level-only check would
            # wrongly flag the directory as empty and try to re-download/rmdir it.
            data_files = [f for f in competition_path.rglob('*') if f.is_file()] if competition_path.exists() else []
            if not competition_path.exists() or not data_files:
                if competition_path.exists() and not data_files:
                    print(f"Competition directory exists but is empty, re-downloading: {challenge_dir}")
                    competition_path.rmdir()
                else:
                    print(f"Competition data not found: {challenge_dir}")
                print(f"Attempting to download from Kaggle...")

                if downloader is None:
                    downloader = KaggleChallengeDownloader(download_dir=str(kaggle_data_dir))

                competition_name = item['challenge_name']
                result = downloader.download_competition_data(competition_name)

                if result['status'] == 'failed':
                    raise RuntimeError(
                        f"Failed to download data for '{competition_name}'. "
                        f"Check your Kaggle credentials (KAGGLE_USERNAME, KAGGLE_KEY in .env) "
                        f"and ensure you have accepted the competition rules at: "
                        f"https://www.kaggle.com/competitions/{competition_name}"
                    )

                # Verify files actually landed
                downloaded_files = [f for f in competition_path.iterdir() if f.is_file()]
                if not downloaded_files:
                    raise RuntimeError(
                        f"Download reported success but no files found in {competition_path}. "
                        f"This usually means the Kaggle API returned an empty response — "
                        f"check that you have accepted the rules at: "
                        f"https://www.kaggle.com/competitions/{competition_name}"
                    )

                print(f"Successfully downloaded {competition_name} ({len(downloaded_files)} files)")

            from ..utils import construct_data_paths
            data_paths = construct_data_paths(
                relative_paths=[challenge_dir],
                dataset_name=f"dspredict-{split}",
                data_root=RAW_DATA_DIR,
                virtual_data_root=self.virtual_data_root
            )
            
            user_content = create_dspredict_prompt(
                challenge_name=item['challenge_name'],
                description=item['description'],
                data_paths=data_paths
            )
            
            extra_info = {
                "challenge_name": item['challenge_name'],
                "docker_challenge_path": item['docker_challenge_path'],
                "data_files": data_paths,
                "question": user_content,
                'index': start_index + idx,
                'source': 'dspredict',
                'metadata_id': item['challenge_name'],
                'query_id': item['challenge_name'],
                'id': item['challenge_name']
            }
            
            # Create standardized sample
            standard_sample = create_standard_task(
                prompt_content=user_content,
                ground_truth="",  # DSPredict competitions don't have ground truth
                extra_info=extra_info,
                system_prompt=SYSTEM_PROMPT_DSPREDICT
            )
            
            samples.append(standard_sample)
        
        self._samples = samples
        return samples
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get a single sample by index."""
        if self._samples is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        
        if index < 0 or index >= len(self._samples):
            raise IndexError(f"Sample index {index} out of range [0, {len(self._samples)})")
        
        return self._samples[index]
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get DSPredict dataset metadata."""
        if self._metadata is None:
            self._metadata = {
                'name': 'DSPredict',
                'description': 'DSPredict competition challenges for machine learning',
                'format': 'json',
                'splits': ['easy', 'hard'],
                'fields': ['challenge_name', 'description', 'docker_challenge_path'],
                'source': 'dspredict'
            }
        
        return self._metadata

    def get_metrics(self) -> List[str]:
        """Get metrics for DSPredict dataset."""
        return ["dspredict_submission"]


    def print_dspredict_results_overview(self, sample_results):
        """Aggregate Kaggle metrics across sample results and print a summary."""
        def _get_metric_dict(sample, metric_name):
            """Extract metric details as a dictionary."""
            metrics = getattr(sample, "metrics", None)
            if metrics is None and isinstance(sample, dict):
                metrics = sample.get("metrics", {})
            metrics = metrics or {}
            metric = metrics.get(metric_name)
            if metric is None:
                return None
            if hasattr(metric, "to_dict"):
                return metric.to_dict()
            return metric
        def _safe_float(value):
            """Convert value to float when possible."""
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                value = value.strip()
                if not value or value.lower() in {"nan", "n/a"}:
                    return None
                try:
                    return float(value)
                except ValueError:
                    return None
            return None
        def _format_pct(numerator, denominator):
            if denominator == 0:
                return "0.0%"
            return f"{(numerator / denominator) * 100:.1f}%"

        total_competitions = len(sample_results)
        total_valid_submissions = 0
        total_above_median = 0
        medal_counts: Counter[str] = Counter()
        leaderboard_data = {
            "public": {
                "competitions": 0,
                "valid_submissions": 0,
                "scores": [],
                "percentiles": [],
                "above_median_total": 0,
                "above_median_count": 0,
                "medals": Counter(),
            },
            "private": {
                "competitions": 0,
                "valid_submissions": 0,
                "scores": [],
                "percentiles": [],
                "above_median_total": 0,
                "above_median_count": 0,
                "medals": Counter(),
            },
        }
        status_counts: Counter[str] = Counter()

        for sample in sample_results:
            metric_info = _get_metric_dict(sample, "kaggle_submission")
            if not metric_info:
                continue

            details = metric_info.get("details") or {}
            status = details.get("status")
            if status:
                status_counts[status] += 1

            leaderboard_stats = details.get("leaderboard_stats") or {}
            for board in ("public", "private"):
                scores_snapshot = leaderboard_stats.get(f"{board}_scores")
                if not scores_snapshot:
                    continue

                board_data = leaderboard_data[board]
                board_data["competitions"] += 1

                score_value = _safe_float(details.get(f"{board}_score"))
                if score_value is not None:
                    board_data["scores"].append(score_value)
                    board_data["valid_submissions"] += 1

                percentile_value = _safe_float(details.get(f"{board}_percentile"))
                if percentile_value is not None:
                    board_data["percentiles"].append(percentile_value)

                above_median_value = details.get(f"{board}_above_median")
                if isinstance(above_median_value, bool):
                    board_data["above_median_total"] += 1
                    if above_median_value:
                        board_data["above_median_count"] += 1

                medal_value = details.get(f"{board}_medal")
                if isinstance(medal_value, str):
                    medal = medal_value.strip().lower()
                    if medal in {"gold", "silver", "bronze"}:
                        medal_counts[medal] += 1
                        board_data["medals"][medal] += 1

        print("\n📝 Kaggle Competition Overview")
        print("-" * 40)
        print(f"Total competitions evaluated : {total_competitions}")
        if status_counts:
            print("Submission status breakdown  :")
            for status, count in sorted(status_counts.items()):
                print(f"  - {status}: {count}")

        for board, label in (("public", "Public"), ("private", "Private")):
            board_data = leaderboard_data[board]
            competitions = board_data["competitions"]
            print(f"\n=== {label} Leaderboard ===")
            if competitions == 0:
                print("  No leaderboard data available.")
                continue

            print(
                f"📊 Coverage               : {competitions} / {total_competitions}  "
                f"({_format_pct(competitions, total_competitions)})"
            )
            print(
                f"✅ Valid submissions       : {board_data['valid_submissions']} / {competitions}  "
                f"({_format_pct(board_data['valid_submissions'], competitions)})"
            )

            if board_data["percentiles"]:
                percentiles = board_data["percentiles"]
                print(f"🎯 Avg percentile          : {mean(percentiles):.1f}")

            if board_data["above_median_total"]:
                above = board_data["above_median_count"]
                total = board_data["above_median_total"]
                print(
                    f"⬆️ Above median            : {above} / {competitions}  "
                    f"({_format_pct(above, competitions)})"
                )

            total_board_medals = sum(board_data["medals"].values())
            if competitions:
                print(
                    f"🏅 Medals                 : {total_board_medals} / {competitions}  "
                    f"({_format_pct(total_board_medals, competitions)})"
                )
                print(
                    f"  🥇 Gold                 : {board_data['medals'].get('gold', 0)}"
                )
                print(
                    f"  🥈 Silver               : {board_data['medals'].get('silver', 0)}"
                )
                print(
                    f"  🥉 Bronze               : {board_data['medals'].get('bronze', 0)}"
                )