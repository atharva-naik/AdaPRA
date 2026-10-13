#!/usr/bin/env python3
"""Per-run breakdown of rubric-store operations for the four adaptive-rubric runs
(AdaPRA and AdaPRA - PRA on DSPredict-Easy and DSPredict-Hard). Companion to
plot_rubric_store_ops.py, which plots the aggregate rates; same RUNS and the same
denominator (tasks that have an op log).

Outputs (our_paper/figures/):
  rubric_store_ops_by_attempt.pdf  2x2, one panel per run: add/update/retire
                                   counts per attempt (stacked)
  rubric_adds_per_task.pdf         share of tasks with 0 / 1 / 2 / 3+ added rubrics
Also prints the per-run theme counts of the added rubrics (THEME below), used for
the appendix table, with example rubric names.

THEME was assigned by hand after reading the name and description of each of the
177 added rubrics (rubric_store.json), keyed by rubric name:
  val   validation and leakage: CV / metric implementation, target leakage,
        integrity of OOF predictions used for model or ensemble selection
  pipe  train-test and submission consistency: the submitted predictions come
        from the validated pipeline and are mapped to the right IDs, format, path
  pred  prediction sanity: dispersion, underfitting, degenerate or constant output
  model features and modeling choices: task-specific features, targets, data use
  bug   crashing code: runtime errors, missing imports, API misuse

Usage: python plot_rubric_store_ops_by_run.py [--outdir our_paper/figures]
"""
import argparse
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_rubric_store_ops import N_ATTEMPTS, OPS, REPO, RUNS

THEMES = [("val", "Validation and leakage"),
          ("pipe", "Train-test and submission consistency"),
          ("pred", "Prediction sanity"),
          ("model", "Features and modeling choices"),
          ("bug", "Crashing code")]

THEME = {
    # validation and leakage
    **dict.fromkeys([
        "calibration_not_insample", "calibration_insample_leakage", "genuine_ensemble_blending",
        "ensemble_uses_valid_oof", "cv_scores_append_not_overwrite", "target_encoding_fold_isolation",
        "cv_fold_stability", "ensemble_oof_integrity", "ensemble_oof_loading_bug",
        "ensemble_oof_validity", "oof_not_fake_targets", "oof_subsample_coverage",
        "oof_mcc_target_mismatch", "metric_unit_consistency", "target_encoding_leakage",
        "target_leakage_in_features", "contributions_leakage_guard", "filtered_species_cv_optimism",
        "per_phone_cv_metric", "target_not_in_features", "ensemble_oof_pipeline_correctness",
        "cv_label_subset_inflation", "batch_boundary_feature_leakage", "train_test_feature_mismatch",
        "oof_test_domain_gap", "test_cell_type_cv_tracking", "cv_clicks_ground_truth",
        "lgbm_cv_session_mismatch", "lgbm_gt_timestamp_leakage", "small_n_cv_score_reliability",
        "camera_geometry_leakage", "perfile_ranking_metric_computation", "oof_file_integrity",
        "oof_id_alignment", "oof_lossless_storage", "oof_is_fold_predictions_not_full_model",
        "group_aware_cv_splitting", "ensemble_oof_alignment_bug", "ensemble_cv_honesty",
        "stacking_requires_oof_base_predictions", "gnss_baseline_feature_leakage",
        "batch_boundary_feature_integrity", "ensemble_oof_merge_bug", "ensemble_script_correctness",
        "label_leakage_in_rf_cv", "label_leakage_via_box_features"], "val"),
    # train-test and submission consistency
    **dict.fromkeys([
        "stacking_test_aggregation_correctness", "oof_sub_distribution_divergence",
        "lag_feature_test_consistency", "submission_file_format", "submission_row_ordering_mismatch",
        "data_centric_schema_compliance", "ensemble_oof_test_model_alignment", "submission_path_bug",
        "train_test_preprocessing_consistency", "prediction_label_inversion",
        "ensemble_oof_test_formula_consistency", "submission_path_correctness",
        "mapk_top3_submission_format", "mlp_submission_decode_bug",
        "fold_ensemble_prediction_aggregation", "single_fold_test_prediction",
        "final_model_rounds_mismatch", "test_aggregation_correctness", "phone_model_key_mismatch",
        "submission_ranking_order", "oof_test_format_consistency",
        "test_prediction_pipeline_scalability", "submission_hierarchy_coverage",
        "submission_pipeline_bugs", "correct_submission_stage", "test_feature_pipeline_integrity",
        "train_test_label_encoding_collapse", "cv_final_model_consistency", "test_feature_degradation",
        "submission_vs_oof_consistency", "rle_relative_offset_correctness",
        "test_prediction_fold_averaging_error", "per_type_test_pred_aggregation",
        "instance_segmentation_output_alignment", "stacking_feature_consistency",
        "xgb_tfidf_train_test_vocabulary_mismatch", "predict_proba_not_predict_for_submission",
        "test_file_matches_competition_format", "submission_coverage_completeness",
        "wrong_submission_ids", "submission_class_mapping_bug", "wrong_submission_path",
        "test_predictions_pipeline_mismatch", "stage2_seeds_data_source",
        "stage2_data_cutoff_alignment", "train_test_feature_consistency",
        "submission_predict_vs_predict_proba", "submission_column_order_swap",
        "test_prediction_clickid_alignment", "ir_submission_column_names", "ranking_index_alignment",
        "rle_prediction_scale_match", "submission_id_format_bug", "test_inference_pipeline_complete"],
        "pipe"),
    # prediction sanity
    **dict.fromkeys([
        "prediction_underdispersion", "residual_signal_check", "prediction_dispersion_health",
        "subgroup_error_balance", "model_underfitting_signal", "test_prediction_distribution_sanity",
        "map3_above_random_baseline", "degenerate_prediction_detection", "residual_feature_signal",
        "prediction_dispersion_vs_target", "non_negative_predictions", "instance_segmentation_quality",
        "submission_prediction_specificity", "median_quantile_calibration", "model_pred_scale_sanity",
        "prediction_scale_consistency", "ir_ranking_topic_diversity", "residual_signal_and_dispersion",
        "model_beats_naive_heuristic", "constant_prediction_across_locations",
        "uniform_prediction_antipattern", "noisy_fragmented_segmentation_masks",
        "near_constant_submission_predictions"], "pred"),
    # features and modeling choices
    **dict.fromkeys([
        "datacentric_cleaning_effectiveness", "bidirectional_outlier_removal",
        "preprocessing_order_correctness", "feature_engineering_wasted", "discrete_target_rounding",
        "log_transform_for_skewed_price", "feature_dtype_integrity",
        "missing_value_indicator_for_key_features", "engineered_features_dropped",
        "missing_encoded_categoricals", "building_level_target_encoding", "outlier_churn_handling",
        "image_download_feasibility", "optimal_k_selection", "primary_modality_usage",
        "pixel_segmentation_vs_image_classification", "embedding_integration_validity",
        "seed_feature_incorporation", "signal_continuity_in_feature_extraction",
        "constant_placeholder_feature", "coupling_type_excluded_from_features",
        "transaction_data_full_coverage", "cnn_forward_classifier_bug", "double_normalization_bug",
        "untrained_cnn_embeddings", "catboost_feature_zero_bug", "constant_target_training_data"],
        "model"),
    # crashing code
    **dict.fromkeys([
        "preprocessing_code_correctness", "training_pipeline_execution", "script_execution_bug",
        "ridge_predict_proba_bug", "ensemble_script_pre_run_validation",
        "training_script_no_undefined_vars", "broken_training_script_before_run",
        "stacking_labelencoder_import_bug", "numeric_dtype_filter_correctness",
        "submission_blocking_code_bugs", "per_type_oof_sampling_crash", "training_code_import_bug",
        "feature_shape_4d_sklearn_incompatibility", "feature_code_correctness",
        "preprocess_import_bug", "training_script_api_correctness", "xgb_api_signature_mismatch"],
        "bug"),
}


def load(run_dir):
    logs = sorted(run_dir.glob("*/reviewer_rubric_store/rubric_oplog.jsonl"))
    by_att = {op: collections.Counter() for op in OPS}
    adds_per_task = {}
    added = []                          # (task, attempt, name)
    for f in logs:
        task = f.parent.parent.name
        adds_per_task[task] = 0
        for line in open(f):
            o = json.loads(line)
            op = o.get("op")
            if op not in OPS:
                continue
            by_att[op][int(o.get("attempt") or 0)] += 1
            if op == "add":
                adds_per_task[task] += 1
                added.append((task, int(o.get("attempt") or 0), o["name"]))
    return len(logs), by_att, adds_per_task, added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(REPO / "our_paper/figures"))
    args = ap.parse_args()
    out = Path(args.outdir)

    stats = []
    for bench, setting, d in RUNS:
        n, by_att, apt, added = load(d)
        label = f"{bench} {setting}"
        stats.append((label, n, by_att, apt, added))
        tot = {op: sum(by_att[op].values()) for op in OPS}
        print(f"== {label.replace('$-$', '-')}: tasks_with_log={n} "
              + " ".join(f"{op}={tot[op]}" for op in OPS))
        for op in OPS:
            print(f"   {op:6s} by attempt: {[by_att[op][a] for a in range(N_ATTEMPTS)]}")
        dist = collections.Counter(min(v, 3) for v in apt.values())
        print(f"   tasks with 0/1/2/3+ adds: {[dist[k] for k in range(4)]}")
        unthemed = sorted({nm for _, _, nm in added if nm not in THEME})
        assert not unthemed, f"unthemed rubric names: {unthemed}"
        th = collections.Counter(THEME[nm] for _, _, nm in added)
        print("   themes: " + ", ".join(f"{k}={th[k]}" for k, _ in THEMES))
        for k, _ in THEMES:
            ex = [f"{nm} ({t.replace('playground-series-', '')})" for t, _, nm in added if THEME[nm] == k]
            print(f"     {k:5s} e.g. {ex[:3]}")

    # Figure 1: per-run ops by attempt, stacked by op type
    op_colors = {"add": "#4c72b0", "update": "#dd8452", "retire": "#55a868"}
    fig, axes = plt.subplots(2, 2, figsize=(3.3, 3.0), sharex=True, sharey=True)
    x = np.arange(N_ATTEMPTS)
    for ax, (label, n, by_att, _, _) in zip(axes.flat, stats):
        bottom = np.zeros(N_ATTEMPTS)
        for op in OPS:
            v = np.array([by_att[op][a] for a in range(N_ATTEMPTS)], dtype=float)
            ax.bar(x, v, 0.7, bottom=bottom, color=op_colors[op], edgecolor="black",
                   linewidth=0.4, label=op.capitalize())
            bottom += v
        ax.set_title(f"{label} ({n} tasks)", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([str(a + 1) for a in range(N_ATTEMPTS)])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6)
    for ax in axes[1]:
        ax.set_xlabel("Attempt", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Operations", fontsize=7)
    axes[0, 1].legend(fontsize=6, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "rubric_store_ops_by_attempt.pdf", bbox_inches="tight")

    # Figure 2: share of tasks by number of added rubrics
    colors = ["#1f77b4", "#9ecae1", "#d62728", "#fc9272"]
    hatches = ["", "//", "", "//"]
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    x = np.arange(4)
    w = 0.2
    for i, (label, n, _, apt, _) in enumerate(stats):
        dist = collections.Counter(min(v, 3) for v in apt.values())
        ax.bar(x + (i - 1.5) * w, [100 * dist[k] / n for k in range(4)], w, label=label,
               color=colors[i], hatch=hatches[i], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(["0", "1", "2", "3+"])
    ax.set_xlabel("Rubrics added to the task's store", fontsize=7)
    ax.set_ylabel("Tasks (%)", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "rubric_adds_per_task.pdf", bbox_inches="tight")
    print("wrote", out / "rubric_store_ops_by_attempt.pdf", out / "rubric_adds_per_task.pdf")


if __name__ == "__main__":
    main()
