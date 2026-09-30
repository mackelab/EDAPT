# %%
"""
Script to generate CSV data for Figure S1 reproduction.
"""

import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from paths import ALIGNMENT_RESULTS, FIGURE_DATA_DIR
from statsmodels.nonparametric.smoothers_lowess import lowess


def filter_most_recent_timestamp(files):
    """Keep only most recent timestamp per experiment directory."""
    by_exp = defaultdict(list)
    for f in files:
        ts = next((p for p in f.parts if re.match(r"^\d{8}_\d{6}(_\d+)?$", p)), "0")
        by_exp[f.parent.parent].append((ts, f))
    return [sorted(v, reverse=True)[0][1] for v in by_exp.values()]

# --- Configuration ---
# Paths are configured in paths.py (override with EDAPT_* environment variables).
BASE_RESULTS_DIR = ALIGNMENT_RESULTS

# Model and dataset configurations
MODEL_NAMES = ["EEGNetv4", "ATCNet", "ShallowConvNet", "DeepConvNet"]

PARADIGMS_MASTER_LIST = {
    "MI": {
        "datasets": [
            {"dataset": "Yang2025", "name_for_plot": "MI (Yang2025)"},
            {"dataset": "Lee2019_MI", "name_for_plot": "MI (Lee2019)"},
            {"dataset": "BNCI2014_001", "name_for_plot": "MI (BNCI2014)"},
        ],
    },
    "P300": {
        "datasets": [
            {"dataset": "BI2015a", "name_for_plot": "P300 (BI2015a)"},
            {"dataset": "Huebner2017", "name_for_plot": "P300 (Huebner2017)"},
            {"dataset": "Huebner2018", "name_for_plot": "P300 (Huebner2018)"},
        ],
    },
    "SSVEP": {
        "datasets": [
            {"dataset": "Lee2019_SSVEP", "name_for_plot": "SSVEP (Lee2019)"},
            {"dataset": "Kalunga2016", "name_for_plot": "SSVEP (Kalunga2016)"},
            {"dataset": "MAMEM2", "name_for_plot": "SSVEP (MAMEM2)"},
        ],
    },
}

DATASET_TRIAL_CUTOFFS = {
    "Yang2025": 600,
    "Lee2019_MI": 200,
    "BNCI2014_001": 250,
    "BI2015a": 1044,
    "Huebner2017": 400,
    "Huebner2018": 500,
    "Lee2019_SSVEP": 200,
    "Kalunga2016": 160,
    "MAMEM2": 125,
}

CONFIG_NAME_MAP = {
    # Pretrained configs
    "FM-None_A-None_AdaBN-F": "PRE-ZS",
    "FM-Full_A-None_AdaBN-F": "PRE+CFT",
    "FM-None_A-Eucl_AdaBN-T": "PRE+UDA",
    "FM-Full_A-Eucl_AdaBN-T": "PRE+UDA+CFT",
    "FM-Dec_A-Eucl_AdaBN-T": "PRE+UDA+CFT-Dec",
    "FM-Dec_A-Eucl_AdaBN-F": "PRE+Align+CFT-Dec",
    "FM-Full_A-Eucl_AdaBN-F": "PRE+UDA-noAdaBN+CFT",
    "FM-Full_A-Riem_AdaBN-T": "PRE+UDA-Riem+CFT",
    # No-pretraining configs
    "NoPre_FM-Full_A-None_AdaBN-F": "CFT-only",
    "NoPre_FM-Full_A-Eucl_AdaBN-T": "UDA+CFT-only",
}

SMOOTHING_FRAC = 0.2
MIN_POINTS_FOR_SMOOTHING = 10


def parse_alignment_studies_dir_name(dir_name_str):
    """Parse directory name to extract model, dataset, and config.

    Handles both pretrained and no-pretrain configs:
    - Pretrained: AlignEval_{model}_{dataset}_FM-{...}
    - NoPre: AlignEval_{model}_{dataset}_NoPre_FM-{...}_no_pretrain
    """
    try:
        # Check for NoPre config first
        nopre_match = re.search(r"_NoPre_(FM-.+?)(?:_no_pretrain)?$", dir_name_str)
        if nopre_match:
            config_name = "NoPre_" + nopre_match.group(1)
            prefix = dir_name_str[: nopre_match.start()]
            parts = prefix.split("_")
            model_name = parts[1]
            dataset_name = "_".join(parts[2:])
            return model_name, dataset_name, config_name

        # Standard pretrained config
        config_start_index = dir_name_str.find("_FM-")
        if config_start_index == -1:
            return None, None, None
        config_name = dir_name_str[config_start_index + 1:]
        prefix = dir_name_str[:config_start_index]
        parts = prefix.split("_")
        model_name = parts[1]
        dataset_name = "_".join(parts[2:])
        return model_name, dataset_name, config_name
    except IndexError:
        return None, None, None


def load_trial_data(base_dir):
    """Load trial data from all experiment directories."""
    all_trial_data_list = []
    if not base_dir.exists():
        print(f"ERROR: Base directory does not exist: {base_dir}")
        return pd.DataFrame()

    valid_align_codes = set(CONFIG_NAME_MAP.keys())

    valid_datasets = set()
    for paradigm, group_details in PARADIGMS_MASTER_LIST.items():
        for d in group_details["datasets"]:
            valid_datasets.add(d["dataset"])

    for exp_dir in base_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        model, dataset, config = parse_alignment_studies_dir_name(exp_dir.name)
        if not (
            model
            and dataset
            and config
            and model in MODEL_NAMES
            and config in valid_align_codes
            and dataset in valid_datasets
        ):
            continue

        csv_files = filter_most_recent_timestamp(exp_dir.rglob("results_trial_metrics.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # New format uses precomputed balanced accuracy columns
                required_cols = [
                    "trial_index",
                    "subject_id",
                    "rolling_balanced_acc",
                    "overall_balanced_acc",
                ]
                if not all(col in df.columns for col in required_cols):
                    continue

                df["model"] = model
                df["dataset"] = dataset
                df["align_code"] = config
                df["method"] = CONFIG_NAME_MAP.get(config)

                # Assign paradigm based on dataset
                paradigm_name = "Unknown"
                for p_key, p_value in PARADIGMS_MASTER_LIST.items():
                    if dataset in [d["dataset"] for d in p_value["datasets"]]:
                        paradigm_name = p_key
                        break
                df["paradigm"] = paradigm_name

                df["subject_id"] = df["subject_id"].astype(str)

                cols_to_keep = [
                    "model",
                    "dataset",
                    "paradigm",
                    "method",
                    "align_code",
                    "subject_id",
                    "trial_index",
                    "rolling_balanced_acc",
                    "overall_balanced_acc",
                ]
                all_trial_data_list.append(df[cols_to_keep])
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")

    return (
        pd.concat(all_trial_data_list, ignore_index=True)
        if all_trial_data_list
        else pd.DataFrame()
    )


def generate_time_series_data(df_trials):
    """Generate smoothed time series data for plotting."""
    time_series_data = []

    dataset_keys = []
    for paradigm, group_details in PARADIGMS_MASTER_LIST.items():
        for d in group_details["datasets"]:
            dataset_keys.append(d["dataset"])

    for dataset in dataset_keys:
        cutoff = DATASET_TRIAL_CUTOFFS.get(dataset)
        df_dataset = df_trials[df_trials["dataset"] == dataset]

        for method in CONFIG_NAME_MAP.values():
            df_method = df_dataset[df_dataset["method"] == method]

            for model in MODEL_NAMES:
                df_model = df_method[df_method["model"] == model].copy()
                if df_model.empty:
                    continue

                if cutoff:
                    subjects_with_enough_trials = df_model.groupby("subject_id")[
                        "trial_index"
                    ].max()
                    valid_subjects = subjects_with_enough_trials[
                        subjects_with_enough_trials >= (cutoff - 1)
                    ].index
                    if len(valid_subjects) > 0:
                        df_model = df_model[
                            (df_model["subject_id"].isin(valid_subjects))
                            & (df_model["trial_index"] < cutoff)
                        ]

                if df_model.empty:
                    continue

                summary_stats = (
                    df_model.groupby("trial_index")["rolling_balanced_acc"]
                    .agg(mean=np.nanmean)
                    .reset_index()
                    .sort_values(by="trial_index")
                )

                if summary_stats.empty:
                    continue

                summary_stats["mean_smoothed"] = summary_stats["mean"]
                if len(summary_stats["mean"].dropna()) >= MIN_POINTS_FOR_SMOOTHING:
                    try:
                        valid_indices = summary_stats["mean"].notna()
                        if valid_indices.any():
                            smoothed = lowess(
                                summary_stats.loc[valid_indices, "mean"],
                                summary_stats.loc[valid_indices, "trial_index"],
                                frac=SMOOTHING_FRAC,
                                return_sorted=False,
                                it=0,
                            )
                            summary_stats.loc[valid_indices, "mean_smoothed"] = smoothed
                    except Exception as e:
                        print(f"Smoothing failed for {model}-{dataset}-{method}: {e}")

                paradigm_name = "Unknown"
                for p_key, p_value in PARADIGMS_MASTER_LIST.items():
                    if dataset in [d["dataset"] for d in p_value["datasets"]]:
                        paradigm_name = p_key
                        break

                for _, row in summary_stats.iterrows():
                    time_series_data.append(
                        {
                            "dataset": dataset,
                            "paradigm": paradigm_name,
                            "method": method,
                            "model": model,
                            "trial_index": row["trial_index"],
                            "mean_accuracy": row["mean"],
                            "smoothed_accuracy": row["mean_smoothed"],
                        }
                    )

    return pd.DataFrame(time_series_data)


def generate_scatter_data(df_trials):
    """Generate subject-level final balanced accuracies for scatter plots."""
    # Get the final balanced accuracy per subject (last trial's overall_balanced_acc)
    final_acc = (
        df_trials.sort_values("trial_index")
        .groupby(["model", "dataset", "paradigm", "method", "subject_id"])
        .last()["overall_balanced_acc"]
        .reset_index()
    )
    final_acc.rename(
        columns={"overall_balanced_acc": "subject_mean_accuracy"}, inplace=True
    )

    scatter_data_list = []

    for (model, dataset, paradigm), group in final_acc.groupby(
        ["model", "dataset", "paradigm"]
    ):
        pivot = group.pivot_table(
            index="subject_id", columns="method", values="subject_mean_accuracy"
        ).reset_index()

        pivot["model"] = model
        pivot["dataset"] = dataset
        pivot["paradigm"] = paradigm
        scatter_data_list.append(pivot)

    scatter_data = pd.concat(scatter_data_list, ignore_index=True)
    scatter_data.columns.name = None

    return scatter_data


def main():
    """Main execution function."""
    print("Loading trial data...")
    df_trials = load_trial_data(BASE_RESULTS_DIR)

    if df_trials.empty:
        print(
            "ERROR: No trial data loaded. Check BASE_RESULTS_DIR and dataset configuration."
        )
        return

    print(f"Loaded {len(df_trials)} trial records.")

    # Generate time series data (using precomputed rolling_balanced_acc)
    print("Generating time series data...")
    time_series_df = generate_time_series_data(df_trials)

    print("Generating scatter plot data...")
    scatter_df = generate_scatter_data(df_trials)

    output_dir = FIGURE_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    time_series_path = output_dir / "fig1_supplements_time_series_data.csv.gz"
    scatter_path = output_dir / "fig1_supplements_scatter_data.csv"

    time_series_df.to_csv(time_series_path, index=False)
    scatter_df.to_csv(scatter_path, index=False)

    print(f"Saved time series data: {time_series_path}")
    print(f"Saved scatter data: {scatter_path}")
    print(f"Time series data shape: {time_series_df.shape}")
    print(f"Scatter data shape: {scatter_df.shape}")


if __name__ == "__main__":
    main()
# %%
