# %%
"""
Generates and submits SLURM jobs for iso-scaling analysis, evaluating the trade-off
between the number of pre-training subjects and trials per subject under a fixed budget.
"""

import argparse
import itertools
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import submitit

# ==========================================================================
# CORE CONFIGURATION
# ==========================================================================

# Repository and environment setup
# Repo root; override with EDAPT_ROOT if launching from elsewhere.
REPO_DIR = Path(os.environ.get("EDAPT_ROOT", Path(__file__).resolve().parent))
CONDA_ENV = os.environ.get("EDAPT_CONDA_ENV", "edapt")
SCRIPT_TO_RUN = REPO_DIR / "train_transfer.py"
BASE_OUTPUT_DIR = REPO_DIR / "results_iso_scaling_fix"

# SLURM & JOB CONFIGURATION
SLURM_PARTITION = os.environ.get("EDAPT_SLURM_PARTITION", "gpu")
MEM_GB_PER_GPU = 96
CPUS_PER_GPU = 8
DEFAULT_DEVICE = "cuda"
PYTHON_EXECUTABLE = "python"


# ==========================================================================
# UTILITY FUNCTIONS
# ==========================================================================


def dict_to_cli_args(args_dict: Dict[str, Any]) -> str:
    """Converts a dictionary to a string of OmegaConf CLI arguments."""
    parts = []
    for key, value in args_dict.items():
        if value is None:
            continue

        if isinstance(value, bool):
            parts.append(f"{key}={str(value).lower()}")
        elif isinstance(value, list):
            formatted_elements = [
                f'"{item}"' if isinstance(item, str) else str(item) for item in value
            ]
            list_str = f"[{','.join(formatted_elements)}]"
            parts.append(f"{key}={shlex.quote(list_str)}")
        else:
            parts.append(f"{key}={shlex.quote(str(value))}")
    return " ".join(parts)


def run_transfer_job(experiment_config: Dict[str, Any]) -> str:
    """Executes the transfer learning script with a specific configuration."""
    exp_name = experiment_config.get("experiment_name", "default_exp")
    output_dir_job = Path(experiment_config.get("base_output_dir"))
    output_dir_job.mkdir(parents=True, exist_ok=True)

    # Prepare CLI arguments, ensuring 'base_output_dir' is correctly passed
    cli_args_to_pass = {k: v for k, v in experiment_config.items() if v is not None}
    cli_args_str = dict_to_cli_args(cli_args_to_pass)

    # Setup logging
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    log_file_dir = REPO_DIR / "job_logs_iso_scaling_fix"
    log_file_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_file_dir / f"iso_scaling_log_{exp_name}_{job_id}.txt"

    # Define the command to be executed
    cmd = f"""
set -e
echo "------ Environment Setup ------" > "{log_file}" && \\
echo "Job ID: {job_id}" >> "{log_file}" && \\
echo "Running on node: $(hostname)" >> "{log_file}" && \\
echo "User: $(whoami)" >> "{log_file}" && \\
echo "Activating Conda env: {CONDA_ENV}..." >> "{log_file}" && \\
source ~/.bashrc && \\
conda activate {CONDA_ENV} && \\
echo "Changing to repo directory: {REPO_DIR}" >> "{log_file}" && \\
cd "{REPO_DIR}" && \\
export MKL_THREADING_LAYER=GNU && \\
echo "------ Running Command ------" >> "{log_file}" && \\
echo "{PYTHON_EXECUTABLE} {SCRIPT_TO_RUN} {cli_args_str}" >> "{log_file}" && \\
echo "------ Script Output ------" >> "{log_file}" && \\
{PYTHON_EXECUTABLE} -u "{SCRIPT_TO_RUN}" {cli_args_str} 2>&1 | tee -a "{log_file}"
"""
    try:
        subprocess.run(cmd, shell=True, check=True, executable="/bin/bash", text=True)
    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: Command failed with exit code {e.returncode}. Check log: {log_file}"
        )
        raise RuntimeError(f"Script for '{exp_name}' failed. Log: {log_file}") from e

    return f"Finished: {exp_name}"


# ==========================================================================
# EXPERIMENT CONFIGURATION
# ==========================================================================

EXPECTED_SUBJECTS = {
    "BI2015a": 43,
    "BNCI2014_001": 9,
    "Huebner2017": 13,
    "Huebner2018": 12,
    "Kalunga2016": 12,
    "Lee2019_MI": 54,
    "Lee2019_SSVEP": 54,
    "MAMEM2": 10,
    "Yang2025": 51,
}


def check_run_complete(exp_name: str, dataset: str) -> bool:
    """Check if a run is complete by examining result files and checking for NaNs."""
    import pandas as pd

    results_dir = BASE_OUTPUT_DIR / "iso_scaling_studies"
    run_dir = results_dir / exp_name
    if not run_dir.exists():
        return False

    # find timestamp subdirs (format: YYYYMMDD_HHMMSS)
    subdirs = []
    for item in run_dir.iterdir():
        if item.is_dir():
            for subitem in item.iterdir():
                if subitem.is_dir() and subitem.name.startswith("202"):
                    subdirs.append(subitem)
            if item.name.startswith("202"):
                subdirs.append(item)

    if not subdirs:
        return False

    latest = sorted(subdirs)[-1]
    csv_file = latest / "results_detailed.csv"
    if not csv_file.exists():
        return False

    try:
        df = pd.read_csv(csv_file)
        expected = EXPECTED_SUBJECTS.get(dataset, 0)
        # check row count
        if len(df) != expected:
            return False
        # check for NaN values in key metric columns
        metric_cols = [
            c for c in df.columns if "accuracy" in c.lower() or "auroc" in c.lower()
        ]
        if metric_cols and df[metric_cols].isna().any().any():
            return False
    except Exception:
        return False

    if not (latest / "results_trial_metrics.csv").exists():
        return False

    return True


def setup_experiment_grid(only_incomplete: bool = False) -> List[Dict[str, Any]]:
    """Configures the experimental grid for iso-scaling analysis."""
    # --- Grid Search Parameters ---
    N_SPLITS = 5
    BASE_GRID_DIR = "iso_scaling_studies"
    DATASETS = [
        ["Yang2025"],
        ["Lee2019_SSVEP"],
        ["BI2015a"],
        ["Lee2019_MI"],
        ["BNCI2014_001"],
        ["Huebner2017"],
        ["Huebner2018"],
        ["Kalunga2016"],
        ["MAMEM2"],
    ]
    MODELS_TO_EVALUATE = ["ShallowConvNet", "EEGNetv4", "ATCNet", "DeepConvNet"]
    PRETRAIN_FLAG = False  # False means 'no_pretrain=False', i.e., use pre-training

    # --- Fixed Configuration for Iso-Scaling ---
    TARGET_CONFIG = {
        "config_name": "FM-Full_A-None_AdaBN-F",
        "finetune_mode": "full",
        "alignment_type": "none",
        "use_adabn": False,
    }

    # --- Iso-Scaling Trade-off Points (Num Subjects, Num Trials) ---
    ISO_TRADE_OFF_POINTS = {
        "Yang2025": {
            "Budget_High_10k": [(15, 600), (20, 450), (30, 300)],
            "Budget_Low_5k": [(10, 450), (15, 300), (30, 150)],
        },
        "Lee2019_SSVEP": {
            "Budget_High_4k": [(20, 200), (25, 160), (40, 100)],
            "Budget_Low_2k": [(10, 200), (20, 100), (40, 50)],
        },
        "BI2015a": {
            "Budget_High_20k": [(15, 1000), (20, 750), (30, 500)],
            "Budget_Low_10k": [(10, 500), (20, 250), (25, 200)],
        },
        "Lee2019_MI": {
            "Budget_High_6k": [(30, 200), (40, 150), (50, 120)],
            "Budget_Low_3k": [(15, 200), (30, 100), (50, 60)],
        },
        "BNCI2014_001": {
            "Budget_High_3k": [(6, 500), (8, 375)],
            "Budget_Low_1.5k": [(3, 500), (5, 300)],
        },
        "Huebner2017": {
            "Budget_High_75k": [(6, 12500), (10, 7500), (12, 6250)],
            "Budget_Low_37.5k": [(3, 12500), (6, 6250), (10, 3750)],
        },
        "Huebner2018": {
            "Budget_High_84k": [(6, 14000), (8, 10500), (12, 7000)],
            "Budget_Low_42k": [(3, 14000), (6, 7000), (10, 4200)],
        },
        "Kalunga2016": {
            "Budget_High_512": [(8, 64), (12, 42)],
            "Budget_Low_256": [(4, 64), (8, 32)],
        },
        "MAMEM2": {
            "Budget_High_600": [(6, 100), (8, 75)],
            "Budget_Low_300": [(3, 100), (6, 50)],
        },
    }

    # --- Base Configuration Shared Across All Jobs ---
    global_base_config = {
        "n_splits": N_SPLITS,
        "device": DEFAULT_DEVICE,
        "save_checkpoints": False,
        "save_results": True,
        "use_tta": True,
        "use_wandb": True,
        "wandb_project": "iso_scaling_analysis_final",
        "seed": 42,
        "lr_finetune": 1e-4,
        "finetune_warmup_trials": 20,
        "tta_buffer_length": 32,
        "adabn_mode": "train_mode",
        "window_size": 100,
        "batch_size_finetune": 100,  # to match window size, so each epoch is full window
    }

    # --- Generate All Experiment Configurations ---
    experiments = []
    for model, dataset_list in itertools.product(MODELS_TO_EVALUATE, DATASETS):
        current_dataset_name = dataset_list[0]
        if current_dataset_name not in ISO_TRADE_OFF_POINTS:
            continue

        for budget_name, points in ISO_TRADE_OFF_POINTS[current_dataset_name].items():
            for num_subj, num_trials in points:
                raw_exp_name = (
                    f"IsoEval_{model}_{current_dataset_name}_"
                    f"NPS{num_subj}_NT{num_trials}_{TARGET_CONFIG['config_name']}"
                )
                exp_name = re.sub(r"[^a-zA-Z0-9_.-]+", "", raw_exp_name)

                # skip complete runs if only_incomplete is set
                if only_incomplete and check_run_complete(
                    exp_name, current_dataset_name
                ):
                    continue

                exp_config = global_base_config.copy()
                exp_config.update(
                    {k: v for k, v in TARGET_CONFIG.items() if k != "config_name"}
                )

                exp_config.update(
                    {
                        "models_to_run": [model],
                        "dataset_names": [current_dataset_name],  # Pass as a list
                        "no_pretrain": PRETRAIN_FLAG,
                        "num_pretrain_subjects": num_subj,
                        "num_trials_per_subject": num_trials,
                        "custom_config_tag": TARGET_CONFIG["config_name"],
                        "wandb_group": f"Iso_{model}_{current_dataset_name}_{budget_name}",
                    }
                )

                exp_config["experiment_name"] = exp_name
                exp_config["base_output_dir"] = str(
                    BASE_OUTPUT_DIR / BASE_GRID_DIR / exp_config["experiment_name"]
                )

                experiments.append(exp_config)

    return experiments


def print_dry_run_summary(experiments: List[Dict[str, Any]]) -> None:
    """Prints a summary of the jobs that would be submitted."""
    total_jobs = len(experiments)
    print(f"\n--- DRY RUN: Would submit {total_jobs} jobs. ---")

    for i, cfg in enumerate(experiments[: min(3, total_jobs)]):
        print(f"\n--- Example Config {i+1} ({cfg['experiment_name']}) ---")
        for key, val in sorted(cfg.items()):
            print(f"    {key}: {val}")
        print(f"    > Output would be in: {cfg['base_output_dir']}")


def submit_experiments(experiments: List[Dict[str, Any]]) -> None:
    """Submits all generated experiment configurations to SLURM."""
    total_jobs = len(experiments)

    for i, exp_config in enumerate(experiments):
        # Sanitize job name for SLURM
        job_name = re.sub(r"[^a-zA-Z0-9_.-]+", "", exp_config["experiment_name"])[:100]

        print(f"\n--- Submitting Job {i+1}/{total_jobs}: {job_name} ---")

        # Setup logging directories
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_folder = REPO_DIR / "slurm_logs_iso_scaling_fix" / date_str / job_name

        executor = submitit.AutoExecutor(folder=str(log_folder))
        executor.update_parameters(
            slurm_partition=SLURM_PARTITION,
            name=job_name,
            slurm_time="2-00:00:00",
            nodes=1,
            tasks_per_node=1,
            slurm_gpus_per_task=1,
            slurm_cpus_per_task=CPUS_PER_GPU,
            mem_gb=MEM_GB_PER_GPU,
        )

        job = executor.submit(run_transfer_job, exp_config)
        print(
            f"  > Submitted Job ID: {job.job_id} for experiment: {exp_config['experiment_name']}"
        )
        print(f"  > SLURM logs will be in: {log_folder}")


# ==========================================================================
# MAIN EXECUTION
# ==========================================================================


def main():
    """Main execution function to generate and submit SLURM jobs."""
    parser = argparse.ArgumentParser(description="Submit Iso-Scaling Analysis Jobs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configurations instead of submitting jobs.",
    )
    parser.add_argument(
        "--resubmit-incomplete",
        action="store_true",
        help="Only submit jobs for incomplete/missing runs.",
    )
    cli_args = parser.parse_args()

    # Generate experiment configurations from the defined grid
    if cli_args.resubmit_incomplete:
        print("Checking for incomplete runs...")
        experiments_to_run = setup_experiment_grid(only_incomplete=True)
        print(f"Found {len(experiments_to_run)} incomplete/missing runs to resubmit.")
    else:
        experiments_to_run = setup_experiment_grid(only_incomplete=False)

    total_jobs = len(experiments_to_run)

    if total_jobs == 0:
        print("No jobs to submit. All runs are complete!")
        sys.exit(0)

    print(
        f"--- Generated {total_jobs} total experiment configurations for iso-scaling analysis. ---"
    )

    if cli_args.dry_run:
        print_dry_run_summary(experiments_to_run)
        sys.exit(0)

    # Submit all generated experiments to SLURM
    submit_experiments(experiments_to_run)
    print(f"\nAll {total_jobs} iso-scaling analysis jobs have been submitted.")


if __name__ == "__main__":
    main()
