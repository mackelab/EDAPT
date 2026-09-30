# %%
"""
Submitit script for hyperparameter sweep over warmup trials, window size,
and finetune epochs for supplementary material. Tests zero shot vs zero shot + CFT
on one representative dataset per paradigm using ShallowConvNet.
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

# %%
# Core configuration

# Repo root; override with EDAPT_ROOT if launching from elsewhere.
REPO_DIR = Path(os.environ.get("EDAPT_ROOT", Path(__file__).resolve().parent))
CONDA_ENV = os.environ.get("EDAPT_CONDA_ENV", "edapt")
SCRIPT_TO_RUN = REPO_DIR / "train_transfer.py"
BASE_OUTPUT_DIR = REPO_DIR / "results"

SLURM_PARTITION = os.environ.get("EDAPT_SLURM_PARTITION", "gpu")
MEM_GB_PER_GPU = 200
CPUS_PER_GPU = 8
DEFAULT_DEVICE = "cuda"
PYTHON_EXECUTABLE = "python"

# %%
# Utility functions


def dict_to_cli_args(args_dict: Dict[str, Any]) -> str:
    """Convert dictionary to OmegaConf CLI argument string."""
    parts = []
    for key, value in args_dict.items():
        if value is None:
            continue

        cli_key = key
        if isinstance(value, bool):
            parts.append(f"{cli_key}={str(value).lower()}")
        elif isinstance(value, list):
            formatted_elements = [
                f'"{item}"' if isinstance(item, str) else str(item) for item in value
            ]
            list_str = f"[{','.join(formatted_elements)}]"
            parts.append(f"{cli_key}={shlex.quote(list_str)}")
        else:
            parts.append(f"{cli_key}={shlex.quote(str(value))}")

    return " ".join(parts)


def generate_job_name(params: Dict[str, Any]) -> str:
    """Generate short, readable SLURM job name."""
    ds_name = params["dataset_names"][0].replace("Lee2019_", "").replace("2025", "")
    config_tag = params.get("custom_config_tag", "Cfg")
    safe_config_tag = re.sub(r"[^a-zA-Z0-9]", "", config_tag)[:30]
    return f"SCN-{ds_name}-{safe_config_tag}"[:80]


def run_transfer_job(experiment_config: Dict[str, Any]) -> str:
    """Execute transfer learning job with environment setup and logging."""
    exp_name = experiment_config.get("experiment_name", "default_exp")
    output_dir = Path(
        experiment_config.get("base_output_dir", BASE_OUTPUT_DIR / exp_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cli_args_str = dict_to_cli_args(
        {k: v for k, v in experiment_config.items() if v is not None}
    )
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    log_file_dir = REPO_DIR / "job_logs"
    log_file_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_file_dir / f"hyperparam_sweep2_{exp_name}_{job_id}.txt"

    cmd = f"""
set -e
echo "------ Environment Setup ------" > "{log_file}" && \
echo "Job ID: {job_id}" >> "{log_file}" && \
echo "Running on node: $(hostname)" >> "{log_file}" && \
echo "User: $(whoami)" >> "{log_file}" && \
echo "Conda Env: {CONDA_ENV}" >> "{log_file}" && \
echo "Activating Conda env..." >> "{log_file}" && \
source ~/.bashrc && \
conda activate {CONDA_ENV} && \
echo "Changing to repo directory..." >> "{log_file}" && \
cd "{REPO_DIR}" && \
echo "Setting MKL environment variables..." >> "{log_file}" && \
export MKL_THREADING_LAYER=GNU && \
echo "------ Running Command ------" >> "{log_file}" && \
echo "{PYTHON_EXECUTABLE} {SCRIPT_TO_RUN} {cli_args_str}" >> "{log_file}" && \
echo "------ Script Output ------" >> "{log_file}" && \
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


# %%
# Experiment configuration

# Datasets: one per paradigm
DATASETS = {
    "MI": "Yang2025",
    "P300": "BI2015a",
    "SSVEP": "Lee2019_SSVEP",
}

# Hyperparameter sweep values
WARMUP_TRIALS = [0, 5, 10, 20, 50]
WINDOW_SIZES = [10, 20, 50, 100]
FINETUNE_EPOCHS = [1, 3, 6]


def get_base_config() -> Dict[str, Any]:
    """Base configuration shared across experiments."""
    return {
        "n_splits": 2,
        "device": DEFAULT_DEVICE,
        "save_checkpoints": False,
        "save_results": True,
        "use_tta": False,
        "use_wandb": True,
        "wandb_project": "hyperparam_sweep2",
        "lr_finetune": 1e-4,
        "models_to_run": ["ShallowConvNet"],
        "alignment_type": "euclidean",
        "use_adabn": False,
    }


def generate_experiment_configs() -> List[Dict[str, Any]]:
    """Generate all experiment configurations for hyperparam sweep."""
    base_grid_dir = "hyperparam_sweep2"
    global_base_config = get_base_config()

    experiments = []

    for paradigm, dataset in DATASETS.items():
        # Zero shot baseline (only one config per dataset)
        exp_config = global_base_config.copy()
        exp_config.update(
            {
                "dataset_names": [dataset],
                "finetune_mode": "none",
                "finetune_warmup_trials": 0,
                "window_size": 50,
                "finetune_epochs": 0,
                "custom_config_tag": "ZeroShot",
                "wandb_group": f"HyperSweep_{paradigm}_ZeroShot",
            }
        )

        raw_exp_name = f"HyperSweep_{paradigm}_{dataset}_ZeroShot"
        exp_config["experiment_name"] = re.sub(r"[^a-zA-Z0-9_.-]+", "", raw_exp_name)
        exp_config["base_output_dir"] = str(
            BASE_OUTPUT_DIR / base_grid_dir / exp_config["experiment_name"]
        )
        experiments.append(exp_config)

        # Zero shot + CFT with hyperparameter sweep
        for warmup, window, epochs in itertools.product(
            WARMUP_TRIALS, WINDOW_SIZES, FINETUNE_EPOCHS
        ):
            exp_config = global_base_config.copy()
            exp_config.update(
                {
                    "dataset_names": [dataset],
                    "finetune_mode": "full",
                    "finetune_warmup_trials": warmup,
                    "window_size": window,
                    "finetune_epochs": epochs,
                    "custom_config_tag": f"CFT_w{warmup}_ws{window}_e{epochs}",
                    "wandb_group": f"HyperSweep_{paradigm}_CFT",
                }
            )

            raw_exp_name = (
                f"HyperSweep_{paradigm}_{dataset}_CFT_w{warmup}_ws{window}_e{epochs}"
            )
            exp_config["experiment_name"] = re.sub(
                r"[^a-zA-Z0-9_.-]+", "", raw_exp_name
            )
            exp_config["base_output_dir"] = str(
                BASE_OUTPUT_DIR / base_grid_dir / exp_config["experiment_name"]
            )
            experiments.append(exp_config)

    return experiments


# %%
# Main execution


def main():
    """Main execution function for job submission."""
    parser = argparse.ArgumentParser(
        description="Submit Hyperparameter Sweep Jobs for Supplementary Analysis"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configurations instead of submitting jobs",
    )
    parser.add_argument(
        "--paradigm",
        type=str,
        choices=["MI", "P300", "SSVEP", "all"],
        default="all",
        help="Run only for specific paradigm",
    )
    cli_args = parser.parse_args()

    experiments_to_run = generate_experiment_configs()

    # Filter by paradigm if specified
    if cli_args.paradigm != "all":
        experiments_to_run = [
            exp
            for exp in experiments_to_run
            if DATASETS[cli_args.paradigm] in exp["dataset_names"]
        ]

    total_jobs = len(experiments_to_run)

    # Summary
    n_zero_shot = sum(1 for exp in experiments_to_run if exp["finetune_mode"] == "none")
    n_cft = total_jobs - n_zero_shot

    print(f"Hyperparameter Sweep Configuration:")
    print(f"  Datasets: {list(DATASETS.values())}")
    print(f"  Model: ShallowConvNet")
    print(f"  Warmup trials: {WARMUP_TRIALS}")
    print(f"  Window sizes: {WINDOW_SIZES}")
    print(f"  Finetune epochs: {FINETUNE_EPOCHS}")
    print(f"\nTotal jobs: {total_jobs} ({n_zero_shot} zero-shot + {n_cft} CFT)")

    if cli_args.dry_run:
        print(f"\nDRY RUN: Would submit {total_jobs} jobs.")
        print("\nExample configurations:")

        for i, cfg in enumerate(experiments_to_run[:5]):
            print(f"\n--- Example Config {i+1} ---")
            for key in [
                "experiment_name",
                "dataset_names",
                "finetune_mode",
                "finetune_warmup_trials",
                "window_size",
                "finetune_epochs",
            ]:
                print(f"    {key}: {cfg.get(key)}")

        if total_jobs > 5:
            print(f"\n... and {total_jobs - 5} more configurations")

        sys.exit(0)

    print(f"\nSubmitting {total_jobs} jobs to SLURM...")

    for i, exp_config in enumerate(experiments_to_run):
        job_name = generate_job_name(exp_config)
        print(f"\nSubmitting Job {i+1}/{total_jobs}: {job_name}")

        log_folder = (
            REPO_DIR
            / "slurm_logs"
            / f"{datetime.now().strftime('%Y-%m-%d')}_{job_name}"
        )
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
        print(f"  Submitted Job ID: {job.job_id}")
        print(f"  Experiment: {exp_config['experiment_name']}")
        print(f"  SLURM logs: {log_folder}")

    print(f"\nAll {total_jobs} jobs submitted successfully.")


if __name__ == "__main__":
    main()
