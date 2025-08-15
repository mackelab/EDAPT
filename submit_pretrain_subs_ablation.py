#%%
"""
Generates and submits SLURM jobs for evaluating model performance across different number of pre-training subjects.
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
REPO_DIR = Path(f"/mnt/lustre/work/macke/{os.environ.get('USER', 'user')}/repos/eegjepa")
CONDA_ENV = "timeseries"
SCRIPT_TO_RUN = REPO_DIR / "EDAPT_neurips/EDAPT/train_transfer.py"
BASE_OUTPUT_DIR = REPO_DIR / "EDAPT_neurips/EDAPT/results/results_scaling_subjects"

# SLURM & JOB CONFIGURATION
SLURM_PARTITION = "a100-galvani"
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
                f'"{item}"' if isinstance(item, str) else str(item)
                for item in value
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
    log_file_dir = REPO_DIR / "job_logs_scaling_subjects"
    log_file_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_file_dir / f"scaling_log_{exp_name}_{job_id}.txt"

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
        print(f"ERROR: Command failed with exit code {e.returncode}. Check log: {log_file}")
        raise RuntimeError(f"Script for '{exp_name}' failed. Log: {log_file}") from e
    
    return f"Finished: {exp_name}"


# ==========================================================================
# EXPERIMENT CONFIGURATION
# ==========================================================================

def setup_experiment_grid() -> List[Dict[str, Any]]:
    """Configures the experimental grid for the pre-training subjects scaling analysis."""
    # --- Grid Search Parameters ---
    N_SPLITS = 5
    BASE_GRID_DIR = "scaling_studies_nps_final_5fold_lr_1e-4_finetune_warmup_20"
    DATASETS = [["Yang2025"], ["Lee2019_SSVEP"], ["BI2015a"], ["Lee2019_MI"], ["BNCI2014_001"], ["Huebner2017"], ["Huebner2018"], ["Kalunga2016"], ["MAMEM2"]]
    MODELS_TO_EVALUATE = ["ShallowConvNet", "EEGNetv4", "ATCNet", "DeepConvNet"]
    PRETRAIN_FLAG = False  # False means 'no_pretrain=False', i.e., use pre-training

    # --- Dataset-specific subject counts for scaling ---
    DATASET_SUBJECT_COUNTS = {
        "Yang2025": [5, 10, 15, 20, 25, 30, 35, 40],
        "Lee2019_SSVEP": [5, 10, 15, 20, 25, 30, 35, 40],
        "BI2015a": [5, 10, 15, 20, 25, 30, 35],
        "Lee2019_MI": [5, 10, 15, 20, 25, 30, 35, 40],
        "BNCI2014_001": [2, 4, 7],
        "Huebner2017": [3, 7, 10],
        "Huebner2018": [3, 6, 9],
        "Kalunga2016": [3, 6, 9],
        "MAMEM2": [3, 5, 8],
    }

    # --- Fixed Alignment Configuration for this study ---
    ALIGNMENT_CONFIG = {
        "config_name": "FM-Full_A-None_AdaBN-F",
        "finetune_mode": "full",
        "alignment_type": "none",
        "use_adabn": False,
    }

    # --- Base Configuration Shared Across All Jobs ---
    global_base_config = {
        "n_splits": N_SPLITS,
        "device": DEFAULT_DEVICE,
        "save_checkpoints": False,
        "save_results": True,
        "use_tta": True,
        "use_wandb": True,
        "wandb_project": "subject_scaling_analysis_final",
        "seed": 42,
        "lr_finetune": 1e-4,
        "finetune_warmup_trials": 20,
    }

    # --- Generate All Experiment Configurations ---
    experiments = []
    for model, dataset_list in itertools.product(MODELS_TO_EVALUATE, DATASETS):
        current_dataset_name = dataset_list[0]
        if current_dataset_name not in DATASET_SUBJECT_COUNTS:
            continue

        subject_counts = DATASET_SUBJECT_COUNTS[current_dataset_name]
        for num_subj in subject_counts:
            exp_config = global_base_config.copy()
            exp_config.update({k: v for k, v in ALIGNMENT_CONFIG.items() if k != "config_name"})

            pretrain_suffix = "_no_pretrain" if PRETRAIN_FLAG else ""
            
            exp_config.update({
                "models_to_run": [model],
                "dataset_names": [current_dataset_name], # Pass as a list
                "no_pretrain": PRETRAIN_FLAG,
                "num_pretrain_subjects": num_subj,
                "custom_config_tag": ALIGNMENT_CONFIG['config_name'],
                "wandb_group": f"ScalingNPS_{model}_{current_dataset_name}_{ALIGNMENT_CONFIG['config_name']}{pretrain_suffix}"
            })
            
            raw_exp_name = (
                f"ScalingNPS_{model}_{current_dataset_name}_"
                f"NPS{num_subj}_{ALIGNMENT_CONFIG['config_name']}{pretrain_suffix}"
            )
            exp_config["experiment_name"] = re.sub(r'[^a-zA-Z0-9_.-]+', '', raw_exp_name)
            exp_config["base_output_dir"] = str(BASE_OUTPUT_DIR / BASE_GRID_DIR / exp_config["experiment_name"])
            
            experiments.append(exp_config)

    return experiments


def print_dry_run_summary(experiments: List[Dict[str, Any]]) -> None:
    """Prints a summary of the jobs that would be submitted."""
    total_jobs = len(experiments)
    print(f"\n--- DRY RUN: Would submit {total_jobs} jobs. ---")
    
    for i, cfg in enumerate(experiments[:min(3, total_jobs)]):
        print(f"\n--- Example Config {i+1} ({cfg['experiment_name']}) ---")
        for key, val in sorted(cfg.items()):
            print(f"    {key}: {val}")
        print(f"    > Output would be in: {cfg['base_output_dir']}")


def submit_experiments(experiments: List[Dict[str, Any]]) -> None:
    """Submits all generated experiment configurations to SLURM."""
    total_jobs = len(experiments)
    
    for i, exp_config in enumerate(experiments):
        job_name = re.sub(r'[^a-zA-Z0-9_.-]+', '', exp_config["experiment_name"])[:100]
        
        print(f"\n--- Submitting Job {i+1}/{total_jobs}: {job_name} ---")

        date_str = datetime.now().strftime('%Y-%m-%d')
        log_folder = REPO_DIR / "slurm_logs_scaling_subjects" / date_str / job_name
        
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
        print(f"  > Submitted Job ID: {job.job_id} for experiment: {exp_config['experiment_name']}")
        print(f"  > SLURM logs will be in: {log_folder}")


# ==========================================================================
# MAIN EXECUTION
# ==========================================================================

def main():
    """Main execution function to generate and submit SLURM jobs."""
    parser = argparse.ArgumentParser(description="Submit Pre-training Subject Scaling Analysis Jobs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configurations instead of submitting jobs."
    )
    cli_args = parser.parse_args()

    experiments_to_run = setup_experiment_grid()
    total_jobs = len(experiments_to_run)
    
    print(f"--- Generated {total_jobs} total experiment configurations for subject scaling analysis. ---")

    if cli_args.dry_run:
        print_dry_run_summary(experiments_to_run)
        sys.exit(0)

    submit_experiments(experiments_to_run)
    print(f"\nAll {total_jobs} subject scaling analysis jobs have been submitted.")


if __name__ == "__main__":
    main()