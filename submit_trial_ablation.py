"""
Generates and submits SLURM jobs for evaluating model performance across different trial counts.
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
from typing import Dict, List, Any

import submitit

# ==========================================================================
# CORE CONFIGURATION
# ==========================================================================

REPO_DIR = Path(f"/mnt/lustre/work/macke/{os.environ.get('USER', 'user')}/repos/eegjepa")
CONDA_ENV = "timeseries"
SCRIPT_TO_RUN = REPO_DIR / "EDAPT_neurips/EDAPT/train_transfer.py"
BASE_OUTPUT_DIR = REPO_DIR / "EDAPT_neurips/EDAPT/results_scaling"

# SLURM & JOB CONFIGURATION
SLURM_PARTITION = "a100-galvani"
MEM_GB_PER_GPU = 96
CPUS_PER_GPU = 8
DEFAULT_DEVICE = "cuda"
PYTHON_EXECUTABLE = "python"


def dict_to_cli_args(args_dict: Dict[str, Any]) -> str:
    """Convert dictionary to OmegaConf CLI arguments string."""
    parts = []
    for key, value in args_dict.items():
        if value is None:
            continue
        
        if isinstance(value, bool):
            parts.append(f"{key}={str(value).lower()}")
        elif isinstance(value, list):
            formatted_elements = []
            for item in value:
                if isinstance(item, str):
                    formatted_elements.append(f'"{item}"')
                else:
                    formatted_elements.append(str(item))
            list_str = f"[{','.join(formatted_elements)}]"
            parts.append(f"{key}={shlex.quote(list_str)}")
        else:
            parts.append(f"{key}={shlex.quote(str(value))}")
    
    return " ".join(parts)


def run_transfer_job(experiment_config: Dict[str, Any]) -> str:
    """Execute transfer learning script with specified configuration."""
    exp_name = experiment_config.get("experiment_name", "default_exp")
    output_dir_job = Path(experiment_config.get("base_output_dir"))
    output_dir_job.mkdir(parents=True, exist_ok=True)

    # Prepare CLI arguments
    cli_args_to_pass = {k: v for k, v in experiment_config.items() 
                       if k != "base_output_dir" and v is not None}
    cli_args_to_pass["base_output_dir"] = str(output_dir_job)
    cli_args_str = dict_to_cli_args(cli_args_to_pass)

    # Setup logging
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    log_file_dir = REPO_DIR / "job_logs_scaling"
    log_file_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_file_dir / f"scaling_log_{exp_name}_{job_id}.txt"

    # Execute command
    cmd = f"""
set -e
echo "------ Environment Setup ------" > "{log_file}" && \\
echo "Job ID: {job_id}" >> "{log_file}" && \\
echo "Running on node: $(hostname)" >> "{log_file}" && \\
echo "User: $(whoami)" >> "{log_file}" && \\
echo "Conda Env: {CONDA_ENV}" >> "{log_file}" && \\
echo "Activating Conda env..." >> "{log_file}" && \\
source ~/.bashrc && \\
conda activate {CONDA_ENV} && \\
echo "Changing to repo directory..." >> "{log_file}" && \\
cd "{REPO_DIR}" && \\
echo "Setting MKL environment variables..." >> "{log_file}" && \\
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


def generate_job_name(params: Dict[str, Any]) -> str:
    """Generate short, readable SLURM job name."""
    model_abbr = {
        "ShallowConvNet": "SCN", 
        "DeepConvNet": "DCN", 
        "EEGNetv4": "EEGNet", 
        "ATCNet": "ATC"
    }
    
    model_name_abbr = model_abbr.get(params.get("models_to_run")[0], "Model")
    ds_name = (params.get("dataset_names", "DS")
               .replace("Lee2019_", "MI")
               .replace("Yang2025", "Y25"))
    config_tag = params.get("custom_config_tag", "Cfg")
    safe_config_tag = re.sub(r'[^a-zA-Z0-9]', '', config_tag)
    num_trials = params.get("num_trials_per_subject", "allT")
    
    return f"Scl-{model_name_abbr}-{ds_name}-NT{num_trials}-{safe_config_tag}"[:80]


def setup_experiment_grid() -> List[Dict[str, Any]]:
    """Configure experimental grid for scaling analysis."""
    # Grid search parameters
    N_SPLITS = 2
    BASE_GRID_DIR = "scaling_studies_trials_lr_1e-4_finetune_warmup_20"
    DATASETS = [["MAMEM2"], ["Kalunga2016"], ["Lee2019_MI"], ["Yang2025"], ["BI2015a"], ["Huebner2017"], ["Huebner2018"]]
    PRETRAIN_FLAG = False  # False means do pretraining
    MODELS_TO_EVALUATE = ["ShallowConvNet", "EEGNetv4", "ATCNet", "DeepConvNet"]

    # Dataset-specific trial counts based on figures
    DATASET_TRIAL_COUNTS = {
        "MAMEM2": [40, 60, 80],
        "Kalunga2016": [30, 40, 50, 64],
        "Lee2019_SSVEP": [50, 100, 150, 200],
        "BNCI2014_001": [50, 150, 300, 450],
        "Lee2019_MI": [50, 100, 150],
        "Yang2025": [50, 150, 300, 450, 600],
        "BI2015a": [50, 250, 500, 750, 1044],
        "Huebner2017": [2000, 4500, 7000, 9500],
        "Huebner2018": [3000, 6000, 9000, 12000],
    }

    # Alignment configurations
    ALIGNMENT_CONFIGURATIONS = [
        {
            "config_name": "FM-Full_A-None_AdaBN-F", 
            "finetune_mode": "full", 
            "alignment_type": "none", 
            "use_adabn": False
        },
    ]

    # Base configuration
    global_base_config = {
        "n_splits": N_SPLITS,
        "device": DEFAULT_DEVICE,
        "save_checkpoints": False,
        "save_results": True,
        "use_tta": True,
        "use_wandb": True,
        "finetune_epochs": 3,
        "finetune_warmup_trials": 20,
        "lr_finetune": 1e-4,
        "wandb_project": "trial_scaling_analysis_final",
        "seed": 42,
    }

    # Generate experiment configurations
    experiments = []
    for alignment_config, model, dataset_list in itertools.product(
        ALIGNMENT_CONFIGURATIONS, MODELS_TO_EVALUATE, DATASETS
    ):
        current_dataset_name = dataset_list[0] if len(dataset_list) == 1 else dataset_list
        trial_counts = DATASET_TRIAL_COUNTS.get(current_dataset_name, [40, 60, 80])
        
        # --- CORRECTED LOGIC ---
        # This loop now correctly creates a full configuration for each trial count.
        for num_trials in trial_counts:
            # Start with a fresh copy of the base config for each experiment
            exp_config = global_base_config.copy()
            
            # Apply alignment settings
            exp_config.update({k: v for k, v in alignment_config.items() if k != "config_name"})

            pretrain_suffix = "_no_pretrain" if PRETRAIN_FLAG else ""

            # Update with specific parameters for this job
            exp_config.update({
                "models_to_run": [model],
                "dataset_names": current_dataset_name,
                "no_pretrain": PRETRAIN_FLAG,
                "num_trials_per_subject": num_trials,
                "custom_config_tag": alignment_config['config_name'],
                "wandb_group": (f"ScalingFinal_{model}_{current_dataset_name}_NT{num_trials}_"
                               f"{alignment_config['config_name']}{pretrain_suffix}")
            })
            
            # Generate unique names and output directories
            raw_exp_name = (f"ScalingEval_{model}_{current_dataset_name}_NT{num_trials}_"
                           f"{alignment_config['config_name']}{pretrain_suffix}")
            exp_config["experiment_name"] = re.sub(r'[^a-zA-Z0-9_.-]+', '', raw_exp_name)
            exp_config["base_output_dir"] = str(BASE_OUTPUT_DIR / BASE_GRID_DIR / exp_config["experiment_name"])
            
            # Add the fully configured experiment to the list
            experiments.append(exp_config)

    return experiments


def print_dry_run_summary(experiments: List[Dict[str, Any]]) -> None:
    """Print experiment configurations for dry run."""
    total_jobs = len(experiments)
    print(f"\n--- DRY RUN: Would submit {total_jobs} jobs. ---")
    
    for i, cfg in enumerate(experiments[:min(3, total_jobs)]):
        print(f"\n--- Example Config {i+1} ({cfg['experiment_name']}) ---")
        for key, val in sorted(cfg.items()):
            print(f"    {key}: {val}")
        print(f"    > Output would be in: {cfg['base_output_dir']}")
        
        cli_args_to_pass = {k: v for k, v in cfg.items() if k != "base_output_dir"}
        cli_args_to_pass["base_output_dir"] = cfg['base_output_dir']
        cmd_preview = f"{PYTHON_EXECUTABLE} {SCRIPT_TO_RUN} {dict_to_cli_args(cli_args_to_pass)}"
        print(f"    > Command preview: {cmd_preview}")


def submit_experiments(experiments: List[Dict[str, Any]]) -> None:
    """Submit all experiment configurations to SLURM."""
    total_jobs = len(experiments)
    
    for i, exp_config in enumerate(experiments):
        job_name = re.sub(r'[^a-zA-Z0-9_.-]+', '', exp_config["experiment_name"])[:100]
        
        print(f"\n--- Submitting Job {i+1}/{total_jobs}: {job_name} ---")

        # Setup logging directories
        date_str = datetime.now().strftime('%Y-%m-%d')
        slurm_log_folder_base = REPO_DIR / "slurm_logs_scaling" / date_str
        log_folder = slurm_log_folder_base / job_name
        
        # Configure executor
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

        # Submit job
        job = executor.submit(run_transfer_job, exp_config)
        print(f"  > Submitted Job ID: {job.job_id} for experiment: {exp_config['experiment_name']}")
        print(f"  > SLURM logs will be in: {log_folder}")
        print(f"  > Experiment results base dir: {exp_config['base_output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit Scaling Analysis Jobs")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Print configurations instead of submitting.")
    cli_args = parser.parse_args()

    # Generate experiment configurations
    experiments_to_run = setup_experiment_grid()
    total_jobs = len(experiments_to_run)
    
    print(f"--- Generated {total_jobs} total experiment configurations for scaling analysis. ---")

    if cli_args.dry_run:
        print_dry_run_summary(experiments_to_run)
        sys.exit(0)

    # Submit all experiments
    submit_experiments(experiments_to_run)
    print(f"\nAll {total_jobs} scaling analysis jobs submitted.")