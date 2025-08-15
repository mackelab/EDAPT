# %%
"""
Generates and submits SLURM jobs for comprehensive evaluation of different
alignment strategies in EEG-based brain-computer interface transfer learning.
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

# %%
# Core configuration

# Repository and environment setup
REPO_DIR = Path(f"/mnt/lustre/work/macke/{os.environ.get('USER', 'user')}/repos/eegjepa")
CONDA_ENV = "timeseries"
SCRIPT_TO_RUN = REPO_DIR / "EDAPT_neurips/EDAPT/train_transfer.py"
BASE_OUTPUT_DIR = REPO_DIR / "EDAPT_neurips/EDAPT/results"

# SLURM configuration
SLURM_PARTITION = "a100-galvani" # or 2080
MEM_GB_PER_GPU = 96
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
                f'"{item}"' if isinstance(item, str) else str(item) 
                for item in value
            ]
            list_str = f"[{','.join(formatted_elements)}]"
            parts.append(f"{cli_key}={shlex.quote(list_str)}")
        else:
            parts.append(f"{cli_key}={shlex.quote(str(value))}")
    
    return " ".join(parts)


def generate_job_name(params: Dict[str, Any]) -> str:
    """Generate short, readable SLURM job name."""
    model_abbr = {
        "ShallowConvNet": "SCN", 
        "DeepConvNet": "DCN", 
        "EEGNetv4": "EEGNet", 
        "ATCNet": "ATC"
    }
    
    model_name = model_abbr.get(params.get("models_to_run")[0], "Model")
    ds_name = params["dataset_names"][0].replace("Lee2019_", "MI").replace("Yang2025", "Y25")
    config_tag = params.get("custom_config_tag", "Cfg")
    safe_config_tag = re.sub(r'[^a-zA-Z0-9]', '', config_tag)
    
    return f"{model_name}-{ds_name}-{safe_config_tag}"[:80]


def run_transfer_job(experiment_config: Dict[str, Any]) -> str:
    """Execute transfer learning job with environment setup and logging."""
    exp_name = experiment_config.get("experiment_name", "default_exp")
    output_dir = Path(experiment_config.get("base_output_dir", BASE_OUTPUT_DIR / exp_name))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cli_args_str = dict_to_cli_args({k: v for k, v in experiment_config.items() if v is not None})
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    
    log_file_dir = REPO_DIR / "job_logs"
    log_file_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_file_dir / f"transfer_log_{exp_name}_{job_id}.txt"

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
        print(f"ERROR: Command failed with exit code {e.returncode}. Check log: {log_file}")
        raise RuntimeError(f"Script for '{exp_name}' failed. Log: {log_file}") from e
    
    return f"Finished: {exp_name}"


# %%
# Experiment configuration

def get_alignment_configurations() -> List[Dict[str, Any]]:
    """Define alignment strategy configurations for evaluation."""
    return [
        {
            "config_name": "FM-None_A-None_AdaBN-F", 
            "finetune_mode": "none", 
            "alignment_type": "none", 
            "use_adabn": False
        },
        {
            "config_name": "FM-None_A-Eucl_AdaBN-T", 
            "finetune_mode": "none", 
            "alignment_type": "euclidean", 
            "use_adabn": True
        },
        {
            "config_name": "FM-Dec_A-None_AdaBN-F", 
            "finetune_mode": "decision_only", 
            "alignment_type": "none", 
            "use_adabn": False
        },
        {
            "config_name": "FM-Dec_A-Riem_AdaBN-T", 
            "finetune_mode": "decision_only", 
            "alignment_type": "riemannian", 
            "use_adabn": True
        },
        {
            "config_name": "FM-Dec_A-Eucl_AdaBN-F", 
            "finetune_mode": "decision_only", 
            "alignment_type": "euclidean", 
            "use_adabn": False
        },
        {
            "config_name": "FM-Full_A-None_AdaBN-F", 
            "finetune_mode": "full", 
            "alignment_type": "none", 
            "use_adabn": False
        },
        {
            "config_name": "FM-Full_A-Riem_AdaBN-T", 
            "finetune_mode": "full", 
            "alignment_type": "riemannian", 
            "use_adabn": True
        },
        {
            "config_name": "FM-Full_A-Eucl_AdaBN-T", 
            "finetune_mode": "full", 
            "alignment_type": "euclidean", 
            "use_adabn": True
        },
        {
            "config_name": "FM-Full_A-Riem_AdaBN-F", 
            "finetune_mode": "full", 
            "alignment_type": "riemannian", 
            "use_adabn": False
        },
        {
            "config_name": "FM-Full_A-Eucl_AdaBN-F", 
            "finetune_mode": "full", 
            "alignment_type": "euclidean", 
            "use_adabn": False
        },
    ]


def get_base_config() -> Dict[str, Any]:
    """Define base configuration shared across all experiments."""
    return {
        "n_splits": 2,
        "device": DEFAULT_DEVICE,
        "save_checkpoints": False,
        "save_results": True,
        "use_tta": True,
        "use_wandb": True,
        "wandb_project": "alignment_studies_final",
        "lr_finetune": 1e-4,
        "finetune_warmup_trials": 20,
    }


def generate_experiment_configs() -> List[Dict[str, Any]]:
    """Generate all experiment configurations for the grid search."""
    # Experiment parameters
    datasets = [
        ["Yang2025"], ["Kalunga2016"], ["BI2015a"], ["Lee2019_MI"], 
        ["BNCI2014_001"], ["Huebner2017"], ["Huebner2018"], ["MAMEM2"]
    ]
    pretrain_settings = [False]  # False means using pretrained models
    models_to_evaluate = ["ShallowConvNet", "EEGNetv4", "ATCNet", "DeepConvNet"]
    alignment_configurations = get_alignment_configurations()
    
    base_grid_dir = "alignment_studies_lr_1e-4_finetune_warmup_20"
    global_base_config = get_base_config()
    
    experiments = []
    
    for model, dataset, no_pretrain, align_conf in itertools.product(
        models_to_evaluate, datasets, pretrain_settings, alignment_configurations
    ):
        exp_config = global_base_config.copy()
        exp_config.update({k: v for k, v in align_conf.items() if k != "config_name"})
        
        pretrain_suffix = "_no_pretrain" if no_pretrain else ""
        
        exp_config.update({
            "models_to_run": [model],
            "dataset_names": dataset,
            "no_pretrain": no_pretrain,
            "custom_config_tag": align_conf['config_name'],
            "wandb_group": f"AlignFinal_{align_conf['config_name']}{pretrain_suffix}"
        })
        
        # Generate safe experiment name
        raw_exp_name = f"AlignEval_{model}_{dataset[0]}_{align_conf['config_name']}{pretrain_suffix}"
        exp_config["experiment_name"] = re.sub(r'[^a-zA-Z0-9_.-]+', '', raw_exp_name)
        exp_config["base_output_dir"] = str(BASE_OUTPUT_DIR / base_grid_dir / exp_config["experiment_name"])
        
        experiments.append(exp_config)
    
    return experiments


# %%
# Main execution

def main():
    """Main execution function for job submission."""
    parser = argparse.ArgumentParser(
        description="Submit Transfer Learning Jobs for Alignment Studies"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Print configurations instead of submitting jobs"
    )
    cli_args = parser.parse_args()
    
    # Generate experiment configurations
    experiments_to_run = generate_experiment_configs()
    total_jobs = len(experiments_to_run)
    
    print(f"Generated {total_jobs} total experiment configurations.")
    
    if cli_args.dry_run:
        print(f"\nDRY RUN: Would submit {total_jobs} jobs.")
        print("\nExample configurations:")
        
        for i, cfg in enumerate(experiments_to_run[:3]):
            print(f"\n--- Example Config {i+1} ---")
            for key, val in sorted(cfg.items()):
                print(f"    {key}: {val}")
        
        sys.exit(0)
    
    # Submit jobs
    print(f"\nSubmitting {total_jobs} jobs to SLURM...")
    
    for i, exp_config in enumerate(experiments_to_run):
        job_name = generate_job_name(exp_config)
        print(f"\nSubmitting Job {i+1}/{total_jobs}: {job_name}")
        
        # Setup logging directory
        log_folder = REPO_DIR / "slurm_logs" / f"{datetime.now().strftime('%Y-%m-%d')}_{job_name}"
        executor = submitit.AutoExecutor(folder=str(log_folder))
        
        # Configure SLURM parameters
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
        print(f"  Submitted Job ID: {job.job_id}")
        print(f"  Experiment: {exp_config['experiment_name']}")
        print(f"  SLURM logs: {log_folder}")
    
    print(f"\nAll {total_jobs} jobs submitted successfully.")


if __name__ == "__main__":
    main()