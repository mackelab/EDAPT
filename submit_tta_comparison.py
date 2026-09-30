#!/usr/bin/env python3
"""
Submitit script for TTA comparison experiments (Euclidean alignment version).
PARALLELIZED BY DATASET + MODEL: 9 methods × 3 datasets × 4 models = 108 jobs

9 unique experiments testing combinations of:
- Alignment: none vs euclidean
- AdaBN: off vs on (buffer=32 new, buffer=1 old)
- CFT: off vs on

Each job runs 1 dataset and 1 model = 108 total jobs (~8h each).
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import submitit

# SLURM Configuration
PARTITION = os.environ.get("EDAPT_SLURM_PARTITION", "gpu")
MEM_GB = 160  # Use higher memory for all (Yang2025 needs it)
CPUS_PER_TASK = 8
GPUS_PER_NODE = 1
TIMEOUT_HOURS = 24  # Reduced timeout since each job only runs 1 dataset + 1 model

# Experiment Configuration
DATASETS = ["Yang2025", "Lee2019_SSVEP", "BI2015a"]
MODELS = ["ShallowConvNet", "EEGNetv4", "DeepConvNet", "ATCNet"]
N_FOLDS = 2
RANDOM_SEED = 42

# Output directory
REPO_DIR = Path(os.environ.get("EDAPT_ROOT", Path(__file__).resolve().parent))
OUTPUT_DIR = REPO_DIR / "results" / "tta_comparison_eucl_new"


def get_experiment_configs() -> List[Dict]:
    """Generate all 108 experiment configurations (9 methods × 3 datasets × 4 models).

    | Exp | Name              | alignment  | adabn | buffer | CFT   |
    |-----|-------------------|------------|-------|--------|-------|
    | 01  | baseline          | none       | False | -      | False |
    | 02  | adabn_noalign     | none       | True  | 32     | False |
    | 03  | adabn_align       | euclidean  | True  | 32     | False |
    | 04  | cft_noalign       | none       | False | -      | True  |
    | 05  | cft_align         | euclidean  | False | -      | True  |
    | 06  | full_new          | euclidean  | True  | 32     | True  |
    | 07  | adabn_cft_noalign | none       | True  | 32     | True  |
    | 08  | adabn_old_align   | euclidean  | True  | 1      | False |
    | 09  | full_old          | euclidean  | True  | 1      | True  |
    """
    configs = []

    # Define all 9 method configurations
    method_configs = [
        # (exp_num, name, alignment, use_adabn, buffer_length, enable_cft)
        (1, "baseline", "none", False, 32, False),
        (2, "adabn_noalign", "none", True, 32, False),
        (3, "adabn_align", "euclidean", True, 32, False),
        (4, "cft_noalign", "none", False, 32, True),
        (5, "cft_align", "euclidean", False, 32, True),
        (6, "full_new", "euclidean", True, 32, True),
        (7, "adabn_cft_noalign", "none", True, 32, True),
        (8, "adabn_old_align", "euclidean", True, 1, False),
        (9, "full_old", "euclidean", True, 1, True),
    ]

    # Generate configs for each method × dataset × model combination
    for (
        exp_num,
        name,
        alignment,
        use_adabn,
        buffer_length,
        enable_cft,
    ) in method_configs:
        for dataset in DATASETS:
            for model in MODELS:
                exp_name = f"{exp_num:02d}_{name}"
                job_name = f"{exp_name}_{dataset}_{model}"

                # use_tta=true only when alignment or adabn is active
                use_tta = alignment != "none" or use_adabn

                configs.append(
                    {
                        "exp_num": exp_num,
                        "exp_name": exp_name,
                        "job_name": job_name,
                        "dataset": dataset,
                        "model": model,
                        "config": {
                            "dataset_names": [dataset],
                            "models_to_run": [model],
                            "n_splits": N_FOLDS,
                            "seed": RANDOM_SEED,
                            "base_output_dir": str(OUTPUT_DIR),
                            "experiment_name": exp_name,
                            # TTA settings
                            "use_tta": use_tta,
                            "alignment_type": alignment,
                            "use_adabn": use_adabn,
                            "adabn_mode": "train_mode",
                            "tta_buffer_length": buffer_length,
                            # CFT settings
                            "finetune_mode": "full" if enable_cft else "none",
                            "finetune_warmup_trials": 20,
                            # Hyperparameters matching main results
                            "window_size": 100,
                            "lr_finetune": 0.0001,
                        },
                    }
                )

    return configs


def build_command(config: Dict) -> List[str]:
    """Build the train_transfer.py command from config dict."""
    return [
        sys.executable,
        str(REPO_DIR / "train_transfer.py"),
        f"dataset_names={config['dataset_names']}",
        f"models_to_run={config['models_to_run']}",
        f"n_splits={config['n_splits']}",
        f"seed={config['seed']}",
        f"base_output_dir={config['base_output_dir']}",
        f"experiment_name={config['experiment_name']}",
        f"use_tta={config['use_tta']}",
        f"alignment_type={config['alignment_type']}",
        f"use_adabn={config['use_adabn']}",
        f"adabn_mode={config['adabn_mode']}",
        f"tta_buffer_length={config['tta_buffer_length']}",
        f"finetune_mode={config['finetune_mode']}",
        f"finetune_warmup_trials={config['finetune_warmup_trials']}",
        f"window_size={config['window_size']}",
        f"lr_finetune={config['lr_finetune']}",
    ]


def run_experiment(job_name: str, config: Dict, stagger_seconds: int = 0):
    """Run a single experiment configuration."""
    import subprocess
    import sys
    import time

    # Stagger start to avoid timestamp collisions in output directory
    if stagger_seconds > 0:
        print(f"Staggering start by {stagger_seconds}s to avoid timestamp collision...")
        time.sleep(stagger_seconds)

    cmd = build_command(config)

    log_file = Path(config["base_output_dir"]) / f"{job_name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running: {job_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    with open(log_file, "w") as f:
        f.write(f"Job: {job_name}\n")
        f.write(f"Started: {datetime.now()}\n")
        f.write(f"Command: {' '.join(cmd)}\n\n")
        f.flush()

        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

        f.write(f"\nFinished: {datetime.now()}\n")
        f.write(f"Return code: {result.returncode}\n")

    return result.returncode == 0


def check_existing_results(configs: List[Dict]) -> List[Dict]:
    """Check which configs have incomplete results."""
    incomplete = []

    for cfg in configs:
        exp_name = cfg["exp_name"]
        dataset = cfg["dataset"]
        model = cfg["model"]
        result_dir = OUTPUT_DIR / exp_name

        # Check if results exist for this specific dataset + model
        has_results = False

        # Check timestamped subdirectories (train_transfer.py default behavior)
        if result_dir.exists():
            for subdir in result_dir.iterdir():
                if subdir.is_dir():
                    # Check if results_detailed.csv contains this dataset and model
                    detailed_csv = subdir / "results_detailed.csv"
                    if detailed_csv.exists():
                        try:
                            with open(detailed_csv, "r") as f:
                                content = f.read()
                                if dataset in content and model in content:
                                    has_results = True
                                    break
                        except Exception:
                            pass

        if not has_results:
            incomplete.append(cfg)

    return incomplete


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Submit TTA comparison jobs (108 parallel jobs)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print jobs without submitting"
    )
    parser.add_argument(
        "--submit-incomplete",
        action="store_true",
        help="Only submit incomplete/missing jobs",
    )
    parser.add_argument(
        "--local", action="store_true", help="Run locally instead of SLURM"
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated list of method numbers to run (e.g., '1,2,3' or '5,6,7,8,9')",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of datasets to run (e.g., 'Yang2025,Lee2019_SSVEP')",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to run (e.g., 'ShallowConvNet,EEGNetv4')",
    )
    args = parser.parse_args()

    # Generate configs
    configs = get_experiment_configs()

    # Filter by methods if specified
    if args.methods:
        method_nums = [int(x.strip()) for x in args.methods.split(",")]
        configs = [c for c in configs if c["exp_num"] in method_nums]

    # Filter by datasets if specified
    if args.datasets:
        dataset_list = [x.strip() for x in args.datasets.split(",")]
        configs = [c for c in configs if c["dataset"] in dataset_list]

    # Filter by models if specified
    if args.models:
        model_list = [x.strip() for x in args.models.split(",")]
        configs = [c for c in configs if c["model"] in model_list]

    # Filter incomplete if requested
    if args.submit_incomplete:
        configs = check_existing_results(configs)
        if not configs:
            print("All jobs already complete!")
            return
        print(f"Found {len(configs)} incomplete jobs to submit")

    print(f"\n{'='*80}")
    print(
        "TTA Comparison Experiment Submission (PARALLELIZED: 9 methods × 3 datasets × 4 models)"
    )
    print(f"{'='*80}")
    print(f"Total jobs: {len(configs)}")
    print(f"Datasets: {DATASETS}")
    print(f"Models: {MODELS}")
    print(f"N_FOLDS: {N_FOLDS}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Timeout: {TIMEOUT_HOURS} hours per job")
    print(f"{'='*80}\n")

    # Submit jobs with staggered starts to avoid timestamp collisions
    jobs = []
    for idx, cfg in enumerate(configs):
        job_name = cfg["job_name"]
        config = cfg["config"]
        dataset = cfg["dataset"]
        stagger_seconds = idx * 2  # 2 second gap between each job

        # Adjust memory based on dataset (Yang2025 needs more)
        mem_gb = MEM_GB if dataset == "Yang2025" else 64

        if args.dry_run:
            cmd = build_command(config)
            print(f"[DRY-RUN] {job_name} (mem={mem_gb}G, stagger={stagger_seconds}s)")
            print(f"  {' '.join(cmd)}")
            print()
        else:
            # Setup executor
            if args.local:
                executor = submitit.LocalExecutor(
                    folder="submitit_logs_tta_eucl"
                )
            else:
                executor = submitit.SlurmExecutor(
                    folder="submitit_logs_tta_eucl"
                )
                executor.update_parameters(
                    partition=PARTITION,
                    mem=f"{mem_gb}G",
                    cpus_per_task=CPUS_PER_TASK,
                    gpus_per_node=GPUS_PER_NODE,
                    time=f"{TIMEOUT_HOURS}:00:00",
                    job_name=job_name,
                )

            job = executor.submit(run_experiment, job_name, config, stagger_seconds)
            jobs.append((job, job_name))
            print(
                f"Submitted: {job_name} (job_id={job.job_id}, stagger={stagger_seconds}s)"
            )

    if not args.dry_run:
        print(f"\n{'='*80}")
        print(f"Submitted {len(jobs)} jobs")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Log directory: submitit_logs_tta_eucl")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
