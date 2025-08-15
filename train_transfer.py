# %%
import argparse
import copy
import logging
import os
import sys
import time
import warnings
import gc
import psutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import lovely_tensors
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

# Import Base Paradigm Classes
from moabb_datasets import *

from omegaconf import ListConfig, OmegaConf
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from tqdm.auto import tqdm
from utils import (ContinuousLearnerBinary, evaluate_single_trial, evaluate_zero_shot, 
                  filter_args_for_model, get_checkpoint_dir, get_model_class, 
                  get_output_dir, save_checkpoint, save_results_df)

from models.builder import build_model
from models.models import (ATCNet, DeepConvNet, EEGNetv4, ShallowConvNet)
from tta_wrapper import TTAWrapper

# Optional imports with availability check
try:
    import pyriemann
    from pyriemann.estimation import Covariances
    from pyriemann.utils.mean import mean_riemann
    PYRIEMANN_AVAILABLE = True
except ImportError:
    PYRIEMANN_AVAILABLE = False
    print("[yellow]Warning: pyriemann not installed. Riemannian initialization will not be available.[/yellow]")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[yellow]Warning: wandb not installed. Skipping wandb logging.[/yellow]")

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
mne.set_log_level("ERROR")
lovely_tensors.monkey_patch()

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# %%
# Data handling and model utilities

class DictDataset(Dataset):
    """Simple dataset wrapper for epochs and labels."""
    
    def __init__(self, epochs_tensor: torch.Tensor, labels_tensor: torch.Tensor):
        """Initialize dataset with epochs and classification labels."""
        self.epochs = epochs_tensor
        
        # Ensure labels are long tensor for classification
        if labels_tensor.dtype != torch.long:
            log.warning(f"Converting labels from {labels_tensor.dtype} to long")
            labels_tensor = labels_tensor.long()
        
        # Remap labels to be 0-indexed if needed
        unique_labels = torch.unique(labels_tensor)
        if len(unique_labels) > 0 and torch.min(unique_labels) > 0:
            label_mapping = {label.item(): idx for idx, label in enumerate(sorted(unique_labels))}
            remapped_labels = [label_mapping[label.item()] for label in labels_tensor]
            self.labels = torch.tensor(remapped_labels, dtype=torch.long)
        else:
            self.labels = labels_tensor
            
        if len(self.epochs) != len(self.labels):
            raise ValueError(f"Epochs and labels length mismatch: {len(self.epochs)} != {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"epoch": self.epochs[index], "label": self.labels[index]}


def create_dataloader(epochs, labels, batch_size, shuffle_data=True):
    """Create a DataLoader for training or evaluation."""
    if epochs is None or labels is None or epochs.size == 0 or labels.size == 0:
        return None
        
    epochs_tensor = torch.from_numpy(epochs).float()
    labels_tensor = torch.from_numpy(labels).long()
    
    dataset = DictDataset(epochs_tensor, labels_tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_data,
        num_workers=0,
        pin_memory=True,
    )


def pretrain_model(model, train_loader, optimizer, n_epochs, device, console, log, run_name_suffix):
    """Pretrain a model using supervised learning."""
    model.to(device)
    model.train()
    
    criterion = torch.nn.CrossEntropyLoss()
    console.print(f"    Starting Pretraining ({run_name_suffix}) for classification task...")
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            X_batch = batch['epoch'].to(device)
            y_batch = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        console.print(f"    Epoch {epoch+1}/{n_epochs} completed. Avg Loss: {avg_epoch_loss:.4f}")
        log.info(f"Pretrain Epoch {epoch+1}/{n_epochs} ({run_name_suffix}): Avg Loss = {avg_epoch_loss:.4f}")
    
    log.info(f"Pretraining completed for {run_name_suffix}.")
    return model


def train_finetuning_step(model, loader, optimizer, device, args, trial_idx, wandb_run):
    """Perform supervised finetuning step."""
    total_loss = 0.0
    total_batches = 0
    
    if len(loader) == 0:
        log.warning(f"Finetune step for trial {trial_idx}: Loader is empty. Skipping.")
        return model, 0.0
    
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(args.finetune_epochs):
        epoch_loss = 0.0
        
        batch_iterator = tqdm(
            loader, 
            desc=f"Finetune Trial {trial_idx} Epoch {epoch+1}/{args.finetune_epochs}", 
            leave=False
        )
        
        for batch_idx, batch_data in enumerate(batch_iterator):
            x_batch = batch_data["epoch"].to(device, non_blocking=True)
            y_batch = batch_data["label"].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch, is_finetuning_batch=True)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_iterator.set_postfix(loss=loss.item())
        
        total_loss += epoch_loss
        total_batches += len(loader)
    
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    
    if wandb_run:
        try:
            wandb_run.log({
                "finetune/step_loss": avg_loss,
                "finetune/lr": optimizer.param_groups[0]["lr"],
            }, step=trial_idx, commit=True)
        except Exception as e:
            log.warning(f"WandB logging failed: {e}")
    
    return model, avg_loss


# %%
# Experiment setup

def setup_experiment(cli_args=None):
    """Parse arguments, setup configuration, and initialize experiment."""
    
    DEFAULT_YAML = """
dataset_names: ["BNCI2014_001"]
subjects: null
data_root: "/mnt/lustre/home/macke/${oc.env:USER}/mne_data"
fmin: 1.0
fmax: 47.0
resample: null
models_to_run:
  - ShallowConvNet
  - DeepConvNet
  - EEGNetv4
  - ATCNet
n_splits: 2
pretrain_epochs: 100
lr_pretrain: 0.0003
optimizer_type_pretrain: "Adam"
weight_decay_pretrain: 0.0
batch_size_pretrain: 64
window_size: 50
finetune_epochs: 3
finetune_warmup_trials: 0
lr_finetune: 0.0003
optimizer_type_finetune: "Adam" # Options: "Adam", "AdamW"
weight_decay_finetune: 0.0
batch_size_finetune: 50 
seed: 42
device: "cuda"
print_dataset_structure_and_exit: false
no_pretrain: false
base_output_dir: "results"
experiment_name: "transfer_kfold"
save_checkpoints: false
save_results: true
use_wandb: false
wandb_project: "kfold_transfer"
wandb_group_prefix: "kfold_transfer"
wandb_run_description: "Default run"
num_pretrain_subjects: "max"
num_finetuning_subjects: null
num_trials_per_subject: null

# TTA Configuration
use_tta: true
alignment_type: "euclidean" # Options: "none", "euclidean", "riemannian" 
alignment_cov_epsilon: 1.0e-6
alignment_transform_epsilon: 1.0e-7
alignment_ref_ema_beta: 0.9
use_adabn: false
finetune_mode: "full" # Options: "full", "decision_only", "none"
"""
    
    # Load default config
    config = OmegaConf.create(DEFAULT_YAML)

    # Suppress verbose MOABB warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="moabb")
    warnings.filterwarnings("ignore", message="warnEpochs*")

    # Set MOABB loggers to ERROR level
    logging.getLogger("moabb").setLevel(logging.ERROR)
    logging.getLogger("moabb.paradigms").setLevel(logging.ERROR)
    logging.getLogger("moabb.datasets").setLevel(logging.ERROR)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="K-Fold Transfer with YAML Config")
    parser.add_argument("-c", "--config", action="append", help="Path to YAML config file(s)", default=[])
    parser.add_argument("--print-dataset-structure-and-exit", action="store_true")
    
    parsed_args, remaining_argv = parser.parse_known_args(args=cli_args)
    
    # Load and merge config files
    if parsed_args.config:
        for config_file in parsed_args.config:
            try:
                user_config = OmegaConf.load(config_file)
                config = OmegaConf.merge(config, user_config)
                print(f"Loaded config from: {config_file}")
            except Exception as e:
                print(f"Failed to load config {config_file}: {e}")
    
    # Apply CLI overrides
    if remaining_argv:
        try:
            cli_conf = OmegaConf.from_cli(remaining_argv)
            if cli_conf:
                config = OmegaConf.merge(config, cli_conf)
        except Exception as e:
            print(f"Failed to parse CLI overrides: {e}")
    
    if parsed_args.print_dataset_structure_and_exit:
        config.print_dataset_structure_and_exit = True
    
    OmegaConf.resolve(config)
    
    # Set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Create output directory
    run_output_dir = get_output_dir(
        base_output_root=config.base_output_dir,
        experiment_name=config.experiment_name,
        timestamp=True,
    )
    console = Console()
    console.print(f"[blue]Output directory: {run_output_dir}[/blue]")
    
    # Save config
    try:
        config_save_path = run_output_dir / "config.yaml"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, config_save_path)
    except Exception as e:
        print(f"Failed to save config: {e}")
    
    # Setup device
    device = torch.device(config.device)
    if config.device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]CUDA not available. Switching to CPU.[/yellow]")
        device = torch.device("cpu")
        config.device = "cpu"
    
    # Environment flags
    config.pyriemann_available = PYRIEMANN_AVAILABLE
    if config.use_wandb and not WANDB_AVAILABLE:
        console.print("[yellow]WandB not available. Disabling.[/yellow]")
        config.use_wandb = False
    config.wandb_available = WANDB_AVAILABLE
    
    # Process subjects and dataset names
    if config.subjects is not None:
        if isinstance(config.subjects, int):
            config.subjects = [config.subjects]
        elif isinstance(config.subjects, (list, ListConfig)):
            config.subjects = [int(s) for s in config.subjects]
    
    if not isinstance(config.dataset_names, (list, ListConfig)):
        if isinstance(config.dataset_names, str):
            config.dataset_names = [d.strip() for d in config.dataset_names.split(",") if d.strip()]
    config.dataset_names = [str(d) for d in config.dataset_names]
    
    return config, device, console, run_output_dir


# %%
# Memory and pretraining utilities

def log_memory_usage(stage: str, log_obj=None):
    """Log current memory usage for debugging."""
    if log_obj:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        log_obj.debug(f"Memory usage at {stage}: {memory_mb:.1f} MB")


def run_fold_pretraining(dataset_name, fold_idx, train_subject_ids, args, device, console, run_output_dir):
    """Load data and pretrain models for one fold."""
    pretrained_models_fold = {}
    n_channels, n_timepoints = -1, -1
    n_outputs_model = 0
    unique_labels_for_return = None,
    apply_trial_ablation = True,

    
    log_memory_usage(f"start_pretraining_fold_{fold_idx+1}", log)
    
    # Determine actual subjects for pretraining
    actual_pretrain_subject_ids = list(train_subject_ids)
    num_subjects_to_pretrain_on = getattr(args, "num_pretrain_subjects", "max")
    
    if isinstance(num_subjects_to_pretrain_on, int) and num_subjects_to_pretrain_on < len(train_subject_ids):
        if num_subjects_to_pretrain_on > 0:
            rng = np.random.RandomState(args.seed + fold_idx)
            actual_pretrain_subject_ids = rng.choice(
                train_subject_ids, size=num_subjects_to_pretrain_on, replace=False
            ).tolist()
            console.print(f"  Sub-sampling: Using {len(actual_pretrain_subject_ids)} subjects for pretraining")
    
    console.print(f"  Loading pretraining data for {len(actual_pretrain_subject_ids)} subjects...")
    
    try:
        # Load data
        paradigm_kwargs = {"fmin": args.fmin, "fmax": args.fmax, "resample": args.resample}
        if hasattr(args, 'channel_subset') and args.channel_subset:
            paradigm_kwargs['channels'] = args.channel_subset
        
        pretrain_epochs_data, pretrain_labels_data, n_channels, n_timepoints, _ = (
            load_cached_pretrain_data(
                dataset_names=[dataset_name],
                subject_ids=actual_pretrain_subject_ids,
                paradigm_kwargs=paradigm_kwargs,
                data_root=args.data_root,
                args=args,
                verbose=False,
                target_type='classification',
                apply_trial_ablation=apply_trial_ablation,
            )
        )
        
        if pretrain_epochs_data is None or pretrain_epochs_data.size == 0:
            console.print(f"[yellow]No pretraining data loaded. Skipping fold.[/yellow]")
            return {}, -1, -1, 0, None, False
        
        # Process labels
        unique_labels = np.unique(pretrain_labels_data)
        n_outputs_model = len(unique_labels)
        unique_labels_for_return = unique_labels.copy()
        
        if n_outputs_model <= 1:
            console.print(f"[red]Only {n_outputs_model} classes found. Need at least 2.[/red]")
            return {}, -1, -1, 0, None, False
        
        console.print(f"    Found {n_outputs_model} classes: {unique_labels.tolist()}")
        
        # Create dataloader
        pretrain_loader = create_dataloader(
            pretrain_epochs_data, pretrain_labels_data, args.batch_size_pretrain, shuffle_data=True
        )
        
        if pretrain_loader is None or len(pretrain_loader) == 0:
            console.print(f"[yellow]Could not create valid dataloader. Skipping.[/yellow]")
            return {}, -1, -1, 0, None, False
        
        del pretrain_epochs_data, pretrain_labels_data
        gc.collect()
        
        # Train models
        console.print(f"    Training models: {args.models_to_run}")
        base_args_dict = OmegaConf.to_container(args, resolve=True)
        printed_summaries = set()
        
        for model_idx, model_name in enumerate(args.models_to_run):
            console.print(f"      Model {model_idx+1}/{len(args.models_to_run)}: [bold yellow]{model_name}[/bold yellow]")
            
            try:
                # Build model
                model_specific_args = filter_args_for_model(
                    base_args_dict, model_name, get_model_class(model_name)
                )
                model_pretrain = build_model(
                    model_name=model_name, n_channels=n_channels, n_times=n_timepoints,
                    n_outputs=n_outputs_model, device=device,
                    model_specific_args=model_specific_args, target_type='classification',
                )
                
                # Setup optimizer
                optimizer_params = {"lr": args.lr_pretrain, "weight_decay": args.weight_decay_pretrain}
                if args.optimizer_type_pretrain.lower() == "adamw":
                    optimizer_pretrain = torch.optim.AdamW(model_pretrain.parameters(), **optimizer_params)
                else:
                    optimizer_pretrain = torch.optim.Adam(model_pretrain.parameters(), **optimizer_params)
                
                # Print model summary
                summary_key = (dataset_name, model_name)
                if summary_key not in printed_summaries and n_channels > 0 and n_timepoints > 0:
                    try:
                        input_size = (1, n_channels, n_timepoints)
                        console.print(f"        Model Summary: Input ({n_channels}, {n_timepoints}), Output ({n_outputs_model})")
                        summary_str = summary(model_pretrain, input_size=input_size, verbose=0)
                        console.print(str(summary_str))
                        printed_summaries.add(summary_key)
                    except Exception:
                        pass
                
                # Setup WandB if enabled
                pretrain_wandb_run = None
                if args.use_wandb and WANDB_AVAILABLE:
                    try:
                        run_name = f"{args.experiment_name}_Pretrain_{dataset_name}_Fold_{fold_idx+1}_{model_name}"
                        config_wandb = OmegaConf.to_container(args, resolve=True)
                        config_wandb.update({
                            "dataset_name": dataset_name, "fold": fold_idx + 1, 
                            "stage": "pretrain", "model_name": model_name,
                        })
                        pretrain_wandb_run = wandb.init(
                            project=args.wandb_project, name=run_name, config=config_wandb,
                            group=f"{args.wandb_group_prefix}_{dataset_name}_Fold_{fold_idx+1}",
                            job_type=f"Pretrain_{model_name}", reinit=True,
                        )
                    except Exception as e:
                        console.print(f"      [yellow]WandB init failed: {e}[/yellow]")
                
                # Train model
                model_pretrain = pretrain_model(
                    model_pretrain, pretrain_loader, optimizer_pretrain, args.pretrain_epochs,
                    device, console, log, f"{dataset_name}_Fold_{fold_idx+1}_{model_name}"
                )
                
                # Save model state
                pretrained_model_state = copy.deepcopy(model_pretrain.state_dict())
                pretrained_models_fold[model_name] = pretrained_model_state
                
                # Save checkpoint if requested
                if args.save_checkpoints:
                    checkpoint_dir = get_checkpoint_dir(run_output_dir)
                    save_path = checkpoint_dir / f"model_{model_name}_ds_{dataset_name}_fold_{fold_idx+1}_pretrained.pt"
                    save_checkpoint({"model_state_dict": pretrained_model_state}, save_path)
                
            except Exception as e:
                log.error(f"Error training {model_name}: {e}")
                console.print(f"      [red]Failed to train {model_name}: {e}[/red]")
            finally:
                if 'pretrain_wandb_run' in locals() and pretrain_wandb_run:
                    pretrain_wandb_run.finish()
                if 'model_pretrain' in locals():
                    del model_pretrain
                if 'optimizer_pretrain' in locals():
                    del optimizer_pretrain
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        if not pretrained_models_fold:
            console.print(f"[red]No models successfully pretrained.[/red]")
            return {}, -1, -1, 0, None, False
        
        console.print(f"    [green]Successfully pretrained {len(pretrained_models_fold)} models.[/green]")
        return pretrained_models_fold, n_channels, n_timepoints, n_outputs_model, unique_labels_for_return, True
        
    except Exception as e:
        log.error(f"Critical error in pretraining: {e}")
        console.print(f"[red]Critical error: {e}[/red]")
        return {}, -1, -1, 0, None, False
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# %%
# Online finetuning simulation

def run_online_finetuning_simulation(model, test_subj_epochs, test_subj_labels, original_unique_labels, 
                                    args, device, console, log_prefix, wandb_run):
    """Run online finetuning simulation with TTA and supervised learning."""
    if test_subj_epochs is None or test_subj_epochs.size == 0:
        log.error(f"Empty test data for {log_prefix}")
        return np.nan, 0.0, []
    
    n_trials_subj = test_subj_epochs.shape[0]
    log.info(f"Starting online simulation for {log_prefix} ({n_trials_subj} trials)")
    
    # Initialize tracking
    optimizer_finetune = None
    learner = None
    trial_times = []
    epoch_buffer = None
    label_buffer = None
    trial_metrics_log = []
    
    try:
        # Setup finetuning if enabled
        if args.finetune_mode != 'none':
            optimizer_params = {"lr": args.lr_finetune, "weight_decay": args.weight_decay_finetune}
            if args.optimizer_type_finetune.lower() == "adamw":
                optimizer_finetune = torch.optim.AdamW(model.parameters(), **optimizer_params)
            else:
                optimizer_finetune = torch.optim.Adam(model.parameters(), **optimizer_params)
            
            if args.finetune_epochs > 0:
                max_window_size = min(args.window_size, n_trials_subj)
                epoch_buffer = deque(maxlen=max_window_size)
                label_buffer = deque(maxlen=max_window_size)
        
        # Setup learner for performance tracking
        learner = ContinuousLearnerBinary(window_size=args.window_size)
        
        # Main trial loop
        online_iterator = tqdm(range(n_trials_subj), desc=f"Online Sim ({log_prefix})", leave=False)
        
        for trial_idx in online_iterator:
            trial_start_time = time.time()
            
            try:
                # Get current trial data
                single_epoch_np = test_subj_epochs[trial_idx]
                single_label_np = test_subj_labels[trial_idx]
                single_epoch_t = torch.from_numpy(single_epoch_np).float().unsqueeze(0).to(device)
                single_label_t = torch.tensor([single_label_np], dtype=torch.long, device=device)
                
                # TTA prediction
                if args.use_tta:
                    logits_tta = model.update_tta_statistics_and_predict(single_epoch_t)
                else:
                    with torch.no_grad():
                        logits_tta = model(single_epoch_t, apply_tta=False)
                
                # Evaluate prediction
                eval_result = evaluate_single_trial(
                    model.get_wrapped_model(), single_epoch_t, single_label_t, 
                    device, original_unique_labels, output_logits=logits_tta
                )
                
                if "is_correct" in eval_result and not np.isnan(eval_result["is_correct"]):
                    learner.update(eval_result["is_correct"])
                
                rolling_accuracy = learner.get_rolling_accuracy()
                overall_accuracy = learner.get_overall_accuracy()
                
                # Supervised finetuning (after prediction)
                step_loss = 0.0
                if (args.finetune_mode != 'none' and epoch_buffer is not None and 
                    trial_idx >= args.finetune_warmup_trials and args.finetune_epochs > 0 and 
                    optimizer_finetune is not None):
                    
                    # Add to buffer
                    epoch_buffer.append(single_epoch_np)
                    label_buffer.append(single_label_np)
                    
                    if epoch_buffer:  # If buffer not empty
                        current_window_epochs = np.array(epoch_buffer)
                        current_window_labels = np.array(label_buffer)
                        
                        window_loader = create_dataloader(
                            current_window_epochs, current_window_labels,
                            min(args.batch_size_finetune, len(epoch_buffer)), shuffle_data=True
                        )
                        
                        if window_loader and len(window_loader) > 0:
                            model.train()
                            _, step_loss = train_finetuning_step(
                                model=model, loader=window_loader, optimizer=optimizer_finetune,
                                device=device, args=args, trial_idx=trial_idx, wandb_run=wandb_run
                            )
                            model.eval()
                
                # Clean up tensors
                del single_epoch_t, single_label_t, logits_tta
                
                if trial_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                trial_end_time = time.time()
                trial_times.append(trial_end_time - trial_start_time)
                
                # Update progress and log metrics
                if not np.isnan(rolling_accuracy):
                    online_iterator.set_postfix(acc=f"{rolling_accuracy:.3f}")
                    trial_metrics_log.append({
                        'trial_idx': trial_idx,
                        'rolling_metric': rolling_accuracy,
                        'overall_metric_at_trial': overall_accuracy,
                    })
                
                # WandB logging
                if wandb_run:
                    try:
                        wandb_run.log({
                            "finetune/trial_time_sec": trial_end_time - trial_start_time,
                            "finetune/step_train_loss": step_loss,
                            "finetune/eval_loss_current_trial": eval_result.get("loss", np.nan),
                            "finetune/eval_is_correct_current_trial": float(eval_result.get("is_correct", 0)),
                            "finetune/eval_rolling_accuracy": rolling_accuracy,
                            "finetune/eval_overall_accuracy": overall_accuracy
                        }, step=trial_idx, commit=True)
                    except Exception:
                        pass
                        
            except Exception as e:
                log.error(f"Error processing trial {trial_idx}: {e}")
                continue
        
        # Final results
        avg_time_per_trial = np.mean(trial_times) if trial_times else 0.0
        final_overall_accuracy = overall_accuracy if not np.isnan(overall_accuracy) else np.nan
        
        log.info(f"Online simulation finished for {log_prefix}. Final Accuracy: {final_overall_accuracy:.4f}")
        
        if wandb_run:
            try:
                wandb_run.log({
                    "finetune/final_avg_trial_time_sec": avg_time_per_trial,
                    "finetune/final_overall_accuracy": final_overall_accuracy,
                    "finetune/final_rolling_accuracy_at_end": rolling_accuracy
                })
            except Exception:
                pass
        
        return final_overall_accuracy, avg_time_per_trial, trial_metrics_log
        
    except Exception as e:
        log.error(f"Critical error in online simulation for {log_prefix}: {e}")
        return np.nan, 0.0, []
    finally:
        # Cleanup
        if optimizer_finetune is not None:
            del optimizer_finetune
        if learner is not None:
            del learner
        if epoch_buffer is not None:
            del epoch_buffer
        if label_buffer is not None:
            del label_buffer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# %%
# Subject evaluation

def run_subject_evaluation(test_subject_id, fold_idx, pretrained_models_fold, n_channels, n_timepoints, 
                          n_outputs, original_unique_labels_fold, args, device, console, run_output_dir, 
                          no_pretrain, dataset_name):
    """Load data for a test subject and run zero-shot and online finetuning evaluation."""
    subject_results = {}
    for model_name in args.models_to_run:
        subject_results[model_name] = {"zero_shot": np.nan, "finetuned": np.nan}
    
    all_models_trial_metrics = {}
    
    console.print(f"  Processing Test Subject {test_subject_id} (Dataset: {dataset_name}, Fold {fold_idx+1})...")
    
    try:
        # Load test data
        console.print(f"    Loading data for subject {test_subject_id}...")
        paradigm_kwargs = {"fmin": args.fmin, "fmax": args.fmax, "resample": args.resample}
        
        if hasattr(args, 'channel_subset') and args.channel_subset:
            paradigm_kwargs['channels'] = args.channel_subset
        
        test_subj_epochs, test_subj_labels, _, _, _ = load_cached_pretrain_data(
            dataset_names=[dataset_name],
            subject_ids=[test_subject_id],
            paradigm_kwargs=paradigm_kwargs,
            data_root=args.data_root,
            args=args,
            verbose=False,
            target_type='classification',
            apply_trial_ablation=False,
        )
        
        if test_subj_epochs is None or test_subj_epochs.size == 0:
            console.print(f"    [yellow]No valid data for subject {test_subject_id}. Skipping.[/yellow]")
            return subject_results, all_models_trial_metrics
        
        # Extract metadata
        sr_hz_eval = None
        for p_name, p_data in PARADIGM_DATA.items():
            if dataset_name in p_data.get("specs", {}):
                dataset_meta_info = p_data["specs"][dataset_name]
                if "sr" in dataset_meta_info:
                    sr_hz_eval = dataset_meta_info["sr"]
                break
        
        console.print(f"    Subject {test_subject_id}: Found {test_subj_epochs.shape[0]} trials")
        
        # Evaluate each model
        for model_idx, model_name in enumerate(args.models_to_run):
            console.print(f"      Evaluating Model {model_idx+1}/{len(args.models_to_run)}: [bold yellow]{model_name}[/bold yellow]")
            
            model_eval = None
            model_eval_wrapped = None
            wandb_run_subject = None
            
            try:
                # Build base model
                model_specific_args = filter_args_for_model(
                    OmegaConf.to_container(args, resolve=True), 
                    model_name, 
                    get_model_class(model_name)
                )
                
                model_eval = build_model(
                    model_name=model_name,
                    n_channels=n_channels,
                    n_times=n_timepoints,
                    n_outputs=n_outputs,
                    device=device,
                    model_specific_args=model_specific_args,
                    target_type='classification'
                )
                
                # Wrap with TTA
                model_eval_wrapped = TTAWrapper(model_eval, args, sr_hz=sr_hz_eval)
                model_eval_wrapped.to(device)
                
                # Load pretrained state and zero-shot evaluation
                zero_shot_accuracy = np.nan
                
                if not no_pretrain and model_name in pretrained_models_fold:
                    try:
                        pretrained_state = copy.deepcopy(pretrained_models_fold[model_name])
                        model_eval_wrapped.get_wrapped_model().load_state_dict(pretrained_state)
                        console.print("        Loaded pretrained state.")
                        
                        console.print(f"        Evaluating Zero-Shot Accuracy...")
                        zero_shot_accuracy, _ = evaluate_zero_shot(
                            model=model_eval_wrapped,
                            test_epochs=test_subj_epochs,
                            test_labels=test_subj_labels,
                            device=device,
                            original_unique_labels=original_unique_labels_fold,
                            batch_size=args.batch_size_finetune
                        )
                        
                        if not np.isnan(zero_shot_accuracy):
                            console.print(f"        Zero-Shot Accuracy: {zero_shot_accuracy:.4f}")
                        else:
                            console.print(f"        [yellow]Zero-Shot Accuracy: Failed (NaN)[/yellow]")
                            
                    except Exception as e:
                        log.error(f"Error during zero-shot evaluation: {e}")
                        console.print(f"        [yellow]Zero-shot evaluation failed: {e}[/yellow]")
                        zero_shot_accuracy = np.nan
                else:
                    console.print("        Skipping zero-shot evaluation (no pretrained state).")
                
                # WandB setup
                if args.use_wandb and WANDB_AVAILABLE:
                    try:
                        subject_run_name = f"{args.experiment_name}_Subj_{test_subject_id}_{dataset_name}_Fold_{fold_idx+1}_{model_name}"
                        config_wandb = OmegaConf.to_container(args, resolve=True)
                        config_wandb.update({
                            "dataset_name": dataset_name,
                            "fold": fold_idx + 1,
                            "subject_id": test_subject_id,
                            "stage": "evaluation",
                            "model_name": model_name,
                        })
                        wandb_run_subject = wandb.init(
                            project=args.wandb_project,
                            name=subject_run_name,
                            config=config_wandb,
                            group=f"{args.wandb_group_prefix}_{dataset_name}_Fold_{fold_idx+1}",
                            job_type=f"Eval_{model_name}",
                            reinit=True,
                        )
                    except Exception as e:
                        log.warning(f"WandB init failed: {e}")
                        wandb_run_subject = None
                
                # Online finetuning simulation
                finetune_log_prefix = f"{dataset_name}_Fold_{fold_idx+1}_Subj_{test_subject_id}_{model_name}"
                console.print(f"        Starting online finetuning simulation...")
                
                final_finetuned_accuracy, avg_time, per_trial_metrics = run_online_finetuning_simulation(
                    model=model_eval_wrapped,
                    test_subj_epochs=test_subj_epochs,
                    test_subj_labels=test_subj_labels,
                    original_unique_labels=original_unique_labels_fold,
                    args=args,
                    device=device,
                    console=console,
                    log_prefix=finetune_log_prefix,
                    wandb_run=wandb_run_subject
                )
                
                if not np.isnan(final_finetuned_accuracy):
                    console.print(f"        [->] {model_name} Final Finetuned Accuracy: {final_finetuned_accuracy:.4f} (Avg Time/trial: {avg_time:.3f}s)")
                else:
                    console.print(f"        [yellow][->] {model_name} Finetuning: Failed (NaN)[/yellow]")
                
                all_models_trial_metrics[model_name] = per_trial_metrics
                
                # Store results
                subject_results[model_name] = {
                    "zero_shot": zero_shot_accuracy,
                    "finetuned": final_finetuned_accuracy,
                }
                
            except Exception as e:
                log.error(f"Error processing model {model_name}: {e}")
                all_models_trial_metrics[model_name] = []
                console.print(f"      [red]Error processing {model_name}: {e}[/red]")
                
            finally:
                # Cleanup
                if wandb_run_subject:
                    wandb_run_subject.finish()
                if model_eval_wrapped is not None:
                    del model_eval_wrapped
                if model_eval is not None:
                    del model_eval
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Final validation
        successful_models = [name for name, results in subject_results.items() 
                           if not (np.isnan(results["zero_shot"]) and np.isnan(results["finetuned"]))]
        
        if successful_models:
            console.print(f"    [green]Successfully evaluated {len(successful_models)}/{len(args.models_to_run)} models for subject {test_subject_id}[/green]")
        else:
            console.print(f"    [yellow]No models completed successfully for subject {test_subject_id}[/yellow]")
        
        return subject_results, all_models_trial_metrics
        
    except Exception as e:
        console.print(f"    [red]Critical error processing subject {test_subject_id}: {e}[/red]")
        log.error(f"Critical error in subject evaluation: {e}")
        return subject_results, all_models_trial_metrics
    finally:
        if 'test_subj_epochs' in locals() and test_subj_epochs is not None:
            del test_subj_epochs
        if 'test_subj_labels' in locals() and test_subj_labels is not None:
            del test_subj_labels
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# %%
# Results aggregation and reporting

def aggregate_and_report_results(results, dataset_names_processed, args, console, run_output_dir):
    """Aggregate results across folds and datasets and generate reports."""
    console.print("\n[bold magenta]===== Aggregated Results Per Dataset =====[/bold magenta]")
    
    all_results_list_for_csv = []
    
    for dataset_name in dataset_names_processed:
        console.print(f"\n[bold green]### Results for Dataset: {dataset_name} (Metric: Accuracy) ###[/bold green]")
        
        results_for_dataset = results.get(dataset_name)
        if not results_for_dataset:
            console.print(f"  [yellow]No results found for dataset {dataset_name}. Skipping.[/yellow]")
            continue
        
        # Calculate fold averages
        dataset_fold_avg_zero_shot = {m: [np.nan] * args.n_splits for m in args.models_to_run}
        dataset_fold_avg_finetuned = {m: [np.nan] * args.n_splits for m in args.models_to_run}
        
        for model_name in args.models_to_run:
            model_results_for_dataset = results_for_dataset.get(model_name)
            if not model_results_for_dataset:
                continue
            
            for fold_idx in range(args.n_splits):
                fold_data = model_results_for_dataset.get(fold_idx)
                if fold_data:
                    fold_subj_zs_metrics = [
                        metric_dict.get("zero_shot", np.nan)
                        for subject_id, metric_dict in fold_data.items()
                        if isinstance(metric_dict, dict)
                    ]
                    fold_subj_ft_metrics = [
                        metric_dict.get("finetuned", np.nan)
                        for subject_id, metric_dict in fold_data.items()
                        if isinstance(metric_dict, dict)
                    ]
                    
                    valid_zs = [m for m in fold_subj_zs_metrics if not np.isnan(m)]
                    valid_ft = [m for m in fold_subj_ft_metrics if not np.isnan(m)]
                    
                    avg_zs_fold = np.mean(valid_zs) if valid_zs else np.nan
                    avg_ft_fold = np.mean(valid_ft) if valid_ft else np.nan
                    
                    dataset_fold_avg_zero_shot[model_name][fold_idx] = avg_zs_fold
                    dataset_fold_avg_finetuned[model_name][fold_idx] = avg_ft_fold
        
        # Create fold table
        fold_table = Table(title=f"Average Accuracy per Fold (Dataset: {dataset_name}) (Zero-Shot / Finetuned)")
        fold_table.add_column("Model", style="cyan")
        for i in range(args.n_splits):
            fold_table.add_column(f"Fold {i+1} (ZS/FT)", style="white")
        fold_table.add_column("Mean Accuracy (ZS)", style="green", justify="right")
        fold_table.add_column("Mean Accuracy (FT)", style="green", justify="right")
        
        # Calculate overall averages
        dataset_overall_avg_zs = {}
        dataset_overall_avg_ft = {}
        
        for model_name in args.models_to_run:
            zs_fold_avgs = dataset_fold_avg_zero_shot.get(model_name, [np.nan] * args.n_splits)
            ft_fold_avgs = dataset_fold_avg_finetuned.get(model_name, [np.nan] * args.n_splits)
            
            fold_zs_metrics_str = [
                f"{metric:.4f}" if not np.isnan(metric) else "[grey]N/A[/]"
                for metric in zs_fold_avgs
            ]
            fold_ft_metrics_str = [
                f"{metric:.4f}" if not np.isnan(metric) else "[grey]N/A[/]"
                for metric in ft_fold_avgs
            ]
            
            fold_combined_metrics = [
                f"{zs} / {ft}" for zs, ft in zip(fold_zs_metrics_str, fold_ft_metrics_str)
            ]
            
            valid_fold_zs = [m for m in zs_fold_avgs if not np.isnan(m)]
            valid_fold_ft = [m for m in ft_fold_avgs if not np.isnan(m)]
            
            mean_metric_zs = np.mean(valid_fold_zs) if valid_fold_zs else np.nan
            mean_metric_ft = np.mean(valid_fold_ft) if valid_fold_ft else np.nan
            
            dataset_overall_avg_zs[model_name] = mean_metric_zs
            dataset_overall_avg_ft[model_name] = mean_metric_ft
            
            fold_table.add_row(
                model_name,
                *fold_combined_metrics,
                f"{mean_metric_zs:.4f}" if not np.isnan(mean_metric_zs) else "[grey]N/A[/]",
                f"{mean_metric_ft:.4f}" if not np.isnan(mean_metric_ft) else "[grey]N/A[/]",
            )
        
        console.print(fold_table)
        
        # Create summary table
        avg_table = Table(title=f"Average Accuracy Across {args.n_splits} Folds (Dataset: {dataset_name})")
        avg_table.add_column("Model", style="cyan")
        avg_table.add_column("Avg Zero-Shot Accuracy", style="white", justify="right")
        avg_table.add_column("Avg Finetuned Accuracy", style="white", justify="right")
        
        # Sort models by finetuned performance
        sorted_models = sorted(
            dataset_overall_avg_ft.items(),
            key=lambda item: item[1] if not np.isnan(item[1]) else -np.inf,
            reverse=True,
        )
        
        for model_name, avg_metric_ft in sorted_models:
            avg_metric_zs = dataset_overall_avg_zs.get(model_name, np.nan)
            avg_table.add_row(
                model_name,
                f"{avg_metric_zs:.4f}" if not np.isnan(avg_metric_zs) else "[grey]N/A[/]",
                f"{avg_metric_ft:.4f}" if not np.isnan(avg_metric_ft) else "[grey]N/A[/]",
            )
        
        console.print(avg_table)
        
        # Collect data for CSV
        if results_for_dataset:
            for model_name, fold_data_dict in results_for_dataset.items():
                for fold_idx, subject_data_dict in fold_data_dict.items():
                    for subject_id, metric_dict in subject_data_dict.items():
                        if isinstance(metric_dict, dict):
                            all_results_list_for_csv.append({
                                "dataset": dataset_name,
                                "model": model_name,
                                "fold": fold_idx + 1,
                                "subject_id": subject_id,
                                "zero_shot_accuracy": metric_dict.get("zero_shot", np.nan),
                                "finetuned_accuracy": metric_dict.get("finetuned", np.nan),
                            })
    
    # Save detailed results
    if args.save_results:
        if all_results_list_for_csv:
            results_df = pd.DataFrame(all_results_list_for_csv)
            results_filename = run_output_dir / "results_detailed.csv"
            try:
                results_filename.parent.mkdir(parents=True, exist_ok=True)
                save_results_df(results_df, results_filename)
                console.print(f"\nDetailed results saved to: {results_filename}")
            except Exception as e:
                console.print(f"[red]Error saving detailed results: {e}[/red]")
        else:
            console.print("\n[yellow]No detailed results to save.[/yellow]")
    else:
        console.print("\n[grey]Results saving skipped.[/grey]")


# %%
# Main execution

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)
    
    # Setup experiment
    try:
        args, device, console, run_output_dir = setup_experiment()
    except Exception as e:
        log.error(f"Experiment setup failed: {e}")
        sys.exit(1)
    
    # Configure file logging
    log_file = run_output_dir / "run.log"
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        )
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        log.info(f"File logging configured to: {log_file}")
    except Exception as e:
        log.error(f"Failed to configure file logging: {e}")
    
    # Log configuration
    log.info("Final Configuration:\n%s", OmegaConf.to_yaml(args))
    console.print("[bold blue]Starting Transfer Experiment[/bold blue]")
    log.info(f"Models to run: {args.models_to_run}")
    log.info(f"Using device: {device}")
    log.info(f"Pretraining enabled: {not args.no_pretrain}")
    
    log_memory_usage("experiment_start", log)
    
    # Print dataset structure if requested
    if args.print_dataset_structure_and_exit:
        log.info("Printing dataset structure and exiting...")
        table = Table(title="Dataset Structure Overview")
        table.add_column("Code", style="cyan")
        table.add_column("Subjects (#)", style="magenta") 
        table.add_column("Task Type", style="yellow")
        table.add_column("Spec Classes", style="green")
        table.add_column("Spec SR (Hz)", style="blue")
        table.add_column("Spec Len (s)", style="blue")
        table.add_column("Epoch Shape (Ch, Time)", style="red")
        
        paradigm_kwargs = {"fmin": args.fmin, "fmax": args.fmax, "resample": args.resample}
        
        for ds_name in args.dataset_names:
            try:
                subjects = get_subject_list_for_datasets([ds_name], args.data_root)
                n_subjects = len(subjects) if subjects else 0
                first_subject = subjects[0] if subjects else None
            except Exception:
                n_subjects = "[red]Error[/red]"
                first_subject = None
            
            # Get dataset specs
            task_type = "classification"  # Default for cleaned version
            dataset_spec = None
            
            for p_name, p_data in PARADIGM_DATA.items():
                if ds_name in p_data.get("class_map", {}):
                    dataset_spec = p_data.get("specs", {}).get(ds_name)
                    break
            
            if not dataset_spec:
                table.add_row(ds_name, str(n_subjects), task_type, "N/A", "N/A", "N/A", "[red]Spec Not Found[/red]")
                continue
            
            spec_n_cls = dataset_spec.get("n_cls", "N/A")
            spec_sr = dataset_spec.get("sr", "N/A") 
            spec_sec = dataset_spec.get("sec", "N/A")
            epoch_shape_str = "N/A"
            
            if first_subject is not None:
                try:
                    epochs_sample, _, _, _, _ = load_cached_pretrain_data(
                        dataset_names=[ds_name], subject_ids=[first_subject],
                        paradigm_kwargs=paradigm_kwargs, data_root=args.data_root,
                        args=args, verbose=False, target_type='classification'
                    )
                    if epochs_sample is not None and epochs_sample.size > 0:
                        epoch_shape_str = str(epochs_sample.shape[1:])
                    else:
                        epoch_shape_str = "[red]Load Failed[/red]"
                except Exception:
                    epoch_shape_str = "[red]Error[/red]"
            
            table.add_row(
                ds_name, str(n_subjects), task_type, str(spec_n_cls), str(spec_sr),
                f"{spec_sec:.2f}" if isinstance(spec_sec, (int, float)) else str(spec_sec),
                epoch_shape_str,
            )
        
        console.print(table)
        sys.exit(0)
    
    # Initialize trial metrics storage
    all_detailed_trial_metrics_for_csv = []
    
    # Main K-Fold Cross-Validation
    try:
        console.print("\n[bold magenta]===== Running K-Fold Cross-Validation =====[/bold magenta]")
        log.info(f"Processing datasets: {args.dataset_names}")
        
        all_datasets_results = {}
        all_datasets_pretrained_models = {}
        processed_dataset_names = []
        
        for dataset_idx, dataset_name in enumerate(args.dataset_names):
            console.print(f"\n[bold cyan]===== Dataset {dataset_idx+1}/{len(args.dataset_names)}: {dataset_name} =====[/bold cyan]")
            log.info(f"Starting dataset: {dataset_name}")
            
            try:
                # Get subjects for this dataset
                current_dataset_subjects = get_subject_list_for_datasets([dataset_name], args.data_root)
                if not current_dataset_subjects:
                    log.warning(f"No subjects for {dataset_name}. Skipping.")
                    continue
                
                if args.subjects:
                    current_dataset_subjects = [s for s in current_dataset_subjects if s in args.subjects]
                    if not current_dataset_subjects:
                        log.warning(f"Requested subjects not in {dataset_name}. Skipping.")
                        continue
                
                console.print(f"   Using subjects for {dataset_name}: {current_dataset_subjects}")
                
                if len(current_dataset_subjects) < args.n_splits:
                    console.print(f"[red]Insufficient subjects ({len(current_dataset_subjects)}) for {args.n_splits} splits. Skipping.[/red]")
                    continue
                
                processed_dataset_names.append(dataset_name)
                
                # Setup K-Fold cross-validation
                kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
                results_current_dataset = {m: {f: {} for f in range(args.n_splits)} for m in args.models_to_run}
                all_datasets_pretrained_models[dataset_name] = {}
                
                # Process each fold
                for fold_idx, (train_indices, test_indices) in enumerate(kf.split(current_dataset_subjects)):
                    train_subject_ids = np.array(current_dataset_subjects)[train_indices].tolist()
                    test_subject_ids = np.array(current_dataset_subjects)[test_indices].tolist()
                    
                    console.print(f"\n   [bold blue]=== {dataset_name} / Fold {fold_idx+1}/{args.n_splits} ===[/bold blue]")
                    console.print(f"     Train Subjects: {train_subject_ids}")
                    console.print(f"     Test Subjects: {test_subject_ids}")
                    
                    try:
                        # Pretraining phase
                        if not args.no_pretrain:
                            console.print(f"     Starting Pretraining Phase...")
                            (fold_pretrained_models, fold_n_channels, fold_n_timepoints, 
                             fold_n_outputs, fold_unique_labels, pretrain_success) = run_fold_pretraining(
                                dataset_name=dataset_name, fold_idx=fold_idx, 
                                train_subject_ids=train_subject_ids, args=args,
                                device=device, console=console, run_output_dir=run_output_dir
                            )
                            
                            if pretrain_success:
                                all_datasets_pretrained_models[dataset_name][fold_idx] = fold_pretrained_models
                                console.print(f"       [green]Pretraining successful.[/green]")
                            else:
                                console.print(f"     [red]Pretraining failed. Skipping evaluation.[/red]")
                                continue
                        else:
                            # Get parameters without pretraining
                            console.print(f"     Skipping Pretraining. Fetching parameters...")
                            if not test_subject_ids:
                                console.print(f"     [red]No test subjects. Skipping fold.[/red]")
                                continue
                                
                            paradigm_kwargs = {"fmin": args.fmin, "fmax": args.fmax, "resample": args.resample}
                            _, proxy_labels, fold_n_channels, fold_n_timepoints, _ = load_cached_pretrain_data(
                                dataset_names=[dataset_name], subject_ids=[test_subject_ids[0]], 
                                paradigm_kwargs=paradigm_kwargs, data_root=args.data_root, 
                                args=args, verbose=False, target_type='classification'
                            )
                            
                            if fold_n_channels > 0 and fold_n_timepoints > 0 and proxy_labels is not None:
                                fold_unique_labels = np.unique(proxy_labels)
                                fold_n_outputs = len(fold_unique_labels)
                                if fold_n_outputs <= 1:
                                    console.print(f"       [red]Only {fold_n_outputs} class(es). Skipping.[/red]")
                                    continue
                                console.print(f"       Params: C={fold_n_channels}, T={fold_n_timepoints}, Out={fold_n_outputs}")
                            else:
                                console.print(f"       [red]Error loading parameters. Skipping fold.[/red]")
                                continue
                        
                        # Evaluation phase
                        console.print(f"     Starting Evaluation Phase...")
                        test_subject_ids_eval = test_subject_ids
                        
                        if (args.num_finetuning_subjects is not None and 
                            args.num_finetuning_subjects > 0 and 
                            len(test_subject_ids) > args.num_finetuning_subjects):
                            console.print(f"       [yellow]Limiting to first {args.num_finetuning_subjects} test subjects.[/yellow]")
                            test_subject_ids_eval = test_subject_ids[:args.num_finetuning_subjects]
                        
                        pretrained_models_for_eval = all_datasets_pretrained_models.get(dataset_name, {}).get(fold_idx, {})
                        
                        for test_subj_idx, test_subject_id in enumerate(test_subject_ids_eval):
                            console.print(f"       Evaluating Subject {test_subject_id} ({test_subj_idx+1}/{len(test_subject_ids_eval)})...")
                            
                            try:
                                subject_eval_results, subject_trial_metrics = run_subject_evaluation(
                                    test_subject_id=test_subject_id, fold_idx=fold_idx,
                                    pretrained_models_fold=pretrained_models_for_eval, 
                                    n_channels=fold_n_channels, n_timepoints=fold_n_timepoints, 
                                    n_outputs=fold_n_outputs, original_unique_labels_fold=fold_unique_labels, 
                                    args=args, device=device, console=console, run_output_dir=run_output_dir, 
                                    no_pretrain=args.no_pretrain, dataset_name=dataset_name
                                )
                                
                                for model_name, res_dict in subject_eval_results.items():
                                    if model_name in results_current_dataset:
                                        results_current_dataset[model_name][fold_idx][test_subject_id] = res_dict
                                
                                # Store trial-by-trial metrics
                                for model_name, trial_data_list in subject_trial_metrics.items():
                                    if trial_data_list:
                                        for trial_entry in trial_data_list:
                                            all_detailed_trial_metrics_for_csv.append({
                                                "dataset": dataset_name,
                                                "model": model_name,
                                                "fold": fold_idx + 1,
                                                "subject_id": test_subject_id,
                                                "trial_index": trial_entry['trial_idx'],
                                                "rolling_metric_value": trial_entry['rolling_metric'],
                                                "overall_metric_at_trial": trial_entry['overall_metric_at_trial'],
                                                "metric_type": "Accuracy"
                                            })
                                            
                            except Exception as e:
                                console.print(f"       [red]Error evaluating subject {test_subject_id}: {e}[/red]")
                                for model_name in args.models_to_run:
                                    if model_name in results_current_dataset:
                                        results_current_dataset[model_name][fold_idx][test_subject_id] = {
                                            "zero_shot": np.nan, "finetuned": np.nan
                                        }
                        
                        console.print(f"     Finished Evaluation for Fold {fold_idx+1}.")
                        
                    except Exception as e:
                        console.print(f"     [red]Error processing Fold {fold_idx+1}: {e}[/red]")
                        log.error(f"Fold processing failed for {dataset_name}, Fold {fold_idx+1}")
                    finally:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                all_datasets_results[dataset_name] = results_current_dataset
                log.info(f"Finished dataset: {dataset_name}")
                
            except Exception as e:
                console.print(f"[red]Error processing dataset {dataset_name}: {e}[/red]")
                log.error(f"Dataset processing failed for {dataset_name}")
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Aggregate and report results
        console.print("\n[bold magenta]===== Aggregating and Reporting Final Results =====[/bold magenta]")
        
        if not all_datasets_results:
            console.print("[red]No results generated. Exiting.[/red]")
            log.warning("No datasets produced results.")
        else:
            aggregate_and_report_results(
                results=all_datasets_results, 
                dataset_names_processed=processed_dataset_names,
                args=args, 
                console=console, 
                run_output_dir=run_output_dir
            )
        
        console.print("\n[bold green]Experiment Complete.[/bold green]")
        log.info("Experiment finished successfully.")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Experiment interrupted by user (Ctrl+C).[/yellow]")
        log.info("Experiment interrupted by user.")
    except Exception as e:
        console.print(f"\n[red]Critical error in main execution: {e}[/red]")
        log.error(f"Critical error in main execution: {e}")
    finally:
        # Save trial-by-trial metrics
        if args.save_results and all_detailed_trial_metrics_for_csv:
            console.print("\n[blue]Saving trial-by-trial metrics...[/blue]")
            try:
                trial_metrics_df = pd.DataFrame(all_detailed_trial_metrics_for_csv)
                trial_metrics_filename = run_output_dir / "results_trial_metrics.csv"
                trial_metrics_filename.parent.mkdir(parents=True, exist_ok=True)
                trial_metrics_df.to_csv(trial_metrics_filename, index=False)
                console.print(f"  Trial-by-trial metrics saved to: {trial_metrics_filename}")
                log.info(f"Trial-by-trial metrics saved to: {trial_metrics_filename}")
            except Exception as e:
                console.print(f"  [red]Error saving trial metrics: {e}[/red]")
                log.error(f"Failed to save trial metrics: {e}")
        elif args.save_results:
            console.print("\n[yellow]No trial-by-trial metrics to save.[/yellow]")
        
        # Final cleanup
        try:
            if 'all_datasets_results' in locals():
                del all_datasets_results
            if 'all_datasets_pretrained_models' in locals():
                del all_datasets_pretrained_models
            if 'all_detailed_trial_metrics_for_csv' in locals():
                del all_detailed_trial_metrics_for_csv
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_memory_usage("final_experiment_cleanup", log)
        except Exception as e:
            log.warning(f"Final cleanup error: {e}")
        
        console.print("\n[dim]Memory cleanup completed.[/dim]")