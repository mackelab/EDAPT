# %%
"""
Measures the computational latency of 4 EEG models across 9 datasets for unsupervised domain adaptation (UDA), prediction and supervised continual finetuning (CFT)
on CPU and GPU hardware. Collects all results and saves them to a single CSV file.
"""

import argparse
import itertools
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import ListConfig, OmegaConf
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from moabb_datasets import PARADIGM_DATA, load_cached_pretrain_data

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    import mne

    mne.set_log_level("ERROR")
except ImportError:
    pass

# Setup project-specific imports (ensure these are in your PYTHONPATH)
try:
    from models.builder import build_model
    from tta_wrapper import TTAWrapper
    from utils import cross_entropy_loss, filter_args_for_model, get_model_class
except ImportError as e:
    print(
        f"ERROR: Could not import a required module. Please ensure 'models', 'tta_wrapper', and 'utils' are accessible. Details: {e}",
        file=sys.stderr,
    )
    sys.exit(1)

# Setup root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ==========================================================================
# 1. DATA HANDLING UTILITIES
# ==========================================================================


class DictDataset(Dataset):
    """A simple PyTorch Dataset that handles dictionary-based data."""

    def __init__(
        self,
        epochs_tensor: torch.Tensor,
        labels_tensor: torch.Tensor,
        target_type: str = "classification",
    ):
        self.epochs = epochs_tensor
        self.target_type = target_type

        if not isinstance(labels_tensor, torch.Tensor):
            labels_tensor = torch.from_numpy(labels_tensor)

        if target_type == "classification":
            self.labels = labels_tensor.long()
            unique_labels = torch.unique(self.labels)
            if len(unique_labels) > 0 and torch.min(unique_labels) > 0:
                log.debug(f"Remapping labels to be 0-indexed.")
                label_mapping = {
                    label.item(): idx for idx, label in enumerate(sorted(unique_labels))
                }
                self.labels = torch.tensor(
                    [label_mapping[lbl.item()] for lbl in self.labels], dtype=torch.long
                )
        elif target_type == "regression":
            self.labels = labels_tensor.float()
            if self.labels.ndim == 1:
                self.labels = self.labels.unsqueeze(1)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        log.debug(
            f"DictDataset created with {len(self)} samples. Target type: {target_type}."
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"epoch": self.epochs[index], "label": self.labels[index]}


def create_dataloader(
    epochs, labels, batch_size, shuffle_data=True, target_type="classification"
):
    """Creates a PyTorch DataLoader from numpy arrays."""
    if epochs is None or labels is None or epochs.size == 0:
        return None
    epochs_tensor = torch.from_numpy(epochs).float()
    dataset = DictDataset(epochs_tensor, labels, target_type=target_type)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_data,
        num_workers=0,
        pin_memory=True,
    )


# ==========================================================================
# 2. CORE LATENCY MEASUREMENT FUNCTIONS
# ==========================================================================


def measure_filtering_latency(
    n_channels: int,
    n_timepoints: int,
    sr_hz: float,
    fmin: float,
    fmax: float,
    n_repeats: int,
):
    """Measures bandpass filtering latency for a single trial."""
    from mne.filter import filter_data

    # Create synthetic trial matching dataset dimensions (filter_data expects 2D: channels x time)
    single_trial = np.random.randn(n_channels, n_timepoints).astype(np.float64)

    # Warm-up
    for _ in range(5):
        _ = filter_data(single_trial, sr_hz, fmin, fmax, verbose=False)

    # Measurement
    latencies = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        _ = filter_data(single_trial, sr_hz, fmin, fmax, verbose=False)
        latencies.append((time.perf_counter() - start) * 1000)

    return np.mean(latencies), np.std(latencies), np.median(latencies)


def measure_baseline_correction_latency(
    n_channels: int,
    n_timepoints: int,
    n_repeats: int,
    baseline_fraction: float = 0.2,
):
    """Measures baseline correction (zero mean) latency for a single trial.

    Baseline correction subtracts mean of baseline period from entire epoch.
    This is how MOABB applies baseline correction in mne.Epochs.
    """
    # Create synthetic epoch (channels x time)
    single_trial = np.random.randn(n_channels, n_timepoints).astype(np.float64)
    baseline_end = int(n_timepoints * baseline_fraction)

    def baseline_correct(data, bline_end):
        """Subtract baseline mean from data (in-place would be faster but we copy for measurement)."""
        baseline_mean = data[:, :bline_end].mean(axis=1, keepdims=True)
        return data - baseline_mean

    # Warm-up
    for _ in range(5):
        _ = baseline_correct(single_trial.copy(), baseline_end)

    # Measurement
    latencies = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        _ = baseline_correct(single_trial.copy(), baseline_end)
        latencies.append((time.perf_counter() - start) * 1000)

    return np.mean(latencies), np.std(latencies), np.median(latencies)


def measure_moabb_preprocessing_latency(
    n_channels: int,
    n_timepoints: int,
    sr_hz: float,
    fmin: float,
    fmax: float,
    n_repeats: int,
    baseline_fraction: float = 0.2,
):
    """Measures combined MOABB preprocessing latency (bandpass + baseline correction)."""
    from mne.filter import filter_data

    single_trial = np.random.randn(n_channels, n_timepoints).astype(np.float64)
    baseline_end = int(n_timepoints * baseline_fraction)

    def preprocess(data, sr, f_lo, f_hi, bline_end):
        # Bandpass filter
        filtered = filter_data(data, sr, f_lo, f_hi, verbose=False)
        # Baseline correction (zero mean)
        baseline_mean = filtered[:, :bline_end].mean(axis=1, keepdims=True)
        return filtered - baseline_mean

    # Warm-up
    for _ in range(5):
        _ = preprocess(single_trial.copy(), sr_hz, fmin, fmax, baseline_end)

    # Measurement
    latencies = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        _ = preprocess(single_trial.copy(), sr_hz, fmin, fmax, baseline_end)
        latencies.append((time.perf_counter() - start) * 1000)

    return np.mean(latencies), np.std(latencies), np.median(latencies)


def measure_single_prediction_latency(
    model: TTAWrapper, single_input: torch.Tensor, device: torch.device, n_repeats: int
):
    """Measures the latency of a single forward pass for inference."""
    model.eval()
    latencies = []
    input_on_device = single_input.to(device)

    # Warm-up runs
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_on_device, apply_tta=True, is_finetuning_batch=False)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Measurement runs
    for _ in range(n_repeats):
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(input_on_device, apply_tta=True, is_finetuning_batch=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start_time) * 1000)

    return np.mean(latencies), np.std(latencies), np.median(latencies)


def measure_tta_update_latency(
    model: TTAWrapper, single_input: torch.Tensor, device: torch.device, n_repeats: int
):
    """Measures the latency of updating TTA alignment statistics."""
    model.eval()
    latencies = []
    input_on_device = single_input.to(device)

    if not hasattr(model, "update_tta_statistics"):
        log.error("Model does not have 'update_tta_statistics' method.")
        return np.nan, np.nan, np.nan

    # Warm-up runs
    for _ in range(10):
        model.update_tta_statistics(input_on_device)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Measurement runs
    for _ in range(n_repeats):
        start_time = time.perf_counter()
        model.update_tta_statistics(input_on_device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start_time) * 1000)

    return np.mean(latencies), np.std(latencies), np.median(latencies)


def measure_finetuning_step_latency(
    model: TTAWrapper,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    n_steps: int,
):
    """Measures the latency of a single supervised finetuning step (one batch)."""
    model.train()
    latencies = []

    if not dataloader:
        return np.nan, np.nan, np.nan

    # Warm-up runs
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        inputs, labels = batch["epoch"].to(device), batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs, is_finetuning_batch=True)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Measurement runs
    steps_measured = 0
    for batch in dataloader:
        if steps_measured >= n_steps:
            break
        inputs, labels = batch["epoch"].to(device), batch["label"].to(device)

        start_time = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs, is_finetuning_batch=True)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start_time) * 1000)
        steps_measured += 1

    if not latencies:
        return np.nan, np.nan, np.nan
    return np.mean(latencies), np.std(latencies), np.median(latencies)


# ==========================================================================
# 3. EXPERIMENT SETUP AND ASSET MANAGEMENT
# ==========================================================================


def setup_experiment(cli_args=None):
    """Parses configs from YAML/CLI and sets up the experiment environment."""
    DEFAULT_YAML = """
    # Data and Model
    data_root: "${oc.env:EDAPT_DATA_ROOT,${oc.env:HOME}/mne_data}"
    fmin: 1.0
    fmax: 47.0
    resample: null

    # TTA Wrapper Configuration
    use_tta: true
    alignment_type: "euclidean"
    alignment_cov_epsilon: 1.0e-6
    alignment_transform_epsilon: 1.0e-7
    alignment_ref_ema_beta: 0.9
    use_adabn: true
    finetune_mode: "full"

    # Latency Measurement Settings
    n_repeats_inference: 100
    n_steps_finetune: 20
    batch_size_latency: 50

    # General
    seed: 42
    base_output_dir: "results_latency"
    experiment_name: "latency_analysis"
    save_results: true
    """
    config = OmegaConf.create(DEFAULT_YAML)
    parser = argparse.ArgumentParser(description="Model Latency Measurement")
    parser.add_argument(
        "-c", "--config", action="append", help="Path to YAML config file.", default=[]
    )
    parsed_args, remaining_argv = parser.parse_known_args(args=cli_args)

    for config_file in parsed_args.config:
        try:
            user_config = OmegaConf.load(config_file)
            config = OmegaConf.merge(config, user_config)
        except Exception as e:
            log.error(f"Failed to load config {config_file}: {e}")

    if remaining_argv:
        cli_conf = OmegaConf.from_cli(remaining_argv)
        config = OmegaConf.merge(config, cli_conf)

    OmegaConf.resolve(config)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_output_dir = Path(config.base_output_dir) / f"{config.experiment_name}_{ts}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, run_output_dir / "config_latency.yaml")

    return config, run_output_dir


def load_latency_data(dataset_name: str, args: OmegaConf):
    """Loads and prepares data for a given dataset for latency measurement."""
    paradigm_info = next(
        (
            p
            for p_name, p in PARADIGM_DATA.items()
            if dataset_name in p.get("class_map", {})
        ),
        None,
    )
    if not paradigm_info:
        log.error(f"No paradigm found for dataset {dataset_name}. Skipping.")
        return (None,) * 7

    specs = paradigm_info.get("specs", {}).get(dataset_name, {})
    effective_target_type = specs.get(
        "target_type", paradigm_info.get("task_type", "classification")
    )
    sr_hz = specs.get("sr", 250)
    n_outputs = specs.get(
        "n_cls", 2 if effective_target_type == "classification" else 1
    )

    paradigm_kwargs = {"fmin": args.fmin, "fmax": args.fmax, "resample": args.resample}

    ds_class = paradigm_info["class_map"][dataset_name]
    subject_to_load = (
        [ds_class().subject_list[0]]
        if hasattr(ds_class(), "subject_list") and ds_class().subject_list
        else [1]
    )

    epochs, labels, n_ch, n_t, _ = load_cached_pretrain_data(
        dataset_names=[dataset_name],
        subject_ids=subject_to_load,
        paradigm_kwargs=paradigm_kwargs,
        data_root=args.data_root,
        args=args,
        verbose=False,
        target_type=effective_target_type,
    )

    if epochs is None or epochs.size == 0:
        log.error(f"Failed to load data for {dataset_name}.")
        return (None,) * 7

    effective_sr_hz = args.resample if args.resample else sr_hz

    ft_dataloader = create_dataloader(
        epochs,
        labels,
        args.batch_size_latency,
        shuffle_data=False,
        target_type=effective_target_type,
    )
    single_input_tensor = (
        torch.from_numpy(epochs[0:1]).float() if epochs.shape[0] > 0 else None
    )

    if effective_target_type == "classification" and labels is not None:
        n_outputs = len(np.unique(labels))

    return (
        ft_dataloader,
        single_input_tensor,
        n_ch,
        n_t,
        n_outputs,
        effective_sr_hz,
        effective_target_type,
    )


def build_latency_model(
    model_name: str,
    n_channels: int,
    n_timepoints: int,
    n_outputs: int,
    sr_hz: float,
    target_type: str,
    device: torch.device,
    args: OmegaConf,
):
    """Builds the specified model and wraps it for TTA."""
    ModelClass = get_model_class(model_name)
    model_specific_args = filter_args_for_model(args, model_name, ModelClass)

    base_model = build_model(
        model_name=model_name,
        n_channels=n_channels,
        n_times=n_timepoints,
        n_outputs=n_outputs,
        device=device,
        model_specific_args=model_specific_args,
        target_type=target_type,
    )

    wrapped_model = TTAWrapper(base_model, args, sr_hz=sr_hz).to(device)

    total_params = sum(p.numel() for p in wrapped_model.parameters())
    trainable_params = sum(
        p.numel() for p in wrapped_model.parameters() if p.requires_grad
    )

    return wrapped_model, total_params, trainable_params


# ==========================================================================
# 4. MAIN EXPERIMENT EXECUTION
# ==========================================================================


def main():
    """Main function to orchestrate the latency measurement experiment."""
    args, run_output_dir = setup_experiment()
    console = Console()

    # Setup file logging
    log_file = run_output_dir / "latency_run.log"
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(lineno)d: %(message)s"
        )
    )
    logging.getLogger().addHandler(file_handler)

    console.print(f"[bold blue]Starting Latency Measurement Analysis[/bold blue]")
    console.print(f"Output directory: [cyan]{run_output_dir.resolve()}[/cyan]")
    log.info(f"Full configuration:\n{OmegaConf.to_yaml(args)}")

    # --- Define Experiment Grid ---
    MODELS = ["ShallowConvNet", "EEGNetv4", "ATCNet", "DeepConvNet"]
    # Corrected: DATASETS must be a flat list of strings, not a list of lists.
    DATASETS = [
        "Yang2025",
        "Lee2019_SSVEP",
        "BI2015a",
        "Lee2019_MI",
        "BNCI2014_001",
        "Huebner2017",
        "Huebner2018",
        "Kalunga2016",
        "MAMEM2",
    ]
    # To run on all available datasets, uncomment the line below
    # DATASETS = sorted({ds for paradigm in PARADIGM_DATA.values() for ds in paradigm.get("class_map", {})})
    DEVICES = [torch.device("cpu")]
    if torch.cuda.is_available():
        DEVICES.append(torch.device("cuda"))

    all_results = []
    combinations = list(itertools.product(DATASETS, MODELS, DEVICES))

    console.print(
        f"Generated [bold yellow]{len(combinations)}[/bold yellow] experiment combinations."
    )

    for dataset, model_name, device in tqdm(combinations, desc="Overall Progress"):
        # The 'dataset' variable is now a string (e.g., "Yang2025"), which is the correct type.
        console.print(
            f"\n[bold green]>>> Testing: [yellow]{model_name}[/yellow] on [cyan]{dataset}[/cyan] with [magenta]{device}[/magenta] <<[/bold green]"
        )

        # --- Data Loading ---
        (ft_loader, s_input, n_ch, n_t, n_out, sr_hz, target_type) = load_latency_data(
            dataset, args
        )
        if s_input is None:
            log.warning(f"Skipping combination due to data loading failure.")
            continue

        console.print(
            f"  Data loaded: C={n_ch}, T={n_t}, N_out={n_out}, SR={sr_hz}Hz, Task={target_type}"
        )

        # --- Model Building ---
        try:
            model, total_p, train_p = build_latency_model(
                model_name, n_ch, n_t, n_out, sr_hz, target_type, device, args
            )
            console.print(
                f"  Model built: Total Params={total_p:,}, Trainable={train_p:,}"
            )
        except Exception as e:
            log.error(f"Model building failed for {model_name}: {e}", exc_info=True)
            all_results.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "device": str(device),
                    "error": str(e),
                }
            )
            continue

        # --- Latency Measurement ---
        # Preprocessing (filtering) latency - measured once per dataset, independent of model/device
        filter_mean, filter_std, filter_med = measure_filtering_latency(
            n_ch, n_t, sr_hz, args.fmin, args.fmax, args.n_repeats_inference
        )
        console.print(
            f"    - Filtering Latency (ms):    Mean={filter_mean:.3f}, Std={filter_std:.3f}, [bold]Median={filter_med:.3f}[/bold]"
        )

        # MOABB preprocessing (bandpass + baseline correction)
        moabb_mean, moabb_std, moabb_med = measure_moabb_preprocessing_latency(
            n_ch, n_t, sr_hz, args.fmin, args.fmax, args.n_repeats_inference
        )
        console.print(
            f"    - MOABB Preproc Latency (ms): Mean={moabb_mean:.3f}, Std={moabb_std:.3f}, [bold]Median={moabb_med:.3f}[/bold]"
        )

        pred_mean, pred_std, pred_med = measure_single_prediction_latency(
            model, s_input, device, args.n_repeats_inference
        )
        console.print(
            f"    - Prediction Latency (ms):   Mean={pred_mean:.3f}, Std={pred_std:.3f}, [bold]Median={pred_med:.3f}[/bold]"
        )

        tta_mean, tta_std, tta_med = measure_tta_update_latency(
            model, s_input, device, args.n_repeats_inference
        )
        console.print(
            f"    - TTA Update Latency (ms):   Mean={tta_mean:.3f}, Std={tta_std:.3f}, [bold]Median={tta_med:.3f}[/bold]"
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss() if target_type == "regression" else cross_entropy_loss
        ft_mean, ft_std, ft_med = measure_finetuning_step_latency(
            model, ft_loader, optimizer, criterion, device, args.n_steps_finetune
        )
        console.print(
            f"    - Finetune Step Latency (ms): Mean={ft_mean:.3f}, Std={ft_std:.3f}, [bold]Median={ft_med:.3f}[/bold]"
        )

        # --- Result Aggregation ---
        all_results.append(
            {
                "dataset": dataset,
                "model": model_name,
                "device": str(device),
                "total_params": total_p,
                "trainable_params": train_p,
                "filter_latency_ms_median": filter_med,
                "filter_latency_ms_mean": filter_mean,
                "filter_latency_ms_std": filter_std,
                "moabb_preproc_latency_ms_median": moabb_med,
                "moabb_preproc_latency_ms_mean": moabb_mean,
                "moabb_preproc_latency_ms_std": moabb_std,
                "pred_latency_ms_median": pred_med,
                "pred_latency_ms_mean": pred_mean,
                "pred_latency_ms_std": pred_std,
                "tta_update_latency_ms_median": tta_med,
                "tta_update_latency_ms_mean": tta_mean,
                "tta_update_latency_ms_std": tta_std,
                "finetune_step_latency_ms_median": ft_med,
                "finetune_step_latency_ms_mean": ft_mean,
                "finetune_step_latency_ms_std": ft_std,
                "n_channels": n_ch,
                "n_timepoints": n_t,
                "sr_hz": sr_hz,
                "batch_size": args.batch_size_latency,
                "target_type": target_type,
                "error": None,
            }
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- Final Results Processing ---
    console.print(
        "\n[bold magenta]Latency Measurement Complete. Final Results:[/bold magenta]"
    )
    if all_results:
        results_df = pd.DataFrame(all_results)

        # Display summary table in console
        summary_table = Table(title="Median Latency (ms) Summary")
        summary_cols = [
            "dataset",
            "model",
            "device",
            "filter_latency_ms_median",
            "moabb_preproc_latency_ms_median",
            "pred_latency_ms_median",
            "tta_update_latency_ms_median",
            "finetune_step_latency_ms_median",
        ]
        for col in summary_cols:
            summary_table.add_column(
                col,
                justify="left" if col in ["dataset", "model", "device"] else "right",
            )

        for _, row in results_df[summary_cols].iterrows():
            summary_table.add_row(
                *[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
            )
        console.print(summary_table)

        # Save detailed results to CSV
        if args.save_results:
            results_path = run_output_dir / "latency_results_detailed.csv"
            results_df.to_csv(results_path, index=False, float_format="%.4f")
            console.print(f"\nDetailed results saved to: [green]{results_path}[/green]")
    else:
        console.print("[yellow]No results were generated.[/yellow]")

    console.print("\n[bold green]Script finished.[/bold green]")


if __name__ == "__main__":
    main()
# %%
