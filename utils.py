#%% utils.py

import datetime
import inspect
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Set up module-level logger
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# Import model mappings
from models.models import MODEL_CLASS_MAP, PREFIX_MAP


def get_username():
    """Get the username from environment variable."""
    return os.environ.get("USER", "jkapoor83")


def replace_username(path_str, username=None):
    """Replace placeholder username in path string."""
    path_str = str(path_str)
    path_str = path_str.replace(
        f"macke/{get_username()}", f"macke/{username if username else get_username()}"
    )
    return Path(path_str)


# Cache directory setup
_repo_root = Path(__file__).parent
_default_cache_dir_template = _repo_root / ".cache" / "moabb"
_raw_cache_dir_source_str = os.environ.get("MOABB_CACHE_DIR", str(_default_cache_dir_template))

CACHE_ROOT_DIR: Optional[Path] = None


def update_cache_root_dir(target_username_for_replacement: str):
    """Update the global CACHE_ROOT_DIR with username replacement."""
    global CACHE_ROOT_DIR
    CACHE_ROOT_DIR = replace_username(_raw_cache_dir_source_str, username=target_username_for_replacement)
    log.info(f"CACHE_ROOT_DIR updated to: {CACHE_ROOT_DIR}")


# Initialize with default username
update_cache_root_dir(target_username_for_replacement="jkapoor83")


def cross_entropy_loss(logits, y_true):
    """Multi-class cross entropy loss."""
    return F.cross_entropy(logits, y_true.long(), reduction="mean")


class ContinuousLearnerBinary:
    """Track rolling binary accuracy over a window and overall accuracy."""

    def __init__(self, window_size=50):
        self.window_size = window_size
        self.correct_history = []
        self.all_correct_flags = []
        self.count = 0

    def update(self, is_correct):
        """Update with new correctness flag."""
        self.correct_history.append(is_correct)
        if len(self.correct_history) > self.window_size:
            self.correct_history.pop(0)
        self.all_correct_flags.append(is_correct)
        self.count += 1

    def get_rolling_accuracy(self):
        """Calculate accuracy over the rolling window."""
        if not self.correct_history:
            return 0.0
        return np.mean(self.correct_history)

    def get_overall_accuracy(self):
        """Calculate accuracy over all trials seen so far."""
        if self.count == 0:
            return 0.0
        return np.mean(self.all_correct_flags)


def evaluate_zero_shot(model, test_epochs, test_labels, device, original_unique_labels, batch_size=64):
    """Evaluate pretrained model on test subject using original label mapping."""
    model.eval()
    all_preds = []
    all_true = []
    all_losses = []

    if test_epochs.size == 0:
        return 0.0, 0.0

    # Create consistent mapping based on original training labels
    labels_np = test_labels
    if original_unique_labels is None or len(original_unique_labels) == 0:
        warnings.warn(
            "Original unique labels not provided to evaluate_zero_shot. Assuming labels are already 0-indexed."
        )
        remapped_labels_np = labels_np
    else:
        # Build mapping from full set of original labels
        label_mapping = {
            label: idx for idx, label in enumerate(sorted(original_unique_labels))
        }
        try:
            remapped_labels_np = np.array([label_mapping[label] for label in labels_np])
            log.debug(
                f"evaluate_zero_shot mapped labels using original mapping: {original_unique_labels} -> {np.unique(remapped_labels_np)}"
            )
        except KeyError as e:
            log.error(
                f"Label {e} found in test set but not in original_unique_labels: {original_unique_labels}. Cannot evaluate."
            )
            return np.nan, np.nan

    eval_dataset = TensorDataset(
        torch.from_numpy(test_epochs).float(),
        torch.from_numpy(remapped_labels_np).long(),
    )
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

    with torch.no_grad():
        for epochs_batch, labels_batch in eval_loader:
            epochs_batch = epochs_batch.to(device)
            labels_batch = labels_batch.to(device)

            logits = model(epochs_batch)
            loss = cross_entropy_loss(logits, labels_batch)

            pred_labels = torch.argmax(logits, dim=1)

            all_preds.extend(pred_labels.cpu().tolist())
            all_true.extend(labels_batch.cpu().tolist())
            all_losses.append(loss.item())

    accuracy = np.mean(np.array(all_preds) == np.array(all_true))
    avg_loss = np.mean(all_losses) if all_losses else 0.0

    return accuracy, avg_loss


def evaluate_single_trial(
    model: nn.Module,
    single_epoch_tensor: torch.Tensor,
    single_label_tensor: torch.Tensor,
    device: torch.device,
    original_unique_labels: Optional[np.ndarray],
    output_logits: Optional[torch.Tensor] = None
) -> Dict[str, any]:
    """Evaluate model on single trial ensuring correct label mapping."""
    if single_epoch_tensor.ndim == 2:
        single_epoch_tensor = single_epoch_tensor.unsqueeze(0)
    if single_label_tensor.ndim == 0:
        single_label_tensor = single_label_tensor.unsqueeze(0)

    # Consistent label remapping
    original_label_int = single_label_tensor.item()
    remapped_label_int = -1

    if original_unique_labels is None or len(original_unique_labels) == 0:
        warnings.warn(
            "evaluate_single_trial: original_unique_labels not provided or empty. "
            "Assuming label is already 0-indexed."
        )
        remapped_label_int = original_label_int
    else:
        label_mapping = {
            label: idx for idx, label in enumerate(sorted(original_unique_labels))
        }
        if original_label_int in label_mapping:
            remapped_label_int = label_mapping[original_label_int]
        else:
            log.error(
                f"Label {original_label_int} found in single trial but not in "
                f"original_unique_labels: {original_unique_labels}. Cannot evaluate."
            )
            num_classes_placeholder = len(original_unique_labels) if original_unique_labels is not None and len(original_unique_labels) > 0 else 2
            return {
                "true_label": original_label_int, "pred_label": np.nan,
                "pred_probs": [np.nan] * num_classes_placeholder,
                "loss": np.nan, "is_correct": np.nan,
            }
            
    remapped_label_tensor = torch.tensor(
        [remapped_label_int], dtype=torch.long, device=device
    )

    # Forward pass if logits not provided
    if output_logits is None:
        log.debug("evaluate_single_trial: output_logits not provided, performing forward pass.")
        model.eval()
        with torch.no_grad():
            logits = model(single_epoch_tensor.to(device))
    else:
        log.debug("evaluate_single_trial: Using provided output_logits.")
        logits = output_logits.to(device)

    # Calculate loss and metrics
    try:
        loss = cross_entropy_loss(logits, remapped_label_tensor)
        pred_probs = F.softmax(logits, dim=1)[0]
        pred_label_remapped = torch.argmax(logits, dim=1)[0].item()
        
        true_label_remapped = remapped_label_tensor.item()
        is_correct = int(pred_label_remapped == true_label_remapped)
    except Exception as e:
        log.error(f"Error during loss/metric calculation in evaluate_single_trial: {e}")
        log.error(f"  Logits shape: {logits.shape if logits is not None else 'None'}, dtype: {logits.dtype if logits is not None else 'N/A'}")
        log.error(f"  Remapped label tensor: {remapped_label_tensor}")
        num_classes_placeholder = logits.shape[1] if logits is not None and logits.ndim > 1 else (len(original_unique_labels) if original_unique_labels is not None and len(original_unique_labels) > 0 else 2)
        return {
            "true_label": original_label_int, "pred_label": np.nan,
            "pred_probs": [np.nan] * num_classes_placeholder,
            "loss": np.nan, "is_correct": np.nan,
        }

    return {
        "true_label": original_label_int,
        "pred_label": pred_label_remapped,
        "pred_probs": pred_probs.cpu().tolist(),
        "loss": loss.item(),
        "is_correct": is_correct,
        "true_label_remapped": true_label_remapped
    }


def get_model_class(model_name: str) -> Type[nn.Module]:
    """Get model class from model name using central map."""
    model_class = MODEL_CLASS_MAP.get(model_name)
    if model_class is None:
        raise ValueError(
            f"Unknown model name: {model_name}. Available models: {list(MODEL_CLASS_MAP.keys())}"
        )
    return model_class


def filter_args_for_model(
    args_dict: Dict[str, Any], model_name: str, model_class: Type[nn.Module]
) -> Dict[str, Any]:
    """Filter arguments to only include those relevant for specific model constructor."""
    model_params = {}

    # Get expected parameters from model's __init__ signature
    try:
        init_signature = inspect.signature(model_class.__init__)
        expected_params = set(init_signature.parameters.keys())
        expected_params.discard("self")
    except ValueError:
        warnings.warn(
            f"Could not inspect signature for {model_name}. Cannot filter args precisely.",
            stacklevel=2,
        )
        expected_params = set()

    # Prefix-based autodiscovery
    prefix = PREFIX_MAP.get(model_name)
    if prefix is not None:
        prefix = prefix.lower()
        for arg_key, arg_val in args_dict.items():
            key_lower = arg_key.lower()
            if key_lower.startswith(prefix):
                stripped_param = arg_key[len(prefix):]
                if not stripped_param:
                    continue
                if not expected_params or stripped_param in expected_params:
                    model_params[stripped_param] = arg_val

    # Direct match
    for arg_key, arg_val in args_dict.items():
        if expected_params and arg_key in expected_params:
            if arg_key not in model_params:
                model_params[arg_key] = arg_val

    return model_params


def get_output_dir(
    base_output_root: Union[str, Path], experiment_name: str, timestamp: bool = True
) -> Path:
    """Create and return unique output directory for experiment run."""
    base_path = Path(base_output_root) / experiment_name
    if timestamp:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_path / timestamp_str
    else:
        run_dir = base_path

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_checkpoint_dir(run_output_dir: Path) -> Path:
    """Get checkpoint directory within run's output directory."""
    chkpt_dir = run_output_dir / "checkpoints"
    chkpt_dir.mkdir(parents=True, exist_ok=True)
    return chkpt_dir


def save_checkpoint(state: dict, path: Union[str, Path]):
    """Save model and optimizer state dictionary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(state, path)
        log.info(f"Checkpoint saved to {path}")
    except Exception as e:
        log.error(f"Failed to save checkpoint to {path}: {e}")


def save_results_df(dataframe, path: Union[str, Path]):
    """Save pandas DataFrame to CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dataframe.to_csv(path, index=False)
        log.info(f"Results DataFrame saved to {path}")
    except Exception as e:
        log.error(f"Failed to save results DataFrame to {path}: {e}")

