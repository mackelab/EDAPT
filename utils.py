# %% utils.py

import datetime
import inspect
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Set up module-level logger
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# Import model mappings
from models.models import MODEL_CLASS_MAP, PREFIX_MAP


# Cache directory for preprocessed MOABB data; override with MOABB_CACHE_DIR.
_repo_root = Path(__file__).resolve().parent
CACHE_ROOT_DIR: Path = Path(
    os.environ.get("MOABB_CACHE_DIR", str(_repo_root / ".cache" / "moabb"))
).expanduser()


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


class ContinuousLearnerMultiMetric:
    """Track predictions and compute multiple metrics over a rolling window."""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.pred_history: List[int] = []
        self.true_history: List[int] = []
        self.prob_history: List[List[float]] = []
        self.all_preds: List[int] = []
        self.all_trues: List[int] = []
        self.all_probs: List[List[float]] = []
        self.count = 0

    def update(
        self, pred_label: int, true_label: int, pred_probs: Optional[List[float]] = None
    ):
        """Update with new prediction."""
        self.pred_history.append(pred_label)
        self.true_history.append(true_label)
        if len(self.pred_history) > self.window_size:
            self.pred_history.pop(0)
            self.true_history.pop(0)
        self.all_preds.append(pred_label)
        self.all_trues.append(true_label)
        if pred_probs is not None:
            self.prob_history.append(pred_probs)
            if len(self.prob_history) > self.window_size:
                self.prob_history.pop(0)
            self.all_probs.append(pred_probs)
        self.count += 1

    def _compute_auroc(self, trues: List[int], probs: List[List[float]]) -> float:
        """Compute AUROC from true labels and probability predictions."""
        if not probs or len(trues) < 2 or len(set(trues)) < 2:
            return np.nan
        try:
            probs_array = np.array(probs)
            n_classes = probs_array.shape[1] if probs_array.ndim > 1 else 1
            if n_classes == 2:
                return roc_auc_score(trues, probs_array[:, 1])
            return roc_auc_score(trues, probs_array, multi_class="ovr", average="macro")
        except Exception:
            return np.nan

    def _compute_balanced_accuracy(self, trues: List[int], preds: List[int]) -> float:
        """Compute balanced accuracy from true and predicted labels."""
        if not trues or len(set(trues)) < 2:
            return np.nan
        try:
            return balanced_accuracy_score(trues, preds)
        except Exception:
            return np.nan

    def get_rolling_accuracy(self) -> float:
        """Calculate accuracy over the rolling window."""
        if not self.pred_history:
            return 0.0
        correct = sum(p == t for p, t in zip(self.pred_history, self.true_history))
        return correct / len(self.pred_history)

    def get_rolling_balanced_accuracy(self) -> float:
        """Calculate balanced accuracy over the rolling window."""
        return self._compute_balanced_accuracy(self.true_history, self.pred_history)

    def get_rolling_auroc(self) -> float:
        """Calculate AUROC over the rolling window."""
        return self._compute_auroc(self.true_history, self.prob_history)

    def get_overall_accuracy(self) -> float:
        """Calculate accuracy over all trials seen so far."""
        if self.count == 0:
            return 0.0
        correct = sum(p == t for p, t in zip(self.all_preds, self.all_trues))
        return correct / self.count

    def get_overall_balanced_accuracy(self) -> float:
        """Calculate balanced accuracy over all trials seen so far."""
        return self._compute_balanced_accuracy(self.all_trues, self.all_preds)

    def get_overall_auroc(self) -> float:
        """Calculate AUROC over all trials seen so far."""
        return self._compute_auroc(self.all_trues, self.all_probs)

    def get_predictions_arrays(self) -> Dict[str, np.ndarray]:
        """Return all predictions as numpy arrays for saving."""
        result = {
            "pred_labels": np.array(self.all_preds),
            "true_labels": np.array(self.all_trues),
        }
        if self.all_probs:
            result["pred_probs"] = np.array(self.all_probs)
        return result


def evaluate_zero_shot(
    model, test_epochs, test_labels, device, original_unique_labels, batch_size=64
):
    """Evaluate pretrained model on test subject using original label mapping.

    Returns:
        Dict with keys: accuracy, balanced_accuracy, auroc, avg_loss,
                        pred_labels, true_labels, pred_probs (for npz saving)
    """
    model.eval()
    all_preds = []
    all_true = []
    all_probs = []
    all_losses = []

    if test_epochs.size == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": np.nan,
            "auroc": np.nan,
            "avg_loss": 0.0,
            "pred_labels": np.array([]),
            "true_labels": np.array([]),
            "pred_probs": np.array([]),
        }

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
            return {
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "auroc": np.nan,
                "avg_loss": np.nan,
                "pred_labels": np.array([]),
                "true_labels": np.array([]),
                "pred_probs": np.array([]),
            }

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
            probs = F.softmax(logits, dim=1)

            pred_labels = torch.argmax(logits, dim=1)

            all_preds.extend(pred_labels.cpu().tolist())
            all_true.extend(labels_batch.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_losses.append(loss.item())

    # Compute metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_true))
    avg_loss = np.mean(all_losses) if all_losses else 0.0

    # Balanced accuracy
    try:
        unique_labels = set(all_true)
        if len(unique_labels) >= 2:
            bal_acc = balanced_accuracy_score(all_true, all_preds)
        else:
            bal_acc = np.nan
    except Exception:
        bal_acc = np.nan

    # AUROC
    try:
        probs_array = np.array(all_probs)
        unique_labels = set(all_true)
        if len(unique_labels) >= 2:
            n_classes = probs_array.shape[1] if probs_array.ndim > 1 else 1
            if n_classes == 2:
                auroc = roc_auc_score(all_true, probs_array[:, 1])
            else:
                auroc = roc_auc_score(
                    all_true, probs_array, multi_class="ovr", average="macro"
                )
        else:
            auroc = np.nan
    except Exception:
        auroc = np.nan

    return {
        "accuracy": accuracy,
        "balanced_accuracy": bal_acc,
        "auroc": auroc,
        "avg_loss": avg_loss,
        "pred_labels": np.array(all_preds),
        "true_labels": np.array(all_true),
        "pred_probs": np.array(all_probs),
    }


def evaluate_single_trial(
    model: nn.Module,
    single_epoch_tensor: torch.Tensor,
    single_label_tensor: torch.Tensor,
    device: torch.device,
    original_unique_labels: Optional[np.ndarray],
    output_logits: Optional[torch.Tensor] = None,
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
            num_classes_placeholder = (
                len(original_unique_labels)
                if original_unique_labels is not None
                and len(original_unique_labels) > 0
                else 2
            )
            return {
                "true_label": original_label_int,
                "pred_label": np.nan,
                "pred_probs": [np.nan] * num_classes_placeholder,
                "loss": np.nan,
                "is_correct": np.nan,
            }

    remapped_label_tensor = torch.tensor(
        [remapped_label_int], dtype=torch.long, device=device
    )

    # Forward pass if logits not provided
    if output_logits is None:
        log.debug(
            "evaluate_single_trial: output_logits not provided, performing forward pass."
        )
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
        log.error(
            f"  Logits shape: {logits.shape if logits is not None else 'None'}, dtype: {logits.dtype if logits is not None else 'N/A'}"
        )
        log.error(f"  Remapped label tensor: {remapped_label_tensor}")
        num_classes_placeholder = (
            logits.shape[1]
            if logits is not None and logits.ndim > 1
            else (
                len(original_unique_labels)
                if original_unique_labels is not None
                and len(original_unique_labels) > 0
                else 2
            )
        )
        return {
            "true_label": original_label_int,
            "pred_label": np.nan,
            "pred_probs": [np.nan] * num_classes_placeholder,
            "loss": np.nan,
            "is_correct": np.nan,
        }

    return {
        "true_label": original_label_int,
        "pred_label": pred_label_remapped,
        "pred_probs": pred_probs.cpu().tolist(),
        "loss": loss.item(),
        "is_correct": is_correct,
        "true_label_remapped": true_label_remapped,
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
                stripped_param = arg_key[len(prefix) :]
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
        # Append PID to avoid collisions when multiple jobs start in the same second
        timestamp_str = f"{timestamp_str}_{os.getpid()}"
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
