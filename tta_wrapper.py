"""Test-time adaptation wrapper for EEG classification models."""

import logging
from typing import Any, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import eigh

try:
    from pyriemann.utils.mean import mean_riemann

    PYRIEMANN_AVAILABLE = True
except ImportError:
    PYRIEMANN_AVAILABLE = False

log = logging.getLogger(__name__)


def _compute_trial_covariances_np(
    trials_np: np.ndarray, cov_epsilon: float = 1e-6
) -> np.ndarray:
    """Compute covariance matrices for trials with regularization."""
    n_trials, n_channels, _ = trials_np.shape
    covs = np.zeros((n_trials, n_channels, n_channels))
    for i in range(n_trials):
        trial_data = trials_np[i]
        cov = np.cov(trial_data)
        # Adaptive regularization
        trace_cov = np.trace(cov)
        adaptive_epsilon = max(cov_epsilon, trace_cov * 1e-6)
        cov += adaptive_epsilon * np.identity(n_channels)
        covs[i] = cov
    return covs


def _compute_reference_covariance_np(
    trial_covs_np: np.ndarray, alignment_type: str = "euclidean"
) -> np.ndarray:
    """Compute reference covariance matrix from trial covariances."""
    if alignment_type == "riemannian":
        if not PYRIEMANN_AVAILABLE:
            raise ImportError("pyriemann required for Riemannian alignment")
        try:
            ref_cov = mean_riemann(trial_covs_np)
        except ValueError as e:
            if "positive definite" in str(e):
                # Apply stronger regularization and retry
                n_trials, n_channels, _ = trial_covs_np.shape
                reg_strength = 1e-5
                for attempt in range(3):
                    try:
                        identity_term = (
                            reg_strength * np.eye(n_channels)[np.newaxis, :, :]
                        )
                        regularized_covs = trial_covs_np + identity_term
                        ref_cov = mean_riemann(regularized_covs)
                        break
                    except ValueError:
                        reg_strength *= 10
                    if attempt == 2:
                        # Fallback to Euclidean
                        ref_cov = np.mean(regularized_covs, axis=0)
                        break
            else:
                raise e
        # Ensure well-conditioned result
        min_eigenval = np.min(np.linalg.eigvals(ref_cov))
        if min_eigenval < 1e-8:
            trace_ref = np.trace(ref_cov)
            safety_epsilon = max(1e-8, trace_ref * 1e-7)
            ref_cov += safety_epsilon * np.identity(ref_cov.shape[0])
        return ref_cov
    elif alignment_type == "euclidean":
        ref_cov = np.mean(trial_covs_np, axis=0)
        # Ensure well-conditioned
        min_eigenval = np.min(np.linalg.eigvals(ref_cov))
        if min_eigenval < 1e-8:
            trace_ref = np.trace(ref_cov)
            safety_epsilon = max(1e-8, trace_ref * 1e-7)
            ref_cov += safety_epsilon * np.identity(ref_cov.shape[0])
        return ref_cov
    else:
        raise ValueError(f"Unsupported alignment type: {alignment_type}")


def _compute_alignment_transform_np(
    ref_cov_np: np.ndarray, transform_epsilon: float = 1e-7
) -> np.ndarray:
    """Compute alignment transform matrix from reference covariance."""
    eigenvalues, eigenvectors = eigh(ref_cov_np)

    # Robust eigenvalue regularization
    max_eigenval = np.max(eigenvalues)
    adaptive_epsilon = max(transform_epsilon, max_eigenval * 1e-12)
    stable_eigenvalues = np.maximum(eigenvalues, adaptive_epsilon)
    # Check condition number
    condition_number = np.max(stable_eigenvalues) / np.min(stable_eigenvalues)
    if condition_number > 1e12:
        stronger_epsilon = max_eigenval * 1e-8
        stable_eigenvalues = np.maximum(eigenvalues, stronger_epsilon)
    # Compute inverse square root
    inv_sqrt_diag_eigenvalues = np.diag(1.0 / np.sqrt(stable_eigenvalues))
    transform_matrix = eigenvectors @ inv_sqrt_diag_eigenvalues @ eigenvectors.T
    return transform_matrix


def _apply_alignment_transform_np(
    trials_np: np.ndarray, transform_matrix_np: np.ndarray
) -> np.ndarray:
    """
    Applies an alignment transform to a batch of trials.
    """
    return np.einsum("jk,ikm->ijm", transform_matrix_np, trials_np)


def set_adabn_status(model: nn.Module, adabn_enabled: bool):
    """Set AdaBN status for all BatchNorm layers in model."""
    for module in model.children():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.track_running_stats = not adabn_enabled
        if len(list(module.children())) > 0:
            set_adabn_status(module, adabn_enabled)


def set_bn_train_mode(model: nn.Module, train_mode: bool):
    """Set BatchNorm layers to train/eval mode (Wimpff BN-1 style)."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.train(train_mode)


class TTAWrapper(nn.Module):
    """Test-time adaptation wrapper for EEG classification models."""

    def __init__(self, model: nn.Module, args: Any, sr_hz: Optional[float] = None):
        """Initialize TTA wrapper with model and configuration."""
        super().__init__()
        self.wrapped_model = model
        self.args = args
        self.sr_hz = sr_hz
        self.device = next(model.parameters()).device

        # TTA state
        self.reference_cov_np: Optional[np.ndarray] = None
        self.alignment_transform_torch: Optional[torch.Tensor] = None
        self.tta_stats_update_counter = 0

        # Buffer for Wimpff-style BN-1 (stores raw trials for batch statistics)
        self.tta_buffer: Optional[torch.Tensor] = None
        self.tta_buffer_length = getattr(args, "tta_buffer_length", 8)

        # Validate Riemannian alignment
        if self.args.alignment_type == "riemannian" and not PYRIEMANN_AVAILABLE:
            log.warning("Riemannian alignment requested but pyriemann not available")

    def forward(
        self,
        x_raw_torch: torch.Tensor,
        apply_tta: bool = False,
        is_finetuning_batch: bool = False,
    ) -> torch.Tensor:
        """Forward pass with optional TTA application."""
        # Supervised finetuning - only apply alignment
        if is_finetuning_batch:
            x_aligned = self._apply_current_alignment_torch(x_raw_torch)
            return self.wrapped_model(x_aligned)
        # TTA prediction - apply alignment + AdaBN
        if apply_tta and self.args.use_tta:
            return self._forward_with_tta(x_raw_torch)
        # Standard forward pass
        return self.wrapped_model(x_raw_torch)

    def _forward_with_tta(self, x_raw_torch: torch.Tensor) -> torch.Tensor:
        """Apply TTA (alignment + AdaBN) and predict.

        AdaBN modes:
        - 'track_running_stats': Sets track_running_stats=False (EDAPT default)
        - 'train_mode': Sets BN layers to train mode (Wimpff BN-1 style)
        """
        was_training = self.wrapped_model.training
        self.wrapped_model.eval()
        try:
            # Apply alignment
            x_aligned = self._apply_current_alignment_torch(x_raw_torch)
            # Apply AdaBN if enabled
            if self.args.use_adabn:
                adabn_mode = getattr(self.args, "adabn_mode", "track_running_stats")
                if adabn_mode == "train_mode":
                    # Wimpff BN-1 style: set BN layers to train mode
                    set_bn_train_mode(self.wrapped_model, train_mode=True)
                    try:
                        logits = self.wrapped_model(x_aligned)
                    finally:
                        set_bn_train_mode(self.wrapped_model, train_mode=False)
                else:
                    # EDAPT default: track_running_stats=False
                    set_adabn_status(self.wrapped_model, adabn_enabled=True)
                    try:
                        logits = self.wrapped_model(x_aligned)
                    finally:
                        set_adabn_status(self.wrapped_model, adabn_enabled=False)
            else:
                logits = self.wrapped_model(x_aligned)
            return logits
        finally:
            if was_training:
                self.wrapped_model.train()

    @torch.no_grad()
    def update_tta_statistics_and_predict(
        self, x_raw_trial: torch.Tensor
    ) -> torch.Tensor:
        """Update TTA statistics with current trial and predict on same trial.

        For BN-1 (AdaBN), uses a buffer of recent trials so BN layers see
        meaningful batch statistics (Wimpff et al. 2024 style).
        """
        # Add current trial to buffer for BN-1
        if self.args.use_adabn and self.tta_buffer_length > 1:
            if self.tta_buffer is None:
                self.tta_buffer = x_raw_trial
            elif self.tta_buffer.shape[0] < self.tta_buffer_length:
                self.tta_buffer = torch.cat([self.tta_buffer, x_raw_trial], dim=0)
            else:
                # Sliding window: remove oldest, add newest
                self.tta_buffer = torch.cat([self.tta_buffer[1:], x_raw_trial], dim=0)

        # Update alignment statistics using current trial (EDAPT-style EMA)
        self.update_tta_statistics(x_raw_trial)

        # Forward pass: use buffer for BN-1 if enabled, otherwise single trial
        if (
            self.args.use_adabn
            and self.tta_buffer is not None
            and self.tta_buffer.shape[0] > 1
        ):
            # Wimpff-style: forward full buffer, return last prediction
            logits = self.forward(self.tta_buffer, apply_tta=True)
            return logits[-1:, :]  # Return only current trial's prediction
        else:
            # Standard: single trial forward
            return self.forward(x_raw_trial, apply_tta=True)

    @torch.no_grad()
    def update_tta_statistics(self, x_raw_unseen_trial_torch: torch.Tensor):
        """Update TTA statistics using new trial data."""
        # If TTA is off or alignment is None, do nothing.
        # MODIFIED: Check for both the string 'none' and the Python object None
        if (
            not self.args.use_tta
            or self.args.alignment_type == "none"
            or self.args.alignment_type is None
        ):
            return

        x_raw_unseen_np = x_raw_unseen_trial_torch.detach().cpu().numpy()
        try:
            # Compute covariance of new trial
            trial_covs_new_data = _compute_trial_covariances_np(
                x_raw_unseen_np, self.args.alignment_cov_epsilon
            )
            # Compute reference from new data
            current_ref_cov_from_new_data = _compute_reference_covariance_np(
                trial_covs_new_data, self.args.alignment_type
            )
        except ImportError as e:
            log.warning(f"TTA statistics update failed: {e}")
            return
        except Exception as e:
            # This is where the "Unsupported alignment type" error was being caught
            log.warning(f"Error computing covariances for TTA update: {e}")
            return
        # Update reference covariance using EMA
        if self.reference_cov_np is None:
            self.reference_cov_np = current_ref_cov_from_new_data
        else:
            beta = self.args.alignment_ref_ema_beta
            self.reference_cov_np = (
                beta * self.reference_cov_np
                + (1 - beta) * current_ref_cov_from_new_data
            )

        # Recompute alignment transform
        try:
            transform_np = _compute_alignment_transform_np(
                self.reference_cov_np, self.args.alignment_transform_epsilon
            )
            self.alignment_transform_torch = (
                torch.from_numpy(transform_np).float().to(self.device)
            )
        except Exception as e:
            log.warning(f"Error computing alignment transform: {e}")
        self.tta_stats_update_counter += 1

    def _apply_current_alignment_torch(self, x_torch: torch.Tensor) -> torch.Tensor:
        """Apply current alignment transform if available and enabled."""
        # Only apply transform if TTA is used, alignment is not None, and transform exists
        # MODIFIED: Check for both the string 'none' and the Python object None
        if (
            self.args.use_tta
            and self.args.alignment_type != "none"
            and self.args.alignment_type is not None
            and self.alignment_transform_torch is not None
        ):
            return torch.einsum("jk,bkm->bjm", self.alignment_transform_torch, x_torch)
        # Otherwise, return the original tensor
        return x_torch

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Return parameters based on finetune_mode."""
        if self.args.finetune_mode == "decision_only":
            # Find last learnable layer
            last_learnable_layer = None
            for module in reversed(list(self.wrapped_model.modules())):
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    if any(p.requires_grad for p in module.parameters()):
                        last_learnable_layer = module
                        break
            if last_learnable_layer:
                return last_learnable_layer.parameters()
            else:
                log.warning("Could not find decision layer. Using all parameters.")
                return self.wrapped_model.parameters(recurse=recurse)
        else:  # 'full' mode
            return self.wrapped_model.parameters(recurse=recurse)

    def train(self, mode: bool = True):
        """Set training mode and configure AdaBN."""
        super().train(mode)
        self.wrapped_model.train(mode)
        if mode:
            # Training mode: ensure AdaBN is OFF
            set_adabn_status(self.wrapped_model, adabn_enabled=False)
        return self

    def eval(self):
        """Set evaluation mode and configure AdaBN."""
        super().eval()
        self.wrapped_model.eval()
        # Evaluation mode: ensure AdaBN is OFF by default
        set_adabn_status(self.wrapped_model, adabn_enabled=False)
        return self

    def get_wrapped_model(self) -> nn.Module:
        """Return the underlying wrapped model."""
        return self.wrapped_model

    def reset_tta_state(self):
        """Reset TTA state (alignment reference and buffer). Call between subjects."""
        self.reference_cov_np = None
        self.alignment_transform_torch = None
        self.tta_stats_update_counter = 0
        self.tta_buffer = None

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """Return named parameters based on finetune_mode."""
        if self.args.finetune_mode == "decision_only":
            # Find last learnable layer and its name
            last_learnable_layer = None
            decision_layer_name = ""
            for module_name, module in reversed(
                list(self.wrapped_model.named_modules())
            ):
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    if any(p.requires_grad for p in module.parameters()):
                        last_learnable_layer = module
                        decision_layer_name = module_name
                        break
            if last_learnable_layer and decision_layer_name:
                param_prefix = f"{prefix}wrapped_model.{decision_layer_name}."
                return last_learnable_layer.named_parameters(
                    prefix=param_prefix, recurse=recurse
                )
            else:
                return self.wrapped_model.named_parameters(
                    prefix=f"{prefix}wrapped_model.", recurse=recurse
                )
        else:  # 'full' mode
            return self.wrapped_model.named_parameters(
                prefix=f"{prefix}wrapped_model.", recurse=recurse
            )
