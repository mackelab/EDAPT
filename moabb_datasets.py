# %%
"""MOABB Datasets Module for EEG Classification."""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from omegaconf import ListConfig
import mne
import numpy as np
import pandas as pd
from moabb.datasets.base import BaseDataset
from moabb.paradigms import P300, SSVEP, MotorImagery
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from tqdm.auto import tqdm

import utils
from yang2025_moabb import Yang2025
from tta_wrapper import (_compute_trial_covariances_np,
                       _compute_reference_covariance_np,
                       _compute_alignment_transform_np,
                       _apply_alignment_transform_np,
                       PYRIEMANN_AVAILABLE)

# Set up module-level logger
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


# %%
# Core Dataset Classes

class EEGDataset(Dataset):
    """Basic EEG Dataset for PyTorch compatibility."""

    def __init__(self, epochs: np.ndarray, labels: np.ndarray, channel_names: Optional[List[str]] = None):
        self.epochs = epochs
        self.labels = labels
        self.channel_names = channel_names

    def __len__(self) -> int:
        """Return the number of trials in the dataset."""
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a single trial and its label."""
        sample = {
            "epoch": self.epochs[index],
            "label": self.labels[index],
        }
        if self.channel_names is not None:
            sample["channel_names"] = self.channel_names
        return sample

    @staticmethod
    def load_subject_data(subject_index: int, data_directory: str, file_name_format: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load EEG data for a single subject from file."""
        file_path = os.path.join(data_directory, file_name_format.format(subject_index))
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Data file not found for subject {subject_index} at {file_path}"
            )
        
        epochs = mne.read_epochs(file_path)
        labels = epochs.events[:, 2] - 1  # Convert labels {1, 2} to {0, 1}
        channel_names = epochs.ch_names
        epochs_data = epochs.get_data().astype(np.float32)

        # Shuffle data for consistency
        epochs_data, labels = shuffle(epochs_data, labels, random_state=42)

        return epochs_data, labels, channel_names


# %%
# Subject Management

def get_subject_list_for_datasets(dataset_names: List[str], data_root: str) -> List[int]:
    """Get subject list from a single MOABB dataset."""
    # Configure MNE data paths
    mne.set_config("MNE_DATA", data_root)
    mne.set_config("MNE_DATASETS_SSVEP_PATH", data_root)
    mne.set_config("MNE_DATASETS_BNCI_PATH", data_root)
    mne.set_config("MNE_DATASETS_MOABB_PATH", data_root)

    if not dataset_names or len(dataset_names) != 1:
        log.error("Function now expects exactly one dataset name.")
        return []
    
    dataset_name = dataset_names[0]
    
    # Determine paradigm
    paradigm_name = None
    for p_name, p_data in PARADIGM_DATA.items():
        if dataset_name in p_data["class_map"]:
            paradigm_name = p_name
            break

    if paradigm_name is None:
        log.error(f"Could not determine paradigm for dataset: {dataset_name}")
        return []

    dataset_class_map = PARADIGM_DATA[paradigm_name]["class_map"]
    dataset_class = dataset_class_map[dataset_name]
    
    try:
        dataset_instance = dataset_class()
        subjects = dataset_instance.subject_list
        log.info(f"Found {len(subjects)} subjects in {dataset_name}")
        return sorted(subjects)
    except Exception as e:
        log.error(f"Could not get subjects for dataset {dataset_name}: {e}")
        return []


# %%
# Data Loading Functions
def load_cached_pretrain_data(
    dataset_names: List[str],
    subject_ids: List[int],
    paradigm_kwargs: Dict[str, Any],
    data_root: str,
    args: Any,
    verbose: bool = True,
    target_type: str = 'classification',
    apply_trial_ablation: bool = False,
    ) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray], Optional[int], Optional[int], Optional[List[str]]
    ]:
    """Load and concatenate EEG data from multiple subjects from a single dataset."""
    all_epochs_list = []
    all_labels_list = []
    n_channels, n_timepoints = None, None
    channel_names = None

    if not dataset_names or len(dataset_names) != 1:
        log.error("Function now expects exactly one dataset name.")
        return None, None, None, None, None
    
    dataset_name = dataset_names[0]
    determined_paradigm_name = None
    for p_name, p_data in PARADIGM_DATA.items():
        if dataset_name in p_data["class_map"]:
            determined_paradigm_name = p_name
            break

    if determined_paradigm_name is None:
        log.error(f"Could not determine paradigm for dataset: {dataset_name}")
        return None, None, None, None, None

    # Use effective paradigm kwargs
    effective_paradigm_kwargs = paradigm_kwargs.copy()

    # Select appropriate caching paradigm class
    if determined_paradigm_name == "MI":
        paradigm_class = CachingMotorImagery
    elif determined_paradigm_name == "P300":
        paradigm_class = CachingP300
    elif determined_paradigm_name == "SSVEP":
        paradigm_class = CachingSSVEP
    else:
        log.error(f"Unknown paradigm name: {determined_paradigm_name}")
        return None, None, None, None, None
    
    if apply_trial_ablation and hasattr(args, 'num_trials_per_subject') and args.num_trials_per_subject is not None:
        effective_paradigm_kwargs['num_trials_per_subject'] = args.num_trials_per_subject
        log.info(f"load_cached_pretrain_data: Applying trial ablation. Passing num_trials_per_subject={args.num_trials_per_subject} to paradigm {paradigm_class.__name__}.")


    # Instantiate paradigm
    log.info(f"Instantiating {paradigm_class.__name__} with kwargs: {effective_paradigm_kwargs}")
    try:
        paradigm = paradigm_class(**effective_paradigm_kwargs)
    except Exception as e:
        log.error(f"Failed to instantiate paradigm {paradigm_class.__name__}: {e}", exc_info=True)
        return None, None, None, None, None

    # Configure MOABB data paths
    mne.set_config("MNE_DATA", data_root)
    mne.set_config("MNE_DATASETS_SSVEP_PATH", data_root)
    mne.set_config("MNE_DATASETS_BNCI_PATH", data_root)
    mne.set_config("MNE_DATASETS_MOABB_PATH", data_root)

    # Load data from the single dataset
    if dataset_name not in PARADIGM_DATA[determined_paradigm_name]["class_map"]:
        log.warning(f"Dataset '{dataset_name}' not found in PARADIGM_DATA.")
        return None, None, None, None, None

    dataset_class = PARADIGM_DATA[determined_paradigm_name]["class_map"][dataset_name]

    try:
        log.debug(f"Instantiating dataset: {dataset_name}")
        dataset = dataset_class()

        # Filter subjects available in this dataset
        available_subjects_in_ds = dataset.subject_list
        subjects_to_load_for_ds = [s for s in subject_ids if s in available_subjects_in_ds]

        if not subjects_to_load_for_ds:
            log.info(f"No requested subjects found in dataset {dataset_name}.")
            return None, None, None, None, None

        log.info(f"Loading data for {len(subjects_to_load_for_ds)} subjects from {dataset_name}")
        
        # Load data using caching paradigm
        epochs_data, y_values, metadata = paradigm.get_data(
            dataset=dataset, subjects=subjects_to_load_for_ds
        )

        if epochs_data is None or epochs_data.size == 0:
            log.warning(f"No data returned from paradigm.get_data for {dataset_name}.")
            return None, None, None, None, None

        # Map string labels to integers for classification
        try:
            event_id_map = dataset.event_id
            if not event_id_map:
                raise ValueError(f"Dataset {dataset_name} has empty event_id for classification.")
            
            labels_numeric = np.array([event_id_map[label] for label in y_values])
            log.debug(f"Mapped classification labels for {dataset_name}. Example: '{y_values[0]}' -> {labels_numeric[0]}")
        except KeyError as e:
            log.error(f"Label mapping error for {dataset_name}: Label '{e}' not found in event_id {event_id_map}.")
            return None, None, None, None, None
        except Exception as e_map:
            log.error(f"Error during label mapping for {dataset_name}: {e_map}.", exc_info=True)
            return None, None, None, None, None

        log.debug(f"Loaded data shape for {dataset_name}: {epochs_data.shape}")

        # Set dimensions
        _, n_channels, n_timepoints = epochs_data.shape

        # Get channel names from paradigm
        if hasattr(paradigm, 'channels') and paradigm.channels is not None:
            channel_names = paradigm.channels
            log.info(f"Using channel subset from paradigm: {channel_names}")
        else:
            log.warning("No channel subset specified. Using placeholder names.")
            channel_names = [f"Ch{i+1}" for i in range(n_channels)]

        all_epochs_list.append(epochs_data)
        all_labels_list.append(labels_numeric)

    except Exception as e:
        log.error(f"Error processing dataset {dataset_name}: {e}", exc_info=True)
        return None, None, None, None, None

    if not all_epochs_list:
        log.warning("No pretraining data could be loaded for specified subjects and datasets.")
        return None, None, None, None, None
    
    if (hasattr(args, 'use_tta') and args.use_tta and
        hasattr(args, 'alignment_type') and args.alignment_type != 'none'
        and args.alignment_type is not None):
        log.info(f"Applying per-subject pretraining alignment (type: {args.alignment_type})")

        aligned_epochs_list = []
        for subject_block_idx, subject_epochs_np in enumerate(all_epochs_list):
            if subject_epochs_np.size == 0:
                aligned_epochs_list.append(subject_epochs_np)
                continue

            try:
                current_alignment_type = args.alignment_type
                if current_alignment_type == 'riemannian' and not PYRIEMANN_AVAILABLE:
                    log.warning("Riemannian alignment requested, falling back to Euclidean.")
                    current_alignment_type = 'euclidean'

                # Compute alignment transformation
                source_trial_covs = _compute_trial_covariances_np(subject_epochs_np, args.alignment_cov_epsilon)
                source_ref_cov = _compute_reference_covariance_np(source_trial_covs, current_alignment_type)
                source_transform = _compute_alignment_transform_np(source_ref_cov, args.alignment_transform_epsilon)
                aligned_subject_epochs = _apply_alignment_transform_np(subject_epochs_np, source_transform)
                aligned_epochs_list.append(aligned_subject_epochs)

            except Exception as e:
                log.warning(f"Alignment failed for subject block {subject_block_idx}. Using original data. Error: {e}")
                aligned_epochs_list.append(subject_epochs_np)

        all_epochs_list = aligned_epochs_list
        log.info("Pretraining alignment complete.")


    # Concatenate data
    try:
        concatenated_epochs = np.concatenate(all_epochs_list, axis=0) if all_epochs_list else np.array([])
        concatenated_labels = np.concatenate(all_labels_list, axis=0) if all_labels_list else np.array([])
        log.info(f"Successfully concatenated data. Final shapes: Epochs {concatenated_epochs.shape}, "
                f"Labels {concatenated_labels.shape}")
        log.info(f"Final labels dtype: {concatenated_labels.dtype}")
    except ValueError as e_concat:
        log.error(f"Failed to concatenate data: {e_concat}", exc_info=True)
        return None, None, None, None, None

    return concatenated_epochs, concatenated_labels, n_channels, n_timepoints, channel_names


# %%
# Custom Paradigm Classes

class CustomMotorImagery(MotorImagery):
    """Custom Motor Imagery paradigm with dynamic event handling and balanced accuracy scoring."""

    def __init__(self, fmin: float = 1, fmax: float = 47, resample: Optional[float] = None, **kwargs):
        """Initialize Custom Motor Imagery paradigm."""
        log.debug(f"CustomMotorImagery init: fmin={fmin}, fmax={fmax}, resample={resample}")
        super().__init__(fmin=fmin, fmax=fmax, resample=resample, **kwargs)

    def used_events(self, dataset: BaseDataset) -> Dict[str, int]:
        """Return dataset-specific event dictionary."""
        if not hasattr(dataset, "paradigm") or dataset.paradigm != "imagery":
            raise ValueError(f"Dataset {dataset.code} is not an 'imagery' paradigm dataset.")

        if not hasattr(dataset, "event_id") or not dataset.event_id:
            raise ValueError(f"Dataset {dataset.code} does not have 'event_id' or it's empty.")

        return dataset.event_id

    @property
    def scoring(self) -> str:
        """Return scoring metric for evaluation."""
        return "balanced_accuracy"


class CustomP300(P300):
    """Custom P300 paradigm with dynamic event handling and balanced accuracy scoring."""

    def __init__(self, fmin: float = 1, fmax: float = 47, resample: Optional[float] = None, **kwargs):
        """Initialize Custom P300 paradigm."""
        log.debug(f"CustomP300 init: fmin={fmin}, fmax={fmax}, resample={resample}")
        super().__init__(fmin=fmin, fmax=fmax, resample=resample, **kwargs)

    def used_events(self, dataset: BaseDataset) -> Dict[str, int]:
        """Return dataset-specific event dictionary for P300 paradigm."""
        if not hasattr(dataset, "paradigm") or dataset.paradigm != "p300":
            raise ValueError(f"Dataset {dataset.code} is not a 'p300' paradigm dataset.")

        if not hasattr(dataset, "event_id") or not dataset.event_id:
            raise ValueError(f"Dataset {dataset.code} does not have 'event_id' or it's empty.")

        return dataset.event_id

    @property
    def scoring(self) -> str:
        """Return scoring metric for evaluation."""
        return "balanced_accuracy"


class CustomSSVEP(SSVEP):
    """Custom SSVEP paradigm with dynamic event handling and balanced accuracy scoring."""

    def __init__(self, fmin: float = 1, fmax: float = 47, resample: Optional[float] = None, **kwargs):
        """Initialize Custom SSVEP paradigm."""
        log.debug(f"CustomSSVEP init: fmin={fmin}, fmax={fmax}, resample={resample}")
        super().__init__(fmin=fmin, fmax=fmax, resample=resample, **kwargs)

    def used_events(self, dataset: BaseDataset) -> Dict[str, int]:
        """Return dataset-specific event dictionary for SSVEP paradigm."""
        if not hasattr(dataset, "paradigm") or dataset.paradigm != "ssvep":
            raise ValueError(f"Dataset {dataset.code} is not an 'ssvep' paradigm dataset.")

        if not hasattr(dataset, "event_id") or not dataset.event_id:
            raise ValueError(f"Dataset {dataset.code} does not have 'event_id' or it's empty.")

        return dataset.event_id

    @property
    def scoring(self) -> str:
        """Return scoring metric for evaluation."""
        return "balanced_accuracy"


# %%
# Caching Paradigm Classes

class CachingMotorImagery(CustomMotorImagery):
    """Motor Imagery paradigm with intelligent caching and data standardization."""

    def __init__(self, fmin: float = 1, fmax: float = 47, resample: Optional[float] = None, 
                 channels: Optional[List[str]] = None, events: Optional[Dict[str, int]] = None,
                 num_trials_per_subject: Optional[int] = None, **kwargs):
        """Initialize Caching Motor Imagery paradigm."""
        self._fmin = fmin
        self._fmax = fmax
        self._resample = resample
        self._channels = channels
        self._events = events
        self._num_trials_per_subject = num_trials_per_subject

        channels_for_super = list(channels) if isinstance(channels, ListConfig) else channels
        
        super_args = dict(
            fmin=fmin, fmax=fmax, resample=resample, channels=channels_for_super, events=events, **kwargs
        )
        super().__init__(**super_args)

    def _get_cache_path(self, dataset: BaseDataset, subject: int, session_name: str = "all_sessions") -> Path:
        """Construct cache file path based on preprocessing parameters."""
        effective_channels = self._channels if self._channels is not None else getattr(self, "channels", None)

        if isinstance(effective_channels, (list, ListConfig)):
            effective_channels_str = "_".join(sorted(list(effective_channels)))
        elif effective_channels is None:
            effective_channels_str = "all"
        else:
            effective_channels_str = str(effective_channels)

        effective_events = getattr(self, "events", self._events)
        if effective_events is None:
            effective_events_str = str(sorted(self.used_events(dataset).items()))
        else:
            effective_events_str = str(sorted(effective_events.items()))

        param_dict = {
            "fmin": self._fmin,
            "fmax": self._fmax,
            "resample": self._resample,
            "channels": effective_channels_str,
            "events": effective_events_str,
            "num_trials": str(self._num_trials_per_subject) if self._num_trials_per_subject is not None else "all",
        }
        
        param_str = "_".join(f"{k}={v}" for k, v in sorted(param_dict.items()))
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

        cache_dir = (
            utils.CACHE_ROOT_DIR 
            / dataset.code
            / f"subject_{subject:03d}"
            / f"session_{session_name}"
        )
        cache_filename = f"params_{param_hash}.npz"
        return cache_dir / cache_filename

    def get_data(self, dataset: BaseDataset, subjects: Optional[List[int]] = None, 
                return_epochs: bool = False, **kwargs) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Retrieve data with caching and standardization."""
        if subjects is None:
            subjects = dataset.subject_list
        elif not isinstance(subjects, list):
            subjects = [subjects]

        all_X = []
        all_labels = []
        all_metadata = []

        for subject in subjects:
            cache_path = self._get_cache_path(dataset, subject)
            X, labels, metadata = None, None, None

            # Try to load from cache
            if cache_path.exists():
                try:
                    cache_data = np.load(cache_path, allow_pickle=True)
                    X = cache_data["X"]
                    labels = cache_data["labels"]
                    if 'metadata_cols' in cache_data and all(f"meta_{col}" in cache_data for col in cache_data['metadata_cols']):
                        metadata_dict = {col: cache_data[f"meta_{col}"] for col in cache_data["metadata_cols"]}
                        metadata = pd.DataFrame(metadata_dict)
                    else:
                        metadata = pd.DataFrame()
                except Exception as e:
                    log.warning(f"[{dataset.code}/Subj {subject}] Cache load error: {e}. Recomputing.")
                    X, labels, metadata = None, None, None
                else:
                    if X is None or X.size == 0 or labels is None:
                        log.warning(f"[{dataset.code}/Subj {subject}] Invalid cached data. Recomputing.")
                        X, labels, metadata = None, None, None

            # Compute data if cache miss or invalid
            if X is None:
                try:
                    X_computed, labels_computed, metadata_computed = super().get_data(
                        dataset, [subject], return_epochs=return_epochs, **kwargs
                    )

                    if X_computed is None or X_computed.size == 0 or labels_computed is None:
                        log.warning(f"super().get_data returned no valid data for subject {subject}. Skipping.")
                        continue
                    
                    X, labels, metadata = X_computed, labels_computed, metadata_computed

                    # Apply trial slicing before caching
                    if self._num_trials_per_subject is not None:
                        if X.shape[0] > self._num_trials_per_subject:
                            log.info(f"Slicing newly computed data for Subj {subject} to {self._num_trials_per_subject} trials.")
                            X = X[:self._num_trials_per_subject]
                            labels = labels[:self._num_trials_per_subject]
                            if not metadata.empty and len(metadata) > self._num_trials_per_subject:
                                metadata = metadata.iloc[:self._num_trials_per_subject]

                    # Save to cache
                    try:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        metadata_save_dict = {f"meta_{col}": metadata[col].values for col in metadata.columns}
                        metadata_save_dict["metadata_cols"] = metadata.columns.tolist()
                        np.savez(cache_path, X=X, labels=labels, **metadata_save_dict)
                    except Exception as e_save:
                        log.error(f"[{dataset.code}/Subj {subject}] Cache save failed: {e_save}")

                except Exception as e_get:
                    log.error(f"[{dataset.code}/Subj {subject}] Data computation failed: {e_get}. Skipping.")
                    continue

            # Apply final trial slicing
            if X is not None and X.size > 0 and labels is not None:
                if self._num_trials_per_subject is not None:
                    if X.shape[0] > self._num_trials_per_subject:
                        log.info(f"[{dataset.code}/Subj {subject}] Final slicing to {self._num_trials_per_subject} trials.")
                        X = X[:self._num_trials_per_subject]
                        labels = labels[:self._num_trials_per_subject]
                        if not metadata.empty and len(metadata) > self._num_trials_per_subject:
                            metadata = metadata.iloc[:self._num_trials_per_subject]
                
                all_X.append(X)
                all_labels.append(labels)
                all_metadata.append(metadata)

        if not all_X:
            log.warning(f"[{dataset.code}] No data processed.")
            return np.array([]), np.array([]), pd.DataFrame()

        # Final concatenation
        try:
            final_X = np.concatenate(all_X, axis=0) if all_X else np.array([])
            final_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
            final_metadata = pd.concat(all_metadata, ignore_index=True) if all_metadata and not all(df.empty for df in all_metadata) else pd.DataFrame()

            log.info(f"[{dataset.code}] Final data: Epochs {final_X.shape if final_X.size > 0 else 'Empty'}, "
                    f"Labels {final_labels.shape if final_labels.size > 0 else 'Empty'}")
            return final_X, final_labels, final_metadata
        except ValueError as ve:
            shapes = [X_s.shape for X_s in all_X if X_s is not None]
            log.error(f"[{dataset.code}] Concatenation failed. Shapes: {shapes}. Error: {ve}")
            return np.array([]), np.array([]), pd.DataFrame()


class CachingP300(CustomP300):
    """P300 paradigm with intelligent caching and data standardization."""

    def __init__(self, fmin: float = 1, fmax: float = 47, resample: Optional[float] = None,
                 channels: Optional[List[str]] = None, events: Optional[Dict[str, int]] = None,
                 num_trials_per_subject: Optional[int] = None, **kwargs):
        """Initialize Caching P300 paradigm."""
        self._fmin = fmin
        self._fmax = fmax
        self._resample = resample
        self._channels_init = channels
        self._events_init = events
        self._num_trials_per_subject = num_trials_per_subject

        super_args = dict(fmin=fmin, fmax=fmax, resample=resample, **kwargs)
        if events is not None:
            super_args['events'] = events
        
        super().__init__(**super_args)

    def _get_cache_path(self, dataset: BaseDataset, subject: int, session_name: str = "all_sessions") -> Path:
        """Construct cache file path for P300 paradigm."""
        effective_channels = getattr(self, "channels", self._channels_init)
        if isinstance(effective_channels, (list, ListConfig)):
            effective_channels_str = "_".join(sorted(list(effective_channels)))
        elif effective_channels is None:
            effective_channels_str = "all"
        else:
            effective_channels_str = str(effective_channels)
        
        effective_events = getattr(self, "events", self._events_init)
        if isinstance(effective_events, dict):
            effective_events_str = str(sorted(effective_events.items()))
        elif effective_events is None:
            effective_events_str = str(sorted(self.used_events(dataset).items()))
        else:
            log.warning(f"Unexpected P300 events type: {type(effective_events)}")
            effective_events_str = str(effective_events)

        param_dict = {
            "fmin": self._fmin,
            "fmax": self._fmax,
            "resample": self._resample,
            "channels": effective_channels_str,
            "events": effective_events_str,
            "num_trials": str(self._num_trials_per_subject) if self._num_trials_per_subject is not None else "all",
        }
        
        param_str = "_".join(f"{k}={v}" for k, v in sorted(param_dict.items()))
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

        cache_dir = (
            utils.CACHE_ROOT_DIR 
            / dataset.code
            / f"subject_{subject:03d}"
            / f"session_{session_name}"
        )
        cache_filename = f"params_{param_hash}.npz"
        return cache_dir / cache_filename

    def get_data(self, dataset: BaseDataset, subjects: Optional[List[int]] = None,
                return_epochs: bool = False, **kwargs) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Retrieve P300 data with caching and standardization."""
        if subjects is None:
            subjects = dataset.subject_list
        elif not isinstance(subjects, list):
            subjects = [subjects]

        all_X = []
        all_labels = []
        all_metadata = []

        for subject in subjects:
            cache_path = self._get_cache_path(dataset, subject)
            X, labels, metadata = None, None, None

            # Try cache load
            if cache_path.exists():
                try:
                    cache_data = np.load(cache_path, allow_pickle=True)
                    X = cache_data["X"]
                    labels = cache_data["labels"]
                    if 'metadata_cols' in cache_data:
                        metadata_dict = {col: cache_data[f"meta_{col}"] for col in cache_data["metadata_cols"]}
                        metadata = pd.DataFrame(metadata_dict)
                    else:
                        metadata = pd.DataFrame()
                except Exception as e:
                    log.warning(f"[{dataset.code}/Subj {subject}] P300 cache error: {e}. Recomputing.")
                    X, labels, metadata = None, None, None
                else:
                    if X is None or X.size == 0 or labels is None:
                        log.warning(f"[{dataset.code}/Subj {subject}] Invalid P300 cache. Recomputing.")
                        X, labels, metadata = None, None, None

            # Compute if needed
            if X is None:
                try:
                    X_computed, labels_computed, metadata_computed = super().get_data(
                        dataset, [subject], return_epochs=return_epochs, **kwargs
                    )

                    if X_computed is None or X_computed.size == 0 or labels_computed is None:
                        log.warning(f"No valid P300 data for subject {subject}. Skipping.")
                        continue
                    
                    X, labels, metadata = X_computed, labels_computed, metadata_computed

                    # Apply trial slicing
                    if self._num_trials_per_subject is not None and X.shape[0] > self._num_trials_per_subject:
                        log.info(f"Slicing P300 data for Subj {subject} to {self._num_trials_per_subject} trials.")
                        X = X[:self._num_trials_per_subject]
                        labels = labels[:self._num_trials_per_subject]
                        if not metadata.empty and len(metadata) > self._num_trials_per_subject:
                            metadata = metadata.iloc[:self._num_trials_per_subject]

                    # Save to cache
                    try:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        metadata_save_dict = {f"meta_{col}": metadata[col].values for col in metadata.columns}
                        metadata_save_dict["metadata_cols"] = metadata.columns.tolist()
                        np.savez(cache_path, X=X, labels=labels, **metadata_save_dict)
                    except Exception as e_save:
                        log.error(f"P300 cache save failed for subject {subject}: {e_save}")

                except Exception as e_get:
                    log.error(f"P300 data computation failed for subject {subject}: {e_get}. Skipping.")
                    continue

            # Final processing
            if X is not None and X.size > 0 and labels is not None:
                if self._num_trials_per_subject is not None and X.shape[0] > self._num_trials_per_subject:
                    log.info(f"Final P300 slicing for Subj {subject} to {self._num_trials_per_subject} trials.")
                    X = X[:self._num_trials_per_subject]
                    labels = labels[:self._num_trials_per_subject]
                    if not metadata.empty and len(metadata) > self._num_trials_per_subject:
                        metadata = metadata.iloc[:self._num_trials_per_subject]
                
                all_X.append(X)
                all_labels.append(labels)
                all_metadata.append(metadata)

        if not all_X:
            log.warning(f"[{dataset.code}] No P300 data processed.")
            return np.array([]), np.array([]), pd.DataFrame()

        # Final concatenation
        try:
            final_X = np.concatenate(all_X, axis=0) if all_X else np.array([])
            final_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
            final_metadata = pd.concat(all_metadata, ignore_index=True) if all_metadata and not all(df.empty for df in all_metadata) else pd.DataFrame()

            log.info(f"[{dataset.code}] Final P300 data: Epochs {final_X.shape if final_X.size > 0 else 'Empty'}")
            return final_X, final_labels, final_metadata
        except ValueError as ve:
            log.error(f"P300 concatenation failed: {ve}")
            return np.array([]), np.array([]), pd.DataFrame()


class CachingSSVEP(CustomSSVEP):
    """SSVEP paradigm with intelligent caching and data standardization."""

    def __init__(self, fmin: float = 1, fmax: float = 47, resample: Optional[float] = None,
                 channels: Optional[List[str]] = None, events: Optional[Dict[str, int]] = None,
                 num_trials_per_subject: Optional[int] = None, **kwargs):
        """Initialize Caching SSVEP paradigm."""
        self._fmin = fmin
        self._fmax = fmax
        self._resample = resample
        self._channels_init = channels
        self._events_init = events
        self._num_trials_per_subject = num_trials_per_subject

        super_args = dict(fmin=fmin, fmax=fmax, resample=resample, **kwargs)
        if events is not None:
            super_args['events'] = events
        
        super().__init__(**super_args)

    def _get_cache_path(self, dataset: BaseDataset, subject: int, session_name: str = "all_sessions") -> Path:
        """Construct cache file path for SSVEP paradigm."""
        effective_channels = getattr(self, "channels", self._channels_init)
        if isinstance(effective_channels, (list, ListConfig)):
            effective_channels_str = "_".join(sorted(list(effective_channels)))
        elif effective_channels is None:
            effective_channels_str = "all"
        else:
            effective_channels_str = str(effective_channels)
        
        effective_events = getattr(self, "events", self._events_init)
        if effective_events is None:
            effective_events_str = str(sorted(self.used_events(dataset).items()))
        elif isinstance(effective_events, dict):
            effective_events_str = str(sorted(effective_events.items()))
        else:
            effective_events_str = str(sorted(list(effective_events)))

        param_dict = {
            "fmin": self._fmin,
            "fmax": self._fmax,
            "resample": self._resample,
            "channels": effective_channels_str,
            "events": effective_events_str,
            "num_trials": str(self._num_trials_per_subject) if self._num_trials_per_subject is not None else "all",
        }
        
        param_str = "_".join(f"{k}={v}" for k, v in sorted(param_dict.items()))
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

        cache_dir = (
            utils.CACHE_ROOT_DIR 
            / dataset.code
            / f"subject_{subject:03d}"
            / f"session_{session_name}"
        )
        cache_filename = f"params_{param_hash}.npz"
        return cache_dir / cache_filename

    def get_data(self, dataset: BaseDataset, subjects: Optional[List[int]] = None,
                return_epochs: bool = False, **kwargs) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Retrieve SSVEP data with caching and standardization."""
        if subjects is None:
            subjects = dataset.subject_list
        elif not isinstance(subjects, list):
            subjects = [subjects]

        all_X = []
        all_labels = []
        all_metadata = []

        for subject in subjects:
            cache_path = self._get_cache_path(dataset, subject)
            X, labels, metadata = None, None, None

            # Try cache load
            if cache_path.exists():
                try:
                    cache_data = np.load(cache_path, allow_pickle=True)
                    X = cache_data["X"]
                    labels = cache_data["labels"]
                    if 'metadata_cols' in cache_data:
                        metadata_dict = {col: cache_data[f"meta_{col}"] for col in cache_data["metadata_cols"]}
                        metadata = pd.DataFrame(metadata_dict)
                    else:
                        metadata = pd.DataFrame()
                except Exception as e:
                    log.warning(f"[{dataset.code}/Subj {subject}] SSVEP cache error: {e}. Recomputing.")
                    X, labels, metadata = None, None, None
                else:
                    if X is None or X.size == 0 or labels is None:
                        log.warning(f"[{dataset.code}/Subj {subject}] Invalid SSVEP cache. Recomputing.")
                        X, labels, metadata = None, None, None

            # Compute if needed
            if X is None:
                try:
                    X_computed, labels_computed, metadata_computed = super().get_data(
                        dataset, [subject], return_epochs=return_epochs, **kwargs
                    )

                    if X_computed is None or X_computed.size == 0 or labels_computed is None:
                        log.warning(f"No valid SSVEP data for subject {subject}. Skipping.")
                        continue
                    
                    X, labels, metadata = X_computed, labels_computed, metadata_computed

                    # Apply trial slicing
                    if self._num_trials_per_subject is not None and X.shape[0] > self._num_trials_per_subject:
                        log.info(f"Slicing SSVEP data for Subj {subject} to {self._num_trials_per_subject} trials.")
                        X = X[:self._num_trials_per_subject]
                        labels = labels[:self._num_trials_per_subject]
                        if not metadata.empty and len(metadata) > self._num_trials_per_subject:
                            metadata = metadata.iloc[:self._num_trials_per_subject]

                    # Save to cache
                    try:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        metadata_save_dict = {f"meta_{col}": metadata[col].values for col in metadata.columns}
                        metadata_save_dict["metadata_cols"] = metadata.columns.tolist()
                        np.savez(cache_path, X=X, labels=labels, **metadata_save_dict)
                    except Exception as e_save:
                        log.error(f"SSVEP cache save failed for subject {subject}: {e_save}")

                except Exception as e_get:
                    log.error(f"SSVEP data computation failed for subject {subject}: {e_get}. Skipping.")
                    continue

            # Final processing
            if X is not None and X.size > 0 and labels is not None:
                if self._num_trials_per_subject is not None and X.shape[0] > self._num_trials_per_subject:
                    log.info(f"Final SSVEP slicing for Subj {subject} to {self._num_trials_per_subject} trials.")
                    X = X[:self._num_trials_per_subject]
                    labels = labels[:self._num_trials_per_subject]
                    if not metadata.empty and len(metadata) > self._num_trials_per_subject:
                        metadata = metadata.iloc[:self._num_trials_per_subject]
                
                all_X.append(X)
                all_labels.append(labels)
                all_metadata.append(metadata)

        if not all_X:
            log.warning(f"[{dataset.code}] No SSVEP data processed.")
            return np.array([]), np.array([]), pd.DataFrame()

        # Final concatenation
        try:
            final_X = np.concatenate(all_X, axis=0) if all_X else np.array([])
            final_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
            final_metadata = pd.concat(all_metadata, ignore_index=True) if all_metadata and not all(df.empty for df in all_metadata) else pd.DataFrame()

            log.info(f"[{dataset.code}] Final SSVEP data: Epochs {final_X.shape if final_X.size > 0 else 'Empty'}")
            return final_X, final_labels, final_metadata
        except ValueError as ve:
            log.error(f"SSVEP concatenation failed: {ve}")
            return np.array([]), np.array([]), pd.DataFrame()


# %%
# Dataset Imports and Configuration

from moabb.datasets import (
    BI2012,
    BNCI2014_001,
    BNCI2014_002,
    BNCI2014_004,
    BNCI2014_008,
    BNCI2014_009,
    BNCI2015_001,
    BNCI2015_003,
    BNCI2015_004,
    EPFLP300,
    MAMEM1,
    MAMEM2,
    MAMEM3,
    AlexMI,
    BI2013a,
    BI2014a,
    BI2014b,
    BI2015a,
    BI2015b,
    Cattan2019_VR,
    Cho2017,
    DemonsP300,
    GrosseWentrup2009,
    Huebner2017,
    Huebner2018,
    Kalunga2016,
    Lee2019_ERP,
    Lee2019_MI,
    Lee2019_SSVEP,
    Liu2024,
    Nakanishi2015,
    Ofner2017,
    PhysionetMI,
    Schirrmeister2017,
    Shin2017A,
    Shin2017B,
    Sosulski2019,
    Stieger2021,
    Wang2016,
    Weibo2014,
    Zhou2016,
)

# %%
# PARADIGM_DATA Configuration

PARADIGM_DATA = {
    "MI": {
        "datasets": [
            "AlexMI",
            "BNCI2014_001",
            "BNCI2014_002",
            "BNCI2014_004",
            "BNCI2015_001",
            "BNCI2015_004",
            "Cho2017",
            "Lee2019_MI",
            "GrosseWentrup2009",
            "Schirrmeister2017",
            "Ofner2017",
            "PhysionetMI",
            "Shin2017A",
            "Shin2017B",
            "Weibo2014",
            "Zhou2016",
            "Stieger2021",
            "Liu2024",
            "Yang2025",
        ],
        "class_map": {
            "AlexMI": AlexMI,
            "BNCI2014_001": BNCI2014_001,
            "BNCI2014_002": BNCI2014_002,
            "BNCI2014_004": BNCI2014_004,
            "BNCI2015_001": BNCI2015_001,
            "BNCI2015_004": BNCI2015_004,
            "Cho2017": Cho2017,
            "Lee2019_MI": Lee2019_MI,
            "GrosseWentrup2009": GrosseWentrup2009,
            "Schirrmeister2017": Schirrmeister2017,
            "Ofner2017": Ofner2017,
            "PhysionetMI": PhysionetMI,
            "Shin2017A": Shin2017A,
            "Shin2017B": Shin2017B,
            "Weibo2014": Weibo2014,
            "Zhou2016": Zhou2016,
            "Stieger2021": Stieger2021,
            "Liu2024": Liu2024,
            "Yang2025": Yang2025,
        },
        "specs": {
            "PhysionetMI": dict(sr=160, sec=3.0, n_cls=4),
            "BNCI2014_001": dict(sr=250, sec=4, n_cls=4),
            "Stieger2021": dict(sr=1000, sec=3, n_cls=4),
            "Cho2017": dict(sr=512, sec=3, n_cls=2),
            "Zhou2016": dict(sr=250, sec=5, n_cls=3),
            "Lee2019_MI": dict(sr=1000, sec=4, n_cls=2),
            "BNCI2014_004": dict(sr=250, sec=4, n_cls=2),
            "GrosseWentrup2009": dict(sr=250, sec=4, n_cls=2),
            "Schirrmeister2017": dict(sr=250, sec=4, n_cls=4),
            "Ofner2017": dict(sr=512, sec=3, n_cls=2),
            "Shin2017A": dict(sr=1000, sec=4, n_cls=2),
            "Shin2017B": dict(sr=1000, sec=4, n_cls=2),
            "Weibo2014": dict(sr=250, sec=4, n_cls=2),
            "Liu2024": dict(sr=1000, sec=3, n_cls=2),
            "Yang2025": dict(sr=1000, sec=4, n_cls=2),
        },
    },
    "P300": {
        "datasets": [
            "BNCI2014_008",
            "BNCI2014_009",
            "BNCI2015_003",
            "BI2012",
            "BI2013a",
            "BI2014a",
            "BI2014b",
            "BI2015a",
            "BI2015b",
            "Cattan2019_VR",
            "Huebner2017",
            "Huebner2018",
            "Sosulski2019",
            "EPFLP300",
            "Lee2019_ERP",
            "DemonsP300",
        ],
        "class_map": {
            "Huebner2017": Huebner2017,
            "Huebner2018": Huebner2018,
            "Lee2019_ERP": Lee2019_ERP,
            "EPFLP300": EPFLP300,
            "DemonsP300": DemonsP300,
            "Cattan2019_VR": Cattan2019_VR,
            "Sosulski2019": Sosulski2019,
            "BI2012": BI2012,
            "BI2013a": BI2013a,
            "BI2014a": BI2014a,
            "BI2014b": BI2014b,
            "BI2015a": BI2015a,
            "BI2015b": BI2015b,
            "BNCI2014_008": BNCI2014_008,
            "BNCI2014_009": BNCI2014_009,
            "BNCI2015_003": BNCI2015_003,
        },
        "specs": {
            "BI2014a": dict(sr=512, sec=1.0, n_cls=2),
            "BrainInvaders2014a": dict(sr=512, sec=1.0, n_cls=2),
            "Huebner2018": dict(sr=1000, sec=0.9, n_cls=2),
            "BI2015a": dict(sr=512, sec=1.0, n_cls=2),
            "BrainInvaders2015a": dict(sr=512, sec=1.0, n_cls=2),
            "Huebner2017": dict(sr=1000, sec=0.9, n_cls=2),
            "Lee2019_ERP": dict(sr=1000, sec=1.0, n_cls=2),
            "BNCI2014_008": dict(sr=2048, sec=1.0, n_cls=2),
            "BNCI2014_009": dict(sr=256, sec=1.0, n_cls=2),
            "BNCI2015_003": dict(sr=512, sec=1.0, n_cls=2),
            "BI2012": dict(sr=512, sec=1.0, n_cls=2),
            "BI2013a": dict(sr=512, sec=1.0, n_cls=2),
            "BI2014b": dict(sr=512, sec=1.0, n_cls=2),
            "BI2015b": dict(sr=512, sec=1.0, n_cls=2),
            "Cattan2019_VR": dict(sr=512, sec=1.0, n_cls=2),
            "Sosulski2019": dict(sr=128, sec=1.0, n_cls=2),
            "EPFLP300": dict(sr=2048, sec=1.0, n_cls=2),
            "DemonsP300": dict(sr=512, sec=1.0, n_cls=2),
        },
    },
    "SSVEP": {
        "datasets": [
            "Lee2019_SSVEP",
            "Kalunga2016",
            "MAMEM1",
            "MAMEM2",
            "MAMEM3",
            "Nakanishi2015",
            "Wang2016",
        ],
        "class_map": {
            "Lee2019_SSVEP": Lee2019_SSVEP,
            "MAMEM3": MAMEM3,
            "Kalunga2016": Kalunga2016,
            "MAMEM1": MAMEM1,
            "MAMEM2": MAMEM2,
            "Nakanishi2015": Nakanishi2015,
            "Wang2016": Wang2016,
        },
        "specs": {
            "MAMEM1": dict(sr=250, sec=3.0, n_cls=5),
            "MAMEM2": dict(sr=250, sec=3, n_cls=5),
            "MAMEM3": dict(sr=128, sec=3, n_cls=5),
            "Nakanishi2015": dict(sr=256, sec=4.15, n_cls=12),
            "Lee2019_SSVEP": dict(sr=1000, sec=4.0, n_cls=4),
            "Kalunga2016": dict(sr=256, sec=5.0, n_cls=4),
            "Wang2016": dict(sr=250, sec=5.0, n_cls=12),
        },
    },
}


# %%
# Module Exports

__all__ = [
    'EEGDataset',
    'get_subject_list_for_datasets',
    'load_cached_pretrain_data',
    'CustomMotorImagery',
    'CustomP300', 
    'CustomSSVEP',
    'CachingMotorImagery',
    'CachingP300',
    'CachingSSVEP',
    'PARADIGM_DATA',
]