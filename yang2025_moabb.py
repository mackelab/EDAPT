"""WBCIC_SHU Motor Imagery dataset (Yang2025) for local data loading."""

import logging
from pathlib import Path

import mne
import numpy as np
from mne.io import read_raw_bdf
from moabb.datasets.base import BaseDataset

log = logging.getLogger(__name__)

# Hardcoded path to raw dataset directory containing sub-001, sub-002, etc.
HARDCODED_RAW_DATA_PATH = Path(
    "/mnt/lustre/work/macke/mwe626/repos/eegjepa/data_prime/Yang2025_MI/WBCIC_SHU Motor Imagery dataset/sourcedata/2C dataset"
)


class Yang2025(BaseDataset):
    """WBCIC_SHU Motor Imagery dataset from Yang et al. 2025.

    Dataset from [1]_. Reads data directly from predefined local path.

    **Events:**
    * `'left_hand'`: Corresponds to annotation '1' in original `evt.bdf` file
    * `'right_hand'`: Corresponds to annotation '2' in original `evt.bdf` file

    **File Structure:**
    Expected BDF format under HARDCODED_RAW_DATA_PATH:
    `sub-XXX/ses-YY/eeg/` containing `data.bdf` and `evt.bdf`

    References
    ----------
    .. [1] Yang, Y., Darki, F., Shen, C. et al. WBCIC SHU, a large-scale motor
           imagery electroencephalography dataset for brain-computer interface.
           Sci Data 12, 275 (2025). https://doi.org/10.1038/s41597-025-04826-y
    """

    def __init__(self):
        """Initialize Yang2025 dataset with local path validation."""
        if not HARDCODED_RAW_DATA_PATH.is_dir():
            log.error(f"Dataset path not found: {HARDCODED_RAW_DATA_PATH}")

        super().__init__(
            subjects=list(range(1, 52)),  # sub-001 to sub-051
            sessions_per_subject=3,  # ses-01, ses-02, ses-03
            events={"left_hand": 1, "right_hand": 2},
            code="Yang2025",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1038/s41597-025-04826-y",
        )
        
        self.sessions_per_subject = 3
        self.drop_chs = ["ECG", "HEOR", "HEOL", "VEOU", "VEOL"]
        self.local_data_path = HARDCODED_RAW_DATA_PATH

    def _get_single_subject_data(self, subject, debug_print=False):
        """Return data for single subject from local path."""
        if debug_print:
            print(f"--- Processing Subject {subject} ---")
            
        subject_dir_str = self.data_path(subject)[0]
        subject_dir = Path(subject_dir_str)
        
        if debug_print:
            print(f"  Subject Dir: {subject_dir}")
            
        sessions = {}

        for session_idx in range(1, self.sessions_per_subject + 1):
            session_str = f"{session_idx-1}"  # MOABB session keys '0', '1', '2'
            session_id = f"ses-{session_idx:02d}"  # Folder names
            session_path = subject_dir / session_id / "eeg"
            
            if debug_print:
                print(f"  Checking Session: {session_id} at {session_path}")

            data_bdf_path = session_path / "data.bdf"
            evt_bdf_path = session_path / "evt.bdf"
            
            if debug_print:
                print(f"    Data: {data_bdf_path}")
                print(f"    Events: {evt_bdf_path}")

            # Check file existence
            if not data_bdf_path.is_file():
                log.warning(f"Data file missing: {data_bdf_path}")
                if debug_print:
                    print("    Data file NOT FOUND")
                continue
                
            if not evt_bdf_path.is_file():
                log.warning(f"Event file missing: {evt_bdf_path}")
                if debug_print:
                    print("    Event file NOT FOUND")
                continue

            if debug_print:
                print("    Files FOUND, loading...")

            try:
                # Load raw data and annotations
                raw = read_raw_bdf(data_bdf_path, preload=True, verbose="WARNING")
                annots = mne.read_annotations(evt_bdf_path)
                raw.set_annotations(annots)
                
                if debug_print:
                    print(f"    Raw loaded: {raw.info['nchan']} channels")
                    if raw.annotations:
                        print(f"    Annotations set: {len(raw.annotations)} events")

                # Rename annotation descriptions
                description_map = {"1": "left_hand", "2": "right_hand"}
                original_descriptions = np.array(raw.annotations.description)
                new_descriptions = np.empty_like(original_descriptions, dtype="<U15")
                
                if debug_print:
                    print(f"    Renaming descriptions: {description_map}")

                ignored_descs = set()
                for i, desc in enumerate(original_descriptions):
                    if desc in description_map:
                        new_descriptions[i] = description_map[desc]
                    else:
                        new_descriptions[i] = desc
                        ignored_descs.add(desc)

                if debug_print and ignored_descs:
                    print(f"    Kept original for: {ignored_descs}")

                raw.annotations.description = new_descriptions
                
                if debug_print:
                    print(f"    Renamed descriptions: {raw.annotations.description[:10]}")

                # Drop non-EEG channels
                ch_to_drop_present = [ch for ch in self.drop_chs if ch in raw.ch_names]
                if ch_to_drop_present:
                    if debug_print:
                        print(f"    Dropping channels: {ch_to_drop_present}")
                    raw.drop_channels(ch_to_drop_present)

                # Add to sessions
                sessions.setdefault(session_str, {})["0"] = raw
                
                if debug_print:
                    print(f"    Successfully loaded session {session_str}")

            except Exception as e:
                log.error(f"Error loading subject {subject}, session {session_id}: {e}")
                if debug_print:
                    print(f"    ERROR: {e}")

        if not sessions:
            log.error(f"No sessions loaded for subject {subject}")
            if debug_print:
                print(f"  No sessions loaded")
            return {}

        if debug_print:
            if "0" in sessions and "0" in sessions["0"]:
                final_raw = sessions["0"]["0"]
                print(f"  Final annotations: {final_raw.annotations.description[:10]}...")
            print(f"  Returning sessions: {list(sessions.keys())}")
            
        return sessions

    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        """Return path to subject's data directory."""
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number: {subject}")

        if not self.local_data_path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.local_data_path}")

        # Construct subject path
        subject_id = f"sub-{subject:03d}"
        subject_path = self.local_data_path / subject_id

        if not subject_path.is_dir():
            log.warning(f"Subject directory not found: {subject_path}")
            raise FileNotFoundError(f"Subject directory not found: {subject_path}")

        return [str(subject_path)]