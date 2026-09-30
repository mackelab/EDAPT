"""Central path configuration for the figure/table pipeline.

Everything is resolved relative to the repository and can be overridden with
environment variables, so nothing here depends on a particular cluster layout.

EDAPT_ROOT          repository root (default: parent of this directory)
EDAPT_RESULTS_DIR   raw experiment outputs written by the submit_*.py scripts
                    (default: <root>/results)
EDAPT_ISO_RESULTS_DIR   raw iso-scaling outputs
                    (default: <root>/results_iso_scaling_fix)
"""

import os
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("EDAPT_ROOT", FIGURES_DIR.parent)).expanduser()
RESULTS_DIR = Path(os.environ.get("EDAPT_RESULTS_DIR", REPO_ROOT / "results")).expanduser()
ISO_RESULTS_DIR = Path(
    os.environ.get("EDAPT_ISO_RESULTS_DIR", REPO_ROOT / "results_iso_scaling_fix")
).expanduser()

# Aggregated CSVs (the shipped inputs of every figure and table notebook).
# The two large time-series tables are gzip-compressed (*.csv.gz); pandas reads them directly.
FIGURE_DATA_DIR = Path(
    os.environ.get("EDAPT_FIGURE_DATA_DIR", FIGURES_DIR / "figure_data")
).expanduser()
# Per-subject, per-fold result tables used by the supplementary scaling figures.
PREPARED_DATA_DIR = Path(
    os.environ.get("EDAPT_PREPARED_DATA_DIR", FIGURES_DIR / "prepared_data")
).expanduser()

# Result sub-directories (names as created by the submit_*.py scripts).
ALIGNMENT_RESULTS = RESULTS_DIR / "alignment_studies_lr_1e-4_finetune_warmup_20"
SUBJECT_SCALING_RESULTS = (
    RESULTS_DIR / "scaling_studies_nps_final_5fold_lr_1e-4_finetune_warmup_20"
)
TRIAL_SCALING_RESULTS = RESULTS_DIR / "trial_scaling_studies"
ISO_SCALING_RESULTS = ISO_RESULTS_DIR / "iso_scaling_studies"
HYPERPARAM_RESULTS = RESULTS_DIR / "hyperparam_sweep2"
TTA_COMPARISON_RESULTS = RESULTS_DIR / "tta_comparison_eucl_new"
