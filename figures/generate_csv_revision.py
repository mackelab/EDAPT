#!/usr/bin/env python3
"""Aggregate raw run outputs into the CSVs behind the revision tables/figures.

Writes to ``figure_data/`` (see ``paths.py``):

* ``table_subject_metrics.csv``  per-subject balanced accuracy / AUROC for every
  (dataset, model, configuration): input of ``tables.ipynb`` (Tables 1-3).
* ``hyperparam_sweep_summary.csv``  mean zero-shot / finetuned balanced accuracy
  per (paradigm, warmup, window, epochs): input of ``fig_hyperparam_sweep.ipynb``.
* ``tta_comparison_subject_accuracy.csv``  per-subject balanced accuracy of the nine
  UDA/CFT component configurations: input of ``fig_tta_comparison.ipynb``.

Usage (from anywhere)::

    python figures/generate_csv_revision.py [--only tables|hyperparam|tta]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from paths import (
    ALIGNMENT_RESULTS,
    FIGURE_DATA_DIR,
    HYPERPARAM_RESULTS,
    TTA_COMPARISON_RESULTS,
)
from sklearn.metrics import balanced_accuracy_score

# Configurations used in Tables 1-3 (directory-name tag of the alignment study).
TABLE_CONFIGS = [
    "FM-None_A-None_AdaBN-F",
    "FM-None_A-Eucl_AdaBN-T",
    "FM-Full_A-None_AdaBN-F",
    "FM-Full_A-Eucl_AdaBN-T",
    "FM-Dec_A-Eucl_AdaBN-T",
    "FM-Dec_A-Eucl_AdaBN-F",
    "FM-Full_A-Eucl_AdaBN-F",
    "FM-Full_A-Riem_AdaBN-T",
    "FM-Dec_A-None_AdaBN-F",
    "FM-Dec_A-Riem_AdaBN-T",
    "FM-Full_A-Riem_AdaBN-F",
    "NoPre_FM-Full_A-None_AdaBN-F",
    "NoPre_FM-Full_A-Eucl_AdaBN-T",
]

# One run of the alignment study finished after Tables 1 and 3 of the manuscript were
# produced, so those tables print "-" for it. It is left out by default to reproduce
# the printed tables; pass --include-late-runs to use every finished run.
LATE_RUNS = {("Huebner2018", "ShallowConvNet", "NoPre_FM-Full_A-None_AdaBN-F")}

HYPERPARAM_DATASETS = {"MI": "Yang2025", "P300": "BI2015a", "SSVEP": "Lee2019_SSVEP"}

TTA_METHOD_DIRS = [
    "01_baseline",
    "02_adabn_noalign",
    "03_adabn_align",
    "04_cft_noalign",
    "05_cft_align",
    "06_full_new",
    "07_adabn_cft_noalign",
    "08_adabn_old_align",
    "09_full_old",
]


def parse_experiment_dir_name(dir_name: str):
    """Return (model, dataset, config) from ``AlignEval_{model}_{dataset}_{config}``."""
    nopre = re.search(r"_NoPre_(FM-.+?)(?:_no_pretrain)?$", dir_name)
    if nopre:
        prefix = dir_name[: nopre.start()].split("_")
        if len(prefix) < 3:
            return None, None, None
        return prefix[1], "_".join(prefix[2:]), "NoPre_" + nopre.group(1)

    idx = -1
    for marker in ("_FM-", "_Eval-"):
        idx = dir_name.rfind(marker)
        if idx != -1:
            break
    if idx == -1:
        return None, None, None
    prefix = dir_name[:idx].split("_")
    if len(prefix) < 3:
        return None, None, None
    return prefix[1], "_".join(prefix[2:]), dir_name[idx + 1 :]


def _first_column(df, candidates):
    return next((c for c in candidates if c in df.columns), None)


def table_subject_metrics(include_late_runs: bool = False) -> pd.DataFrame:
    """Per-subject balanced accuracy and AUROC from the alignment study."""
    rows = []
    for exp_dir in sorted(ALIGNMENT_RESULTS.iterdir()):
        if not exp_dir.is_dir():
            continue
        model, dataset, config = parse_experiment_dir_name(exp_dir.name)
        if not (model and dataset and config) or config not in TABLE_CONFIGS:
            continue
        if not include_late_runs and (dataset, model, config) in LATE_RUNS:
            continue
        csv_files = sorted(exp_dir.rglob("results_detailed.csv"))
        if not csv_files:
            continue
        df = pd.read_csv(csv_files[-1])  # most recent timestamp
        acc_col = _first_column(
            df,
            [
                "finetuned_balanced_accuracy",
                "finetuned_accuracy",
                "zero_shot_balanced_accuracy",
            ],
        )
        if acc_col is None or "subject_id" not in df.columns:
            continue
        auc_col = _first_column(df, ["finetuned_auroc", "zero_shot_auroc"])
        for _, r in df.iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "config": config,
                    "subject_id": str(r["subject_id"]),
                    "balanced_accuracy": r[acc_col],
                    "auroc": r[auc_col] if auc_col else np.nan,
                }
            )
    return pd.DataFrame(rows)


def hyperparam_summary() -> pd.DataFrame:
    rows = []
    for exp_dir in sorted(HYPERPARAM_RESULTS.iterdir()):
        if not exp_dir.is_dir():
            continue
        csv_files = sorted(exp_dir.glob("**/results_detailed.csv"))
        if not csv_files:
            continue
        paradigm = next(
            (p for p, d in HYPERPARAM_DATASETS.items() if d in exp_dir.name), None
        )
        m = re.search(r"CFT_w(\d+)_ws(\d+)_e(\d+)", exp_dir.name)
        if paradigm is None or m is None:
            continue
        df = pd.read_csv(csv_files[-1])
        zs = df["zero_shot_balanced_accuracy"].mean()
        ft = df["finetuned_balanced_accuracy"].mean()
        rows.append(
            {
                "paradigm": paradigm,
                "dataset": HYPERPARAM_DATASETS[paradigm],
                "warmup_trials": int(m.group(1)),
                "window_size": int(m.group(2)),
                "finetune_epochs": int(m.group(3)),
                "zs_accuracy": zs,
                "cft_accuracy": ft,
                "delta_accuracy": ft - zs,
            }
        )
    return pd.DataFrame(rows)


def tta_subject_accuracy() -> pd.DataFrame:
    """Balanced accuracy per (method, dataset, model, fold, subject)."""
    rows = []
    for method in TTA_METHOD_DIRS:
        method_dir = TTA_COMPARISON_RESULTS / method
        files = sorted(method_dir.rglob("results_trial_metrics.csv"))
        if not files:
            print(f"No results for {method}")
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        for (dataset, model, fold, subj), g in df.groupby(
            ["dataset", "model", "fold", "subject_id"]
        ):
            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "model": model,
                    "fold": fold,
                    "subject_id": subj,
                    "balanced_acc": balanced_accuracy_score(
                        g["true_label"].astype(int), g["pred_label"].astype(int)
                    ),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", choices=["tables", "hyperparam", "tta"])
    parser.add_argument(
        "--include-late-runs",
        action="store_true",
        help="also use runs that finished after the manuscript tables were made",
    )
    args = parser.parse_args()

    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    jobs = {
        "tables": (lambda: table_subject_metrics(args.include_late_runs), "table_subject_metrics.csv"),
        "hyperparam": (hyperparam_summary, "hyperparam_sweep_summary.csv"),
        "tta": (tta_subject_accuracy, "tta_comparison_subject_accuracy.csv"),
    }
    for name, (fn, out_name) in jobs.items():
        if args.only and args.only != name:
            continue
        df = fn()
        out = Path(FIGURE_DATA_DIR) / out_name
        df.to_csv(out, index=False, float_format="%.10g")
        print(f"Saved {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
