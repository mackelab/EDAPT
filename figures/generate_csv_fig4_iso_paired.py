#!/usr/bin/env python
# %%
"""
Generate iso-ablation CSVs for Figure 4 with pairing preserved.

This script produces:
- figure_data/fig4_iso_ablation_long.csv: per-(dataset, model, fold, subject_id, budget, nps, nt) rows
- figure_data/fig4_iso_ablation_data_paired.csv: aggregated mean/SEM matching legacy fig4 iso CSV

It also validates that the aggregated mean/SEM match the legacy output
figure_data/fig4_iso_ablation_data.csv (from generate_csv_fig4.py) within tolerance.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from paths import FIGURE_DATA_DIR, ISO_SCALING_RESULTS

# --- Metric toggle ---
# Balanced accuracy is the headline metric of the paper. Set the environment
# variable EDAPT_METRIC=accuracy to write the plain-accuracy variant instead
# (files then carry an "_accuracy" suffix).
USE_BALANCED_ACCURACY = os.environ.get("EDAPT_METRIC", "balanced") != "accuracy"
FT_ACC_COL = (
    "finetuned_balanced_accuracy" if USE_BALANCED_ACCURACY else "finetuned_accuracy"
)
METRIC_TAG = "balanced" if USE_BALANCED_ACCURACY else "accuracy"

# --- Configuration (mirrors figures/generate_csv_fig4.py) ---
# Use repo-relative paths to avoid dependence on environment variables under conda run.
RESULTS_BASE_DIR_FIG6 = ISO_SCALING_RESULTS
OUTPUT_DATA_DIR = FIGURE_DATA_DIR

MODEL_COL, DATASET_COL, CONFIG_TAG_COL = "model", "dataset", "config_tag"
FT_CONFIG_TAG = "FM-Full_A-None_AdaBN-F"
NPS_COL, NT_COL = "num_pretrain_subjects", "num_trials"

ISO_TRADE_OFF_POINTS = {
    "BI2015a": {
        "High": [(15, 1000), (20, 750), (30, 500)],
        "Low": [(10, 500), (20, 250), (25, 200)],
    },
    "Yang2025": {
        "High": [(15, 600), (20, 450), (30, 300)],
        "Low": [(10, 450), (15, 300), (30, 150)],
    },
    "Lee2019_SSVEP": {
        "High": [(20, 200), (25, 160), (40, 100)],
        "Low": [(10, 200), (20, 100), (40, 50)],
    },
}


def _filter_most_recent_timestamp(files: Iterable[Path]) -> list[Path]:
    """Keep only most recent timestamp per experiment directory."""
    by_exp: dict[Path, list[tuple[str, Path]]] = defaultdict(list)
    for f in files:
        ts = next((p for p in f.parts if re.match(r"^\d{8}_\d{6}(_\d+)?$", p)), "0")
        by_exp[f.parent.parent].append((ts, f))
    return [sorted(v, reverse=True)[0][1] for v in by_exp.values()]


def _parse_iso_exp_name(results_file: Path) -> dict | None:
    match = re.search(
        r"IsoEval_(?P<model>[^_]+)_(?P<dataset>.+?)_NPS(?P<nps>\d+)_NT(?P<nt>\d+)_(?P<config>.*)",
        results_file.parent.parent.name,
    )
    if not match:
        return None
    d = match.groupdict()
    d["config"] = d["config"].replace("_SI-F", "")
    return {
        MODEL_COL: d["model"],
        DATASET_COL: d["dataset"],
        NPS_COL: int(d["nps"]),
        NT_COL: int(d["nt"]),
        CONFIG_TAG_COL: d["config"],
    }


def _get_budget_type(dataset: str, nps: int, nt: int) -> str:
    points = ISO_TRADE_OFF_POINTS.get(dataset, {})
    for budget, coords in points.items():
        if (nps, nt) in coords:
            return budget
    return "Unknown"


def load_iso_detailed(base_dir: Path) -> pd.DataFrame:
    """Load per-subject detailed iso results, keeping only latest timestamp per experiment."""
    all_dfs: list[pd.DataFrame] = []
    all_files = list(base_dir.rglob("**/results_detailed.csv"))
    print(f"Found {len(all_files)} results_detailed.csv files (pre-filter).")
    files = _filter_most_recent_timestamp(all_files)
    print(f"Using {len(files)} results_detailed.csv files (post-filter).")
    parse_fail = 0
    for f in files:
        meta = _parse_iso_exp_name(f)
        if not meta:
            parse_fail += 1
            continue
        df = pd.read_csv(f)
        for k, v in meta.items():
            df[k] = v
        all_dfs.append(df)
    if parse_fail:
        print(f"Warning: failed to parse {parse_fail} iso experiment names.")
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def build_long_iso_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Create long iso DF with pairing keys preserved."""
    if df_raw.empty:
        return df_raw
    df = df_raw[df_raw[CONFIG_TAG_COL] == FT_CONFIG_TAG].copy()
    # require pairing keys and the balanced accuracy used by legacy generator
    required = {
        "fold",
        "subject_id",
        FT_ACC_COL,
        DATASET_COL,
        MODEL_COL,
        NPS_COL,
        NT_COL,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in iso detailed results: {sorted(missing)}"
        )
    df["budget"] = [
        _get_budget_type(ds, int(nps), int(nt))
        for ds, nps, nt in zip(df[DATASET_COL], df[NPS_COL], df[NT_COL])
    ]
    keep_cols = [
        DATASET_COL,
        MODEL_COL,
        "budget",
        NPS_COL,
        NT_COL,
        "fold",
        "subject_id",
        FT_ACC_COL,
    ]
    return df[keep_cols].rename(columns={FT_ACC_COL: "accuracy"}).copy()


def aggregate_iso_df(df_long: pd.DataFrame) -> pd.DataFrame:
    """Aggregate long iso DF to match legacy fig4 iso output."""
    if df_long.empty:
        return pd.DataFrame()
    df_agg = (
        df_long.groupby([DATASET_COL, MODEL_COL, "budget", NPS_COL, NT_COL])["accuracy"]
        .agg(["mean", "sem"])
        .reset_index()
        .rename(columns={"mean": "mean_accuracy", "sem": "sem_accuracy"})
    )
    return df_agg


def _assert_close_match(
    df_new: pd.DataFrame,
    df_legacy: pd.DataFrame,
    atol: float = 2e-3,
    rtol: float = 2e-3,
) -> None:
    keys = [DATASET_COL, MODEL_COL, "budget", NPS_COL, NT_COL]
    dfn = df_new.copy()
    dfl = df_legacy.copy()
    for c in keys:
        if c not in dfl.columns:
            raise ValueError(f"Legacy iso CSV missing column: {c}")
    dfm = dfl.merge(
        dfn, on=keys, how="outer", suffixes=("_legacy", "_new"), indicator=True
    )
    if (dfm["_merge"] != "both").any():
        bad = dfm[dfm["_merge"] != "both"][keys + ["_merge"]]
        raise AssertionError(
            f"Row key mismatch between legacy and new iso CSV:\\n{bad.to_string(index=False)}"
        )
    for col in ["mean_accuracy", "sem_accuracy"]:
        a = dfm[f"{col}_legacy"].to_numpy(dtype=float)
        b = dfm[f"{col}_new"].to_numpy(dtype=float)
        if not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
            dif = np.nanmax(np.abs(a - b))
            raise AssertionError(
                f"{col} mismatch: max abs diff={dif} (atol={atol}, rtol={rtol})"
            )


def main() -> None:
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("--- Creating paired iso-ablation CSVs for Figure 4 ---")
    print(
        f"Iso results dir: {RESULTS_BASE_DIR_FIG6} (exists={RESULTS_BASE_DIR_FIG6.exists()})"
    )
    df_raw = load_iso_detailed(RESULTS_BASE_DIR_FIG6)
    print(f"Loaded {len(df_raw)} detailed iso rows.")
    if df_raw.empty:
        raise RuntimeError(
            "No iso detailed rows loaded. Check that results exist under "
            f"{RESULTS_BASE_DIR_FIG6} and match the expected naming convention."
        )
    df_long = build_long_iso_df(df_raw)
    print(f"Built long iso DF with {len(df_long)} rows.")
    df_agg = aggregate_iso_df(df_long)
    print(f"Aggregated to {len(df_agg)} rows.")

    # The headline (balanced-accuracy) files carry no suffix; the plain-accuracy
    # variant is written with an "_accuracy" suffix.
    suffix = "" if USE_BALANCED_ACCURACY else f"_{METRIC_TAG}"
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    long_path = OUTPUT_DATA_DIR / f"fig4_iso_ablation_long{suffix}.csv"
    agg_path = OUTPUT_DATA_DIR / f"fig4_iso_ablation_data_paired{suffix}.csv"
    df_long.to_csv(long_path, index=False)
    df_agg.to_csv(agg_path, index=False)
    print(f"Wrote {long_path}")
    print(f"Wrote {agg_path}")

    # Validate against legacy only when a matching legacy file exists.
    legacy_candidates = [OUTPUT_DATA_DIR / f"fig4_iso_ablation_data{suffix}.csv"]
    legacy_path = next((p for p in legacy_candidates if p.exists()), None)
    if legacy_path:
        df_legacy = pd.read_csv(legacy_path)
        if df_agg.empty:
            raise RuntimeError(
                "Aggregated iso DF is empty; cannot validate against legacy output."
            )
        _assert_close_match(df_new=df_agg, df_legacy=df_legacy)
        print(f"OK: aggregated mean/SEM match legacy iso CSV at {legacy_path}.")
    else:
        print("Note: no legacy iso CSV found for this metric tag; skipped validation.")


if __name__ == "__main__":
    main()
