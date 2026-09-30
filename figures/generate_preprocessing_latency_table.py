#!/usr/bin/env python3
"""Preprocessing-latency table (median band-pass filtering + re-referencing time per trial).

Reads figure_data/latency_rerun_2026-01_Yang2025.csv (written by analyze_latency.py)
and writes table_preprocessing_latency.tex.
"""

import argparse
from pathlib import Path

import pandas as pd
from paths import FIGURE_DATA_DIR, FIGURES_DIR

DATASETS = ["Yang2025", "BI2015a", "Lee2019_SSVEP"]
TRIAL_LEN_S = {"Yang2025": 4.0, "BI2015a": 1.0, "Lee2019_SSVEP": 4.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=FIGURE_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    df = pd.read_csv(args.data_dir / "latency_rerun_2026-01_Yang2025.csv")
    cpu = df[df["device"] == "cpu"].groupby("dataset").first()

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Dataset & \\# Channels & Fs (Hz) & Trial len. (s) & CPU (ms) \\\\",
        "\\midrule",
    ]
    for ds in DATASETS:
        r = cpu.loc[ds]
        name = ds.replace("_", "\\_")
        lines.append(
            f"{name} & {int(r['n_channels'])} & {int(r['sr_hz'])} & "
            f"{TRIAL_LEN_S[ds]:.1f} & {r['filter_latency_ms_median']:.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "table_preprocessing_latency.tex"
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
