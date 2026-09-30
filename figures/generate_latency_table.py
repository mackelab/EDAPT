#!/usr/bin/env python3
"""Main latency table (prediction / UDA update / finetune epoch, CPU and GPU, 9 datasets x 4 models).

Reads figure_data/latency_published.csv (values transcribed from the published table)
and writes table_latency_main.tex.
"""

import argparse
from pathlib import Path

import pandas as pd
from paths import FIGURE_DATA_DIR, FIGURES_DIR

CAPTION = (
    "\\textbf{Latency benchmarks for EDAPT's online components.} Median computational time in "
    "milliseconds (ms) for the three online stages of the EDAPT framework: prediction (a single "
    "forward pass), the unsupervised domain adaptation (UDA) update, and one continual finetuning "
    "(CFT) epoch. Latencies were measured for all nine datasets and four network architectures, "
    "benchmarked on an 8-core CPU and a consumer-grade GPU (NVIDIA GeForce RTX 2080Ti, 11 GB)."
)
HEADER = r"""\begin{table}[htbp]
\centering
\caption{%s}
\resizebox{\textwidth}{!}{
\begin{tabular}{lrr l rrrrrr}
\toprule
\multirow{2}{*}{\begin{tabular}[c]{@{}c@{}}Dataset \\ (Paradigm)\end{tabular}} & \multirow{2}{*}{\begin{tabular}[c]{@{}c@{}}Trial \\ length (s)\end{tabular}} & \multirow{2}{*}{\begin{tabular}[c]{@{}c@{}}Sampling \\ rate (Hz)\end{tabular}} & \multirow{2}{*}{Model} & \multicolumn{2}{c}{Prediction (ms)} & \multicolumn{2}{c}{UDA update (ms)} & \multicolumn{2}{c}{\begin{tabular}[c]{@{}c@{}}Finetune \\ epoch (ms)\end{tabular}} \\
\cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}
 &  &  &  & CPU & GPU & CPU & GPU & CPU & GPU \\
\midrule"""
VALUE_COLS = ["pred_ms_cpu", "pred_ms_gpu", "uda_ms_cpu", "uda_ms_gpu", "cft_ms_cpu", "cft_ms_gpu"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=FIGURE_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    df = pd.read_csv(args.data_dir / "latency_published.csv")
    lines = [HEADER % CAPTION]
    datasets = list(dict.fromkeys(df["dataset"]))  # keep the order of the CSV
    for k, ds in enumerate(datasets):
        block = df[df["dataset"] == ds]
        first = block.iloc[0]
        name = ds.replace("_", "\\_")
        span = (
            f"\\multirow{{4}}{{*}}{{\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{name} \\\\ ({first['paradigm']})\\end{{tabular}}}} & "
            f"\\multirow{{4}}{{*}}{{{first['trial_length_s']:g}}} & \\multirow{{4}}{{*}}{{{int(first['sr_hz'])}}}"
        )
        for j, (_, r) in enumerate(block.iterrows()):
            vals = " & ".join(f"{r[c]:.2f}" for c in VALUE_COLS)
            lead = span if j == 0 else " & &"
            lines.append(f"{lead} & {r['model']} & {vals} \\\\")
        lines.append("\\midrule" if k < len(datasets) - 1 else "\\bottomrule")
    lines += ["\\end{tabular}%", "}", "\\label{tab:latency_summary_cpu_gpu}", "\\end{table}"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "table_latency_main.tex"
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
