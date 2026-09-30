#!/usr/bin/env python
# %%
"""
Generate LaTeX tables for iso-ablation significance (Figure 4 and Figures S4-9).

Tables show, for each dataset/model/budget iso point:
- Δ = mean paired difference vs the minimum-NPS configuration within the same budget
- BH-corrected significance stars: *, **, ***

Inputs (see paths.py):
- Figure 4: figure_data/fig4_iso_ablation_long.csv (paired long iso CSV)
- Supp: prepared_data/fig6_iso_ablation_complete.csv (raw iso complete CSV)

Outputs (written to --out-dir, default: this directory):
- table_fig4_iso_significance.tex, table_fig4_iso_significance_zs.tex
- table_figs4_9_iso_significance.tex, table_figs4_9_iso_significance_zs.tex

Usage: python figures/generate_iso_significance_tables.py [--data-dir D] [--prepared-dir P] [--out-dir O]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from paths import FIGURE_DATA_DIR, FIGURES_DIR, PREPARED_DATA_DIR

DATA_DIR = FIGURE_DATA_DIR
PREPARED_DIR = PREPARED_DATA_DIR

FT_CONFIG_TAG = "FM-Full_A-None_AdaBN-F"
MODEL_ORDER = ["EEGNetv4", "ATCNet", "ShallowConvNet", "DeepConvNet"]
FIG4_DATASETS = ["Yang2025", "BI2015a", "Lee2019_SSVEP"]


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
    "Lee2019_MI": {
        "High": [(30, 200), (40, 150), (50, 120)],
        "Low": [(15, 200), (30, 100), (50, 60)],
    },
    "BNCI2014_001": {"High": [(6, 500), (8, 375)], "Low": [(3, 500), (5, 300)]},
    "Huebner2017": {
        "High": [(6, 12500), (10, 7500), (12, 6250)],
        "Low": [(3, 12500), (6, 6250), (10, 3750)],
    },
    "Huebner2018": {
        "High": [(6, 14000), (8, 10500), (12, 7000)],
        "Low": [(3, 14000), (6, 7000), (10, 4200)],
    },
    "Kalunga2016": {"High": [(8, 64), (12, 42)], "Low": [(4, 64), (8, 32)]},
    "MAMEM2": {"High": [(6, 100), (8, 75)], "Low": [(3, 100), (6, 50)]},
}


def _budget_for(dataset: str, nps: int, nt: int) -> str:
    points = ISO_TRADE_OFF_POINTS.get(dataset, {})
    for budget, coords in points.items():
        if (nps, nt) in coords:
            return budget
    return "Unknown"


def _stars(p: float) -> str:
    if p < 0.001:
        return "$^{***}$"
    if p < 0.01:
        return "$^{**}$"
    if p < 0.05:
        return "$^{*}$"
    return ""


@dataclass(frozen=True)
class IsoKey:
    dataset: str
    model: str
    budget: str
    nps: int
    nt: int


def _paired_delta_and_p(
    base: pd.Series,
    cmp: pd.Series,
) -> tuple[float, float] | tuple[float, float]:
    """Return (mean_delta, pvalue) for paired t-test (2-sided)."""
    common = base.index.intersection(cmp.index)
    aligned = (
        base.loc[common]
        .rename("base")
        .to_frame()
        .join(cmp.loc[common].rename("cmp"), how="inner")
        .dropna()
    )
    if len(aligned) < 2:
        return (np.nan, np.nan)
    d = aligned["cmp"] - aligned["base"]
    mean_d = float(d.mean())
    p = float(
        stats.ttest_rel(aligned["cmp"], aligned["base"], nan_policy="omit").pvalue
    )
    return (mean_d, p)


def compute_iso_table_cells(
    df_long: pd.DataFrame, datasets: list[str]
) -> dict[IsoKey, tuple[float, float]]:
    """Compute mean delta and BH-corrected p per iso point."""
    # filter only iso tradeoff points for requested datasets
    df = df_long[df_long["dataset"].isin(datasets)].copy()
    df = df[df["budget"].isin(["Low", "High"])].copy()

    out: dict[IsoKey, tuple[float, float]] = {}
    for dataset in datasets:
        for model in MODEL_ORDER:
            for budget in ["Low", "High"]:
                pts = ISO_TRADE_OFF_POINTS.get(dataset, {}).get(budget, [])
                if not pts:
                    continue

                df_g = df[
                    (df["dataset"] == dataset)
                    & (df["model"] == model)
                    & (df["budget"] == budget)
                ].copy()
                if df_g.empty:
                    continue

                # baseline is the minimum NPS point within this budget’s tradeoff list
                base_nps, base_nt = sorted(pts, key=lambda t: t[0])[0]
                base = df_g[
                    (df_g["num_pretrain_subjects"] == base_nps)
                    & (df_g["num_trials"] == base_nt)
                ].set_index(["fold", "subject_id"])["accuracy"]

                # compute raw p-values for BH within this dataset+model+budget
                raw_p = []
                keys = []
                deltas = []
                for nps, nt in pts:
                    cmp = df_g[
                        (df_g["num_pretrain_subjects"] == nps)
                        & (df_g["num_trials"] == nt)
                    ].set_index(["fold", "subject_id"])["accuracy"]
                    mean_d, p = _paired_delta_and_p(base, cmp)
                    k = IsoKey(dataset, model, budget, int(nps), int(nt))
                    deltas.append(mean_d)
                    keys.append(k)
                    raw_p.append(p)

                # BH correction (ignore NaNs)
                pvals = np.asarray(raw_p, dtype=float)
                ok = np.isfinite(pvals)
                p_adj = np.full_like(pvals, np.nan)
                if ok.sum() > 0:
                    p_adj[ok] = multipletests(pvals[ok], method="fdr_bh")[1]

                for k, d, pa in zip(keys, deltas, p_adj):
                    out[k] = (d, float(pa) if np.isfinite(pa) else np.nan)

    return out


def _format_delta(d: float) -> str:
    if not np.isfinite(d):
        return "-"
    return f"{d:+.3f}"


def render_latex_table(
    cells: dict[IsoKey, tuple[float, float]],
    datasets: list[str],
    out_path: Path,
    caption: str,
    label: str,
) -> None:
    """Render a LaTeX table with columns: datasets -> models; rows: budget + iso points."""
    # build column spec
    n_models = len(MODEL_ORDER)
    col_spec = "ll" + ("l" * (len(datasets) * n_models))

    lines = []
    lines.append("\\\\begin{table}[htbp]")
    lines.append("\\\\centering")
    lines.append("\\\\resizebox{\\\\textwidth}{!}{%")
    lines.append(f"\\\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\\\toprule")

    # header row 1: dataset multicolumns
    hdr1 = ["Budget", "Case"]
    for ds in datasets:
        hdr1.append(f"\\\\multicolumn{{{n_models}}}{{c}}{{{ds}}}")
    lines.append(" & ".join(hdr1) + " \\\\")

    # header row 2: model names
    hdr2 = ["", ""]
    for _ds in datasets:
        hdr2.extend(MODEL_ORDER)
    lines.append(" & ".join(hdr2) + " \\\\")
    lines.append("\\\\midrule")

    # rows per budget using "case index" to avoid sparse union rows
    pts_by_ds: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for ds in datasets:
        pts_by_ds[ds] = {
            b: sorted(
                ISO_TRADE_OFF_POINTS.get(ds, {}).get(b, []), key=lambda t: (t[0], t[1])
            )
            for b in ["Low", "High"]
        }

    for budget in ["Low", "High"]:
        max_cases = max((len(pts_by_ds[ds][budget]) for ds in datasets), default=0)
        if max_cases == 0:
            continue

        # derive human-readable case labels from the first dataset that has that index
        case_labels: list[str] = []
        for case_idx in range(max_cases):
            label = f"case{case_idx+1}"
            for ds in datasets:
                pts = pts_by_ds[ds][budget]
                if case_idx < len(pts):
                    nps, nt = pts[case_idx]
                    label = f"NPS={nps}, NT={nt}"
                    break
            case_labels.append(label)

        for case_idx in range(max_cases):
            budget_cell = budget if case_idx == 0 else ""
            case_cell = case_labels[case_idx]
            row = [budget_cell, case_cell]
            for ds in datasets:
                pts = pts_by_ds[ds][budget]
                if case_idx >= len(pts):
                    row.extend(["-"] * len(MODEL_ORDER))
                    continue
                nps, nt = pts[case_idx]
                for model in MODEL_ORDER:
                    k = IsoKey(ds, model, budget, int(nps), int(nt))
                    d, p = cells.get(k, (np.nan, np.nan))
                    row.append(_format_delta(d) + (_stars(p) if np.isfinite(p) else ""))
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\\\midrule")

    lines[-1] = "\\\\bottomrule"
    lines.append("\\\\end{tabular}%")
    lines.append("}")
    lines.append(f"\\\\caption{{{caption}}}")
    lines.append(f"\\\\label{{{label}}}")
    lines.append("\\\\end{table}")

    out_path.write_text("\n".join(lines) + "\n")


def load_fig4_long() -> pd.DataFrame:
    path = DATA_DIR / "fig4_iso_ablation_long.csv"
    df = pd.read_csv(path)
    # expected columns: dataset, model, budget, num_pretrain_subjects, num_trials, fold, subject_id, accuracy
    return df


def load_figs4_9_long(metric: str = "finetuned") -> pd.DataFrame:
    """Load iso-ablation data in long format.
    
    Args:
        metric: "finetuned" or "zero_shot"
    """
    path = PREPARED_DIR / "fig6_iso_ablation_complete.csv"
    df = pd.read_csv(path)
    df = df[df["config_tag"] == FT_CONFIG_TAG].copy()
    # derive budget and normalize to long schema
    df["budget"] = [
        _budget_for(ds, int(nps), int(nt))
        for ds, nps, nt in zip(
            df["dataset"], df["num_pretrain_subjects"], df["num_trials"]
        )
    ]
    acc_col = f"{metric}_balanced_accuracy"
    df_long = df.rename(columns={acc_col: "accuracy"})[
        [
            "dataset",
            "model",
            "budget",
            "num_pretrain_subjects",
            "num_trials",
            "fold",
            "subject_id",
            "accuracy",
        ]
    ].copy()
    return df_long


def main() -> None:
    global DATA_DIR, PREPARED_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--prepared-dir", type=Path, default=PREPARED_DIR)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    DATA_DIR, PREPARED_DIR, out_dir = args.data_dir, args.prepared_dir, args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 4 table (CFT)
    df4 = load_fig4_long()
    cells4 = compute_iso_table_cells(df4, datasets=FIG4_DATASETS)
    render_latex_table(
        cells=cells4,
        datasets=FIG4_DATASETS,
        out_path=out_dir / "table_fig4_iso_significance.tex",
        caption="Iso-ablation (CFT): mean paired Δ vs min-NPS within budget, with BH-corrected paired t-test significance.",
        label="tab:fig4_iso_sig",
    )

    # Supplement table (all datasets, CFT)
    ds_all = list(ISO_TRADE_OFF_POINTS.keys())
    dfS_cft = load_figs4_9_long(metric="finetuned")
    cellsS_cft = compute_iso_table_cells(dfS_cft, datasets=ds_all)
    render_latex_table(
        cells=cellsS_cft,
        datasets=ds_all,
        out_path=out_dir / "table_figs4_9_iso_significance.tex",
        caption="Iso-ablation (all datasets, CFT): mean paired Δ vs min-NPS within budget, with BH-corrected paired t-test significance.",
        label="tab:figs4_9_iso_sig",
    )

    # Zero-shot tables
    dfS_zs = load_figs4_9_long(metric="zero_shot")
    
    # Figure 4 ZS table (3 main datasets)
    cells4_zs = compute_iso_table_cells(
        dfS_zs[dfS_zs["dataset"].isin(FIG4_DATASETS)], datasets=FIG4_DATASETS
    )
    render_latex_table(
        cells=cells4_zs,
        datasets=FIG4_DATASETS,
        out_path=out_dir / "table_fig4_iso_significance_zs.tex",
        caption="Iso-ablation (ZS): mean paired Δ vs min-NPS within budget, with BH-corrected paired t-test significance.",
        label="tab:fig4_iso_sig_zs",
    )

    # Supplement ZS table (all datasets)
    cellsS_zs = compute_iso_table_cells(dfS_zs, datasets=ds_all)
    render_latex_table(
        cells=cellsS_zs,
        datasets=ds_all,
        out_path=out_dir / "table_figs4_9_iso_significance_zs.tex",
        caption="Iso-ablation (all datasets, ZS): mean paired Δ vs min-NPS within budget, with BH-corrected paired t-test significance.",
        label="tab:figs4_9_iso_sig_zs",
    )

    print("Wrote:")
    print(out_dir / "table_fig4_iso_significance.tex")
    print(out_dir / "table_figs4_9_iso_significance.tex")
    print(out_dir / "table_fig4_iso_significance_zs.tex")
    print(out_dir / "table_figs4_9_iso_significance_zs.tex")


if __name__ == "__main__":
    main()

# %%
