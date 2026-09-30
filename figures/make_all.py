#!/usr/bin/env python3
"""Regenerate every figure and table of the paper from the shipped CSVs.

    python figures/make_all.py [--out figures/out] [--only fig2,tables,...]

The notebooks are executed headlessly in a scratch copy of the inputs (figure_data/*.csv and *.csv.gz,
prepared_data/*.csv), so nothing in the repository is modified. All PDFs / .tex files end up in ``--out``.

Steps (names usable with --only):
  fig2 fig3 fig4 figs1 figs4-9      main + supplementary figures
  tables                            Tables 1-3 (table1_/table2_/table3_*.tex)
  hyperparam                        hyperparam_heatmap_all.pdf
  tta                               fig_tta_comparison_combined.pdf
  iso                               iso-scaling significance tables
  latency_main                      main latency table (table_latency_main.tex)
  latency                           preprocessing-latency table
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import FIGURE_DATA_DIR, PREPARED_DATA_DIR  # noqa: E402

NOTEBOOKS = {
    "fig2": "fig2.ipynb",
    "fig3": "fig3.ipynb",
    "fig4": "fig4.ipynb",
    "figs1": "figs1.ipynb",
    "figs4-9": "figs4-9.ipynb",
    "tables": "tables.ipynb",
    "hyperparam": "fig_hyperparam_sweep.ipynb",
    "tta": "fig_tta_comparison.ipynb",
    "iso_fractions": "analyze_iso_significance_fractions.ipynb",
}
SCRIPTS = {
    "iso": "generate_iso_significance_tables.py",
    "latency_main": "generate_latency_table.py",
    "latency": "generate_preprocessing_latency_table.py",
}


def run_notebook(name: str, workdir: Path) -> None:
    nb = nbformat.read(HERE / NOTEBOOKS[name], as_version=4)
    client = NotebookClient(
        nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(workdir)}}
    )
    client.execute()
    (workdir / "executed").mkdir(exist_ok=True)
    nbformat.write(nb, workdir / "executed" / NOTEBOOKS[name])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out", type=Path, default=HERE / "out")
    parser.add_argument("--only", default=None, help="comma-separated step names")
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # scratch copy of the inputs: notebooks read ./figure_data, ./prepared_data, ./matplotlibrc
    for src, dst in [(FIGURE_DATA_DIR, "figure_data"), (PREPARED_DATA_DIR, "prepared_data")]:
        shutil.copytree(src, out / dst, dirs_exist_ok=True)
    shutil.copy(HERE / "matplotlibrc", out / "matplotlibrc")

    steps = list(NOTEBOOKS) + list(SCRIPTS)
    if args.only:
        steps = [s.strip() for s in args.only.split(",")]
    for step in steps:
        print(f"== {step}", flush=True)
        if step in NOTEBOOKS:
            run_notebook(step, out)
        elif step in SCRIPTS:
            cmd = [sys.executable, str(HERE / SCRIPTS[step]), "--data-dir", str(out / "figure_data"), "--out-dir", str(out)]
            if step == "iso":
                cmd += ["--prepared-dir", str(out / "prepared_data")]
            subprocess.run(cmd, check=True)
        else:
            sys.exit(f"unknown step {step!r}")
    print(f"done; outputs in {out}")


if __name__ == "__main__":
    main()
