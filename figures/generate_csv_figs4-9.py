#%%
'''
Script to generate CSV data for Figure 4-9 supplements reproduction.
'''

import re
from pathlib import Path
import pandas as pd
import os
from collections import defaultdict

# --- Consolidated Configurations (Copied directly from original script) ---
# --- General ---
USERNAME = os.environ.get('USER', 'default_user')
PLOTS_OUTPUT_DIR = Path("./final_figure_plots_by_paradigm") # Not used here, but kept for consistency

# --- Models and Datasets ---
MODEL_NAMES = ["EEGNetv4", "ATCNet", "ShallowConvNet", "DeepConvNet"]
DATASET_MAX_TRIALS_FIG5 = { "Yang2025": 600, "Kalunga2016": 64, "Lee2019_SSVEP": 200, "BI2015a": 1044, "Lee2019_MI": 200, "BNCI2014_001": 576, "Huebner2017": 12500, "Huebner2018": 14000, "MAMEM2": 100 }

# --- Data Source Paths ---
# IMPORTANT: Update these paths to point to your actual data directories.
RESULTS_BASE_DIR_FIG3 = Path(f"/mnt/lustre/work/macke/{USERNAME}/repos/eegjepa/EDAPT_neurips/results_scaling/scaling_studies_nps_final_5fold_lr_1e-4_finetune_warmup_20")
SCALING_RESULTS_BASE_DIR_FIG5 = Path(f"/mnt/lustre/work/macke/{USERNAME}/repos/eegjepa/EDAPT_neurips/results_scaling/trial_ablation_final")
ALLTRIALS_RESULTS_BASE_DIR_FIG5 = Path(f"/mnt/lustre/work/macke/{USERNAME}/repos/eegjepa/EDAPT_neurips/results/alignment_studies_NMI_lr_1e-4_finetune_warmup_20")
RESULTS_BASE_DIR_FIG6 = Path(f"/mnt/lustre/work/macke/{USERNAME}/repos/eegjepa/EDAPT_neurips/results_scaling/iso_scaling_studies_final_2.0")

# --- Column Names & Configs ---
MODEL_COL = "model"
DATASET_COL = "dataset"
CONFIG_TAG_COL = "config_tag"
NPS_COL_FIG3 = "num_pretrain_subjects"
NT_COL_FIG5 = "num_trials"
NPS_COL_FIG6 = "num_pretrain_subjects"
NT_COL_FIG6 = "num_trials"

# --- Data Loading and Parsing Functions (Copied directly from original script) ---
def _fig3_parse_experiment_name(results_file_path: Path):
    try:
        for part in [results_file_path.parent.parent.parent.name, results_file_path.parent.parent.name]:
            match = re.search(r"ScalingEval_(?P<model>[^_]+)_(?P<dataset>.+?)_NPS(?P<nps>[^_]+)_(?P<config>.*)", part)
            if match:
                details = match.groupdict()
                details["config"] = details["config"].replace("_SI-F", "")
                return {MODEL_COL: details["model"], DATASET_COL: details["dataset"], NPS_COL_FIG3: details["nps"], CONFIG_TAG_COL: details["config"]}
        return None
    except Exception: return None

def _fig5_parse_experiment_name(results_file_path: Path):
    for part in [results_file_path.parent.name, results_file_path.parent.parent.name]:
        scaling_match = re.search(r"ScalingEval_(?P<model>[^_]+)_(?P<dataset>.+?)_NT(?P<nt>\d+)_(?P<config>FM-.*|Eval-.*)", part)
        if scaling_match:
            details = scaling_match.groupdict()
            details["config"] = details["config"].replace("_SI-F", "")
            return {MODEL_COL: details["model"], DATASET_COL: details["dataset"], CONFIG_TAG_COL: details["config"], NT_COL_FIG5: int(details["nt"])}
        
        align_match = re.search(r"AlignEval_(?P<model>[^_]+)_(P<dataset>.+?)_(?P<config>FM-.*|Eval-.*)", part)
        if align_match:
            details = align_match.groupdict()
            details["config"] = details["config"].replace("_SI-F", "")
            if max_trials := DATASET_MAX_TRIALS_FIG5.get(details["dataset"]):
                return {MODEL_COL: details["model"], DATASET_COL: details["dataset"], CONFIG_TAG_COL: details["config"], NT_COL_FIG5: max_trials}
    return None

def _fig6_parse_iso_exp_name(results_file_path: Path):
    exp_name = results_file_path.parent.parent.name
    match = re.search(r"IsoEval_(?P<model>[^_]+)_(?P<dataset>.+?)_NPS(?P<nps>\d+)_NT(?P<nt>\d+)_(?P<config>.*)", exp_name)
    if match:
        details = match.groupdict()
        details["config"] = details["config"].replace("_SI-F", "")
        try:
            return {MODEL_COL: details["model"], DATASET_COL: details["dataset"], NPS_COL_FIG6: int(details["nps"]), NT_COL_FIG6: int(details["nt"]), CONFIG_TAG_COL: details["config"]}
        except (ValueError, TypeError): return None
    return None

def _fig3_load_all_results(results_base_dir: Path) -> pd.DataFrame:
    all_dfs = []
    found_files = list(results_base_dir.rglob("**/results_detailed.csv"))
    print(f"   Found {len(found_files)} potential files for subject ablation.")
    for csv_file in found_files:
        exp_details = _fig3_parse_experiment_name(csv_file)
        if exp_details:
            try:
                df = pd.read_csv(csv_file)
                for col, val in exp_details.items(): df[col] = val
                all_dfs.append(df)
            except Exception as e: print(f"   Warning: Could not load {csv_file}: {e}"); continue
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def _fig3_preprocess_nps_column(df: pd.DataFrame) -> pd.DataFrame:
    if NPS_COL_FIG3 not in df.columns: return df
    df['nps_numeric'] = pd.to_numeric(df[NPS_COL_FIG3], errors='coerce')
    df.dropna(subset=['nps_numeric'], inplace=True)
    df['nps_numeric'] = df['nps_numeric'].astype(int)
    df['nps_categorical'] = df['nps_numeric'].astype(str)
    return df.sort_values(by=[DATASET_COL, 'nps_numeric'])

def _fig5_load_all_results(base_dirs: list[Path]) -> pd.DataFrame:
    all_dfs = []
    for current_base_dir in base_dirs:
        found_files = list(current_base_dir.rglob("**/results_detailed.csv"))
        print(f"   Found {len(found_files)} potential files in {current_base_dir.name}.")
        for csv_file_path in found_files:
            try:
                exp_details = _fig5_parse_experiment_name(csv_file_path)
                if exp_details:
                    temp_df = pd.read_csv(csv_file_path)
                    for col, val in exp_details.items(): temp_df[col] = val
                    all_dfs.append(temp_df)
            except Exception as e: print(f"   Warning: Could not load {csv_file_path}: {e}"); continue
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def _fig5_preprocess_nt_column(df: pd.DataFrame) -> pd.DataFrame:
    if NT_COL_FIG5 not in df.columns: return df
    df['nt_numeric'] = pd.to_numeric(df[NT_COL_FIG5], errors='coerce')
    df.dropna(subset=['nt_numeric'], inplace=True)
    df['nt_numeric'] = df['nt_numeric'].astype(int)
    df['nt_categorical'] = df['nt_numeric'].astype(str)
    return df.sort_values(by=[DATASET_COL, 'nt_numeric'])

def _fig6_load_all_results(base_dir: Path) -> pd.DataFrame:
    all_dfs = []
    found_files = list(base_dir.rglob("**/results_detailed.csv"))
    print(f"   Found {len(found_files)} potential files for iso-ablation.")
    for csv_file in found_files:
        details = _fig6_parse_iso_exp_name(csv_file)
        if details:
            try:
                temp_df = pd.read_csv(csv_file)
                for col, val in details.items(): temp_df[col] = val
                all_dfs.append(temp_df)
            except Exception as e: print(f"   Warning: Could not load {csv_file}: {e}"); continue
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# --- Main Execution Block ---
if __name__ == "__main__":
    OUTPUT_CSV_DIR = Path("./prepared_data")
    OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    print(f"--- Starting Data Preparation. Output will be saved to '{OUTPUT_CSV_DIR}' ---")

    # --- 1. Process Subject Ablation Data ---
    print("\n[1/3] Loading and preparing subject ablation data...")
    df_fig3 = _fig3_load_all_results(RESULTS_BASE_DIR_FIG3)
    if not df_fig3.empty:
        df_fig3 = df_fig3[~df_fig3[CONFIG_TAG_COL].str.contains('A-Eucl', na=False)]
        df_fig3 = _fig3_preprocess_nps_column(df_fig3)
        output_path = OUTPUT_CSV_DIR / "fig3_subject_ablation_complete.csv"
        df_fig3.to_csv(output_path, index=False)
        print(f"    Saved {len(df_fig3)} rows to {output_path}")
    else:
        print("    No subject ablation data found.")

    # --- 2. Process Trial Ablation Data ---
    print("\n[2/3] Loading and preparing trial ablation data...")
    df_fig5 = _fig5_load_all_results([SCALING_RESULTS_BASE_DIR_FIG5, ALLTRIALS_RESULTS_BASE_DIR_FIG5])
    if not df_fig5.empty:
        df_fig5 = df_fig5[~df_fig5[CONFIG_TAG_COL].str.contains('A-Eucl', na=False)]
        df_fig5 = _fig5_preprocess_nt_column(df_fig5)
        output_path = OUTPUT_CSV_DIR / "fig5_trial_ablation_complete.csv"
        df_fig5.to_csv(output_path, index=False)
        print(f"    Saved {len(df_fig5)} rows to {output_path}")
    else:
        print("    No trial ablation data found.")

    # --- 3. Process Iso-Ablation Data ---
    print("\n[3/3] Loading and preparing iso-ablation data...")
    df_fig6 = _fig6_load_all_results(RESULTS_BASE_DIR_FIG6)
    if not df_fig6.empty:
        df_fig6 = df_fig6[~df_fig6[CONFIG_TAG_COL].str.contains('A-Eucl', na=False)]
        output_path = OUTPUT_CSV_DIR / "fig6_iso_ablation_complete.csv"
        df_fig6.to_csv(output_path, index=False)
        print(f"    Saved {len(df_fig6)} rows to {output_path}")
    else:
        print("    No iso-ablation data found.")

    print("\n--- Data preparation script finished successfully. ---")
# %%
