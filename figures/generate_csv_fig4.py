#%%
import re
from pathlib import Path
import pandas as pd
import numpy as np
import os

# --- Configuration (Adapted from fig4.py) ---
# NOTE: Update USERNAME and base paths if necessary to match your system.
USERNAME = os.environ.get('USER', 'default_user')

# --- Data Source Paths ---
RESULTS_BASE_DIR_FIG3 = Path(f"/mnt/lustre/work/macke/{USERNAME}/repos/eegjepa/EDAPT_neurips/results_scaling/scaling_studies_nps_final_5fold_lr_1e-4_finetune_warmup_20")
SCALING_RESULTS_BASE_DIR_FIG5 = Path(f"/mnt/lustre/work/macke/{USERNAME}/repos/eegjepa/EDAPT_neurips/results_scaling/trial_ablation_final")
ALLTRIALS_RESULTS_BASE_DIR_FIG5 = Path(f"/mnt/lustre/work/macke/{USERNAME}/repos/eegjepa/EDAPT_neurips/results/alignment_studies_NMI_lr_1e-4_finetune_warmup_20")
RESULTS_BASE_DIR_FIG6 = Path(f"/mnt/lustre/work/macke/{USERNAME}/repos/eegjepa/EDAPT_neurips/results_scaling/iso_scaling_studies_final_2.0")

# --- Output Directory ---
OUTPUT_DATA_DIR = Path("figure_data")

# --- Models and Datasets ---
MODEL_NAMES = ["EEGNetv4", "ATCNet", "ShallowConvNet", "DeepConvNet"]
PREFERRED_DATASET_ORDER = ["Yang2025", "BI2015a", "Lee2019_SSVEP"]

# --- Column Names & Configs ---
MODEL_COL, DATASET_COL, CONFIG_TAG_COL = "model", "dataset", "config_tag"
FT_CONFIG_TAG = "FM-Full_A-None_AdaBN-F"
NPS_COL, NT_COL = "num_pretrain_subjects", "num_trials"
DATASET_MAX_TRIALS_FIG5 = { "Yang2025": 600, "Kalunga2016": 64, "Lee2019_SSVEP": 200, "BI2015a": 1044 }
ISO_TRADE_OFF_POINTS = {
    "BI2015a": { "High": [(15, 1000), (20, 750), (30, 500)], "Low": [(10, 500), (20, 250), (25, 200)] },
    "Yang2025": { "High": [(15, 600), (20, 450), (30, 300)], "Low": [(10, 450), (15, 300), (30, 150)] },
    "Lee2019_SSVEP": { "High": [(20, 200), (25, 160), (40, 100)], "Low": [(10, 200), (20, 100), (40, 50)] }
}

# --- Data Loading & Parsing Functions (from fig4.py) ---
def _fig3_parse_experiment_name(path: Path):
    """
    Parses experiment details for subject ablation.
    This function is corrected to handle non-integer values for 'nps'.
    """
    for part in [path.parent.parent.parent.name, path.parent.parent.name]:
        match = re.search(r"ScalingEval_(?P<model>[^_]+)_(?P<dataset>.+?)_NPS(?P<nps>[^_]+)_(?P<config>.*)", part)
        if match:
            d = match.groupdict()
            # FIX: Check if 'nps' is a digit before converting to int.
            if d["nps"].isdigit():
                d["config"] = d["config"].replace("_SI-F", "")
                return {MODEL_COL: d["model"], DATASET_COL: d["dataset"], NPS_COL: int(d["nps"]), CONFIG_TAG_COL: d["config"]}
            else:
                # If 'nps' is not a digit (e.g., 'max'), skip this file by returning None.
                return None
    return None

def _fig5_parse_experiment_name(path: Path):
    for part in [path.parent.name, path.parent.parent.name]:
        s_match = re.search(r"ScalingEval_(?P<model>[^_]+)_(?P<dataset>.+?)_NT(?P<nt>\d+)_(?P<config>FM-.*|Eval-.*)", part)
        if s_match:
            d = s_match.groupdict()
            d["config"] = d["config"].replace("_SI-F", "")
            return {MODEL_COL: d["model"], DATASET_COL: d["dataset"], CONFIG_TAG_COL: d["config"], NT_COL: int(d["nt"])}
        a_match = re.search(r"AlignEval_(?P<model>[^_]+)_(?P<dataset>.+?)_(?P<config>FM-.*|Eval-.*)", part)
        if a_match:
            d = a_match.groupdict()
            d["config"] = d["config"].replace("_SI-F", "")
            if max_trials := DATASET_MAX_TRIALS_FIG5.get(d["dataset"]):
                return {MODEL_COL: d["model"], DATASET_COL: d["dataset"], CONFIG_TAG_COL: d["config"], NT_COL: max_trials}
    return None

def _fig6_parse_iso_exp_name(path: Path):
    match = re.search(r"IsoEval_(?P<model>[^_]+)_(?P<dataset>.+?)_NPS(?P<nps>\d+)_NT(?P<nt>\d+)_(?P<config>.*)", path.parent.parent.name)
    if match:
        d = match.groupdict()
        d["config"] = d["config"].replace("_SI-F", "")
        return {MODEL_COL: d["model"], DATASET_COL: d["dataset"], NPS_COL: int(d["nps"]), NT_COL: int(d["nt"]), CONFIG_TAG_COL: d["config"]}
    return None

def load_all_results(base_dir, parse_fn):
    all_dfs = []
    for f in base_dir.rglob("**/results_detailed.csv"):
        if details := parse_fn(f):
            try:
                df = pd.read_csv(f)
                for col, val in details.items(): df[col] = val
                all_dfs.append(df)
            except Exception as e: print(f"Warning: Could not load {f}: {e}")
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# --- Data Processing Functions ---
def process_subject_ablation_data(df_raw, output_dir):
    df = df_raw[df_raw[CONFIG_TAG_COL] == FT_CONFIG_TAG].copy()
    if df.empty:
        print("No subject ablation data to process.")
        return
    df_agg = df.groupby([DATASET_COL, MODEL_COL, NPS_COL])['finetuned_accuracy'].agg(['mean', 'sem']).reset_index()
    df_agg.rename(columns={'mean': 'mean_accuracy', 'sem': 'sem_accuracy'}, inplace=True)
    df_agg.to_csv(output_dir / "fig4_subject_ablation_data.csv", index=False)
    print(f"Saved subject ablation data to {output_dir / 'fig4_subject_ablation_data.csv'}")

def process_trial_ablation_data(df_raw, output_dir):
    df = df_raw[df_raw[CONFIG_TAG_COL] == FT_CONFIG_TAG].copy()
    if df.empty:
        print("No trial ablation data to process.")
        return
    df_agg = df.groupby([DATASET_COL, MODEL_COL, NT_COL])['finetuned_accuracy'].agg(['mean', 'sem']).reset_index()
    df_agg.rename(columns={'mean': 'mean_accuracy', 'sem': 'sem_accuracy'}, inplace=True)
    df_agg.to_csv(output_dir / "fig4_trial_ablation_data.csv", index=False)
    print(f"Saved trial ablation data to {output_dir / 'fig4_trial_ablation_data.csv'}")

def process_iso_ablation_data(df_raw, output_dir):
    df = df_raw[df_raw[CONFIG_TAG_COL] == FT_CONFIG_TAG].copy()
    if df.empty:
        print("No iso-ablation data to process.")
        return
    
    def get_budget_type(row):
        points = ISO_TRADE_OFF_POINTS.get(row[DATASET_COL], {})
        for budget, coords in points.items():
            if (row[NPS_COL], row[NT_COL]) in coords: return budget
        return "Unknown"
    
    df['budget'] = df.apply(get_budget_type, axis=1)
    df_agg = df.groupby([DATASET_COL, MODEL_COL, 'budget', NPS_COL, NT_COL])['finetuned_accuracy'].agg(['mean', 'sem']).reset_index()
    df_agg.rename(columns={'mean': 'mean_accuracy', 'sem': 'sem_accuracy'}, inplace=True)
    df_agg.to_csv(output_dir / "fig4_iso_ablation_data.csv", index=False)
    print(f"Saved iso-ablation data to {output_dir / 'fig4_iso_ablation_data.csv'}")

def calculate_efficiency_ratios(df_raw, scaling_col, tolerance=0.01):
    all_points = []
    for (dataset, model), group in df_raw.groupby([DATASET_COL, MODEL_COL]):
        df_agg = group.groupby(scaling_col)[['finetuned_accuracy', 'zero_shot_accuracy']].median().reset_index()
        cft_points = df_agg[[scaling_col, 'finetuned_accuracy']].dropna().to_numpy()
        zs_points_df = df_agg[[scaling_col, 'zero_shot_accuracy']].dropna()
        if len(cft_points) == 0 or zs_points_df.empty: continue

        for n_cft, perf_cft in cft_points:
            if n_cft <= 0: continue
            points_in_window = zs_points_df[(zs_points_df['zero_shot_accuracy'] >= perf_cft - tolerance) & (zs_points_df['zero_shot_accuracy'] <= perf_cft + tolerance)].copy()
            
            point_info = {'dataset': dataset, 'model': model, 'finetuned_accuracy': perf_cft, 'cft_data_points': n_cft}
            
            if not points_in_window.empty:
                points_in_window['diff'] = (points_in_window['zero_shot_accuracy'] - perf_cft).abs()
                best_match = points_in_window.loc[points_in_window['diff'].idxmin()]
                point_info.update({'status': 'matched', 'zs_data_points': best_match[scaling_col], 'efficiency_ratio': n_cft / best_match[scaling_col]})
            elif perf_cft > zs_points_df['zero_shot_accuracy'].max() + tolerance:
                point_info.update({'status': 'unmatched_cft_is_better', 'zs_data_points': np.nan, 'efficiency_ratio': np.inf})
            else: # CFT performance is lower than any ZS performance, not plotted in original
                continue
            all_points.append(point_info)
            
    return pd.DataFrame(all_points)

def process_efficiency_data(df_subjects, df_trials, output_dir):
    if df_subjects.empty or df_trials.empty:
        print("Missing subject or trial data for efficiency calculation.")
        return
        
    df_subjects_ft = df_subjects[df_subjects[CONFIG_TAG_COL] == FT_CONFIG_TAG]
    df_trials_ft = df_trials[df_trials[CONFIG_TAG_COL] == FT_CONFIG_TAG]

    subject_ratios = calculate_efficiency_ratios(df_subjects_ft, NPS_COL)
    trial_ratios = calculate_efficiency_ratios(df_trials_ft, NT_COL)
    
    if not subject_ratios.empty: subject_ratios['scaling_type'] = 'subject'
    if not trial_ratios.empty: trial_ratios['scaling_type'] = 'trial'

    combined_df = pd.concat([subject_ratios, trial_ratios], ignore_index=True)
    combined_df.to_csv(output_dir / "fig4_efficiency_data.csv", index=False)
    print(f"Saved efficiency data to {output_dir / 'fig4_efficiency_data.csv'}")

# --- Main Execution ---
if __name__ == "__main__":
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("--- Creating CSV data files for Figure 4 ---")

    # Load all raw data
    print("\n1. Loading subject ablation data...")
    df_fig3 = load_all_results(RESULTS_BASE_DIR_FIG3, _fig3_parse_experiment_name)
    print(f"   ...loaded {len(df_fig3)} rows.")
    
    print("\n2. Loading trial ablation data...")
    df_fig5_raw = load_all_results(SCALING_RESULTS_BASE_DIR_FIG5, _fig5_parse_experiment_name)
    df_fig5_alltrials = load_all_results(ALLTRIALS_RESULTS_BASE_DIR_FIG5, _fig5_parse_experiment_name)
    df_fig5 = pd.concat([df_fig5_raw, df_fig5_alltrials], ignore_index=True)
    print(f"   ...loaded {len(df_fig5)} rows.")

    print("\n3. Loading iso-ablation data...")
    df_fig6 = load_all_results(RESULTS_BASE_DIR_FIG6, _fig6_parse_iso_exp_name)
    print(f"   ...loaded {len(df_fig6)} rows.")

    # Process and save aggregated data
    print("\n4. Processing and saving aggregated CSVs...")
    process_subject_ablation_data(df_fig3, OUTPUT_DATA_DIR)
    process_trial_ablation_data(df_fig5, OUTPUT_DATA_DIR)
    process_iso_ablation_data(df_fig6, OUTPUT_DATA_DIR)
    process_efficiency_data(df_fig3, df_fig5, OUTPUT_DATA_DIR)

    print("\n--- Script finished successfully. ---")
# %%
