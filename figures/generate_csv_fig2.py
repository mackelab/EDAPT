#%%
#!/usr/bin/env python3
"""
Script to generate CSV data for Figure 2 reproduction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess
import os

# Configuration - Update these paths to match your setup
USERNAME = os.environ.get('USER', 'mwe626')
REPO_ROOT = Path(f"/mnt/lustre/work/macke/{USERNAME}/repos/eegjepa")
BASE_RESULTS_DIR = REPO_ROOT / "EDAPT_neurips/results" / "alignment_studies_NMI_lr_1e-4_finetune_warmup_20"

MODEL_NAMES = ["EEGNetv4", "ATCNet", "ShallowConvNet", "DeepConvNet"]

# Dataset configuration
DATASETS = {
    "Yang2025": {"paradigm": "MI", "trial_cutoff": 600},
    "BI2015a": {"paradigm": "P300", "trial_cutoff": 1044},
    "Lee2019_SSVEP": {"paradigm": "SSVEP", "trial_cutoff": 200},
}

CONFIG_NAME_MAP = {
    "FM-None_A-None_AdaBN-F_SI-F": "TL-ZS",
    "FM-Full_A-None_AdaBN-F_SI-F": "TL+CFT",
}

SMOOTHING_FRAC = 0.2
MIN_POINTS_FOR_SMOOTHING = 10

def parse_alignment_studies_dir_name(dir_name_str):
    """Parse directory name to extract model, dataset, and config."""
    try:
        config_start_index = dir_name_str.find("_FM-")
        if config_start_index == -1:
            return None, None, None
        config_name = dir_name_str[config_start_index+1:]
        prefix = dir_name_str[:config_start_index]
        parts = prefix.split('_')
        model_name = parts[1]
        dataset_name = "_".join(parts[2:])
        return model_name, dataset_name, config_name
    except IndexError:
        return None, None, None

def load_trial_data(base_dir):
    """Load trial data from all experiment directories."""
    all_trial_data_list = []
    if not base_dir.exists():
        print(f"ERROR: Base directory does not exist: {base_dir}")
        return pd.DataFrame()

    valid_align_codes = set(CONFIG_NAME_MAP.keys())
    valid_datasets = set(DATASETS.keys())

    for exp_dir in base_dir.iterdir():
        if not exp_dir.is_dir():
            continue
            
        model, dataset, config = parse_alignment_studies_dir_name(exp_dir.name)
        if not (model and dataset and config and
                model in MODEL_NAMES and
                config in valid_align_codes and
                dataset in valid_datasets):
            continue

        csv_files = list(exp_dir.rglob("results_trial_metrics.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                required_cols = ['trial_index', 'overall_metric_at_trial', 'metric_type', 'subject_id']
                if not all(col in df.columns for col in required_cols):
                    continue

                df = df[df['metric_type'] == 'Accuracy']
                if df.empty:
                    continue

                df['model'] = model
                df['dataset'] = dataset
                df['align_code'] = config
                df['method'] = CONFIG_NAME_MAP[config]
                df['paradigm'] = DATASETS[dataset]['paradigm']
                df['subject_id'] = df['subject_id'].astype(str)
                
                all_trial_data_list.append(df[['model', 'dataset', 'paradigm', 'method', 'align_code', 
                                             'subject_id', 'trial_index', 'overall_metric_at_trial']])
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
    
    return pd.concat(all_trial_data_list, ignore_index=True) if all_trial_data_list else pd.DataFrame()

def convert_cumulative_to_trialwise(df):
    """Convert cumulative accuracy to trial-wise outcomes."""
    if 'overall_metric_at_trial' not in df.columns:
        return df

    grouping_cols = ['model', 'dataset', 'align_code', 'subject_id']
    df_sorted = df.sort_values(by=grouping_cols + ['trial_index']).copy()
    
    df_sorted['total_correct'] = df_sorted['overall_metric_at_trial'] * (df_sorted['trial_index'] + 1)
    trial_outcomes = df_sorted.groupby(grouping_cols)['total_correct'].diff().fillna(df_sorted['total_correct'])
    df_sorted['trial_wise_accuracy'] = trial_outcomes.round().clip(0, 1).astype(int)
    
    return df_sorted.drop(columns=['total_correct'])

def generate_time_series_data(df_trials):
    """Generate smoothed time series data for plotting."""
    time_series_data = []
    
    for dataset in DATASETS.keys():
        cutoff = DATASETS[dataset]['trial_cutoff']
        df_dataset = df_trials[df_trials['dataset'] == dataset]
        
        for method in CONFIG_NAME_MAP.values():
            df_method = df_dataset[df_dataset['method'] == method]
            
            for model in MODEL_NAMES:
                df_model = df_method[df_method['model'] == model].copy()
                if df_model.empty:
                    continue
                
                # Apply trial cutoff
                if cutoff:
                    subjects_with_enough_trials = df_model.groupby('subject_id')['trial_index'].max()
                    valid_subjects = subjects_with_enough_trials[subjects_with_enough_trials >= (cutoff - 1)].index
                    if len(valid_subjects) > 0:
                        df_model = df_model[
                            (df_model['subject_id'].isin(valid_subjects)) &
                            (df_model['trial_index'] < cutoff)
                        ]
                
                if df_model.empty:
                    continue
                
                # Calculate mean accuracy per trial
                summary_stats = df_model.groupby('trial_index')['trial_wise_accuracy'].agg(
                    mean=np.mean
                ).reset_index().sort_values(by='trial_index')
                
                if summary_stats.empty:
                    continue
                
                # Apply smoothing
                summary_stats['mean_smoothed'] = summary_stats['mean']
                if len(summary_stats['mean'].dropna()) >= MIN_POINTS_FOR_SMOOTHING:
                    try:
                        valid_indices = summary_stats['mean'].notna()
                        if valid_indices.any():
                            smoothed = lowess(
                                summary_stats.loc[valid_indices, 'mean'],
                                summary_stats.loc[valid_indices, 'trial_index'],
                                frac=SMOOTHING_FRAC, return_sorted=False, it=0
                            )
                            summary_stats.loc[valid_indices, 'mean_smoothed'] = smoothed
                    except Exception as e:
                        print(f"Smoothing failed for {model}-{dataset}-{method}: {e}")
                
                # Add metadata
                for _, row in summary_stats.iterrows():
                    time_series_data.append({
                        'dataset': dataset,
                        'paradigm': DATASETS[dataset]['paradigm'],
                        'method': method,
                        'model': model,
                        'trial_index': row['trial_index'],
                        'mean_accuracy': row['mean'],
                        'smoothed_accuracy': row['mean_smoothed']
                    })
    
    return pd.DataFrame(time_series_data)

def generate_scatter_data(df_trials):
    """Generate subject-level mean accuracies for scatter plots."""
    subject_means = df_trials.groupby(
        ['model', 'dataset', 'paradigm', 'method', 'subject_id'], as_index=False
    )['trial_wise_accuracy'].mean()
    subject_means.rename(columns={'trial_wise_accuracy': 'subject_mean_accuracy'}, inplace=True)
    
    # Pivot to get TL-ZS and TL+CFT as separate columns
    scatter_data = subject_means.pivot_table(
        index=['model', 'dataset', 'paradigm', 'subject_id'],
        columns='method', 
        values='subject_mean_accuracy'
    ).reset_index()
    
    # Flatten column names
    scatter_data.columns.name = None
    
    return scatter_data

def main():
    """Main execution function."""
    print("Loading trial data...")
    df_trials = load_trial_data(BASE_RESULTS_DIR)
    
    if df_trials.empty:
        print("ERROR: No trial data loaded. Check BASE_RESULTS_DIR and dataset configuration.")
        return
    
    print(f"Loaded {len(df_trials)} trial records.")
    
    # Convert cumulative to trial-wise accuracy
    df_trials = convert_cumulative_to_trialwise(df_trials)
    
    # Generate time series data
    print("Generating time series data...")
    time_series_df = generate_time_series_data(df_trials)
    
    # Generate scatter plot data
    print("Generating scatter plot data...")
    scatter_df = generate_scatter_data(df_trials)
    
    # Save CSV files
    output_dir = Path("figure_data")
    output_dir.mkdir(exist_ok=True)
    
    time_series_path = output_dir / "figure2_time_series_data.csv"
    scatter_path = output_dir / "figure2_scatter_data.csv"
    
    time_series_df.to_csv(time_series_path, index=False)
    scatter_df.to_csv(scatter_path, index=False)
    
    print(f"Saved time series data: {time_series_path}")
    print(f"Saved scatter data: {scatter_path}")
    print(f"Time series data shape: {time_series_df.shape}")
    print(f"Scatter data shape: {scatter_df.shape}")

if __name__ == "__main__":
    main()

