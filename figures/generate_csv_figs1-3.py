#%%
"""
Script to generate CSV data for Figure S1 reproduction.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess
import os

# --- Configuration ---
# Set the username and repository root directory
USERNAME = os.environ.get('USER', 'default_user')
REPO_ROOT = Path(f"/mnt/lustre/work/macke/{USERNAME}/repos/eegjepa")
BASE_RESULTS_DIR = REPO_ROOT / "EDAPT_neurips/results" / "alignment_studies_NMI_lr_1e-4_finetune_warmup_20"

# Model and dataset configurations
MODEL_NAMES = ["EEGNetv4", "ATCNet", "ShallowConvNet", "DeepConvNet"]

PARADIGMS_MASTER_LIST = {
    "MI": {
        "datasets": [
            {"dataset": "Yang2025", "name_for_plot": "MI (Yang2025)"},
            {"dataset": "Lee2019_MI", "name_for_plot": "MI (Lee2019)"},
            {"dataset": "BNCI2014_001", "name_for_plot": "MI (BNCI2014)"},
        ],
    },
    "P300": {
        "datasets": [
            {"dataset": "BI2015a", "name_for_plot": "P300 (BI2015a)"},
            {"dataset": "Huebner2017", "name_for_plot": "P300 (Huebner2017)"},
            {"dataset": "Huebner2018", "name_for_plot": "P300 (Huebner2018)"},
        ],
    },
    "SSVEP": {
        "datasets": [
            {"dataset": "Lee2019_SSVEP", "name_for_plot": "SSVEP (Lee2019)"},
            {"dataset": "Kalunga2016", "name_for_plot": "SSVEP (Kalunga2016)"},
            {"dataset": "MAMEM2", "name_for_plot": "SSVEP (MAMEM2)"},
        ],
    },
}

DATASET_TRIAL_CUTOFFS = {
    "Yang2025": 600,
    "Lee2019_MI": 800,
    "BNCI2014_001": 250,
    "BI2015a": 1044,
    "Huebner2017": 400,
    "Huebner2018": 500,
    "Lee2019_SSVEP": 200,
    "Kalunga2016": 240,
    "MAMEM2": 300,
}

CONFIG_NAME_MAP = {
    "FM-None_A-None_AdaBN-F_SI-F": "TL-ZS",
    "FM-Full_A-None_AdaBN-F_SI-F": "TL+CFT",
    "FM-None_A-Eucl_AdaBN-T_SI-F": "TL+UDA",
    "FM-Full_A-Eucl_AdaBN-T_SI-F": "TL+UDA+CFT",
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
    
    valid_datasets = set()
    for paradigm, group_details in PARADIGMS_MASTER_LIST.items():
        for d in group_details['datasets']:
            valid_datasets.add(d['dataset'])

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
                df['method'] = CONFIG_NAME_MAP.get(config)
                
                # Assign paradigm based on dataset
                paradigm_name = "Unknown"
                for p_key, p_value in PARADIGMS_MASTER_LIST.items():
                    if dataset in [d['dataset'] for d in p_value['datasets']]:
                        paradigm_name = p_key
                        break
                df['paradigm'] = paradigm_name

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
    
    dataset_keys = []
    for paradigm, group_details in PARADIGMS_MASTER_LIST.items():
        for d in group_details['datasets']:
            dataset_keys.append(d['dataset'])

    for dataset in dataset_keys:
        cutoff = DATASET_TRIAL_CUTOFFS.get(dataset)
        df_dataset = df_trials[df_trials['dataset'] == dataset]
        
        for method in CONFIG_NAME_MAP.values():
            df_method = df_dataset[df_dataset['method'] == method]
            
            for model in MODEL_NAMES:
                df_model = df_method[df_method['model'] == model].copy()
                if df_model.empty:
                    continue
                
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
                
                summary_stats = df_model.groupby('trial_index')['trial_wise_accuracy'].agg(
                    mean=np.mean
                ).reset_index().sort_values(by='trial_index')
                
                if summary_stats.empty:
                    continue
                
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
                
                paradigm_name = "Unknown"
                for p_key, p_value in PARADIGMS_MASTER_LIST.items():
                    if dataset in [d['dataset'] for d in p_value['datasets']]:
                        paradigm_name = p_key
                        break

                for _, row in summary_stats.iterrows():
                    time_series_data.append({
                        'dataset': dataset,
                        'paradigm': paradigm_name,
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
    
    scatter_data_list = []
    
    for (model, dataset, paradigm), group in subject_means.groupby(['model', 'dataset', 'paradigm']):
        pivot = group.pivot_table(
            index='subject_id',
            columns='method', 
            values='subject_mean_accuracy'
        ).reset_index()
        
        pivot['model'] = model
        pivot['dataset'] = dataset
        pivot['paradigm'] = paradigm
        scatter_data_list.append(pivot)
        
    scatter_data = pd.concat(scatter_data_list, ignore_index=True)
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
    
    df_trials = convert_cumulative_to_trialwise(df_trials)
    
    print("Generating time series data...")
    time_series_df = generate_time_series_data(df_trials)
    
    print("Generating scatter plot data...")
    scatter_df = generate_scatter_data(df_trials)
    
    output_dir = Path("figure_data")
    output_dir.mkdir(exist_ok=True)
    
    time_series_path = output_dir / "fig1_supplements_time_series_data.csv"
    scatter_path = output_dir / "fig1_supplements_scatter_data.csv"
    
    time_series_df.to_csv(time_series_path, index=False)
    scatter_df.to_csv(scatter_path, index=False)
    
    print(f"Saved time series data: {time_series_path}")
    print(f"Saved scatter data: {scatter_path}")
    print(f"Time series data shape: {time_series_df.shape}")
    print(f"Scatter data shape: {scatter_df.shape}")

if __name__ == "__main__":
    main()
# %%
