# EDAPT: Towards Calibration-Free BCIs with Continual Online Adaptation

![EDAPT Overview](./assets/edapt_overview.png)

This repository contains research code for the *Journal of Neural Engineering* paper:
[***EDAPT: Towards Calibration-Free BCIs with Continual Online Adaptation***](https://iopscience.iop.org/article/10.1088/1741-2552/ae5689)
by Haxel\*, Kapoor\*, [Ziemann](https://ziemannlab.com) and [Macke](https://mackelab.org) (2026) (\* equal contribution; [arXiv preprint](https://arxiv.org/abs/2508.10474)).

EDAPT pretrains an EEG decoder on data from many users and then keeps adapting it to a new user during use, with supervised continual finetuning (CFT) after every trial and optional unsupervised domain adaptation (UDA). No calibration session is needed.

## Installation

We recommend a conda environment. A GPU is recommended but not necessary.

```bash
git clone https://github.com/mackelab/EDAPT.git
cd EDAPT
conda create --name edapt python=3.11
conda activate edapt
pip install -r requirements.txt
```

### Data

We use nine public EEG datasets across three paradigms: motor imagery (Yang2025, Lee2019_MI, BNCI2014_001), P300 (BI2015a, Huebner2017, Huebner2018) and SSVEP (Lee2019_SSVEP, MAMEM2, Kalunga2016). All but Yang2025 are downloaded automatically through [MOABB](https://moabb.neurotechx.com). Download Yang2025 from [its data paper](https://www.nature.com/articles/s41597-025-04826-y) and set `YANG2025_DATA_DIR` to its location.

Paths default to the repository and can be changed with environment variables (`EDAPT_DATA_ROOT`, `YANG2025_DATA_DIR`, `EDAPT_RESULTS_DIR`, `EDAPT_SLURM_PARTITION`, `EDAPT_CONDA_ENV`).

## Running the experiments

The model code is in [`train_transfer.py`](train_transfer.py) (pretraining and online adaptation) and [`tta_wrapper.py`](tta_wrapper.py) (unsupervised adaptation). Experiments are launched on a SLURM cluster with [submitit](https://github.com/facebookincubator/submitit); add `--dry-run` to any launcher to print the jobs without submitting them.

| Experiment | Command |
|------------|---------|
| main results: PRE / UDA / CFT ablation (Fig. 2, Table 3) | `python submit_component_ablation.py --include-no-pretrain` |
| scaling with number of pretraining subjects (Fig. 4) | `python submit_pretrain_subs_ablation.py` |
| scaling with trials per subject (Fig. 4) | `python submit_trial_ablation.py` |
| fixed data budget: subjects vs trials (Fig. 4) | `python submit_isoscaling_analysis.py` |
| hyperparameter sensitivity | `python submit_hyperparam_sweep.py` |
| UDA components, incl. the BN-1 baseline of Wimpff et al. (2024) | `python submit_tta_comparison.py` |
| latency | `python analyze_latency.py` |

### Figures and tables

All figures and tables of the paper can be regenerated from the aggregated results shipped in [`figures/figure_data`](figures/figure_data) and [`figures/prepared_data`](figures/prepared_data), without re-running any experiment (a few minutes on a CPU):

```bash
python figures/make_all.py --out figures/out
```

After re-running experiments, rebuild these files from your own results with the `figures/generate_csv_*.py` scripts.

## Citation

```bibtex
@article{haxel2026edapt,
    title   = {EDAPT: towards calibration-free BCIs with continual online adaptation},
    author  = {Haxel, Lisa and Kapoor, Jaivardhan and Ziemann, Ulf and Macke, Jakob H.},
    journal = {Journal of Neural Engineering},
    year    = {2026},
    volume  = {23},
    number  = {2},
    pages   = {026025},
    doi     = {10.1088/1741-2552/ae5689}
}
```

## Contact

Please open a GitHub issue for questions, or send an email to [lisa.haxel@uni-tuebingen.de](mailto:lisa.haxel@uni-tuebingen.de) or [jaivardhan.kapoor@uni-tuebingen.de](mailto:jaivardhan.kapoor@uni-tuebingen.de).

## License

MIT, see [LICENSE](LICENSE).
