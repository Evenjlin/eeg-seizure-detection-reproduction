# EEG Seizure Detection — Reproduction of Kashefi Amiri et al. (2025)

Reproduction and validation study of:

> Kashefi Amiri, H., Zarei, M., & Daliri, M. R. (2025). Epileptic seizure
> detection from electroencephalogram signals based on 1D CNN-LSTM deep
> learning model using discrete wavelet transform. *Scientific Reports*,
> 15, 32820. https://doi.org/10.1038/s41598-025-18479-9

## Goal

Faithfully reproduce the paper's DWT + 1D CNN-LSTM pipeline on the Bonn,
CHB-MIT, and TUSZ datasets, validate whether the reported accuracy holds
under a patient-independent train/test split, and use the findings to
identify a genuine research gap.


## Status

- [x] Bonn: loader, z-score, DWT (db1, level 3), PyTorch CNN-LSTM verified against Table 2; ABCD vs E: 99.2% accuracy (10-fold, 300 epochs)
- [x] CHB-MIT: all 24 patients extracted (11,910 windows, 23 subjects) with patient/recording metadata
- [x] Finding: the published CHB-MIT features are 18 raw samples plus 54 zeros; the architecture cannot run on that input without same-padding
- [x] CHB-MIT accuracy (paper 96.94%) not reproduced: 74-85% under several variants vs 73.9% majority baseline
- [x] Evaluation-protocol ladder, CNN-LSTM full-window features, 3 seeds (AUC): random 0.86, recording-wise 0.81, subject-wise 0.71
- [x] Random Forest proxy ladder and per-patient leave-one-subject-out analysis; exploratory per-patient calibration (+0.11 pooled AUC)
- [ ] Remaining Bonn cases (multi-class), patient-level uncertainty, write-up
- [ ] Request exact CHB-MIT code from the authors

## Project structure

eeg_base_paper/
├── data/ # not tracked in git — see .gitignore
├── src/ # all pipeline code
├── notebooks/ # exploration notebooks
├── configs/ # training configs
├── outputs/ # checkpoints, figures, metrics, logs — not tracked
└── requirements.txt

## Setup

```bash
conda create -n eeg_env python=3.10 -y
conda activate eeg_env
pip install -r requirements.txt
```

## Known deviations from the original paper's code

- The authors' own binary-classification code uses `L2(0.0003)` on the
  64-unit dense layer, while their paper's Table 2 and their own
  multi-class code both state `L2(0.03)`. We use 0.03, matching the
  published table.
- The authors' CV strategy (`sklearn.model_selection.KFold` applied to
  pooled segments) does not group by patient for CHB-MIT/TUSZ, creating
  a segment-leakage risk. We reproduce this faithfully as "Experiment A"
  and additionally run a patient-independent "Experiment B" for
  CHB-MIT/TUSZ (not possible for Bonn — no public patient ID per segment).