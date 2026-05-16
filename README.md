# SAR-Based Soil Moisture Estimation with Prediction Intervals

Experiments for an IEEE paper on estimating soil moisture (SM1 %) from Synthetic Aperture Radar (SAR) backscatter using EOS-04 and Sentinel-1 satellites, with uncertainty quantification via Conformalized Quantile Regression (CQR).

---

## Overview

Soil moisture is a critical variable for agriculture, hydrology, and climate modelling. SAR satellites can retrieve it at field scale regardless of cloud cover, but point estimates alone are insufficient for decision-making — reliable prediction intervals are required.

This repository covers the full pipeline from raw SAR backscatter to calibrated 95% prediction intervals, across 8 experimental phases:

| Phase | What it does |
|-------|-------------|
| 0 | Data preparation — raw Excel → enhanced CSV with engineered features |
| 1 | Classical ML regression (RF, XGBoost, AdaBoost, SVR) |
| 2 | 4-class soil moisture classification (Low / Medium / High / Very High) |
| 3 | ANN point estimation (multiple architectures) |
| 4 | ANN quantile regression — raw prediction intervals |
| 5 | Conformal regression via MAPIE (QuantileReg, GBR, HistGBR) |
| 6 | Conformalized Quantile Regression — 4 methods (SVM Split, GBM CQR, ANN Split, ANN CQR dual) |
| 7 | Tau hyperparameter tuning for ANN CQR with cal-based selection |
| 8 | Quantile SVR gamma hyperparameter grid search |

---

## Satellites & Dataset

| Satellite | Polarizations | Raw samples | After filter |
|-----------|--------------|-------------|-------------|
| EOS-04 (RISAT-1A) | HH, HV | 2530 | **2528** |
| Sentinel-1 | VH, VV | 1818 | **1816** |

**Field site:** Agricultural plots, Gujarat, India (~22.52° N, 72.76° E)
**Ground truth:** IoT LoRa soil moisture sensors (SM1, 0–60%)
**SM1 filter applied:** `0 < SM1 ≤ 60` — removes sensor saturation artifacts (SM1 = 50) and invalid readings
**Period:** June 2022 – July 2023

---

## Feature Engineering

Six input features per sample:

| Feature | Description |
|---------|-------------|
| `HH-pol` / `VH-pol` | Co-polarization backscatter (dB) |
| `HV-pol` / `VV-pol` | Cross-polarization backscatter (dB) |
| `cross_pol_ratio` | HH/HV or VH/VV ratio — dielectric sensitivity proxy |
| `month_sin`, `month_cos` | Cyclical seasonal encoding (captures monsoon pattern) |
| `crop_encoded` | Ordinal-encoded crop type (surface roughness proxy) |

Target: `SM1 (%)` — volumetric soil moisture at 0–5 cm depth

---

## Key Results

### Phase 1 — Classical ML Regression

| Model | EOS-04 R² | EOS-04 RMSE | Sentinel-1 R² | Sentinel-1 RMSE |
|-------|----------|------------|--------------|----------------|
| Random Forest | 0.637 | 9.877 | 0.479 | 10.827 |
| XGBoost | 0.628 | 9.991 | 0.487 | 10.737 |
| AdaBoost | 0.634 | 9.911 | 0.328 | 12.290 |
| SVR | 0.586 | 10.539 | 0.444 | 11.176 |

### Phase 2 — Classification (SVC best model)

| Satellite | Accuracy |
|-----------|----------|
| EOS-04 | **57.9%** |
| Sentinel-1 | **58.0%** |

### Phase 3 — ANN Point Estimation (best architecture)

| Satellite | Architecture | R² | MAE |
|-----------|-------------|-----|-----|
| EOS-04 | 16→Dropout→1 | **0.588** | 8.28 |
| Sentinel-1 | 16→D→8→D→1 | **0.327** | 9.82 |

### Phase 4 — Prediction Intervals (ANN Quantile, τ=0.025/0.975)

| Satellite | Architecture | PICP | MPIW |
|-----------|-------------|------|------|
| EOS-04 | 16, 1 | **0.9763** | **39.57** |
| Sentinel-1 | 4, 1 | **0.9725** | 46.11 |

### Phase 6 — Conformalized Quantile Regression (all 4 methods, α=0.05)

| Method | EOS-04 PICP | EOS-04 MPIW | S1 PICP | S1 MPIW | Crossing |
|--------|------------|------------|---------|---------|---------|
| SVM Split Conformal | 0.9073 | 37.28 | 0.9608 | 42.30 | — |
| **GBM CQR** | **0.9610** | **35.91** | 0.9346 | **33.85** | 0.0% |
| ANN Split Conformal | 0.9317 | 36.73 | **0.9673** | 40.57 | — |
| ANN CQR (dual backbone) | 0.9171 | 41.00 | **0.9739** | 41.30 | 0.0% |

**GBM CQR achieves the lowest MPIW on both satellites. ANN CQR (dual) achieves 0% interval inversions.**

### Phase 7 — Best Tau (cal-based selection, RawCal PICP ≥ 0.95)

| Satellite | Best τ_lo | Best τ_hi | CQR PICP | CQR MPIW |
|-----------|----------|----------|---------|---------|
| EOS-04 | 0.01 | 0.96 | 0.9268 | 37.06 |
| Sentinel-1 | 0.02 | 0.97 | 0.9673 | 40.38 |

---

## Methodological Notes

### CQR Implementation (correct split)

Conformal prediction requires strict separation of train, calibration, and test sets. This repo uses a **70/10/10/10 split**:

```
70% → model training
10% → validation  (early stopping only — never seen by calibration)
10% → calibration (conformal scores only — never seen during training)
10% → test        (final evaluation only)
```

Using the same set for validation and calibration violates the exchangeability assumption that underlies the coverage guarantee and produces optimistic PICP estimates.

### Dual-Output ANN (no interval inversions)

Training two independent ANN models for lower and upper quantiles can produce `lower_bound > upper_bound` inversions. This repo uses a **shared-backbone dual-output model**:

```
Input → Dense(16, relu) → Dropout(0.09) → Dense(8, relu) → Dropout(0.09) → ┬→ Dense(1)  [lo head]
                                                                              └→ Dense(1)  [hi head]
Combined pinball loss: L_total = L_pinball(τ_lo) + L_pinball(τ_hi)
```

Both heads share the same feature representation, eliminating structural inversions (0.0% crossing confirmed in all runs).

### Tau Selection (no test-set snooping)

Best τ is selected by finding the minimum `RawCal_MPIW` where `RawCal_PICP ≥ 0.95` on the **calibration set**. The test set is never consulted during hyperparameter selection.

---

## Repository Structure

```
experiments/classification_new_data/
├── code/
│   ├── constants.py                                        # paths, column names
│   ├── model_experiments.py                                # experiment classes (all phases)
│   ├── run_all_enhanced.py                                 # headless runner — all 8 phases
│   ├── run_phases_467.py                                   # focused runner — phases 4/6/7
│   ├── run_experiments_sequence.sh                         # notebook runner (nbconvert)
│   ├── exploration_eos.ipynb / exploration_sentinel.ipynb  # Phase 0 — EDA
│   ├── classical_ml_*.ipynb                                # Phase 1
│   ├── classification_*.ipynb                              # Phase 2
│   ├── ann_*.ipynb                                         # Phase 3
│   ├── pi_estimation_*.ipynb                               # Phase 4
│   ├── conformal_regression_*.ipynb                        # Phase 5
│   ├── conformalized_quantile_regression_*.ipynb           # Phase 6
│   ├── quantile_regression_tau_tuning_*.ipynb              # Phase 7
│   └── quantile_svr_HP_tuning.ipynb                        # Phase 8
├── data/
│   ├── EOS-04_datasheet.xlsx                               # raw EOS-04 field data
│   ├── sentinel-1.xlsx                                     # raw Sentinel-1 field data
│   ├── eos-04-enhanced.csv                                 # processed + feature-engineered
│   └── sentinel-1-enhanced.csv                             # processed + feature-engineered
└── output/
    ├── ml_experiment_uncensored/                           # Phase 1 metrics
    ├── classification_uncensored/                          # Phase 2 metrics
    ├── ann_experiments_uncensored/                         # Phase 3 metrics + plots
    ├── pi_estimation_uncensored/                           # Phase 4 metrics + plots
    ├── conformal_regression_uncensored/                    # Phase 5 metrics + plots
    └── conformal_results/                                  # Phase 6/7 metrics + plots
```

---

## How to Run

### Option A — Full headless run (fastest, recommended)

```bash
cd experiments/classification_new_data/code
uv run python run_all_enhanced.py
```

Runs all 8 phases, writes JSON metrics and PNG plots to `output/`. Takes ~30–90 min depending on hardware.

### Option B — Phases 4/6/7 only (CQR pipeline)

```bash
cd experiments/classification_new_data/code
uv run python run_phases_467.py
```

### Option C — Notebook sequence (with cell-level outputs)

```bash
cd experiments/classification_new_data/code
bash run_experiments_sequence.sh
```

Requires the `ml-experiments` Jupyter kernel to be registered first:

```bash
uv run python -m ipykernel install --user --name ml-experiments --display-name "ml-experiments"
```

---

## Dependencies

Managed via `uv`. Key packages:

```
tensorflow >= 2.15
scikit-learn >= 1.4
xgboost
mapie
pandas
numpy
matplotlib
seaborn
tqdm
jupyter
ipykernel
```

---

## Reproducibility

All stochastic operations are seeded:

```python
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED']       = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)   # called inside every training function
```

---

## Citation

> Naitik Thakor et al., "SAR-Based Soil Moisture Estimation with Calibrated Prediction Intervals using Conformalized Quantile Regression," *IEEE Geoscience and Remote Sensing Letters* (under review).
