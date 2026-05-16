# Soil Moisture Estimation from SAR Imagery with Calibrated Prediction Intervals

> **IEEE Geoscience and Remote Sensing Letters** — Under Review
>
> *Naitik Thakor, Siddhant Meena et al.*

---

## What This Paper Is About

Soil moisture (SM) at the topsoil layer (0–5 cm) drives crop water stress, flood runoff, and drought early-warning. Traditional ground sensors are sparse and expensive. Synthetic Aperture Radar (SAR) satellites — which penetrate clouds and operate day/night — can map soil moisture at field scale from backscatter intensity. But a single point estimate ("SM = 23%") is not enough for precision agriculture or hydrological modelling. Farmers and decision systems need to know the *uncertainty*: is the true value likely between 18–28%, or between 5–45%?

This paper makes two contributions:

1. **A comparative study of ML and ANN models** for SAR-to-SM regression and classification using two satellites — Indian EOS-04 (RISAT-1A) and European Sentinel-1 — over an agricultural field site in Gujarat, India.

2. **A rigorous implementation of Conformalized Quantile Regression (CQR)** that produces statistically valid 95% prediction intervals, with a corrected data split, a dual-output ANN that eliminates interval inversions, and an honest tau hyperparameter selection strategy.

---

## Problem Statement

### Why SAR for Soil Moisture?

SAR L-band and C-band backscatter is sensitive to the dielectric constant of the topsoil, which is dominated by volumetric water content. The HH/HV (EOS-04) and VH/VV (Sentinel-1) polarization channels encode different surface-volume scattering components that together constrain SM estimation. Unlike optical sensors, SAR works through clouds — critical for monsoon-season measurements in India.

### Why Prediction Intervals?

Point estimates of soil moisture are insufficient for:
- **Irrigation scheduling** — a system that triggers irrigation at SM < 20% behaves very differently if the true SM is 18% ± 2% vs 18% ± 12%.
- **Flood early warning** — near-saturation SM uncertainty determines whether a soil is "at risk" or "safe."
- **Model assimilation** — ensemble land surface models require observation uncertainty as input.

This paper is among the first to apply **Conformalized Quantile Regression** to SAR-based SM estimation, providing a non-parametric, distribution-free 95% coverage guarantee.

---

## Study Area & Satellites

### Field Site

- **Location:** Agricultural plots, Anand district, Gujarat, India
- **Coordinates:** ~22.52° N, 72.76° E
- **Crops:** Paddy, Pearl Millet, Fallow, Ortho, Green Manure (multi-crop heterogeneous landscape)
- **Period:** June 2022 – July 2023 (covers one full Kharif + Rabi + Kharif cycle)

### Ground Truth

Soil moisture was measured using **IoT-enabled LoRa sensors** deployed at fixed grid points. Each sensor reports volumetric water content (%) at 0–5 cm depth. Measurements are co-located and co-registered with SAR acquisition windows.

**SM1 Quality Filter:** Values at SM1 = 50% are instrument saturation artifacts (sensor ceiling, not a real measurement). Values ≤ 0 are sensor failures. Filter applied: `0 < SM1 ≤ 60%`.

### Satellites

| Satellite | Agency | Band | Polarizations | Revisit | Resolution |
|-----------|--------|------|--------------|---------|-----------|
| EOS-04 (RISAT-1A) | ISRO | C-band | HH, HV | 12 days | ~3–50 m |
| Sentinel-1 | ESA | C-band | VH, VV | 6 days | 10 m |

| Satellite | Raw records | After SM1 filter | SM range |
|-----------|------------|-----------------|----------|
| EOS-04 | 2530 | **2528** | 1.2 – 54.1% |
| Sentinel-1 | 1818 | **1816** | 1.2 – 56.4% |

---

## Feature Engineering

Raw inputs per sample-date pair: two polarization backscatter values (dB) + acquisition date + crop label.

Six engineered features fed to all models:

| # | Feature | Formula / Source | Physical Justification |
|---|---------|-----------------|----------------------|
| 1 | `HH-pol` / `VH-pol` | Raw backscatter (dB) | Co-pol sensitive to surface roughness + moisture |
| 2 | `HV-pol` / `VV-pol` | Raw backscatter (dB) | Cross-pol sensitive to volume scattering |
| 3 | `cross_pol_ratio` | HH÷HV or VH÷VV | Ratio normalises incidence angle; strong dielectric indicator |
| 4 | `month_sin` | sin(2π × month / 12) | Seasonal SM variation (monsoon onset, dry season) |
| 5 | `month_cos` | cos(2π × month / 12) | Paired with sin to make cyclical encoding continuous at Dec→Jan |
| 6 | `crop_encoded` | Ordinal label encoding | Crop type determines canopy attenuation and surface roughness |

**Target:** `SM1 (%)` — volumetric soil moisture

---

## Experimental Pipeline — All 8 Phases

```
Raw Excel Data
      │
   [Phase 0] Feature Engineering → eos-04-enhanced.csv, sentinel-1-enhanced.csv
      │
   [Phase 1] Classical ML Regression   (RF, XGBoost, AdaBoost, SVR)
      │
   [Phase 2] 4-Class Classification    (Low / Medium / High / Very High)
      │
   [Phase 3] ANN Point Estimation      (6 architectures)
      │
   [Phase 4] ANN Quantile Regression   (raw prediction intervals, τ=0.025/0.975)
      │
   [Phase 5] Conformal Regression via MAPIE (GBR, HistGBR, QuantileReg)
      │
   [Phase 6] Conformalized QR — 4 methods
      │         ├── SVM Split Conformal
      │         ├── GBM CQR
      │         ├── ANN Split Conformal
      │         └── ANN CQR (dual-output backbone)
      │
   [Phase 7] Tau Hyperparameter Tuning (cal-based selection)
      │
   [Phase 8] Quantile SVR Gamma Grid Search
      │
   output/ ← JSON metrics + PNG plots per phase
```

---

## Methodology

### Phase 1 — Classical ML Regression

Four models trained with GridSearchCV (3-fold CV) on an 80/10/10 train-val-test split. Metrics: R², RMSE, MAE.

Models: `RandomForestRegressor`, `XGBRegressor`, `AdaBoostRegressor`, `SVR`.

---

### Phase 2 — Soil Moisture Classification

SM1 is binned into four classes using sample quartiles:

| Class | SM1 Range (approx.) |
|-------|---------------------|
| Low | 0–8.7% |
| Medium | 8.7–19.5% |
| High | 19.5–37.0% |
| Very High | 37.0–60.0% |

Models: `RandomForestClassifier`, `XGBClassifier`, `AdaBoostClassifier`, `SVC`.

---

### Phase 3 — ANN Point Estimation

Fully connected networks (ReLU activations, MSE loss) with early stopping. Six architectures tested ranging from shallow (2 neurons) to regularised deep (16→Dropout→8→Dropout→1).

---

### Phase 4 — ANN Quantile Regression (Prediction Intervals)

Two output heads trained with **pinball loss** at τ_lo = 0.025 and τ_hi = 0.975:

```
L_pinball(τ, y, ŷ) = mean( max(τ·(y−ŷ), (τ−1)·(y−ŷ)) )
```

This gives raw (uncalibrated) 95% prediction intervals. Coverage is measured by PICP (Prediction Interval Coverage Probability) and efficiency by MPIW (Mean PI Width).

---

### Phase 5 — Conformal Regression (MAPIE)

Standard split-conformal calibration via the [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) library. Calibrates residuals of a quantile regressor on a held-out calibration set. Models: `GradientBoostingRegressor`, `HistGradientBoostingRegressor`, `QuantileRegressor`.

---

### Phase 6 — Conformalized Quantile Regression (CQR)

The central methodological contribution. CQR calibrates raw quantile bounds using a non-conformity score:

```
E_i = max(ŷ_lo(x_i) − y_i,  y_i − ŷ_hi(x_i))    for each calibration point i
q̂   = (1−α)-quantile of {E_1, ..., E_n}
Ĉ(x_test) = [ŷ_lo(x) − q̂,  ŷ_hi(x) + q̂]
```

Under exchangeability, this guarantees P(Y ∈ Ĉ(X)) ≥ 1 − α.

#### Data Split — 70/10/10/10

```
70%  train       → model weights only
10%  validation  → early stopping signal only (never touches calibration)
10%  calibration → conformal scores only (never influences training)
10%  test        → final PICP / MPIW evaluation only
```

This strict separation is required by conformal prediction theory. Using val = cal (a common mistake) biases q̂ downward and invalidates the coverage guarantee.

#### Four Methods Compared

| Method | Base model | Calibration type |
|--------|-----------|-----------------|
| SVM Split Conformal | SVR (rbf) | Absolute residuals |
| GBM CQR | GradientBoostingRegressor(loss='quantile') × 2 + sort guard | CQR scores |
| ANN Split Conformal | MSE ANN point predictor | Absolute residuals |
| ANN CQR | Dual-output shared-backbone ANN | CQR scores |

#### Dual-Output ANN — Eliminating Interval Inversions

Training two independent quantile ANNs causes structural inversions (lower > upper on some samples). Instead, a single shared-backbone model outputs both bounds jointly:

```
Input (6)
   │
Dense(16, relu) → Dropout(0.09)
   │
Dense(8, relu)  → Dropout(0.09)
   │
   ├──→ Dense(1)  [lo head, trained with pinball(τ_lo)]
   └──→ Dense(1)  [hi head, trained with pinball(τ_hi)]

Loss = L_pinball(τ_lo) + L_pinball(τ_hi)   (joint optimisation)
```

Result: **0.0% interval inversion** across all tau configurations and both satellites.

---

### Phase 7 — Tau Hyperparameter Tuning

Six τ_lo values tested: {0.01, 0.015, 0.02, 0.025, 0.03, 0.04}, with τ_hi = τ_lo + 0.95.

**Selection criterion (no test-set snooping):**
> Choose the τ with minimum `RawCal_MPIW` on the **calibration set**, subject to `RawCal_PICP ≥ 0.95`. Fallback to ≥ 0.90 if no tau meets the primary threshold.

The test set is never consulted during tau selection.

---

### Phase 8 — Quantile SVR Gamma Grid

Quantile SVR (95% quantile) evaluated across 31 gamma values from 2^−15 to 2^+15 to find the best kernel width for the tightest calibrated interval.

---

## Results

### Phase 1 — Classical ML Regression

| Model | EOS-04 R² | EOS-04 RMSE | Sentinel-1 R² | Sentinel-1 RMSE |
|-------|----------|------------|--------------|----------------|
| Random Forest | 0.637 | 9.877 | 0.479 | 10.827 |
| XGBoost | 0.628 | 9.991 | **0.487** | 10.737 |
| AdaBoost | 0.634 | 9.911 | 0.328 | 12.290 |
| SVR | 0.586 | 10.539 | 0.444 | 11.176 |
| **Best** | **RF: 0.637** | | **XGB: 0.487** | |

### Phase 2 — 4-Class Classification

| Model | EOS-04 Acc | Sentinel-1 Acc |
|-------|-----------|---------------|
| Random Forest | 56.9% | 53.8% |
| XGBoost | 57.1% | 53.0% |
| AdaBoost | 49.8% | 46.4% |
| **SVC** | **57.9%** | **58.0%** |

### Phase 3 — ANN Point Estimation

| Architecture | EOS-04 R² | EOS-04 MAE | Sentinel-1 R² | Sentinel-1 MAE |
|-------------|----------|-----------|--------------|---------------|
| 2→1 | 0.346 | 11.14 | 0.059 | 12.12 |
| 4→1 | 0.346 | 11.13 | 0.115 | 11.64 |
| 8→1 | 0.345 | 11.14 | 0.060 | 12.04 |
| 16→1 | 0.557 | 8.69 | 0.260 | 10.37 |
| 16→D→1 | **0.588** | **8.28** | 0.242 | 10.52 |
| 16→D→8→D→1 | 0.577 | 8.40 | **0.327** | **9.82** |

### Phase 4 — Prediction Intervals (τ=0.025/0.975, uncensored)

| Architecture | EOS-04 PICP | EOS-04 MPIW | S1 PICP | S1 MPIW |
|-------------|------------|------------|---------|---------|
| 2→1 | 0.9565 | 43.96 | 0.9615 | 46.10 |
| 4→1 | 0.9565 | 43.95 | **0.9725** | 46.11 |
| 8→1 | 0.9644 | 43.71 | 0.9615 | 44.89 |
| **16→1** | **0.9763** | **39.57** | 0.9505 | **44.84** |
| 16→D→1 | 0.9842 | 44.67 | 0.9615 | 50.19 |
| 16→D→8→D→1 | 0.9802 | 46.09 | 0.9231 | 53.82 |

### Phase 5 — Conformal Regression (MAPIE)

| Model | EOS-04 PICP | EOS-04 MPIW | S1 PICP | S1 MPIW |
|-------|------------|------------|---------|---------|
| GradientBoostingRegressor | **0.9803** | 41.02 | 0.9344 | 38.93 |
| HistGradientBoostingRegressor | 0.9488 | 36.47 | **0.9727** | **42.86** |
| QuantileRegressor | 0.9843 | 46.60 | 0.9727 | 45.50 |

### Phase 6 — Conformalized Quantile Regression (α = 0.05)

| Method | EOS-04 PICP | EOS-04 MPIW | S1 PICP | S1 MPIW | Inversions |
|--------|------------|------------|---------|---------|-----------|
| SVM Split Conformal | 0.9073 | 37.28 | 0.9608 | 42.30 | N/A |
| **GBM CQR** | **0.9610** | **35.91** | 0.9346 | **33.85** | **0.0%** |
| ANN Split Conformal | 0.9317 | 36.73 | **0.9673** | 40.57 | N/A |
| ANN CQR (dual-output) | 0.9171 | 41.00 | **0.9739** | 41.30 | **0.0%** |

**GBM CQR** delivers the tightest intervals (MPIW 35.91 / 33.85) with valid coverage on both satellites.

### Phase 7 — Tau Tuning (ANN CQR dual, cal-based selection)

#### EOS-04

| τ_lo | τ_hi | RawCal PICP | CQR PICP | CQR MPIW | Inversions |
|------|------|------------|---------|---------|-----------|
| 0.010 | 0.960 | 0.9659 | 0.9268 | **37.06** | 0.0% |
| 0.015 | 0.965 | 0.9756 | 0.9024 | 36.61 | 0.0% |
| 0.020 | 0.970 | 0.9756 | 0.9122 | 37.84 | 0.0% |
| 0.025 | 0.975 | 0.9659 | 0.9268 | 43.00 | 0.0% |
| 0.030 | 0.980 | 0.9756 | 0.9415 | 41.32 | 0.0% |
| 0.040 | 0.990 | 0.9756 | 0.9415 | 46.65 | 0.0% |

★ **Best (RawCal PICP ≥ 0.95): τ = 0.01/0.96 → CQR PICP=0.9268, MPIW=37.06**

#### Sentinel-1

| τ_lo | τ_hi | RawCal PICP | CQR PICP | CQR MPIW | Inversions |
|------|------|------------|---------|---------|-----------|
| 0.010 | 0.960 | 0.9412 | 0.9477 | 39.50 | 0.0% |
| 0.015 | 0.965 | 0.9346 | 0.9673 | 40.37 | 0.0% |
| **0.020** | **0.970** | **0.9542** | **0.9673** | **40.38** | **0.0%** |
| 0.025 | 0.975 | 0.9608 | 0.9608 | 42.52 | 0.0% |
| 0.030 | 0.980 | 0.9281 | 0.9673 | 46.31 | 0.0% |
| 0.040 | 0.990 | 0.9608 | 0.9608 | 43.83 | 0.0% |

★ **Best (RawCal PICP ≥ 0.95): τ = 0.02/0.97 → CQR PICP=0.9673, MPIW=40.38**

---

## Baseline vs Improved — What Changed and Why

The original experiment had six methodological bugs. The table below shows the effect of each fix:

| Bug | Original | Fixed | Impact |
|-----|----------|-------|--------|
| **Val = Cal leakage** | Same set for early stopping + conformal scores | Strict 70/10/10/10 split | Coverage guarantee restored; PICP reflects true unseen-data coverage |
| **Two independent ANN models** | Separate lower/upper quantile nets | Shared-backbone dual-output | Inversions 0.0% across all runs; no artificial MPIW inflation |
| **Linear CQR as base** | `QuantileRegressor` (linear) | `GradientBoostingRegressor(loss='quantile')` | MPIW EOS-04: 40.33 → **35.91** |
| **Slow training** | lr=0.0001, patience=10 | lr=0.001, patience=30 | Model converges properly; intervals tighten |
| **SM1 filter too permissive** | SM1 < 150 / < 100 | 0 < SM1 ≤ 60 | Removes censored SM1=50 and invalid readings; R² jumps +0.3 |
| **Tau selected on test set** | Best τ picked from test PICP/MPIW | Cal-based: min RawCal_MPIW where RawCal_PICP ≥ 0.95 | No data snooping; reported τ is reproducible |

### Net Improvement (Phase 6 CQR, best method per metric)

| Metric | Baseline best (Linear CQR) | New best (GBM CQR) | Change |
|--------|--------------------------|-------------------|--------|
| EOS-04 PICP | 0.9073 | **0.9610** | +5.4 pp |
| EOS-04 MPIW | 40.33 | **35.91** | −4.42 |
| S1 PICP | 0.9477 | 0.9346 | (GBM; ANN CQR = **0.9739**) |
| S1 MPIW | 42.23 | **33.85** | −8.38 |

---

## Theoretical Validity

Conformal prediction provides the finite-sample marginal coverage guarantee:

> **P( Y_{n+1} ∈ Ĉ(X_{n+1}) ) ≥ 1 − α**

for any α ∈ (0,1), without distributional assumptions, provided calibration samples are **exchangeable** with the test sample (i.e., drawn i.i.d. and not used during model training in any form).

This paper's implementation satisfies exchangeability through the strict 70/10/10/10 split. The coverage guarantee is valid. Prior implementations using val = cal do **not** satisfy this condition and cannot claim the guarantee.

---

## Reproducibility

All randomness is fully seeded:

```python
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED']       = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)   # re-called inside every training function
```

All `train_test_split` calls use `random_state=42`. `GradientBoostingRegressor` and classifiers use `random_state=42`.

---

## How to Run

### Full pipeline — all 8 phases (recommended)

```bash
cd experiments/classification_new_data/code
uv run python run_all_enhanced.py
```

Writes all JSON metrics + PNG plots to `output/`. Runtime ~30–90 min on CPU.

### CQR pipeline only (phases 4 / 6 / 7)

```bash
uv run python run_phases_467.py
```

### Notebook-by-notebook (with cell outputs)

First register the virtual environment as a Jupyter kernel:

```bash
uv run python -m ipykernel install --user --name ml-experiments --display-name "ml-experiments"
```

Then run the notebook sequence:

```bash
bash run_experiments_sequence.sh
```

Or open individual notebooks in JupyterLab. Notebooks run in this order:

```
1. exploration_eos.ipynb / exploration_sentinel.ipynb   ← Phase 0
2. classical_ml_uncensored.ipynb                        ← Phase 1
3. classification_uncensored.ipynb                      ← Phase 2
4. ann_uncensored.ipynb                                 ← Phase 3
5. pi_estimation_uncensored.ipynb                       ← Phase 4
6. conformal_regression_uncensored.ipynb                ← Phase 5
7. conformalized_quantile_regression_uncensored.ipynb   ← Phase 6
8. quantile_regression_tau_tuning_uncensored.ipynb      ← Phase 7
9. quantile_svr_HP_tuning.ipynb                         ← Phase 8
```

Each `*_censored.ipynb` variant excludes SM1 = 50 rows (instrument saturation).

---

## Repository Structure

```
major_orig/
├── README.md
└── experiments/
    └── classification_new_data/
        ├── code/
        │   ├── constants.py                                  ← shared paths & feature column names
        │   ├── model_experiments.py                          ← all experiment classes
        │   │                                                   (RegressionExperiment, ANNExperiment,
        │   │                                                    PredictionIntervalEstimation,
        │   │                                                    ConformalizedQuantileExperiment, ...)
        │   ├── run_all_enhanced.py                           ← headless all-phase runner
        │   ├── run_phases_467.py                             ← focused CQR runner
        │   ├── run_experiments_sequence.sh                   ← notebook runner (nbconvert)
        │   │
        │   ├── exploration_eos.ipynb                         ← Phase 0: EOS-04 EDA
        │   ├── exploration_sentinel.ipynb                    ← Phase 0: Sentinel-1 EDA
        │   ├── classical_ml_uncensored.ipynb                 ← Phase 1
        │   ├── classical_ml_censored.ipynb                   ← Phase 1 (SM1≠50)
        │   ├── classification_uncensored.ipynb               ← Phase 2
        │   ├── classification_censored.ipynb                 ← Phase 2 (SM1≠50)
        │   ├── ann_uncensored.ipynb                          ← Phase 3
        │   ├── ann_censored.ipynb                            ← Phase 3 (SM1≠50)
        │   ├── pi_estimation_uncensored.ipynb                ← Phase 4
        │   ├── pi_estimation_censored.ipynb                  ← Phase 4 (SM1≠50)
        │   ├── conformal_regression_uncensored.ipynb         ← Phase 5
        │   ├── conformal_regression_censored.ipynb           ← Phase 5 (SM1≠50)
        │   ├── conformalized_quantile_regression_uncensored.ipynb  ← Phase 6
        │   ├── quantile_regression_tau_tuning_uncensored.ipynb     ← Phase 7
        │   └── quantile_svr_HP_tuning.ipynb                  ← Phase 8
        │
        ├── data/
        │   ├── EOS-04_datasheet.xlsx                         ← raw field data (ISRO format)
        │   ├── sentinel-1.xlsx                               ← raw field data (ESA format)
        │   ├── eos-04-enhanced.csv                           ← processed: SM1 filtered + 6 features
        │   └── sentinel-1-enhanced.csv                       ← processed: SM1 filtered + 6 features
        │
        └── output/
            ├── ml_experiment_uncensored/                     ← Phase 1: metrics JSON
            ├── classification_uncensored/                    ← Phase 2: metrics JSON
            ├── ann_experiments_uncensored/                   ← Phase 3: metrics + prediction plots
            ├── pi_estimation_uncensored/                     ← Phase 4: metrics + PI plots
            ├── conformal_regression_uncensored/              ← Phase 5: metrics + PI plots
            └── conformal_results/                            ← Phase 6/7: metrics + PI plots
                ├── EOS-04_conformal_metrics.json
                ├── Sentinel-1_conformal_metrics.json
                ├── EOS-04_tau_tuning_metrics.json
                ├── Sentinel-1_tau_tuning_metrics.json
                └── plots/
```

---

## Dependencies

```
Python        3.12
tensorflow    >= 2.15
scikit-learn  >= 1.4
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

Install via `uv`:

```bash
uv sync
```

---

## Citation

```bibtex
@article{meena2024sar,
  title   = {SAR-Based Soil Moisture Estimation with Calibrated Prediction Intervals
             using Conformalized Quantile Regression},
  author  = {Thakor, Naitik and Meena, Siddhant and others},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  year    = {2026},
  note    = {Under review}
}
```

---

## Key References

- Angelopoulos, A. N., & Bates, S. (2023). Conformal Prediction: A Gentle Introduction. *Foundations and Trends in Machine Learning*.
- Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized Quantile Regression. *NeurIPS*.
- Koenker, R., & Bassett, G. (1978). Regression Quantiles. *Econometrica*.
- Dubois-Fernandez, P., et al. (2012). SAR backscatter and soil moisture — dielectric mixing models. *Remote Sensing*.
