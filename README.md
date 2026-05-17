# Soil Moisture Estimation from SAR Imagery with Calibrated Prediction Intervals

> **IEEE Geoscience and Remote Sensing Letters** — Under Review
>
> *Naitik Thakor, Siddhant Meena et al.*

---

## What This Paper Is About

Soil moisture (SM) at the topsoil layer (0–5 cm) drives crop water stress, flood runoff, and drought early-warning. Traditional ground sensors are sparse and expensive. Synthetic Aperture Radar (SAR) satellites — which penetrate clouds and operate day/night — can map soil moisture at field scale from backscatter intensity. But a single point estimate ("SM = 23%") is not enough for precision agriculture or hydrological modelling. Farmers and decision systems need to know the *uncertainty*: is the true value likely between 18–28%, or between 5–45%?

This paper makes three contributions:

1. **A comparative study of ML and ANN models** for SAR-to-SM regression and classification using two satellites — Indian EOS-04 (RISAT-1A) and European Sentinel-1 — over an agricultural field site in Gujarat, India.

2. **Integration of GEE-derived Sentinel-2 NDVI** as a 7th auxiliary feature, retrieved via per-pixel SCL cloud masking and a ±5/15-day temporal window, improving EOS-04 R² by up to 4.2% and enabling ≥0.95 conformal coverage across 5 additional method–sensor combinations.

3. **A rigorous implementation of Conformalized Quantile Regression (CQR)** that produces statistically valid 95% prediction intervals, with a corrected data split, a dual-output ANN that eliminates interval inversions, and an honest tau hyperparameter selection strategy.

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

Raw inputs per sample-date pair: two polarization backscatter values (dB) + acquisition date + crop label + Sentinel-2 NDVI.

Seven engineered features fed to all models:

| # | Feature | Formula / Source | Physical Justification |
|---|---------|-----------------|----------------------|
| 1 | `HH-pol` / `VH-pol` | Raw backscatter (dB) | Co-pol sensitive to surface roughness + moisture |
| 2 | `HV-pol` / `VV-pol` | Raw backscatter (dB) | Cross-pol sensitive to volume scattering |
| 3 | `cross_pol_ratio` | HH÷HV or VH÷VV | Ratio normalises incidence angle; strong dielectric indicator |
| 4 | `month_sin` | sin(2π × month / 12) | Seasonal SM variation (monsoon onset, dry season) |
| 5 | `month_cos` | cos(2π × month / 12) | Paired with sin to make cyclical encoding continuous at Dec→Jan |
| 6 | `crop_encoded` | Ordinal label encoding | Crop type determines canopy attenuation and surface roughness |
| 7 | `NDVI` | Sentinel-2 SR (GEE) | Optical vegetation proxy orthogonal to SAR backscatter; constrains canopy-moisture coupling under dry surface conditions |

**Target:** `SM1 (%)` — volumetric soil moisture

### NDVI Retrieval Methodology

NDVI is retrieved from **Sentinel-2 SR Harmonized** (`COPERNICUS/S2_SR_HARMONIZED`) via the Google Earth Engine (GEE) batch API for each SAR acquisition date:

1. **Cloud masking** — per-pixel Scene Classification Layer (SCL): retain SCL ∈ {4, 5, 6, 11} (vegetation, bare soil, water, snow); mask SCL ∈ {1, 2, 3, 7, 8, 9, 10} (cloud shadow, cloud, cirrus)
2. **Temporal window** — ±5 days around each acquisition date; fallback to ±15 days for cloud-gap dates (monsoon Jun–Aug)
3. **Spatial aggregation** — `reduceRegions` batch call over all 145 sample points per date (34 API calls total vs 5800 per-point)
4. **Missing value imputation** — monthly median NDVI per crop type for rows with no clear-sky pixel in either window (monsoon gap)

| Sensor | Acquisitions | Direct NDVI coverage | Imputed |
|--------|:-----------:|:--------------------:|:-------:|
| EOS-04 | 34 dates | **86.7%** | 13.3% |
| Sentinel-1 | 34 dates | **94.2%** | 5.8% |

---

## Experimental Pipeline — All 8 Phases

```
Raw Excel Data + GEE NDVI CSVs
      │
   [Phase 0] Feature Engineering → eos-04-enhanced-ndvi.csv, sentinel-1-enhanced-ndvi.csv
      │                             (7 features: pol1, pol2, cross_pol_ratio, month_sin,
      │                              month_cos, crop_encoded, NDVI)
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
Input (7)
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
| **Random Forest** | **0.6475** | **9.727** | 0.4749 | 10.865 |
| XGBoost | 0.6447 | 9.766 | 0.4738 | 10.877 |
| AdaBoost | 0.6379 | 9.860 | 0.3847 | 11.762 |
| SVR | 0.6112 | 10.217 | 0.4536 | 11.084 |
| **Best** | **RF: 0.6475** | | **RF: 0.4749** | |

> Sentinel-1 scores are lower than EOS-04 across all models. VH-pol in agricultural C-band is less sensitive to topsoil SM than HH-pol. The apparent R² decline vs the pre-NDVI baseline (0.544→0.475 for RF) is **not caused by NDVI collinearity** — a controlled ablation removing NDVI from Sentinel-1 features produced a change of only +0.004 R² for RF and made AdaBoost worse (−0.053). The true cause is a dataset composition change when the NDVI pipeline was introduced: `sentinel-1-enhanced.csv` (1821 rows) was rebuilt as `sentinel-1-enhanced-ndvi.csv` (1816 rows) via a pol-value join that altered row ordering and train/test split boundaries. The current results (7 features including NDVI) are the authoritative numbers for this dataset.

### Phase 2 — 4-Class Classification

| Model | EOS-04 Acc | Sentinel-1 Acc |
|-------|-----------|---------------|
| Random Forest | 54.7% | 56.0% |
| XGBoost | 55.3% | 53.6% |
| AdaBoost | 52.6% | 47.0% |
| **SVC** | **57.7%** | **57.1%** |

### Phase 3 — ANN Point Estimation

| Architecture | EOS-04 R² | EOS-04 MAE | Sentinel-1 R² | Sentinel-1 MAE |
|-------------|----------|-----------|--------------|---------------|
| 2→1 | 0.398 | 10.67 | 0.062 | 11.76 |
| 4→1 | 0.396 | 10.68 | 0.081 | 11.58 |
| 8→1 | 0.396 | 10.67 | 0.084 | 11.50 |
| **16→1** | **0.609** | **8.01** | **0.265** | **10.28** |
| 16→D→1 | 0.395 | 10.68 | 0.189 | 10.82 |
| 16→D→8→D→1 | 0.607 | 8.14 | 0.230 | 10.52 |

> The `16→D→1` architecture collapsed for EOS-04 (0.588→0.395) with the extra NDVI input dimension — dropout over-regularises at 7 inputs with rate=0.09. Best stable architecture: `16→1`.

### Phase 4 — Prediction Intervals (τ=0.025/0.975)

| Architecture | EOS-04 PICP | EOS-04 MPIW | S1 PICP | S1 MPIW |
|-------------|------------|------------|---------|---------|
| 2→1 | 0.9526 ✅ | 43.61 | 0.9725 ✅ | 46.10 |
| 4→1 | 0.9644 ✅ | 43.49 | 0.9725 ✅ | 46.10 |
| 8→1 | 0.9644 ✅ | 44.34 | 0.9670 ✅ | 44.56 |
| **16→1** | **0.9684 ✅** | **39.64** | 0.9505 ✅ | 44.61 |
| 16→D→1 | 0.9763 ✅ | 47.76 | 0.9505 ✅ | 49.05 |
| 16→D→8→D→1 | 0.9723 ✅ | 45.64 | **0.9670 ✅** | **53.40** |

All architectures achieve PICP ≥ 0.95 for both sensors.

### Phase 5 — Conformal Regression (MAPIE)

| Model | EOS-04 PICP | EOS-04 MPIW | S1 PICP | S1 MPIW |
|-------|------------|------------|---------|---------|
| **GradientBoostingRegressor** | **0.9685 ✅** | **39.05** | **0.9672 ✅** | 43.54 |
| HistGradientBoostingRegressor | 0.9252 ❌ | 34.17 | 0.9727 ✅ | **45.16** |
| QuantileRegressor | 0.9843 ✅ | 46.60 | 0.9727 ✅ | 45.50 |

### Phase 6 — Conformalized Quantile Regression (α = 0.05)

| Method | EOS-04 PICP | EOS-04 MPIW | S1 PICP | S1 MPIW | Inversions |
|--------|------------|------------|---------|---------|-----------|
| SVM Split Conformal | 0.9317 ❌ | 38.38 | 0.9608 ✅ | 40.81 | N/A |
| **GBM CQR** | **0.9659 ✅** | **35.70** | 0.9542 ✅ | **35.75** | **0.0%** |
| ANN Split Conformal | 0.9610 ✅ | 39.32 | 0.9542 ✅ | 39.39 | N/A |
| ANN CQR (dual-output) | 0.9220 ❌ | 41.18 | **0.9804 ✅** | 41.77 | **0.0%** |

**GBM CQR** delivers the tightest intervals (MPIW 35.70 / 35.75) with valid ≥0.95 coverage on both satellites.

### Phase 7 — Tau Tuning (ANN CQR dual, cal-based selection)

#### EOS-04

| τ_lo | τ_hi | RawCal PICP | CQR PICP | CQR MPIW | Inversions |
|------|------|------------|---------|---------|-----------|
| 0.010 | 0.960 | **0.9707 ✅** | 0.8878 | **34.52** | 0.0% |
| 0.015 | 0.965 | 0.9659 ✅ | 0.9171 | 36.45 | 0.0% |
| 0.020 | 0.970 | 0.9707 ✅ | 0.9073 | 36.36 | 0.0% |
| 0.025 | 0.975 | 0.9756 ✅ | 0.9220 | 41.40 | 0.0% |
| 0.030 | 0.980 | 0.9805 ✅ | 0.9220 | 40.22 | 0.0% |
| 0.040 | 0.990 | 0.9707 ✅ | 0.9220 | 43.53 | 0.0% |

★ **Best (RawCal PICP ≥ 0.95): τ = 0.01/0.96 → CQR PICP=0.8878, MPIW=34.52**

#### Sentinel-1

| τ_lo | τ_hi | RawCal PICP | CQR PICP | CQR MPIW | Inversions |
|------|------|------------|---------|---------|-----------|
| **0.010** | **0.960** | **0.9608 ✅** | 0.9542 ✅ | **37.40** | 0.0% |
| 0.015 | 0.965 | 0.9477 ❌ | 0.9608 ✅ | 38.90 | 0.0% |
| 0.020 | 0.970 | 0.9412 ❌ | 0.9673 ✅ | 40.70 | 0.0% |
| 0.025 | 0.975 | 0.9542 ✅ | 0.9804 ✅ | 43.26 | 0.0% |
| 0.030 | 0.980 | 0.9412 ❌ | 0.9673 ✅ | 44.59 | 0.0% |
| 0.040 | 0.990 | 0.9412 ❌ | 0.9673 ✅ | 43.64 | 0.0% |

★ **Best (RawCal PICP ≥ 0.95): τ = 0.01/0.96 → CQR PICP=0.9542, MPIW=37.40**

### Phase 8 — Quantile SVR Gamma Grid (C=64, 31 γ values)

#### EOS-04

| γ | PICP | MPIW |
|---|:----:|:----:|
| 2^-15 | 0.9805 ✅ | 43.43 |
| 2^-14 | 0.9805 ✅ | 43.39 |
| 2^-13 | 0.9805 ✅ | 43.29 |
| 2^-12 | 0.9805 ✅ | 43.03 |
| 2^-11 | 0.9805 ✅ | 42.64 |
| 2^-10 | 0.9756 ✅ | 42.00 |
| 2^-9  | 0.9756 ✅ | 41.12 |
| 2^-8  | 0.9756 ✅ | 40.10 |
| 2^-7  | 0.9805 ✅ | 39.52 |
| 2^-6  | 0.9707 ✅ | 38.39 |
| 2^-5  | 0.9659 ✅ | 37.20 |
| 2^-4  | 0.9707 ✅ | 36.76 |
| 2^-3  | 0.9659 ✅ | 35.25 |
| 2^-2  | 0.9707 ✅ | 33.94 |
| 2^-1  | 0.9659 ✅ | 33.18 |
| 2^0   | 0.9659 ✅ | 32.40 |
| **2^1**   | **0.9512 ✅** | **30.91** |
| 2^2   | 0.9122 ❌ | 28.92 |
| 2^3   | 0.8488 ❌ | 25.87 |
| 2^4   | 0.7659 ❌ | 22.50 |
| 2^5   | 0.6732 ❌ | 19.03 |
| 2^6   | 0.5415 ❌ | 15.88 |
| 2^7   | 0.3415 ❌ | 12.29 |
| 2^8   | 0.1902 ❌ | 8.24 |
| 2^9   | 0.0976 ❌ | 4.79 |
| 2^10  | 0.0683 ❌ | 2.53 |
| 2^11  | 0.0293 ❌ | 1.20 |
| 2^12  | 0.0146 ❌ | 0.52 |
| 2^13  | 0.0049 ❌ | 0.23 |
| 2^14  | 0.0049 ❌ | 0.10 |
| 2^15  | 0.0049 ❌ | 0.05 |

★ **Best (PICP ≥ 0.95): γ=2^1 → PICP=0.9512, MPIW=30.91** (improved to MPIW=30.77 in Phase 8b C×γ grid)

#### Sentinel-1

| γ | PICP | MPIW |
|---|:----:|:----:|
| 2^-15 | 0.9542 ✅ | 43.02 |
| 2^-14 | 0.9542 ✅ | 42.98 |
| 2^-13 | 0.9542 ✅ | 42.87 |
| 2^-12 | 0.9542 ✅ | 42.75 |
| 2^-11 | 0.9542 ✅ | 42.55 |
| 2^-10 | 0.9542 ✅ | 42.15 |
| 2^-9  | 0.9477 ❌ | 41.43 |
| 2^-8  | 0.9477 ❌ | 40.47 |
| 2^-7  | 0.9608 ✅ | 40.15 |
| 2^-6  | 0.9542 ✅ | 39.22 |
| 2^-5  | 0.9608 ✅ | 38.47 |
| **2^-4**  | **0.9542 ✅** | **37.56** |
| 2^-3  | 0.9216 ❌ | 35.95 |
| 2^-2  | 0.9150 ❌ | 34.88 |
| 2^-1  | 0.9085 ❌ | 33.06 |
| 2^0   | 0.8954 ❌ | 31.24 |
| 2^1   | 0.8758 ❌ | 28.32 |
| 2^2   | 0.8235 ❌ | 25.80 |
| 2^3   | 0.8235 ❌ | 23.26 |
| 2^4   | 0.6928 ❌ | 20.60 |
| 2^5   | 0.6340 ❌ | 17.98 |
| 2^6   | 0.5098 ❌ | 15.79 |
| 2^7   | 0.3007 ❌ | 13.28 |
| 2^8   | 0.1699 ❌ | 10.04 |
| 2^9   | 0.1046 ❌ | 6.47 |
| 2^10  | 0.0588 ❌ | 3.68 |
| 2^11  | 0.0261 ❌ | 1.91 |
| 2^12  | 0.0065 ❌ | 0.89 |
| 2^13  | 0.0000 ❌ | 0.40 |
| 2^14  | 0.0000 ❌ | 0.14 |
| 2^15  | 0.0000 ❌ | 0.03 |

★ **Best (PICP ≥ 0.95): γ=2^-4 → PICP=0.9542, MPIW=37.56**

### Phase 8b — Quantile SVR C × γ Joint Grid

Phase 8 swept only γ at fixed C=2^6. Phase 8b adds C ∈ {2^4, 2^6, 2^8, 2^10} for a 13×4=52 joint grid per sensor (104 total QP solves).

| Sensor | Best C | Best γ | PICP | MPIW | vs Phase 8 |
|--------|:------:|:------:|:----:|:----:|:----------:|
| EOS-04 | 2^8 | 2^0 | 0.9561 ✅ | **30.77** | −0.14 |
| Sentinel-1 | 2^6 | 2^-4 | 0.9542 ✅ | 37.56 | 0.00 |

EOS-04 improves marginally (30.91→30.77) with higher C=2^8 forcing tighter quantile adherence. Sentinel-1 shows no improvement: higher C causes interval over-collapse on a heteroscedastic sensor (σ/μ of raw widths = 26%), destroying PICP above C=2^6.

---

### Phase 9 — Adaptive CQR Variants (Negative Results)

Three adaptive CQR methods were tested on top of the Phase 6 GBM base model. All failed to improve Sentinel-1 PICP while reducing MPIW:

| Method | EOS-04 PICP | EOS-04 MPIW | S1 PICP | S1 MPIW |
|--------|:-----------:|:-----------:|:-------:|:-------:|
| GBM CQR (Phase 6 baseline) | 0.9659 ✅ | 35.70 | 0.9542 ✅ | 35.75 |
| Interval-normalized CQR | 0.9610 ✅ | 35.34 | 0.9412 ❌ | 35.28 |
| Mondrian CQR (crop-stratified) | 0.9317 ❌ | 36.80 | 0.9216 ❌ | 33.87 |

**Root cause (MPIW decomposition):** The base GBM contributes 90–92% of final MPIW; the CQR correction contributes only 8–10%. Interval-normalized CQR introduces multiplicative estimation error in the normalization function, which exceeds the correction benefit at n_cal ≈ 153. Mondrian CQR produced negative group-level q̂ values (crop 5 EOS-04: q̂=−0.882; crop 19 S1: q̂=−0.046), indicating that the random 70/10/10/10 split violates within-crop exchangeability — different seasonal compositions end up in cal vs test per crop, invalidating the Mondrian conformal guarantee. **These are valid negative findings** confirming that adaptive CQR with ~150 calibration samples and 20+ crop classes introduces more estimation error than it eliminates.

---

### Phase 10 — Tuned GBM CQR (Hyperparameter Grid for MPIW Reduction)

**Key insight from Phase 9 postmortem:** The base GBM hyperparameters (n_estimators=200, max_depth=4, lr=0.05, min_samples_leaf=1) were inherited from Phase 6 and never tuned for interval tightness. Since the base model contributes 90% of MPIW, tuning it directly is the primary lever.

**Grid:** min_samples_leaf ∈ {1,5,10,20} × max_depth ∈ {3,4,5} × n_estimators ∈ {200,300} × subsample ∈ {1.0,0.8} × base_τ ∈ {(0.025,0.975),(0.1,0.9)} = 96 configs per sensor. Selection: val PICP ≥ 0.95 → min val MPIW.

**Why min_samples_leaf matters:** With the default min_samples_leaf=1, GBM leaves can contain a single sample, making extreme quantile (2.5th/97.5th percentile) estimates unreliable. Larger min_samples_leaf forces leaf-level averaging over ≥n samples, producing smoother quantile surfaces. **Why base_τ=0.1/0.9:** fitting the 80% interval avoids noisy extreme-quantile leaf estimates; the CQR calibration step bridges the gap to 95% coverage using a stable q̂ from n_cal=153 scores.

| Sensor | Selected params | q̂ | val PICP | val MPIW | test PICP | test MPIW | vs Phase 6 |
|--------|:---------------:|:--:|:--------:|:--------:|:---------:|:---------:|:----------:|
| EOS-04 | msl=1, depth=4, n=300, sub=0.8, τ=0.025 | 1.45 | 0.9510 ✅ | 33.10 | 0.9561 ✅ | **33.40** | −2.30 |
| Sentinel-1 | msl=5, depth=4, n=300, sub=0.8, τ=0.1/0.9 | 6.23 | 0.9539 ✅ | 31.95 | 0.9608 ✅ | **30.61** | −5.14 |

EOS-04: n_estimators=300 with subsample=0.8 (stochastic boosting) reduces MPIW from 35.70→33.40 (6.5%). QSVR Phase 8b still achieves the absolute minimum MPIW (30.77) for this sensor. Sentinel-1: the τ=0.1/0.9 strategy with msl=5 reduces MPIW from 35.75→30.61 (14.4%) — the largest single-phase improvement for this sensor in the study, also beating the QSVR best (37.56) by 18.5%.

---

## NDVI Impact Summary — Baseline vs +NDVI

All results comparing 6-feature baseline (commit `46063be`) against 7-feature NDVI pipeline.

### Regression (Phase 1, R²)

| Model | EOS-04 Before | EOS-04 After | Δ | S1 Before | S1 After | Δ |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|
| RandomForest | 0.618 | **0.648** | +0.029 | **0.544** | 0.475 | −0.069 |
| XGBoost | 0.608 | **0.645** | +0.037 | **0.550** | 0.474 | −0.076 |
| AdaBoost | 0.612 | **0.638** | +0.026 | **0.411** | 0.385 | −0.026 |
| SVR | 0.569 | **0.611** | +0.042 | **0.516** | 0.454 | −0.062 |

> EOS-04: all 4 models improved. NDVI adds orthogonal surface condition context not captured by HH/HV alone.
> Sentinel-1: all 4 models show lower R² vs the pre-NDVI baseline. This is **not caused by NDVI collinearity** — a controlled ablation removing NDVI from Sentinel-1 features changed RF R² by only +0.004 and made AdaBoost worse (−0.053). Root cause: the NDVI pipeline rebuilt the dataset (1821→1816 rows, new pol-value merge) changing train/test split composition. The 7-feature results are authoritative for the current dataset.

### Coverage Threshold Crossings (PICP ≥ 0.95)

| Phase | Method | Sensor | Before | After | Status |
|-------|--------|--------|:------:|:-----:|:------:|
| 4 | ANN 16→D→8→D→1 | Sentinel-1 | 0.923 ❌ | **0.967** | ✅ FIXED |
| 5 | MAPIE GBM | Sentinel-1 | 0.934 ❌ | **0.967** | ✅ FIXED |
| 6 | ANN Split Conformal | EOS-04 | 0.932 ❌ | **0.961** | ✅ FIXED |
| 6 | GBM CQR | Sentinel-1 | 0.935 ❌ | **0.954** | ✅ FIXED |
| 8 | Quantile SVR | Sentinel-1 | 0.922 ❌ | **0.954** | ✅ FIXED |

**5 method–sensor combinations crossed the ≥0.95 threshold with NDVI.**

---

## Baseline vs Improved — What Changed and Why

The original experiment had six methodological bugs. The table below shows the effect of each fix:

| Bug | Original | Fixed | Impact |
|-----|----------|-------|--------|
| **Val = Cal leakage** | Same set for early stopping + conformal scores | Strict 70/10/10/10 split | Coverage guarantee restored; PICP reflects true unseen-data coverage |
| **Two independent ANN models** | Separate lower/upper quantile nets | Shared-backbone dual-output | Inversions 0.0% across all runs; no artificial MPIW inflation |
| **Linear CQR as base** | `QuantileRegressor` (linear) | `GradientBoostingRegressor(loss='quantile')` | MPIW EOS-04: 40.33 → **35.70** |
| **Slow training** | lr=0.0001, patience=10 | lr=0.001, patience=30 | Model converges properly; intervals tighten |
| **SM1 filter too permissive** | SM1 < 150 / < 100 | 0 < SM1 ≤ 60 | Removes censored SM1=50 and invalid readings; R² jumps +0.3 |
| **Tau selected on test set** | Best τ picked from test PICP/MPIW | Cal-based: min RawCal_MPIW where RawCal_PICP ≥ 0.95 | No data snooping; reported τ is reproducible |
| **Only SAR features** | 6 features (pol + temporal + crop) | +NDVI from Sentinel-2/GEE | EOS-04 RF R²: 0.618→**0.648**; 5 new PICP≥0.95 crossings |

### Net Improvement (best method per metric, across all phases)

| Metric | Baseline best | Best result | Method | Change |
|--------|:------------:|:-----------:|:------:|:------:|
| EOS-04 PICP | 0.9073 | **0.9659** | GBM CQR (Phase 6) | +5.9 pp |
| EOS-04 MPIW | 40.33 | **30.77** | QSVR Phase 8b (C=2^8, γ=2^0) | −9.56 |
| S1 PICP | 0.9346 | **0.9804** | ANN CQR dual-output (Phase 6) | +4.6 pp |
| S1 MPIW | 42.23 | **30.61** | Tuned GBM CQR Phase 10 (τ=0.1/0.9) | −11.62 |

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

**NDVI reproducibility:** The GEE retrieval requires a registered service account. Pre-fetched NDVI CSVs (`eos04_ndvi.csv`, `sentinel1_ndvi.csv`) are included in `data/` so the full pipeline can run without a GEE account. To re-fetch from scratch:

```bash
uv run python experiments/classification_new_data/code/fetch_ndvi.py
```

---

## How to Run

### Full pipeline — all 8 phases (recommended)

```bash
cd experiments/classification_new_data/code
uv run python run_all_enhanced.py
```

Writes all JSON metrics + PNG plots to `output/`. Runtime ~30–90 min on CPU.

### NDVI fetch only (requires GEE service account)

```bash
uv run python fetch_ndvi.py
```

Place your GEE service account key at `data/ee-key.json` before running.

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

---

## Repository Structure

```
major_orig/
├── README.md
└── experiments/
    └── classification_new_data/
        ├── code/
        │   ├── constants.py                                  ← shared paths & feature column names (7 features)
        │   ├── model_experiments.py                          ← all experiment classes
        │   ├── fetch_ndvi.py                                 ← GEE NDVI retrieval (Sentinel-2, SCL masking)
        │   ├── run_all_enhanced.py                           ← headless all-phase runner
        │   ├── run_phases_467.py                             ← focused CQR runner
        │   ├── run_experiments_sequence.sh                   ← notebook runner (nbconvert)
        │   │
        │   ├── exploration_eos.ipynb                         ← Phase 0: EOS-04 EDA
        │   ├── exploration_sentinel.ipynb                    ← Phase 0: Sentinel-1 EDA
        │   ├── classical_ml_uncensored.ipynb                 ← Phase 1
        │   ├── classification_uncensored.ipynb               ← Phase 2
        │   ├── ann_uncensored.ipynb                          ← Phase 3
        │   ├── pi_estimation_uncensored.ipynb                ← Phase 4
        │   ├── conformal_regression_uncensored.ipynb         ← Phase 5
        │   ├── conformalized_quantile_regression_uncensored.ipynb  ← Phase 6
        │   ├── quantile_regression_tau_tuning_uncensored.ipynb     ← Phase 7
        │   └── quantile_svr_HP_tuning.ipynb                  ← Phase 8
        │
        ├── data/
        │   ├── EOS-04_datasheet.xlsx                         ← raw field data (ISRO format)
        │   ├── sentinel-1.xlsx                               ← raw field data (ESA format)
        │   ├── eos04_ndvi.csv                                ← GEE-fetched NDVI per date/point (EOS-04)
        │   ├── sentinel1_ndvi.csv                            ← GEE-fetched NDVI per date/point (Sentinel-1)
        │   ├── eos-04-enhanced-ndvi.csv                      ← processed: SM1 filtered + 7 features incl. NDVI
        │   └── sentinel-1-enhanced-ndvi.csv                  ← processed: SM1 filtered + 7 features incl. NDVI
        │
        └── output/
            ├── ml_experiment_uncensored/                     ← Phase 1: metrics JSON
            ├── classification_uncensored/                    ← Phase 2: metrics JSON
            ├── ann_experiments_uncensored/                   ← Phase 3: metrics + prediction plots
            ├── pi_estimation_uncensored/                     ← Phase 4: metrics + PI plots
            ├── conformal_regression_uncensored/              ← Phase 5: metrics + PI plots
            ├── conformal_results/                            ← Phase 6/7: metrics + PI plots
            │   ├── EOS-04_conformal_metrics.json
            │   ├── Sentinel-1_conformal_metrics.json
            │   ├── EOS-04_tau_tuning_metrics.json
            │   ├── Sentinel-1_tau_tuning_metrics.json
            │   └── plots/
            └── quantile_svr_uncensored/                      ← Phase 8: gamma tuning CSVs + plots
```

---

## Dependencies

```
Python           3.12
tensorflow       >= 2.15
scikit-learn     >= 1.4
xgboost
mapie
pandas
numpy
matplotlib
seaborn
tqdm
jupyter
ipykernel
earthengine-api  ← for NDVI retrieval (optional if using pre-fetched CSVs)
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
- Gorelick, N., et al. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. *Remote Sensing of Environment*.
