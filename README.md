# SAR Soil Moisture Estimation with Prediction Intervals

A machine learning research project for estimating surface soil moisture from SAR (Synthetic Aperture Radar) satellite data, with a focus on **calibrated prediction interval (PI) estimation** — producing uncertainty bounds alongside each prediction.

## Research Overview

**Satellites**: EOS-04 (C/X-band, HH+HV) and Sentinel-1 (C-band, VH+VV)  
**Target**: Surface soil moisture `SM1 (%)` across agricultural field plots in India  
**Key question**: Which uncertainty quantification method produces the tightest prediction intervals while maintaining ≥95% coverage probability (PICP)?

### Methods Compared

| Method | Approach |
|---|---|
| Quantile Regression ANN | Pinball loss with tunable τ lower/upper quantiles |
| Tube Loss ANN | Custom loss penalising predictions outside a confidence tube |
| Conformal Regression | Distribution-free coverage via MAPIE `ConformalizedQuantileRegressor` |
| Conformalized Quantile Regression (CQR) | Quantile ANN + conformal calibration step |
| Quantile SVR | Support vector quantile regression with 2D C×gamma hyperparameter sweep |

### Input Features (7 per satellite)

| Satellite | Features |
|---|---|
| EOS-04 | `HH-pol`, `HV-pol`, `NDVI`, `DpRVI`, `Depolarization_Rate`, `RFDI`, `RVI4S1` |
| Sentinel-1 | `VH-pol`, `VV-pol`, `NDVI`, `DpRVI`, `Depolarization_Rate`, `RFDI`, `RVI4S1` |

**NDVI** — Sentinel-2 NDVI extracted via Google Earth Engine, matched by `(Latitude, Longitude)` + acquisition date.  
**DpRVI** — Dual-pol Radar Vegetation Index (Mandal et al. 2020); derived from polarization ratio, range [0, 1].  
**Depolarization Rate** — Linear cross-pol / co-pol ratio `q`.  
**RFDI** — Radar Forest Degradation Index: `(1−q)/(1+q)`, range [−1, 1]; higher values indicate soil-dominant scattering.  
**RVI4S1** — Dual-pol vegetation index for Sentinel-1: `4q/(1+q)`, range [0, 1]; increases with vegetation volume scattering.

### Key Metrics

- **PICP** — Prediction Interval Coverage Probability: fraction of true values inside `[y_lower, y_upper]`. Target ≥ 0.95.
- **MPIW** — Mean Prediction Interval Width: average interval size. Lower is better given sufficient PICP.

---

## Environment

**Platform**: Ubuntu 26.04 · NVIDIA RTX 4060 · CUDA 12.4 · cuDNN 9  
**Python**: 3.12 via [`uv`](https://github.com/astral-sh/uv) (installed via `snap install astral-uv`)  
**GPU stack**: TensorFlow 2.21 + XGBoost 3.0 (`device='cuda'`) + LightGBM (`device='gpu'`) + CatBoost (`task_type='GPU'`)

### Setup

```bash
# 1. Install Python dependencies
/snap/bin/astral-uv.uv sync

# 2. Ensure cuDNN is on the library path (add to ~/.bashrc permanently)
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 3. Launch Jupyter
/snap/bin/astral-uv.uv run jupyter notebook
```

---

## Repository Layout

```
experiments/
  classification_new_data/
    code/
      model_experiments.py          # All experiment classes (central module)
      constants.py                  # Paths, feature columns (7 each), target column
      export_to_excel.py            # Export all metrics + plots → .xlsx workbooks
      run_all.ipynb                 # Master notebook — runs all 15 experiments in order
      add_ndvi.ipynb                # GEE pipeline: extract + merge Sentinel-2 NDVI
      exploration_eos.ipynb         # EDA + CSV generation for EOS-04 (DpRVI, RFDI, RVI4S1)
      exploration_sentinel.ipynb    # EDA + CSV generation for Sentinel-1 (DpRVI, RFDI, RVI4S1)
      ann_*.ipynb                   # ANN regression (censored / uncensored)
      classical_ml_*.ipynb          # RF, XGB, AdaBoost, SVR, LightGBM, CatBoost regression
      classification_*.ipynb        # 4-class SM classification
      pi_estimation_*.ipynb         # Quantile regression prediction intervals
      conformal_regression_*.ipynb  # Conformal PI estimation
      conformalized_quantile_regression_uncensored.ipynb
      quantile_regression_tau_tuning_uncensored.ipynb
      quantile_svr_HP_tuning.ipynb  # 2D C×gamma hyperparameter sweep
    data/
      EOS-04_datasheet.xlsx         # Raw measurements — 20 date sheets, all 7 features stored
      sentinel-1.xlsx               # Raw measurements — 14 date sheets, all 7 features stored
      eos-04-processed.csv          # Cleaned, 7-feature dataset (~1953 rows)
      sentinel-1-processed.csv      # Cleaned, 7-feature dataset (~1575 rows)
      ndvi_cache_eos.csv            # GEE NDVI cache (avoids re-fetching)
      ndvi_cache_sentinel.csv       # GEE NDVI cache
      ee-key.json                   # GEE service account key (not committed)
    output/                         # Generated plots + JSON metrics (not committed)
pyproject.toml
uv.lock
```

---

## Running Experiments

### Recommended: Jupyter Notebook (interactive)

```bash
/snap/bin/astral-uv.uv run jupyter notebook
```

Open `experiments/classification_new_data/code/run_all.ipynb` and run all cells. This executes all 15 notebooks in dependency order and prints a pass/fail/elapsed-time summary.

### Headless (CLI)

```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
/snap/bin/astral-uv.uv run python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  --ExecutePreprocessor.kernel_name=python3 \
  experiments/classification_new_data/code/run_all.ipynb
```

### Pipeline runner (from a specific stage)

```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
/snap/bin/astral-uv.uv run python experiments/classification_new_data/code/run_pipeline.py \
  --from ann_censored
```

### Single notebook

```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
/snap/bin/astral-uv.uv run python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  --ExecutePreprocessor.kernel_name=python3 \
  experiments/classification_new_data/code/ann_uncensored.ipynb
```

### Export all results to Excel

After running experiments, export every metric table and plot to `.xlsx`:

```bash
/snap/bin/astral-uv.uv run python experiments/classification_new_data/code/export_to_excel.py
```

Produces one workbook per experiment folder under `output/`:
- **Metrics sheet** — all JSON metrics flattened into a styled table (handles nested dicts and lists)
- **One sheet per plot** — PNG image embedded directly in the sheet

---

## Model Architecture (`model_experiments.py`)

All experiment classes inherit from `Experiment` (handles train/val/test splits):

| Class | Type | Tuning | GPU |
|---|---|---|---|
| `ClassificationExperiment` | RF, XGB, AdaBoost, SVC | `RandomizedSearchCV` | XGB only |
| `RegressionExperiment` | RF, XGB, AdaBoost, SVR, LightGBM, CatBoost | `RandomizedSearchCV` | XGB, LGBM, CatBoost |
| `ANNExperiment` | Keras MLP — MSE loss | — | Yes |
| `PredictionIntervalEstimation` | Quantile ANN — pinball loss | — | Yes |
| `TubeLossPredictionInterval` | Tube loss ANN | — | Yes |
| `ConformalRegression` | MAPIE + 4 quantile estimators | `RandomizedSearchCV` per estimator | LGBM only |
| `ConformalizedQuantileExperiment` | CQR + SVM split-conformal | — | ANN part only |

**ANN training settings:**
- `batch_size=64` — 20 gradient steps per epoch (better convergence than 256)
- `EarlyStopping(patience=50)` for `ANNExperiment`; `patience=35` for quantile/tube models
- `ReduceLROnPlateau(factor=0.3, patience=10, min_lr=1e-7)` — aggressive LR decay
- `mixed_float16` precision policy **removed** — float16 degraded regression accuracy on SM (%) target

---

## Feature Engineering

### NDVI (via Google Earth Engine)
Sentinel-2 NDVI = (B8 − B4) / (B8 + B4), extracted per field point by lat/lon + date.  
Cloud-covered dates filled with per-crop median, then global median.

### DpRVI — Dual-pol Radar Vegetation Index
*Mandal et al. 2020, Remote Sensing of Environment*

```python
q = 10 ** ((cross_dB - co_dB) / 10)   # linear cross/co ratio
DpRVI = q * (q + 3) / (q + 1) ** 2    # range [0, 1]
```

- EOS-04: cross = HV-pol, co = HH-pol  
- Sentinel-1: cross = VH-pol, co = VV-pol

### Depolarization Rate
```python
Depolarization_Rate = q   # same q — dimensionless linear ratio
```

### RFDI — Radar Forest Degradation Index
```python
RFDI = (1 - q) / (1 + q)   # range [-1, 1]
```
High RFDI → co-pol dominates → soil/bare scattering. Low RFDI → cross-pol dominates → vegetation volume.

### RVI4S1 — Dual-pol Vegetation Index for Sentinel-1
*Mandal et al. 2020*
```python
RVI4S1 = 4 * q / (1 + q)   # range [0, 1]
```
Increases with vegetation density. Complementary to DpRVI for linear/kernel models that cannot internally learn non-linear transforms of q.

---

## Data Notes

- **Censored vs uncensored**: SM values at detection threshold (SM = 50) kept in `*_censored` notebooks, dropped in `*_uncensored`.
- **Exploration notebooks regenerate CSVs**: Always run `exploration_eos` and `exploration_sentinel` first after any feature changes — they write the 7-feature processed CSVs used by all downstream notebooks.
- **Polarization values are in dB** (negative floats, e.g. −13.5, −19.4). DpRVI/RFDI/RVI4S1 computations convert to linear internally.
- **xlsx raw data**: RFDI and RVI4S1 are pre-computed and stored alongside DpRVI in all sheets of both workbooks.
