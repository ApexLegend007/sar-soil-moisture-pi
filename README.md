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
| Quantile SVR | Support vector quantile regression with hyperparameter sweep |

### Input Features (5 per satellite)

| Satellite | Features |
|---|---|
| EOS-04 | `HH-pol`, `HV-pol`, `NDVI`, `DpRVI`, `Depolarization_Rate` |
| Sentinel-1 | `VH-pol`, `VV-pol`, `NDVI`, `DpRVI`, `Depolarization_Rate` |

**NDVI** — Sentinel-2 NDVI extracted via Google Earth Engine, matched by `(Latitude, Longitude)` + acquisition date.  
**DpRVI** — Dual-pol Radar Vegetation Index (Mandal et al. 2020); derived from polarization ratio, range [0, 1].  
**Depolarization Rate** — Linear cross-pol / co-pol ratio; measures canopy/soil depolarization effect.

### Key Metrics

- **PICP** — Prediction Interval Coverage Probability: fraction of true values inside `[y_lower, y_upper]`. Target ≥ 0.95.
- **MPIW** — Mean Prediction Interval Width: average interval size. Lower is better given sufficient PICP.

---

## Environment

**Platform**: Ubuntu 26.04 · NVIDIA RTX 4060 · CUDA 12.4 · cuDNN 9  
**Python**: 3.12 via [`uv`](https://github.com/astral-sh/uv) (installed via `snap install astral-uv`)  
**GPU stack**: TensorFlow 2.21 + XGBoost 3.0 (`device='cuda'`)

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
      constants.py                  # Paths, feature columns (5 each), target column
      export_to_excel.py            # Export all metrics + plots → .xlsx workbooks
      run_all.ipynb                 # Master notebook — runs all 15 experiments in order
      add_ndvi.ipynb                # GEE pipeline: extract + merge Sentinel-2 NDVI
      exploration_eos.ipynb         # EDA + CSV generation for EOS-04 (computes DpRVI)
      exploration_sentinel.ipynb    # EDA + CSV generation for Sentinel-1 (computes DpRVI)
      ann_*.ipynb                   # ANN regression (censored / uncensored)
      classical_ml_*.ipynb          # RF, XGB, AdaBoost, SVR regression
      classification_*.ipynb        # 4-class SM classification
      pi_estimation_*.ipynb         # Quantile regression prediction intervals
      conformal_regression_*.ipynb  # Conformal PI estimation
      conformalized_quantile_regression_uncensored.ipynb
      quantile_regression_tau_tuning_uncensored.ipynb
      quantile_svr_HP_tuning.ipynb
    data/
      EOS-04_datasheet.xlsx         # Raw measurements — 20 date sheets, includes DpRVI
      sentinel-1.xlsx               # Raw measurements — 14 date sheets, includes DpRVI
      eos-04-processed.csv          # Cleaned, 5-feature dataset (1953 rows)
      sentinel-1-processed.csv      # Cleaned, 5-feature dataset (1575 rows)
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
- **Metrics sheet** — all JSON metrics flattened into a styled table (source file · metric path · value)
- **One sheet per plot** — PNG image embedded directly in the sheet

---

## Model Architecture (`model_experiments.py`)

All experiment classes inherit from `Experiment` (handles train/val/test splits):

| Class | Type | GPU |
|---|---|---|
| `ClassificationExperiment` | RF, XGB, AdaBoost, SVC | XGB only |
| `RegressionExperiment` | RF, XGB, AdaBoost, SVR | XGB only |
| `ANNExperiment` | Keras MLP — MSE loss | Yes |
| `PredictionIntervalEstimation` | Quantile ANN — pinball loss | Yes |
| `TubeLossPredictionInterval` | Tube loss ANN | Yes |
| `ConformalRegression` | MAPIE conformal | No (sklearn) |
| `ConformalizedQuantileExperiment` | CQR + SVM split-conformal | ANN part only |

**Performance optimisations applied:**
- TF mixed precision `float16` — ~2× ANN training speed on RTX 4060
- ANN default `batch_size=256` — better GPU utilisation
- `RandomForest n_jobs=-1` — all CPU cores
- `XGBRegressor(device='cuda')`
- Quantile SVR gamma sweep parallelised via `joblib.Parallel(n_jobs=-1)`

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
Depolarization_Rate = q   # same q as above — dimensionless linear ratio
```

---

## NDVI Impact (adding Sentinel-2 NDVI as 3rd feature)

| Stage | Metric | EOS-04 | Sentinel-1 |
|---|---|---|---|
| Classical ML | R² | +0.14–0.15 | +0.12–0.13 |
| Classical ML | MAE | −2.1 units | −1.2 units |
| ANN | R² | +0.086–0.089 | +0.081–0.131 |
| Quantile ANN PI | MPIW | −9 to −10 units | minimal |
| Conformal GBR | MPIW | −6 to −8 units | −2 units |

---

## Data Notes

- **Censored vs uncensored**: SM values at detection threshold (SM = 50) kept in `*_censored` notebooks, dropped in `*_uncensored`.
- **Exploration notebooks regenerate CSVs**: Always run `exploration_eos` and `exploration_sentinel` first after any feature changes — they write the processed CSVs used by all downstream notebooks.
- **Polarization values are in dB** (negative floats, e.g. −13.5, −19.4). DpRVI computation converts to linear internally.
