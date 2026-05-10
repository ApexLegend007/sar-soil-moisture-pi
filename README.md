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

### Input Features

| Satellite | Features |
|---|---|
| EOS-04 | `HH-pol`, `HV-pol`, `NDVI` |
| Sentinel-1 | `VH-pol`, `VV-pol`, `NDVI` |

NDVI is extracted from Sentinel-2 imagery via Google Earth Engine, matched per field point by `(Latitude, Longitude)` and acquisition date.

### Key Metrics

- **PICP** — Prediction Interval Coverage Probability: fraction of true values within `[y_lower, y_upper]`. Target ≥ 0.95.
- **MPIW** — Mean Prediction Interval Width: average interval size. Lower is better given sufficient PICP.

---

## Environment

**Platform**: Ubuntu (NVIDIA GPU — RTX 4060)  
**Python**: 3.12 via [`uv`](https://github.com/astral-sh/uv) (installed via `snap install astral-uv`)  
**GPU stack**: CUDA 12.4 + cuDNN 9 + TensorFlow 2.21 + XGBoost 3.0 (`device='cuda'`)

```bash
# Install dependencies
/snap/bin/astral-uv.uv sync

# Launch Jupyter
/snap/bin/astral-uv.uv run jupyter notebook

# Run all experiments end-to-end
/snap/bin/astral-uv.uv run python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  --ExecutePreprocessor.kernel_name=python3 \
  experiments/classification_new_data/code/run_all.ipynb
```

> **cuDNN path**: Add `export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH` to `~/.bashrc` if TensorFlow does not detect the GPU.

---

## Repository Layout

```
experiments/
  classification_new_data/
    code/
      model_experiments.py       # All experiment classes (central module)
      constants.py               # Paths, feature columns, target column
      export_to_excel.py         # Export all metrics + plots to .xlsx workbooks
      run_all.ipynb              # Master notebook — runs all 15 experiments in order
      add_ndvi.ipynb             # GEE pipeline to extract + merge Sentinel-2 NDVI
      exploration_eos.ipynb      # EDA + processed CSV generation for EOS-04
      exploration_sentinel.ipynb # EDA + processed CSV generation for Sentinel-1
      ann_*.ipynb                # ANN regression (censored / uncensored)
      classical_ml_*.ipynb       # RF, XGB, AdaBoost, SVR regression
      classification_*.ipynb     # 4-class SM classification
      pi_estimation_*.ipynb      # Quantile regression prediction intervals
      conformal_regression_*.ipynb
      conformalized_quantile_regression_uncensored.ipynb
      quantile_regression_tau_tuning_uncensored.ipynb
      quantile_svr_HP_tuning.ipynb
    data/
      EOS-04_datasheet.xlsx      # Raw field measurements (20 date sheets)
      sentinel-1.xlsx            # Raw field measurements (14 date sheets)
      eos-04-processed.csv       # Cleaned + feature-engineered (1953 rows)
      sentinel-1-processed.csv   # Cleaned + feature-engineered (1575 rows)
      ndvi_cache_eos.csv         # GEE NDVI cache (skip re-fetching)
      ndvi_cache_sentinel.csv    # GEE NDVI cache
      ee-key.json                # GEE service account key (not committed)
    output/                      # Generated plots + JSON metrics (not committed)
pyproject.toml
uv.lock
```

---

## Running Experiments

### Full pipeline

```bash
/snap/bin/astral-uv.uv run python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  --ExecutePreprocessor.kernel_name=python3 \
  experiments/classification_new_data/code/run_all.ipynb
```

`run_all.ipynb` executes all 15 notebooks in dependency order and prints a pass/fail/elapsed-time summary.

### Single notebook

```bash
/snap/bin/astral-uv.uv run python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  --ExecutePreprocessor.kernel_name=python3 \
  experiments/classification_new_data/code/ann_uncensored.ipynb
```

### Export results to Excel

```bash
/snap/bin/astral-uv.uv run python experiments/classification_new_data/code/export_to_excel.py
```

Produces one `.xlsx` workbook per experiment folder under `output/`, with a styled metrics table sheet and one sheet per embedded plot image.

---

## Model Architecture (`model_experiments.py`)

All experiment classes inherit from `Experiment` (base class handling train/val/test splits):

| Class | Type | GPU |
|---|---|---|
| `ClassificationExperiment` | RF, XGB, AdaBoost, SVC | XGB only |
| `RegressionExperiment` | RF, XGB, AdaBoost, SVR | XGB only |
| `ANNExperiment` | Keras MLP (MSE loss) | Yes |
| `PredictionIntervalEstimation` | Quantile ANN (pinball loss) | Yes |
| `TubeLossPredictionInterval` | Tube loss ANN | Yes |
| `ConformalRegression` | MAPIE conformal | No (sklearn) |
| `ConformalizedQuantileExperiment` | CQR + SVM split-conformal | ANN part only |

**Performance optimisations applied:**
- TF mixed precision (`float16`) — ~2× ANN training speed on RTX 4060
- ANN batch size 256 — better GPU utilisation
- `RandomForest n_jobs=-1` — all CPU cores
- XGBoost `device='cuda'`
- Quantile SVR gamma sweep parallelised via `joblib.Parallel`

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

- **Censored vs uncensored**: SM values at the detection threshold (SM = 50) are kept in `*_censored` notebooks and dropped in `*_uncensored` notebooks.
- **NDVI cloud fill**: Dates with heavy monsoon cloud cover have `NaN` NDVI filled with per-crop-type median, then global median.
- **Exploration notebooks regenerate CSVs**: Always run `exploration_eos` and `exploration_sentinel` before any ML notebook if the processed CSVs are stale.
