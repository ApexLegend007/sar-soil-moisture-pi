# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **machine learning research project** for soil moisture estimation using SAR (Synthetic Aperture Radar) satellite data (EOS-04 and Sentinel-1). The core research focus is on **prediction interval (PI) estimation** — producing not just a point prediction but a calibrated uncertainty range for each soil moisture estimate.

The project compares multiple uncertainty quantification approaches: quantile regression, tube loss, and conformal prediction. NDVI (from Sentinel-2 via Google Earth Engine) has been added as a third input feature alongside SAR polarization values.

## Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management (Python 3.12).

```bash
# Install dependencies and register constants + model_experiments as importable modules
uv sync

# Launch Jupyter (notebooks can now import constants/model_experiments from any working directory)
uv run jupyter notebook

# Run a Python script directly
uv run python experiments/classification_new_data/code/model_experiments.py

# Execute a single notebook programmatically
uv run python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  --ExecutePreprocessor.kernel_name=python3 \
  experiments/classification_new_data/code/<notebook>.ipynb
```

There is no build step, Makefile, or test suite — all experiments are driven by Jupyter notebooks.

## Repository Layout

```
experiments/
  classification_new_data/
    code/                        # model_experiments.py, constants.py, add_ndvi.ipynb, run_all.ipynb + 35+ notebooks
    data/
      sentinel-1-processed.csv   # includes NDVI column
      eos-04-processed.csv       # includes NDVI column
      EOS-04_datasheet.xlsx      # raw data with NDVI added per sheet (matched by lat/lon + date)
      sentinel-1.xlsx            # raw data with NDVI added per sheet (matched by lat/lon + date)
      ndvi_cache_eos.csv         # GEE extraction cache — skip re-fetching already-processed dates
      ndvi_cache_sentinel.csv    # GEE extraction cache
      ee-key.json                # GEE service account key (not committed to git)
    output/
```

## Core Module Architecture

### `classification_new_data/code/model_experiments.py`

The central module. All experiment classes inherit from `Experiment`, a base class that handles train/val/test splitting with configurable ratios. Each subclass accepts a `type='censored'/'uncensored'` parameter that controls the output subdirectory under `output/`.

| Class | Purpose |
|---|---|
| `Experiment` | Base class — flexible train/val/test splitting, data validation |
| `ClassificationExperiment` | RF, XGB, AdaBoost, SVC classification with ordinal encoding |
| `RegressionExperiment` | Classical ML regression (RF, XGB, AdaBoost, SVR) with grid search |
| `ANNExperiment` | Basic ANN regression (MSE loss) |
| `PredictionIntervalEstimation` | Quantile regression PI via pinball loss; tunable τ |
| `TubeLossPredictionInterval` | PI via custom tube loss; hyperparams `q`, `r`, `delta` |
| `ConformalRegression` | Conformal prediction using MAPIE `ConformalizedQuantileRegressor` |
| `ConformalizedQuantileExperiment` | Extends `PredictionIntervalEstimation`; adds CQR calibration, SVM split-conformal, and τ hyperparameter tuning |

All classes share the same interface: construct with `(X, y, satellite, ...)`, then call `.run_experiment(...)`. Metrics are saved as JSON and plots are saved to subdirectories under `output/`.

### `classification_new_data/code/constants.py`

Defines dataset paths, feature column names, and the target column. Paths are derived relative to the file's location using `Path(__file__).resolve().parent.parent`, so no manual updates are needed when cloning to a new machine.

```python
X_cols_eos      = ['HH-pol', 'HV-pol', 'NDVI']
X_cols_sentinel = ['VH-pol', 'VV-pol', 'NDVI']
y_col           = ['SM1 (%)']
```

## Data & Feature Schema

- **EOS-04 features**: `HH-pol`, `HV-pol`, `NDVI`
- **Sentinel-1 features**: `VH-pol`, `VV-pol`, `NDVI`
- **Target**: `SM1 (%)` — surface soil moisture percentage
- **NDVI**: Sentinel-2 NDVI extracted via Google Earth Engine, matched per field point using `(Latitude, Longitude)` coordinates and acquisition date. Dates with heavy cloud cover (monsoon months) have `NaN` NDVI filled with per-crop-type median.

## NDVI Pipeline (`add_ndvi.ipynb`)

Extracts Sentinel-2 NDVI for all field measurement points and merges it into the dataset.

**Authentication**: Uses a GEE service account key at `data/ee-key.json` with project `sharp-weft-236811`.

```python
credentials = ee.ServiceAccountCredentials(email=sa_email, key_file='data/ee-key.json')
ee.Initialize(credentials=credentials, project='sharp-weft-236811')
```

**Flow**:
1. Load raw xlsx files (lat/lon/date per row)
2. GEE: one `sampleRegions` call per acquisition date (~34 total) — results cached to `ndvi_cache_eos.csv` / `ndvi_cache_sentinel.csv`
3. Merge into processed CSVs joined on `(Latitude, Longitude)` per sheet
4. Overwrite original xlsx files in-place (NDVI as last column, matched by lat/lon per sheet)
5. Update `constants.py` to add `'NDVI'` to both `X_cols_*` lists

**IAM requirement**: The service account needs `roles/serviceusage.serviceUsageConsumer` on the GCP project. Grant at: `console.developers.google.com/iam-admin/iam?project=sharp-weft-236811`

**Re-running safely**: The cache files skip already-processed dates, so interruptions are safe.

## Master Notebook (`run_all.ipynb`)

Runs all 15 experiment notebooks in correct order via `nbconvert --execute --inplace`. Prints a pass/fail/skip summary table with elapsed times. Exploration notebooks are skipped automatically if the raw xlsx files are absent.

**Execution order**:
```
exploration_eos → exploration_sentinel →
classification_censored → classification_uncensored →
classical_ml_censored → classical_ml_uncensored →
ann_censored → ann_uncensored →
pi_estimation_censored → pi_estimation_uncensored →
conformal_regression_censored → conformal_regression_uncensored →
conformalized_quantile_regression_uncensored →
quantile_regression_tau_tuning_uncensored →
quantile_svr_HP_tuning
```

⚠️ The exploration notebooks regenerate the processed CSVs — they must run before any ML stage so the NDVI column is present.

## Key ML Concepts in Use

- **Prediction Intervals (PI)**: Bounds `[y_lower, y_upper]` around a point estimate.
- **PICP** (Prediction Interval Coverage Probability): Fraction of true values falling within the PI — target ≥ 0.95.
- **MPIW** (Mean Prediction Interval Width): Average PI width — lower is better given adequate PICP.
- **Conformal Prediction** (via MAPIE): Distribution-free coverage guarantees.
- **Tube loss**: Custom loss that penalises predictions outside a confidence tube; controlled by `q` (target coverage), `r` (tube movement), and `delta` (recalibration penalty).
- **Censored data**: Some SM values are below detection threshold (SM = 50); censored notebooks retain these rows, uncensored notebooks drop them.

## NDVI Impact on Model Performance (with vs without NDVI)

Adding NDVI as a 3rd feature improved all models:

| Stage | Metric | EOS-04 improvement | Sentinel-1 improvement |
|---|---|---|---|
| Classical ML | R² | +0.14 to +0.15 | +0.12 to +0.13 |
| Classical ML | MAE | −2.1 units | −1.2 units |
| ANN | R² | +0.086 to +0.089 | +0.081 to +0.131 |
| Quantile ANN PI | MPIW | −9 to −10 (EOS) | minimal change |
| Conformal GBR | MPIW | −6 to −8 units tighter | −2 units tighter |

## Notebook Conventions

Notebooks follow the naming pattern `{method}_{censored|uncensored}.ipynb` (e.g., `ann_censored.ipynb`, `conformal_regression_uncensored.ipynb`).

ANN model architectures use `n_features = X_eos.shape[1]` (dynamically set) for the Keras `Input(shape=(n_features,))` layer — do not hardcode `shape=(2,)` as the feature count is now 3.

## ⏳ Next Task — GPU Migration on Ubuntu 26.04

The project is being migrated from Windows (CPU-only) to **Ubuntu 26.04** where full NVIDIA GPU support is available via CUDA. The current `pyproject.toml` uses `tensorflow-cpu` and `xgboost-cpu` — both must be swapped to their GPU variants.

### Step 1 — Install system dependencies

```bash
# NVIDIA driver (check your GPU model first)
ubuntu-drivers autoinstall

# CUDA 12.x toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt install -y cuda-toolkit-12-6

# cuDNN 9 (for TensorFlow 2.x)
sudo apt install -y libcudnn9-cuda-12

# Verify
nvidia-smi
nvcc --version
```

### Step 2 — Install `uv` on Ubuntu

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### Step 3 — Update `pyproject.toml`

Replace CPU-only packages with GPU versions:

```toml
# Remove:
"tensorflow-cpu>=2.20.0"
"xgboost-cpu==3.0.0"

# Add:
"tensorflow>=2.20.0"
"xgboost==3.0.0"
```

Then sync:

```bash
uv sync
```

### Step 4 — Verify GPU is detected

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  # should show your GPU

import xgboost as xgb
# XGBoost GPU: pass device='cuda' in model params
```

### Step 5 — Enable GPU memory growth (optional but recommended)

Add to the top of `model_experiments.py` or any notebook before TF imports:

```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Step 6 — Enable XGBoost GPU in `model_experiments.py`

In `RegressionExperiment`, the XGBoost param grid should add `device='cuda'`:

```python
'XGBoost': {
    'model': XGBRegressor(device='cuda', ...),
    ...
}
```

### Step 7 — Re-run `run_all.ipynb`

```bash
uv run python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  --ExecutePreprocessor.kernel_name=python3 \
  experiments/classification_new_data/code/run_all.ipynb
```

### CUDA ↔ TensorFlow compatibility

| TensorFlow | CUDA | cuDNN |
|---|---|---|
| 2.18 – 2.21 | 12.x | 9.x |
| 2.13 – 2.17 | 11.8 | 8.6 |

Use TF 2.21 (current) with CUDA 12.6 + cuDNN 9.

---

## Gotchas

- Every experiment class prints `Results → <path>` on construction so you can verify the `type` parameter routed output to the right folder before training starts.
- `ConformalizedQuantileExperiment` overrides `results_path` after calling `super().__init__`, so the parent's print fires first (showing `pi_estimation_<type>`) and is immediately superseded by the child's print (`conformal_results_<type>`). The second path is the one that's actually used.
- **ANN notebooks**: All Keras `Input(shape=...)` layers must use `n_features = X_eos.shape[1]` — not the hardcoded value `2`. This was fixed in all notebooks after NDVI was added as a 3rd feature.
- **Exploration notebooks write processed CSVs**: `exploration_eos.ipynb` and `exploration_sentinel.ipynb` regenerate `eos-04-processed.csv` and `sentinel-1-processed.csv`. They include NDVI from the raw xlsx (which has NDVI per sheet). If you add new features, update the `save_cols` list in the `to_csv` cell of both exploration notebooks.
- **`pi_estimation_uncensored.ipynb`**: Contains hardcoded `OUTPUT_PATH / "pi_estimation_uncensored"` path in the JSON-saving cells — already fixed but watch for regressions.
- **`quantile_svr_HP_tuning.ipynb`**: Data loading uses `DATA_PATH` from `constants` — do not revert to relative `data/` paths.
- **GPU on Windows (deprecated)**: TensorFlow ≥ 2.11 has no native Windows GPU support; DirectML plugin requires Python ≤ 3.10 (incompatible with this project). Project has moved to Ubuntu 26.04 for GPU support.
