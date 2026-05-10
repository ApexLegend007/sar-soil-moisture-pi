# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Machine learning research project for **soil moisture estimation** using SAR satellite data (EOS-04 and Sentinel-1). Core research focus: **prediction interval (PI) estimation** — calibrated uncertainty ranges alongside each soil moisture estimate.

Methods compared: quantile regression, tube loss, conformal prediction. Input features: SAR polarization values + NDVI (Sentinel-2 via GEE) + DpRVI + Depolarization Rate.

## Environment

**Platform**: Ubuntu 26.04 · NVIDIA RTX 4060 · CUDA 12.4 · cuDNN 9  
**Python**: 3.12 · `uv` installed via `snap install astral-uv`  
**uv binary**: `/snap/bin/astral-uv.uv` (not plain `uv` — not on PATH)

```bash
# Install dependencies
/snap/bin/astral-uv.uv sync

# Launch Jupyter (preferred way to run experiments)
/snap/bin/astral-uv.uv run jupyter notebook

# Headless notebook execution
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
/snap/bin/astral-uv.uv run python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  --ExecutePreprocessor.kernel_name=python3 \
  experiments/classification_new_data/code/<notebook>.ipynb
```

> **cuDNN path**: Libraries live in `/usr/lib/x86_64-linux-gnu/`. The export is in `~/.bashrc` but always prefix headless runs with `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH` to be safe.

## Repository Layout

```
experiments/
  classification_new_data/
    code/
      model_experiments.py       # Central module — all experiment classes
      constants.py               # Paths, X_cols (5 features each), y_col
      export_to_excel.py         # Export metrics + plots → one .xlsx per output folder
      run_all.ipynb              # Master: runs all 15 notebooks in order
      add_ndvi.ipynb             # GEE pipeline for Sentinel-2 NDVI extraction
      exploration_eos.ipynb      # EDA + CSV generation (computes DpRVI + Depolarization_Rate)
      exploration_sentinel.ipynb # Same for Sentinel-1
      ann_*.ipynb                # ANN regression (censored / uncensored)
      classical_ml_*.ipynb       # RF, XGB, AdaBoost, SVR
      classification_*.ipynb     # 4-class SM label classification
      pi_estimation_*.ipynb      # Quantile regression PI
      conformal_regression_*.ipynb
      conformalized_quantile_regression_uncensored.ipynb
      quantile_regression_tau_tuning_uncensored.ipynb
      quantile_svr_HP_tuning.ipynb
    data/
      EOS-04_datasheet.xlsx      # Raw data — 20 date sheets, DpRVI column present
      sentinel-1.xlsx            # Raw data — 14 date sheets, DpRVI column present
      eos-04-processed.csv       # 5-feature processed dataset (1953 rows)
      sentinel-1-processed.csv   # 5-feature processed dataset (1575 rows)
      ndvi_cache_eos.csv         # GEE cache — do not delete
      ndvi_cache_sentinel.csv    # GEE cache — do not delete
      ee-key.json                # GEE service account key (gitignored)
    output/                      # Generated plots + JSONs (gitignored)
```

## Core Module — `model_experiments.py`

All classes inherit from `Experiment` (train/val/test split base). Each accepts `type='censored'/'uncensored'` which controls the output subdirectory.

| Class | Purpose |
|---|---|
| `Experiment` | Base — configurable splits, data validation |
| `ClassificationExperiment` | RF, XGB, AdaBoost, SVC classification |
| `RegressionExperiment` | RF, XGB, AdaBoost, SVR with GridSearchCV |
| `ANNExperiment` | Keras MLP — MSE loss |
| `PredictionIntervalEstimation` | Quantile ANN — pinball loss, tunable τ |
| `TubeLossPredictionInterval` | Tube loss ANN; params `q`, `r`, `delta` |
| `ConformalRegression` | MAPIE `ConformalizedQuantileRegressor` |
| `ConformalizedQuantileExperiment` | CQR + SVM split-conformal + τ tuning |

Interface: `__init__(X, y, satellite, ...)` → `.run_experiment(...)`. Metrics → JSON, plots → `output/<folder>/plots/`.

## `constants.py`

```python
X_cols_eos      = ['HH-pol', 'HV-pol', 'NDVI', 'DpRVI', 'Depolarization_Rate']
X_cols_sentinel = ['VH-pol', 'VV-pol', 'NDVI', 'DpRVI', 'Depolarization_Rate']
y_col           = ['SM1 (%)']
```

Paths derived from `Path(__file__).resolve().parent.parent` — no manual edits needed on new machines.

## `export_to_excel.py`

Standalone script. For each folder under `output/`, produces `<folder>_results.xlsx` with:
- **Metrics sheet**: all JSON metrics flattened into a styled table
- **One sheet per PNG plot**: image embedded directly

```bash
/snap/bin/astral-uv.uv run python experiments/classification_new_data/code/export_to_excel.py
```

## Data & Feature Schema

| Feature | Satellite | Formula / Source |
|---|---|---|
| `HH-pol` | EOS-04 | Raw SAR backscatter (dB) |
| `HV-pol` | EOS-04 | Raw SAR backscatter (dB) |
| `VH-pol` | Sentinel-1 | Raw SAR backscatter (dB) |
| `VV-pol` | Sentinel-1 | Raw SAR backscatter (dB) |
| `NDVI` | Both | Sentinel-2 via GEE, cloud-fill with crop median |
| `DpRVI` | Both | `q*(q+3)/(q+1)²`, q = 10^((cross−co)/10), range [0,1] |
| `Depolarization_Rate` | Both | `q` — linear cross/co ratio |

**All pol values are in dB** (negative floats). DpRVI/Depolarization_Rate convert to linear internally.

Target: `SM1 (%)` — surface soil moisture percentage.

## GPU Configuration (fully applied — do not revert)

| Setting | Location | Value |
|---|---|---|
| TF package | `pyproject.toml` | `tensorflow>=2.20.0` (not `-cpu`) |
| XGBoost package | `pyproject.toml` | `xgboost==3.0.0` (not `-cpu`) |
| GPU memory growth | `model_experiments.py` top | `set_memory_growth(gpu, True)` |
| Mixed precision | `model_experiments.py` top | `set_global_policy('mixed_float16')` |
| XGBoost device | `RegressionExperiment` | `XGBRegressor(device='cuda')` |
| RF parallelism | Both RF classes | `n_jobs=-1` |
| ANN batch size | All TF classes | `batch_size=256` (default) |
| SVR gamma loop | `quantile_svr_HP_tuning.ipynb` | `joblib.Parallel(n_jobs=-1)` |

### GPU vs CPU breakdown

| Models | GPU | Why |
|---|---|---|
| TF/Keras ANNs | Yes | TF 2.21 + CUDA 12.4 |
| XGBoost | Yes | `device='cuda'` |
| RandomForest, AdaBoost, SVR, GBR | No | scikit-learn is CPU-only |
| Quantile SVR (cvxopt) | No | Custom QP solver, no GPU support |

## NDVI Pipeline (`add_ndvi.ipynb`)

GEE service account key: `data/ee-key.json`, project: `sharp-weft-236811`.

```python
credentials = ee.ServiceAccountCredentials(email=sa_email, key_file='data/ee-key.json')
ee.Initialize(credentials=credentials, project='sharp-weft-236811')
```

Flow: load xlsx → GEE `sampleRegions` per date (~34 calls, cached) → merge by lat/lon → overwrite xlsx in-place (NDVI as last column). Cache files prevent redundant GEE calls on re-run.

IAM: service account needs `roles/serviceusage.serviceUsageConsumer` on `sharp-weft-236811`.

## DpRVI + Depolarization Rate (implemented)

Computed in `exploration_eos.ipynb` / `exploration_sentinel.ipynb` from existing dB pol columns:

```python
# EOS-04
q = 10 ** ((df['HV-pol'] - df['HH-pol']) / 10)
# Sentinel-1
q = 10 ** ((df['VH-pol'] - df['VV-pol']) / 10)

df['DpRVI'] = q * (q + 3) / (q + 1) ** 2   # Mandal et al. 2020
df['Depolarization_Rate'] = q
```

Also pre-computed and stored in all xlsx sheets. After any change to features, re-run exploration notebooks to regenerate processed CSVs before running ML notebooks.

## Master Notebook (`run_all.ipynb`)

Runs 15 notebooks via `nbconvert --execute --inplace`. Prints pass/fail/skip table with elapsed times.

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

⚠️ Exploration notebooks must run first — they write the processed CSVs all downstream notebooks read.

## Key Concepts

- **PICP**: Fraction of true values inside `[y_lower, y_upper]` — target ≥ 0.95.
- **MPIW**: Mean interval width — lower is better given PICP ≥ 0.95.
- **Conformal prediction** (MAPIE): distribution-free coverage guarantee.
- **Tube loss**: custom loss with params `q` (target coverage), `r` (tube movement), `delta` (recalibration penalty).
- **Censored**: SM = 50 rows retained. **Uncensored**: dropped.

## Gotchas

- Every experiment class prints `Results → <path>` on construction — verify `type` is correct before training.
- `ConformalizedQuantileExperiment` prints two paths on init — the second one (`conformal_results_<type>`) is the one actually used.
- **ANN `Input(shape=...)`**: always use `n_features = X.shape[1]` — never hardcode a number since feature count changed (2 → 3 → 5).
- **Exploration notebooks write CSVs**: if you add features, update `save_cols` in the `to_csv` cell of both exploration notebooks.
- **cuDNN not found**: prefix command with `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH` or source `~/.bashrc`.
- **`quantile_svr_HP_tuning.ipynb`**: uses `DATA_PATH` from `constants` — do not revert to relative paths.
- **`pi_estimation_uncensored.ipynb`**: had hardcoded output path — fixed, watch for regressions.

## ⚠️ OPEN ISSUES — Resume After Restart

### 1. cuDNN version mismatch (BLOCKING — all GPU/ANN notebooks fail)

**Error**: `Loaded runtime CuDNN library: 9.0.0 but source was compiled with: 9.3.0`  
**Cause**: TF 2.21 was compiled against cuDNN 9.3.0; system has 9.0.0 only.  
**Fix needed** (choose one after restart):
- **Option A — Downgrade TF**: change `pyproject.toml` to `tensorflow==2.18.0` (compiled against cuDNN 9.0) and run `/snap/bin/astral-uv.uv sync`
- **Option B — Upgrade cuDNN**: install cuDNN 9.3 from NVIDIA's repo (requires adding nvidia apt source)
- **Option C — Force CPU for ANN**: set `CUDA_VISIBLE_DEVICES=""` in run_notebook() env for GPU-only notebooks (loses GPU speed)

After fix, re-run pipeline from `ann_censored`:
```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
/snap/bin/astral-uv.uv run python experiments/classification_new_data/code/run_pipeline.py \
--from ann_censored
```

### 2. MAPIE `alpha_name` ValueError (conformal_regression notebooks 11–12)

**Error**: `ValueError: The matching parameter 'alpha_name' for estimator does not...`  
**Cause**: MAPIE API changed — `alpha_name` parameter renamed or removed in newer MAPIE version.  
**Fix needed**: Check `conformal_regression_censored.ipynb` and `conformal_regression_uncensored.ipynb` — find where `alpha_name` is passed and update to current MAPIE API.

### Current output state (after last run)

Notebooks **1–6 have results** (classification + classical ML):
- `output/classification_censored/` ✓ metrics_EOS-04.json, metrics_Sentinel-1.json
- `output/classification_uncensored/` ✓
- `output/ml_experiment_censored/` ✓
- `output/ml_experiment_uncensored/` ✓

Notebooks **7–15 have NO results** (all failed with cuDNN error or MAPIE error).

### Pipeline fixes already applied (do NOT revert)

- `run_pipeline.py`: non-interactive stdin auto-continue (`sys.stdin.isatty()` check)
- `model_experiments.py`: `try/except RuntimeError` around `set_memory_growth` (handles TF pre-initialized by notebook)
- `model_experiments.py`: `os.makedirs(self.results_path, exist_ok=True)` in all `__init__` methods
- `model_experiments.py`: `fit_grid_search` yellowbrick removed; only `make_plot` called
- `model_experiments.py`: CQR finite-sample correction `(1-α)(1+1/n)` quantile
