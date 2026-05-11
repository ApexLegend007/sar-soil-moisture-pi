# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Machine learning research project for **soil moisture estimation** using SAR satellite data (EOS-04 and Sentinel-1). Core research focus: **prediction interval (PI) estimation** — calibrated uncertainty ranges alongside each soil moisture estimate.

Methods compared: quantile regression, tube loss, conformal prediction. Input features: SAR polarization values + NDVI (Sentinel-2 via GEE) + DpRVI + Depolarization Rate + RFDI + RVI4S1.

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
      constants.py               # Paths, X_cols (7 features each), y_col
      export_to_excel.py         # Export metrics + plots → one .xlsx per output folder
      run_all.ipynb              # Master: runs all 15 notebooks in order
      add_ndvi.ipynb             # GEE pipeline for Sentinel-2 NDVI extraction
      exploration_eos.ipynb      # EDA + CSV generation (computes DpRVI, RFDI, RVI4S1)
      exploration_sentinel.ipynb # Same for Sentinel-1
      ann_*.ipynb                # ANN regression (censored / uncensored)
      classical_ml_*.ipynb       # RF, XGB, AdaBoost, SVR, LightGBM, CatBoost
      classification_*.ipynb     # 4-class SM label classification
      pi_estimation_*.ipynb      # Quantile regression PI
      conformal_regression_*.ipynb
      conformalized_quantile_regression_uncensored.ipynb
      quantile_regression_tau_tuning_uncensored.ipynb
      quantile_svr_HP_tuning.ipynb
    data/
      EOS-04_datasheet.xlsx      # Raw data — 20 date sheets, all derived features present
      sentinel-1.xlsx            # Raw data — 14 date sheets, all derived features present
      eos-04-processed.csv       # 7-feature processed dataset (~1953 rows)
      sentinel-1-processed.csv   # 7-feature processed dataset (~1575 rows)
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
| `ClassificationExperiment` | RF, XGB, AdaBoost, SVC — all tuned via `RandomizedSearchCV` |
| `RegressionExperiment` | RF, XGB, AdaBoost, SVR, LightGBM, CatBoost — all tuned via `RandomizedSearchCV` |
| `ANNExperiment` | Keras MLP — MSE loss |
| `PredictionIntervalEstimation` | Quantile ANN — pinball loss, tunable τ |
| `TubeLossPredictionInterval` | Tube loss ANN; params `q`, `r`, `delta` |
| `ConformalRegression` | MAPIE `ConformalizedQuantileRegressor` — hyperparameter-tuned estimators |
| `ConformalizedQuantileExperiment` | CQR + SVM split-conformal + τ tuning |

Interface: `__init__(X, y, satellite, ...)` → `.run_experiment(...)`. Metrics → JSON, plots → `output/<folder>/plots/`.

## `constants.py`

```python
X_cols_eos      = ['HH-pol', 'HV-pol', 'NDVI', 'DpRVI', 'Depolarization_Rate', 'RFDI', 'RVI4S1']
X_cols_sentinel = ['VH-pol', 'VV-pol', 'NDVI', 'DpRVI', 'Depolarization_Rate', 'RFDI', 'RVI4S1']
y_col           = ['SM1 (%)']
```

Paths derived from `Path(__file__).resolve().parent.parent` — no manual edits needed on new machines.

## `export_to_excel.py`

Standalone script. For each folder under `output/`, produces `<folder>_results.xlsx` with:
- **Metrics sheet**: all JSON metrics flattened into a styled table (handles nested dicts and lists)
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
| `RFDI` | Both | `(1−q)/(1+q)`, range [−1, 1] — soil scattering dominance |
| `RVI4S1` | Both | `4q/(1+q)`, range [0, 1] — dual-pol vegetation index (Mandal 2020) |

**All pol values are in dB** (negative floats). DpRVI/RFDI/RVI4S1 derive from the same `q` computed in exploration notebooks.

Target: `SM1 (%)` — surface soil moisture percentage.

## GPU Configuration (fully applied — do not revert)

| Setting | Location | Value |
|---|---|---|
| TF package | `pyproject.toml` | `tensorflow>=2.20.0` (not `-cpu`) |
| XGBoost package | `pyproject.toml` | `xgboost==3.0.0` (not `-cpu`) |
| CatBoost package | `pyproject.toml` | `catboost>=1.2` |
| GPU memory growth | `model_experiments.py` top | `set_memory_growth(gpu, True)` in try/except |
| XGBoost device | `RegressionExperiment` | `XGBRegressor(device='cuda')` |
| LightGBM device | `RegressionExperiment` | `LGBMRegressor(device='gpu')` |
| CatBoost device | `RegressionExperiment` | `CatBoostRegressor(task_type='GPU')` |
| RF parallelism | Both RF classes | `n_jobs=-1` |
| ANN batch size | All TF classes | `batch_size=64` (default) |

> **Note**: `mixed_float16` was removed — float16 degrades regression accuracy on small-range targets like SM (%).

### GPU vs CPU breakdown

| Models | GPU | Why |
|---|---|---|
| TF/Keras ANNs | Yes | TF 2.21 + CUDA 12.4 |
| XGBoost | Yes | `device='cuda'` |
| LightGBM | Yes | `device='gpu'` |
| CatBoost | Yes | `task_type='GPU'` |
| RandomForest, AdaBoost, SVR, GBR, QuantileRegressor | No | scikit-learn is CPU-only |
| Quantile SVR (cvxopt) | No | Custom QP solver, no GPU support |

## Hyperparameter Tuning — Applied Everywhere

All sklearn models use `RandomizedSearchCV(n_iter=50, cv=5, scoring='neg_mean_absolute_error', random_state=42)`:

- **`RegressionExperiment`**: RF, XGB, AdaBoost, SVR, LightGBM, CatBoost — broad param grids
- **`ClassificationExperiment`**: RF, XGB, AdaBoost, SVC — `scoring='accuracy'`
- **`ConformalRegression`**: each quantile estimator tuned on training data only (conf set kept clean); best params passed as unfitted clone to MAPIE with `prefit=False`

## ANN Training Settings

| Parameter | ANNExperiment | PredictionIntervalEstimation | TubeLoss | run_ann_cqr |
|---|---|---|---|---|
| epochs (default) | 300 | 300 | 300 | 250 |
| batch_size | 64 | 64 | 64 | 64 |
| EarlyStopping patience | 50 | 35 | 35 | — |
| ReduceLR factor | 0.3 | 0.3 | 0.3 | — |
| ReduceLR patience | 10 | 10 | 10 | — |
| ReduceLR min_lr | 1e-7 | 1e-7 | 1e-7 | — |
| learning_rate | via optimizer | 0.0001 (notebooks pass Adam) | via optimizer | 0.0005 |

`run_ann_tuning_experiment` default: `epochs=250, learning_rate=0.0005`.

## NDVI Pipeline (`add_ndvi.ipynb`)

GEE service account key: `data/ee-key.json`, project: `sharp-weft-236811`.

```python
credentials = ee.ServiceAccountCredentials(email=sa_email, key_file='data/ee-key.json')
ee.Initialize(credentials=credentials, project='sharp-weft-236811')
```

Flow: load xlsx → GEE `sampleRegions` per date (~34 calls, cached) → merge by lat/lon → overwrite xlsx in-place (NDVI as last column). Cache files prevent redundant GEE calls on re-run.

IAM: service account needs `roles/serviceusage.serviceUsageConsumer` on `sharp-weft-236811`.

## Feature Derivation (all implemented)

Computed in `exploration_eos.ipynb` / `exploration_sentinel.ipynb` from existing dB pol columns:

```python
# EOS-04: cross=HV-pol, co=HH-pol
q = 10 ** ((df['HV-pol'] - df['HH-pol']) / 10)
# Sentinel-1: cross=VH-pol, co=VV-pol
q = 10 ** ((df['VH-pol'] - df['VV-pol']) / 10)

df['DpRVI']              = q * (q + 3) / (q + 1) ** 2   # Mandal et al. 2020
df['Depolarization_Rate'] = q
df['RFDI']               = (1 - q) / (1 + q)             # Radar Forest Degradation Index
df['RVI4S1']             = 4 * q / (1 + q)               # RVI for Sentinel-1
```

All 7 derived features are also stored in the xlsx raw data sheets. After any change to features, re-run exploration notebooks to regenerate processed CSVs before running ML notebooks.

## `quantile_svr_HP_tuning.ipynb`

Searches over a 2D grid of C × gamma values using the custom cvxopt QP solver:

```python
C_VALUES     = [2**i for i in range(1, 10, 2)]   # [2, 8, 32, 128, 512]
GAMMA_VALUES = [2**i for i in range(-15, 16)]     # 31 values, 2^-15 to 2^15
```

Total: 5 × 31 = 155 combinations per satellite. Results saved as `tuning_summary_2D.csv`.

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

⚠️ Exploration notebooks must run first — they write the 7-feature processed CSVs all downstream notebooks read.

## Key Concepts

- **PICP**: Fraction of true values inside `[y_lower, y_upper]` — target ≥ 0.95.
- **MPIW**: Mean interval width — lower is better given PICP ≥ 0.95.
- **Conformal prediction** (MAPIE): distribution-free coverage guarantee.
- **Tube loss**: custom loss with params `q` (target coverage), `r` (tube movement), `delta` (recalibration penalty).
- **Censored**: SM = 50 rows retained. **Uncensored**: dropped.
- **CQR calibration**: non-conformity scores on calibration set; q_hat at `(1−α)(1+1/n)` quantile (finite-sample correction).

## Gotchas

- Every experiment class prints `Results → <path>` on construction — verify `type` is correct before training.
- `ConformalizedQuantileExperiment` prints two paths on init — the second one (`conformal_results_<type>`) is the one actually used.
- **ANN `Input(shape=...)`**: always use `n_features = X.shape[1]` — never hardcode since feature count is now 7.
- **Exploration notebooks write CSVs**: if you add features, update `save_cols` in the `to_csv` cell of both exploration notebooks AND update `X_cols_*` in `constants.py`.
- **cuDNN not found**: prefix command with `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH` or source `~/.bashrc`.
- **`pi_estimation_uncensored.ipynb`**: had hardcoded output path — fixed, watch for regressions.
- **`export_to_excel.py`**: `_flatten()` handles list-valued JSON entries (e.g., tau tuning results) — do not simplify.
- **Plot annotations**: all plots use `fig.text(0.5, -0.02, ...)` with `bbox_inches='tight'` — metrics appear below the axes, not inside overlapping the data.
- **`ConformalRegression` tuning**: best params found on train data only; conf set must stay unseen until `regressor.conformalize()` — do not pass combined train+conf to `_find_best_params`.

## Applied Fixes Log (do NOT revert)

- `run_pipeline.py`: non-interactive stdin auto-continue (`sys.stdin.isatty()` check)
- `model_experiments.py`: `try/except RuntimeError` around `set_memory_growth` (handles TF pre-initialized by notebook)
- `model_experiments.py`: `os.makedirs(self.results_path, exist_ok=True)` in all `__init__` methods
- `model_experiments.py`: `fit_grid_search` uses `RandomizedSearchCV` with MAE scoring (yellowbrick removed)
- `model_experiments.py`: CQR finite-sample correction `(1−α)(1+1/n)` quantile
- `model_experiments.py`: `mixed_float16` removed (degraded regression accuracy)
- `model_experiments.py`: `batch_size` 256 → 64 (20 gradient steps/epoch vs 5)
- `model_experiments.py`: `ANNExperiment` plot restored to `plot_line_comparison` (sample-index scatter)
- `model_experiments.py`: all `ax.annotate` calls replaced with `fig.text(0.5, -0.02, ...)` below axes
- `export_to_excel.py`: `_flatten()` extended to recurse into list values with `[i]` index prefix
- `exploration_*.ipynb`: RFDI and RVI4S1 computed and added to `save_cols`
- `constants.py`: `X_cols_*` expanded to 7 features each
- `EOS-04_datasheet.xlsx` / `sentinel-1.xlsx`: RFDI and RVI4S1 columns added to all sheets
- `quantile_svr_HP_tuning.ipynb`: 2D C×gamma search; plot annotation fixed to below-axes style
