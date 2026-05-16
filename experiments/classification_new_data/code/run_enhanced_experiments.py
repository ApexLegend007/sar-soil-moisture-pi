"""
Enhanced experiment runner.
Adds three features to the original 2-channel SAR inputs:
  - cross_pol_ratio  : HH-HV (EOS-04) or VH-VV (Sentinel-1) in dB
  - month_sin / month_cos : cyclical encoding of acquisition month
  - crop_encoded     : label-encoded crop type

Runs classical ML regression (RF, XGB, AdaBoost, SVR) headlessly and saves
metrics to output/ml_experiment_enhanced/.
"""

import matplotlib
matplotlib.use('Agg')

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_absolute_percentage_error,
                             r2_score, root_mean_squared_error)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data'
OUTPUT_PATH = ROOT / 'output' / 'ml_experiment_enhanced'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ── data loading ────────────────────────────────────────────────────────────

def _load_sheets(xlsx_path: Path, drop_sheet: str = 'UniqueCrops') -> pd.DataFrame:
    """Concatenate all sheets from an xlsx file, skipping drop_sheet."""
    xl = pd.ExcelFile(xlsx_path)
    frames = []
    for sheet in xl.sheet_names:
        if sheet.strip() == drop_sheet:
            continue
        df = xl.parse(sheet)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_eos() -> pd.DataFrame:
    df = _load_sheets(DATA_PATH / 'EOS-04_datasheet.xlsx')
    # Match original processing: drop any row with ANY NaN column,
    # then remove physically impossible SM values (sensor errors, saturation artefacts).
    df = df.dropna()
    df['date'] = pd.to_datetime(df['Sample Date & Time'], errors='coerce')
    df['Month'] = df['date'].dt.month
    df = df[df['date'].dt.year > 1990]
    df = df[(df['SM1 (%)'] > 0) & (df['SM1 (%)'] <= 60)]
    return df


def load_sentinel() -> pd.DataFrame:
    df = _load_sheets(DATA_PATH / 'sentinel-1.xlsx')
    df = df.dropna()
    df['date'] = pd.to_datetime(df['Sample Date & Time'], errors='coerce')
    df['Month'] = df['date'].dt.month
    df = df[df['date'].dt.year > 1990]
    df = df[(df['SM1 (%)'] > 0) & (df['SM1 (%)'] <= 60)]
    return df


# ── feature engineering ─────────────────────────────────────────────────────

def engineer(df: pd.DataFrame, pol1: str, pol2: str) -> tuple[pd.DataFrame, LabelEncoder]:
    df = df.copy()
    df['cross_pol_ratio'] = df[pol1] - df[pol2]
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    le = LabelEncoder()
    df['crop_encoded'] = le.fit_transform(df['Crop Name'].fillna('Unknown').str.strip())
    return df, le


# ── metrics helper ──────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        'MAE':  round(float(mean_absolute_error(y_true, y_pred)), 4),
        'MSE':  round(float(mean_squared_error(y_true, y_pred)), 4),
        'RMSE': round(float(root_mean_squared_error(y_true, y_pred)), 4),
        'R2':   round(float(r2_score(y_true, y_pred)), 4),
        'MAPE': round(float(mean_absolute_percentage_error(y_true, y_pred)), 4),
    }


# ── plot ────────────────────────────────────────────────────────────────────

def save_scatter_plot(y_test, y_pred, satellite, model_name, metrics):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.3,
               color='steelblue', label='Prediction', s=20)

    lo = min(y_test.min(), y_pred.min()) * 0.95
    hi = max(y_test.max(), y_pred.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='Identity (perfect)')

    coeffs = np.polyfit(y_test, y_pred, 1)
    xs = np.linspace(lo, hi, 200)
    ax.plot(xs, np.poly1d(coeffs)(xs), 'r-', lw=1.5, label='Best fit')

    txt = f"MAE:  {metrics['MAE']:.2f}\nRMSE: {metrics['RMSE']:.2f}\nR²:   {metrics['R2']:.3f}"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

    ax.set_xlabel('Actual SM1 (%)', fontsize=13)
    ax.set_ylabel('Predicted SM1 (%)', fontsize=13)
    ax.set_title(f'{satellite} — {model_name} (enhanced features)', fontsize=14)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

    plot_dir = OUTPUT_PATH / 'plots'
    plot_dir.mkdir(exist_ok=True)
    fig.savefig(plot_dir / f'{satellite}_{model_name}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


# ── single model runner ──────────────────────────────────────────────────────

def run_grid_search(name, model, param_grid, X_train, X_test, y_train, y_test):
    print(f"  [{name}] grid search …", flush=True)
    gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='r2', verbose=0)
    gs.fit(X_train, y_train)
    y_pred = gs.best_estimator_.predict(X_test)
    m = compute_metrics(y_test, y_pred)
    print(f"  [{name}] R²={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}")
    return m, y_pred, gs.best_params_


# ── full experiment ──────────────────────────────────────────────────────────

def run_experiment(satellite: str, X_cols: list, df: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"  Satellite: {satellite}  |  Features: {X_cols}")
    print(f"{'='*60}")

    X = df[X_cols].values
    y = df['SM1 (%)'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42
    )

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        'RandomForest': (
            RandomForestRegressor(random_state=10),
            {
                'n_estimators':    [100, 200, 500],
                'max_depth':       [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf':  [1, 2, 4],
            }
        ),
        'XGBoost': (
            XGBRegressor(random_state=10, objective='reg:squarederror', verbosity=0),
            {
                'n_estimators':  [100, 200, 500],
                'max_depth':     [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample':     [0.8, 1.0],
            }
        ),
        'AdaBoost': (
            AdaBoostRegressor(
                estimator=DecisionTreeRegressor(random_state=10), random_state=10
            ),
            {
                'n_estimators':          [50, 100, 200],
                'learning_rate':         [0.01, 0.05, 0.1, 1.0],
                'estimator__max_depth':  [2, 3, 5],
            }
        ),
        'SVR': (
            SVR(),
            {
                'kernel':   ['rbf'],
                'C':        [0.1, 1, 10, 100],
                'gamma':    ['scale', 'auto', 0.01, 0.1, 1],
                'epsilon':  [0.01, 0.1, 0.2, 0.5],
            }
        ),
    }

    results = {}
    for name, (model, grid) in models.items():
        m, y_pred, best_params = run_grid_search(
            name, model, grid, X_train_s, X_test_s, y_train, y_test
        )
        results[name] = m
        save_scatter_plot(y_test, y_pred, satellite, name, m)

    out_file = OUTPUT_PATH / f'metrics_{satellite}.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n  Saved → {out_file}")
    return results


# ── main ────────────────────────────────────────────────────────────────────

def main():
    # ── EOS-04 ──
    eos_df, eos_le = engineer(load_eos(), 'HH-pol', 'HV-pol')
    eos_cols = ['HH-pol', 'HV-pol', 'cross_pol_ratio', 'month_sin', 'month_cos', 'crop_encoded']
    eos_results = run_experiment('EOS-04', eos_cols, eos_df)

    # ── Sentinel-1 ──
    sen_df, sen_le = engineer(load_sentinel(), 'VH-pol', 'VV-pol')
    sen_cols = ['VH-pol', 'VV-pol', 'cross_pol_ratio', 'month_sin', 'month_cos', 'crop_encoded']
    sen_results = run_experiment('Sentinel-1', sen_cols, sen_df)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  SUMMARY — Best R² per satellite")
    print(f"{'='*60}")
    for sat, res in [('EOS-04', eos_results), ('Sentinel-1', sen_results)]:
        best = max(res.items(), key=lambda x: x[1]['R2'])
        print(f"  {sat:12s}  best={best[0]:15s}  R²={best[1]['R2']:.4f}  "
              f"RMSE={best[1]['RMSE']:.4f}  MAE={best[1]['MAE']:.4f}")

    print(f"\nAll plots → {OUTPUT_PATH / 'plots'}")
    print("Done.")


if __name__ == '__main__':
    main()
