"""
Full enhanced experiment runner — runs ALL experiments headlessly.

Phases:
  0  Prepare enhanced CSVs (raw Excel → add features → save)
  1  Classical ML regression
  2  Classification (4-class)
  3  ANN point estimation
  4  Prediction Interval estimation (ANN quantile)
  5  Conformal Regression (QuantileReg / GBR / HistGBR via MAPIE)
  6  Conformalized Quantile Regression (SVM Split + GBM CQR + ANN Split + ANN CQR dual)
  7  Tau hyperparameter tuning (ANN CQR)
  8  Quantile SVR hyperparameter tuning (gamma grid)

Results overwrite the output/ folder (baseline is in output_baseline_backup/).
"""

# ── backend must be first ────────────────────────────────────────────────────
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DETERMINISTIC_OPS']  = '1'
os.environ['PYTHONHASHSEED']        = '42'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda: None  # kill all interactive show() calls globally

import random
random.seed(42)

import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import numpy as np
np.random.seed(42)
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             root_mean_squared_error, r2_score,
                             mean_absolute_percentage_error,
                             accuracy_score, classification_report)
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor,
                               RandomForestClassifier, AdaBoostClassifier,
                               GradientBoostingRegressor,
                               HistGradientBoostingRegressor)
from sklearn.linear_model import QuantileRegressor
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor, XGBClassifier

# add code dir to path so model_experiments imports correctly
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

# ── paths ────────────────────────────────────────────────────────────────────
ROOT        = CODE_DIR.parent
DATA_PATH   = ROOT / 'data'
OUTPUT_PATH = ROOT / 'output'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ── shared constants ─────────────────────────────────────────────────────────
X_COLS_EOS = ['HH-pol', 'HV-pol', 'cross_pol_ratio', 'month_sin', 'month_cos', 'crop_encoded', 'NDVI']
X_COLS_SEN = ['VH-pol', 'VV-pol', 'cross_pol_ratio', 'month_sin', 'month_cos', 'crop_encoded', 'NDVI']
Y_COL      = 'SM1 (%)'
CLASS_LABELS = ['Low', 'Medium', 'High', 'Very High']
INPUT_DIM  = 6
RANDOM_SEED = 42

print("=" * 70)
print("  ENHANCED EXPERIMENT RUNNER — ALL PHASES")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Prepare enhanced CSVs
# ══════════════════════════════════════════════════════════════════════════════

def _load_sheets(xlsx_path, drop='UniqueCrops'):
    xl = pd.ExcelFile(xlsx_path)
    return pd.concat(
        [xl.parse(s) for s in xl.sheet_names if s.strip() != drop],
        ignore_index=True
    )


def prepare_enhanced(raw_path, pol1, pol2, out_path, ndvi_csv=None):
    df = _load_sheets(raw_path)
    df = df.dropna()
    df['date'] = pd.to_datetime(df['Sample Date & Time'], errors='coerce')
    df = df[df['date'].dt.year > 1990]
    df = df[(df[Y_COL] > 0) & (df[Y_COL] <= 60)]

    df['Month'] = df['date'].dt.month
    df['cross_pol_ratio'] = df[pol1] - df[pol2]
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    le = LabelEncoder()
    df['crop_encoded'] = le.fit_transform(df['Crop Name'].str.strip())

    df['label'] = pd.qcut(df[Y_COL], q=4, labels=CLASS_LABELS)

    # Merge NDVI if available
    if ndvi_csv is not None and ndvi_csv.exists():
        ndvi = pd.read_csv(ndvi_csv)
        ndvi['month'] = pd.to_datetime(ndvi['date']).dt.month
        monthly_med = ndvi.groupby('month')['NDVI'].median()
        ndvi['NDVI'] = ndvi['NDVI'].fillna(ndvi['month'].map(monthly_med))
        ndvi[pol1] = ndvi[pol1].round(4)
        ndvi[pol2] = ndvi[pol2].round(4)
        df[pol1]   = df[pol1].round(4)
        df[pol2]   = df[pol2].round(4)
        df = df.merge(ndvi[[pol1, pol2, 'NDVI']].drop_duplicates(subset=[pol1, pol2]),
                      on=[pol1, pol2], how='left')
        still_nan = df['NDVI'].isna().sum()
        if still_nan > 0:
            df['NDVI'] = df['NDVI'].fillna(df['Month'].map(monthly_med))
    else:
        df['NDVI'] = np.nan

    keep = [pol1, pol2, 'cross_pol_ratio', 'month_sin', 'month_cos',
            'crop_encoded', 'NDVI', Y_COL, 'Crop Name', 'label']
    df[keep].to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path.name}")
    return df[keep]


print("\n[Phase 0] Preparing enhanced CSVs …")
eos_df = prepare_enhanced(
    DATA_PATH / 'EOS-04_datasheet.xlsx', 'HH-pol', 'HV-pol',
    DATA_PATH / 'eos-04-enhanced-ndvi.csv',
    ndvi_csv=DATA_PATH / 'eos04_ndvi.csv',
)
sen_df = prepare_enhanced(
    DATA_PATH / 'sentinel-1.xlsx', 'VH-pol', 'VV-pol',
    DATA_PATH / 'sentinel-1-enhanced-ndvi.csv',
    ndvi_csv=DATA_PATH / 'sentinel1_ndvi.csv',
)

X_eos = eos_df[X_COLS_EOS].values
y_eos = eos_df[Y_COL].values
X_sen = sen_df[X_COLS_SEN].values
y_sen = sen_df[Y_COL].values

print(f"  EOS-04: {X_eos.shape[0]} samples, SM range {y_eos.min():.1f}–{y_eos.max():.1f}%")
print(f"  Sentinel-1: {X_sen.shape[0]} samples, SM range {y_sen.min():.1f}–{y_sen.max():.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def reg_metrics(y_true, y_pred):
    return {
        'MAE':  round(float(mean_absolute_error(y_true, y_pred)), 4),
        'MSE':  round(float(mean_squared_error(y_true, y_pred)), 4),
        'RMSE': round(float(root_mean_squared_error(y_true, y_pred)), 4),
        'R2':   round(float(r2_score(y_true, y_pred)), 4),
        'MAPE': round(float(mean_absolute_percentage_error(y_true, y_pred)), 4),
    }


def split_80_10_10(X, y, seed=RANDOM_SEED):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, train_size=0.8, random_state=seed)
    X_v, X_te, y_v, y_te    = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed)
    return X_tr, X_v, X_te, y_tr, y_v, y_te


def split_70_10_10_10(X, y, seed=RANDOM_SEED):
    """70% train | 10% val (early stopping) | 10% cal (conformal) | 10% test."""
    X_tr,  X_tmp,  y_tr,  y_tmp  = train_test_split(X, y, train_size=0.7, random_state=seed)
    X_v,   X_tmp2, y_v,   y_tmp2 = train_test_split(X_tmp, y_tmp, train_size=1/3, random_state=seed)
    X_cal, X_te,   y_cal, y_te   = train_test_split(X_tmp2, y_tmp2, test_size=0.5, random_state=seed)
    return X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te


def scale_X(X_tr, X_v, X_te):
    sc = MinMaxScaler()
    return sc.fit_transform(X_tr), sc.transform(X_v), sc.transform(X_te), sc


def scale_X_4(X_tr, X_v, X_cal, X_te):
    sc = MinMaxScaler()
    return sc.fit_transform(X_tr), sc.transform(X_v), sc.transform(X_cal), sc.transform(X_te), sc


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)


def pi_metrics(y_true, lo, hi):
    covered = np.sum((y_true >= lo) & (y_true <= hi))
    return {
        'PICP': round(float(covered / len(y_true)), 6),
        'MPIW': round(float(np.mean(hi - lo)), 6),
    }


def save_pi_plot(y_true, lo, hi, title, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = np.arange(len(y_true))
    m = pi_metrics(y_true, lo, hi)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(idx, y_true, 'o', color='steelblue', ms=3, alpha=0.6, label='Actual')
    ax.plot(idx, lo, 'r--', lw=1, label='Lower bound')
    ax.plot(idx, hi, color='orange', ls='--', lw=1, label='Upper bound')
    ax.fill_between(idx, lo, hi, alpha=0.15, color='gray', label='95% PI')
    ax.text(0.02, 0.97, f"PICP: {m['PICP']*100:.2f}%\nMPIW: {m['MPIW']:.2f}",
            transform=ax.transAxes, va='top', fontsize=13,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Sample Index'); ax.set_ylabel('SM1 (%)')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Classical ML Regression
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Phase 1] Classical ML Regression …")

ML_GRIDS = {
    'RandomForest': (
        RandomForestRegressor(random_state=10),
        {'n_estimators': [100, 200, 500], 'max_depth': [None, 5, 10, 20],
         'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
    ),
    'XGBoost': (
        XGBRegressor(random_state=10, objective='reg:squarederror', verbosity=0),
        {'n_estimators': [100, 200, 500], 'max_depth': [3, 5, 7],
         'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.8, 1.0]}
    ),
    'AdaBoost': (
        AdaBoostRegressor(estimator=DecisionTreeRegressor(random_state=10), random_state=10),
        {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.05, 0.1, 1.0],
         'estimator__max_depth': [2, 3, 5]}
    ),
    'SVR': (
        SVR(),
        {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100],
         'gamma': ['scale', 'auto', 0.01, 0.1, 1], 'epsilon': [0.01, 0.1, 0.2, 0.5]}
    ),
}


def run_ml_regression(X, y, satellite, out_dir):
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_80_10_10(X, y)
    X_tr_s, _, X_te_s, _ = scale_X(X_tr, X_v, X_te)
    results = {}
    for name, (model, grid) in ML_GRIDS.items():
        print(f"  [{satellite}] {name} …", end=' ', flush=True)
        gs = GridSearchCV(model, grid, cv=3, n_jobs=-1, scoring='r2', verbose=0)
        gs.fit(X_tr_s, y_tr)
        y_pred = gs.best_estimator_.predict(X_te_s)
        m = reg_metrics(y_te, y_pred)
        results[name] = m
        print(f"R²={m['R2']:.4f}  RMSE={m['RMSE']:.4f}")
    save_json(results, out_dir / f"metrics_{satellite}.json")
    return results


for sat, X, y in [('EOS-04', X_eos, y_eos), ('Sentinel-1', X_sen, y_sen)]:
    run_ml_regression(X, y, sat, OUTPUT_PATH / 'ml_experiment_uncensored')

print("  Phase 1 complete.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Classification (4-class)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Phase 2] Classification …")

CLASS_MODELS = {
    'rf':  RandomForestClassifier(random_state=42),
    'xgb': XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0),
    'ada': AdaBoostClassifier(random_state=42),
    'svc': SVC(probability=True, random_state=42),
}


def run_classification(df, X_cols, satellite, out_dir):
    X = df[X_cols].values
    y_str = df['label'].astype(str).values

    le_cls = LabelEncoder()
    y_int = le_cls.fit_transform(y_str)  # XGB requires integer labels

    X_tr, X_te, y_tr_int, y_te_int = train_test_split(X, y_int, train_size=0.8, random_state=RANDOM_SEED)
    y_tr_str = le_cls.inverse_transform(y_tr_int)
    y_te_str = le_cls.inverse_transform(y_te_int)
    sc = MinMaxScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)

    results = {}
    for name, model in CLASS_MODELS.items():
        # XGB needs integer labels; others accept strings
        if name == 'xgb':
            model.fit(X_tr_s, y_tr_int)
            y_pred_int = model.predict(X_te_s)
            y_pred = le_cls.inverse_transform(y_pred_int)
        else:
            model.fit(X_tr_s, y_tr_str)
            y_pred = model.predict(X_te_s)
        acc = accuracy_score(y_te_str, y_pred)
        report = classification_report(y_te_str, y_pred, zero_division=0,
                                        output_dict=True, target_names=CLASS_LABELS)
        results[name] = report
        print(f"  [{satellite}] {name} accuracy={acc:.4f}")

    save_json(results, out_dir / f"metrics_{satellite}.json")
    return results


for sat, df_sat, xcols in [('EOS-04', eos_df, X_COLS_EOS), ('Sentinel-1', sen_df, X_COLS_SEN)]:
    run_classification(df_sat, xcols, sat, OUTPUT_PATH / 'classification_uncensored')

print("  Phase 2 complete.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — ANN Point Estimation
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Phase 3] ANN Point Estimation …")

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

tf.get_logger().setLevel('ERROR')
tf.random.set_seed(42)


def build_ann_models(input_dim):
    D = tf.keras.layers.Dense
    Do = tf.keras.layers.Dropout
    S = tf.keras.Sequential
    return {
        '2, 1':                   S([D(2,  activation='relu', input_shape=(input_dim,)), D(1)]),
        '4, 1':                   S([D(4,  activation='relu', input_shape=(input_dim,)), D(1)]),
        '8, 1':                   S([D(8,  activation='relu', input_shape=(input_dim,)), D(1)]),
        '16, 1':                  S([D(16, activation='relu', input_shape=(input_dim,)), D(1)]),
        '16, Dropout':            S([D(16, activation='relu', input_shape=(input_dim,)), Do(0.1), D(1)]),
        '16, Dropout, 8, Dropout':S([D(16, activation='relu', input_shape=(input_dim,)), Do(0.1),
                                     D(8,  activation='relu'), Do(0.1), D(1)]),
    }


def run_ann(X, y, satellite, out_dir):
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_80_10_10(X, y)
    X_tr_s, X_v_s, X_te_s, _ = scale_X(X_tr, X_v, X_te)

    models = build_ann_models(X.shape[1])
    results = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)

    for name, model in models.items():
        tf.keras.backend.clear_session()
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss='mse', metrics=['mae'])
        cb = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
        model.fit(X_tr_s, y_tr, epochs=300, batch_size=32,
                  validation_data=(X_v_s, y_v), verbose=0, callbacks=cb)

        y_pred_te = model.predict(X_te_s, verbose=0).flatten()
        y_pred_v  = model.predict(X_v_s,  verbose=0).flatten()
        te = {'MAE': float(mean_absolute_error(y_te, y_pred_te)),
              'MSE': float(mean_squared_error(y_te, y_pred_te)),
              'R2':  float(r2_score(y_te, y_pred_te))}
        va = {'MAE': float(mean_absolute_error(y_v, y_pred_v)),
              'MSE': float(mean_squared_error(y_v, y_pred_v)),
              'R2':  float(r2_score(y_v, y_pred_v))}
        results[name] = {'Test': te, 'Val': va}
        print(f"  [{satellite}] {name}: test R²={te['R2']:.4f}  MAE={te['MAE']:.4f}")

        # scatter plot
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_te, y_pred_te, alpha=0.5, s=15, color='steelblue', edgecolors='k', lw=0.2)
        lo, hi = min(y_te.min(), y_pred_te.min())*0.95, max(y_te.max(), y_pred_te.max())*1.05
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5)
        ax.set_xlabel('Actual SM1 (%)'); ax.set_ylabel('Predicted SM1 (%)')
        ax.set_title(f'{satellite} ANN — {name}')
        ax.text(0.05, 0.95, f"R²={te['R2']:.3f}\nMAE={te['MAE']:.2f}",
                transform=ax.transAxes, va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        fig.savefig(plot_dir / f"{satellite}_{name}_prediction_error.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    save_json(results, out_dir / f"{satellite}_metrics.json")
    return results


for sat, X, y in [('EOS-04', X_eos, y_eos), ('Sentinel-1', X_sen, y_sen)]:
    run_ann(X, y, sat, OUTPUT_PATH / 'ann_experiments_uncensored')

print("  Phase 3 complete.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Prediction Interval Estimation (ANN Quantile)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Phase 4] Prediction Interval Estimation …")


def pinball(tau):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(tau * e, (tau - 1) * e))
    return loss


def build_dual_ann(input_dim):
    """Shared backbone → two quantile heads. Prevents independent-model crossing."""
    inp    = tf.keras.Input(shape=(input_dim,))
    x      = tf.keras.layers.Dense(16, activation='relu')(inp)
    x      = tf.keras.layers.Dropout(0.09)(x)
    x      = tf.keras.layers.Dense(8,  activation='relu')(x)
    x      = tf.keras.layers.Dropout(0.09)(x)
    lo_out = tf.keras.layers.Dense(1, name='lo')(x)
    hi_out = tf.keras.layers.Dense(1, name='hi')(x)
    return tf.keras.Model(inputs=inp, outputs=[lo_out, hi_out])


def train_dual_quantile(X_tr, X_v, y_tr, y_v, lo_tau, hi_tau, epochs, input_dim,
                        lr=1e-3, patience=30, batch_size=32):
    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)
    model = build_dual_ann(input_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=[pinball(lo_tau), pinball(hi_tau)],
                  loss_weights=[1.0, 1.0])
    cb = [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
    model.fit(X_tr, [y_tr, y_tr], validation_data=(X_v, [y_v, y_v]),
              epochs=epochs, batch_size=batch_size, verbose=0, callbacks=cb)
    return model


def predict_dual(model, X):
    """Return (lo, hi) with sample-wise crossing correction."""
    preds = model.predict(X, verbose=0)
    lo = preds[0].flatten(); hi = preds[1].flatten()
    crossed = lo > hi
    return np.where(crossed, hi, lo), np.where(crossed, lo, hi)


def build_ann_regressor_sc(input_dim):
    inp = tf.keras.Input(shape=(input_dim,))
    x   = tf.keras.layers.Dense(16, activation='relu')(inp)
    x   = tf.keras.layers.Dropout(0.09)(x)
    x   = tf.keras.layers.Dense(8,  activation='relu')(x)
    x   = tf.keras.layers.Dropout(0.09)(x)
    out = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=inp, outputs=out)


def train_ann_regressor_sc(X_tr, X_v, y_tr, y_v, epochs, input_dim,
                            lr=1e-3, patience=30, batch_size=32):
    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)
    model = build_ann_regressor_sc(input_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
    cb = [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
    model.fit(X_tr, y_tr, validation_data=(X_v, y_v),
              epochs=epochs, batch_size=batch_size, verbose=0, callbacks=cb)
    return model


def report_crossing_p6(lo, hi):
    rate = float((lo > hi).mean())
    if rate > 0.05:
        print(f"  ℹ  crossing {rate*100:.1f}% — correction applied", end='')
    return rate


def _select_best_tau(tuning_results):
    for threshold in [0.95, 0.90]:
        valid = [r for r in tuning_results if r['RawCal_PICP'] >= threshold]
        if valid:
            return min(valid, key=lambda r: r['RawCal_MPIW']), threshold
    return min(tuning_results, key=lambda r: r['CQR_MPIW']), None


def run_pi(X, y, satellite, out_dir, tau_lo=0.025, tau_hi=0.975):
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_80_10_10(X, y)
    X_tr_s, X_v_s, X_te_s, _ = scale_X(X_tr, X_v, X_te)

    models = build_ann_models(X.shape[1])
    results = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)

    for name, model_tmpl in models.items():
        tf.keras.backend.clear_session()
        lower_model = tf.keras.models.clone_model(model_tmpl)
        upper_model = tf.keras.models.clone_model(model_tmpl)

        lower_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=pinball(tau_lo))
        upper_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=pinball(tau_hi))

        cb = [EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)]
        upper_model.fit(X_tr_s, y_tr, epochs=1000, batch_size=32,
                        validation_data=(X_v_s, y_v), verbose=0, callbacks=cb)
        lower_model.fit(X_tr_s, y_tr, epochs=1000, batch_size=32,
                        validation_data=(X_v_s, y_v), verbose=0, callbacks=cb)

        lo_te = lower_model.predict(X_te_s, verbose=0).flatten()
        hi_te = upper_model.predict(X_te_s, verbose=0).flatten()
        lo_v  = lower_model.predict(X_v_s,  verbose=0).flatten()
        hi_v  = upper_model.predict(X_v_s,  verbose=0).flatten()

        te_m = pi_metrics(y_te, lo_te, hi_te)
        va_m = pi_metrics(y_v,  lo_v,  hi_v)
        results[name] = {'test': te_m, 'val': va_m}
        print(f"  [{satellite}] {name}: PICP={te_m['PICP']:.4f}  MPIW={te_m['MPIW']:.4f}")

        save_pi_plot(y_te, lo_te, hi_te,
                     f'{satellite} PI — {name}',
                     plot_dir / f"{satellite}_{name}.png")

    save_json(results, out_dir / f"{satellite}_metrics.json")
    return results


for sat, X, y in [('EOS-04', X_eos, y_eos), ('Sentinel-1', X_sen, y_sen)]:
    run_pi(X, y, sat, OUTPUT_PATH / 'pi_estimation_uncensored')

print("  Phase 4 complete.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Conformal Regression (MAPIE)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Phase 5] Conformal Regression (MAPIE) …")

from mapie.utils import train_conformalize_test_split
from mapie.regression import ConformalizedQuantileRegressor


def run_conformal_regression(X, y, satellite, out_dir):
    y_flat = y.flatten()
    X_tr, X_cal, X_te, y_tr, y_cal, y_te = train_conformalize_test_split(
        X, y_flat, train_size=0.8, conformalize_size=0.1, test_size=0.1, random_state=RANDOM_SEED
    )

    conf_models = {
        'GradientBoostingRegressor':     GradientBoostingRegressor(loss='quantile'),
        'HistGradientBoostingRegressor': HistGradientBoostingRegressor(loss='quantile'),
        'QuantileRegressor':             QuantileRegressor(solver='highs'),
    }

    results = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)

    for name, model in conf_models.items():
        print(f"  [{satellite}] {name} …", end=' ', flush=True)
        reg = ConformalizedQuantileRegressor(estimator=model, confidence_level=0.95, prefit=False)
        reg.fit(X_tr, y_tr)
        reg.conformalize(X_cal, y_cal)
        _, intervals = reg.predict_interval(X_te, minimize_interval_width=True)
        intervals = intervals.squeeze()
        lo, hi = intervals[:, 0], intervals[:, 1]
        m = pi_metrics(y_te, lo, hi)
        results[name] = m
        print(f"PICP={m['PICP']:.4f}  MPIW={m['MPIW']:.4f}")
        save_pi_plot(y_te, lo, hi, f'{satellite} Conformal — {name}',
                     plot_dir / f"{satellite}_{name}.png")

    save_json(results, out_dir / f"{satellite}_metrics.json")
    return results


for sat, X, y in [('EOS-04', X_eos, y_eos), ('Sentinel-1', X_sen, y_sen)]:
    run_conformal_regression(X, y, sat, OUTPUT_PATH / 'conformal_regression_uncensored')

print("  Phase 5 complete.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — CQR: ANN + SVM Split + Linear (3-method comparison)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Phase 6] Conformalized Quantile Regression (3 methods) …")


def apply_cqr(y_cal, lo_cal, hi_cal, lo_te, hi_te, alpha=0.05):
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    q = np.quantile(scores, 1 - alpha, method='higher')
    return lo_te - q, hi_te + q


def apply_split_conformal(y_cal, pred_cal, pred_te, alpha=0.05):
    q = np.quantile(np.abs(y_cal - pred_cal), 1 - alpha, method='higher')
    return pred_te - q, pred_te + q


def run_cqr_suite(X, y, satellite, out_dir):
    mask = y != 50
    X, y = X[mask], y[mask]

    X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te = split_70_10_10_10(X, y)
    X_tr_s, X_v_s, X_cal_s, X_te_s, _ = scale_X_4(X_tr, X_v, X_cal, X_te)

    results = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'plots').mkdir(exist_ok=True)

    # --- SVM Split Conformal ---
    print(f"  [{satellite}] SVM Split …", end=' ', flush=True)
    svr = SVR(kernel='rbf')
    svr.fit(X_tr_s, y_tr)
    lo, hi = apply_split_conformal(y_cal, svr.predict(X_cal_s), svr.predict(X_te_s))
    m = pi_metrics(y_te, lo, hi)
    results['SVM_Split_Conformal'] = m
    print(f"PICP={m['PICP']:.4f}  MPIW={m['MPIW']:.4f}")
    save_pi_plot(y_te, lo, hi, f'{satellite} SVM Split Conformal',
                 out_dir / 'plots' / f"{satellite}_SVM_Split_Conformal_plot.png")

    # --- GBM CQR (replaces Linear CQR) ---
    print(f"  [{satellite}] GBM CQR …", end=' ', flush=True)
    gbm_lo = GradientBoostingRegressor(loss='quantile', alpha=0.025, n_estimators=200,
                                        max_depth=4, learning_rate=0.05,
                                        random_state=RANDOM_SEED)
    gbm_hi = GradientBoostingRegressor(loss='quantile', alpha=0.975, n_estimators=200,
                                        max_depth=4, learning_rate=0.05,
                                        random_state=RANDOM_SEED)
    gbm_lo.fit(X_tr_s, y_tr); gbm_hi.fit(X_tr_s, y_tr)
    p_lo_cal = gbm_lo.predict(X_cal_s); p_hi_cal = gbm_hi.predict(X_cal_s)
    p_lo_te  = gbm_lo.predict(X_te_s);  p_hi_te  = gbm_hi.predict(X_te_s)
    cross_gbm = report_crossing_p6(p_lo_te, p_hi_te)
    p_lo_cal, p_hi_cal = np.minimum(p_lo_cal, p_hi_cal), np.maximum(p_lo_cal, p_hi_cal)
    p_lo_te,  p_hi_te  = np.minimum(p_lo_te,  p_hi_te),  np.maximum(p_lo_te,  p_hi_te)
    lo, hi = apply_cqr(y_cal, p_lo_cal, p_hi_cal, p_lo_te, p_hi_te)
    m = pi_metrics(y_te, lo, hi)
    m['crossing_rate'] = round(cross_gbm, 4)
    results['GBM_CQR'] = m
    print(f"PICP={m['PICP']:.4f}  MPIW={m['MPIW']:.4f}  crossing={cross_gbm*100:.1f}%")
    save_pi_plot(y_te, lo, hi, f'{satellite} GBM CQR',
                 out_dir / 'plots' / f"{satellite}_GBM_CQR_plot.png")

    # --- ANN Split Conformal ---
    print(f"  [{satellite}] ANN Split Conformal …", end=' ', flush=True)
    ann_reg = train_ann_regressor_sc(X_tr_s, X_v_s, y_tr, y_v, 500, X.shape[1])
    pred_cal_ann = ann_reg.predict(X_cal_s, verbose=0).flatten()
    pred_te_ann  = ann_reg.predict(X_te_s,  verbose=0).flatten()
    lo, hi = apply_split_conformal(y_cal, pred_cal_ann, pred_te_ann)
    m = pi_metrics(y_te, lo, hi)
    results['ANN_Split_Conformal'] = m
    print(f"PICP={m['PICP']:.4f}  MPIW={m['MPIW']:.4f}")
    save_pi_plot(y_te, lo, hi, f'{satellite} ANN Split Conformal',
                 out_dir / 'plots' / f"{satellite}_ANN_Split_Conformal_plot.png")

    # --- ANN CQR (dual-output shared backbone) ---
    print(f"  [{satellite}] ANN CQR …", end=' ', flush=True)
    dual_m = train_dual_quantile(X_tr_s, X_v_s, y_tr, y_v, 0.025, 0.975, 500, X.shape[1])
    lo_cal_d, hi_cal_d = predict_dual(dual_m, X_cal_s)
    lo_te_d,  hi_te_d  = predict_dual(dual_m, X_te_s)
    cross_ann = report_crossing_p6(lo_te_d, hi_te_d)
    lo, hi = apply_cqr(y_cal, lo_cal_d, hi_cal_d, lo_te_d, hi_te_d)
    m = pi_metrics(y_te, lo, hi)
    m['crossing_rate'] = round(cross_ann, 4)
    results['ANN_CQR'] = m
    print(f"PICP={m['PICP']:.4f}  MPIW={m['MPIW']:.4f}  crossing={cross_ann*100:.1f}%")
    save_pi_plot(y_te, lo, hi, f'{satellite} ANN CQR (dual)',
                 out_dir / 'plots' / f"{satellite}_ANN_CQR_plot.png")

    save_json(results, out_dir / f"{satellite}_conformal_metrics.json")
    return results


for sat, X, y in [('EOS-04', X_eos, y_eos), ('Sentinel-1', X_sen, y_sen)]:
    run_cqr_suite(X, y, sat, OUTPUT_PATH / 'conformal_results')

print("  Phase 6 complete.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — Tau Hyperparameter Tuning
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Phase 7] Tau Hyperparameter Tuning (ANN CQR) …")

TAU_LOWERS = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04]


def run_tau_tuning(X, y, satellite, out_dir):
    mask = y != 50
    X, y = X[mask], y[mask]
    X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te = split_70_10_10_10(X, y)
    X_tr_s, X_v_s, X_cal_s, X_te_s, _ = scale_X_4(X_tr, X_v, X_cal, X_te)
    tuning_results = []

    for lo_tau in TAU_LOWERS:
        hi_tau = round(lo_tau + 0.95, 3)
        pair   = f"Low:{lo_tau} - High:{hi_tau}"
        print(f"  [{satellite}] tau {pair} …", end=' ', flush=True)

        dual_m = train_dual_quantile(X_tr_s, X_v_s, y_tr, y_v, lo_tau, hi_tau,
                                      500, X.shape[1])
        lo_cal_d, hi_cal_d = predict_dual(dual_m, X_cal_s)
        lo_te_d,  hi_te_d  = predict_dual(dual_m, X_te_s)
        cross_rate = report_crossing_p6(lo_te_d, hi_te_d)

        raw_cal_m = pi_metrics(y_cal, lo_cal_d, hi_cal_d)
        raw_te_m  = pi_metrics(y_te,  lo_te_d,  hi_te_d)
        clo, chi  = apply_cqr(y_cal, lo_cal_d, hi_cal_d, lo_te_d, hi_te_d)
        cqr_m     = pi_metrics(y_te, clo, chi)

        tuning_results.append({
            'Tau_Pair':    pair,
            'RawCal_PICP': raw_cal_m['PICP'],
            'RawCal_MPIW': raw_cal_m['MPIW'],
            'Raw_PICP':    raw_te_m['PICP'],
            'Raw_MPIW':    raw_te_m['MPIW'],
            'CQR_PICP':    cqr_m['PICP'],
            'CQR_MPIW':    cqr_m['MPIW'],
            'crossing_pct': round(cross_rate * 100, 2),
        })
        print(f"RawCal PICP={raw_cal_m['PICP']:.4f}  "
              f"CQR PICP={cqr_m['PICP']:.4f}  MPIW={cqr_m['MPIW']:.4f}  "
              f"crossing={cross_rate*100:.1f}%")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(tuning_results, out_dir / f"{satellite}_tau_tuning_metrics.json")

    best, threshold = _select_best_tau(tuning_results)
    if threshold:
        print(f"  [{satellite}] Best tau (RawCal_PICP ≥ {threshold}): {best['Tau_Pair']}"
              f"  CQR PICP={best['CQR_PICP']:.4f}  CQR MPIW={best['CQR_MPIW']:.4f}")
    else:
        print(f"  [{satellite}] ⚠ No tau met RawCal_PICP ≥ 0.90 — fallback: {best['Tau_Pair']}"
              f"  CQR PICP={best['CQR_PICP']:.4f}  CQR MPIW={best['CQR_MPIW']:.4f}")
    return tuning_results


for sat, X, y in [('EOS-04', X_eos, y_eos), ('Sentinel-1', X_sen, y_sen)]:
    run_tau_tuning(X, y, sat, OUTPUT_PATH / 'conformal_results')

print("  Phase 7 complete.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 8 — Quantile SVR Hyperparameter Tuning (gamma grid)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Phase 8] Quantile SVR Gamma Tuning …")

from scipy.spatial.distance import cdist
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

C_VALUE   = 2 ** 6
# Focus on 2^{-5} → 2^{8} (14 values covering the winning region from baseline)
GAMMA_VALUES = [2 ** i for i in range(-15, 16)]


def rbf_kernel(X, Y, gamma):
    return np.exp(-gamma * cdist(X, Y, 'sqeuclidean'))


def fit_qsvr(X, y, gamma, C, tau):
    n  = X.shape[0]
    H  = rbf_kernel(X, X, gamma)
    Hb = np.block([[H, -H], [-H, H]])
    c  = np.concatenate([(1 - tau) * np.zeros(n) - y,
                          tau       * np.zeros(n) + y])
    vub = np.concatenate([tau * C * np.ones(n), (1 - tau) * C * np.ones(n)])
    I   = np.eye(2 * n)
    sol = solvers.qp(matrix(Hb), matrix(c),
                     matrix(np.vstack([-I, I])),
                     matrix(np.concatenate([np.zeros(2 * n), vub])))
    alpha = np.array(sol['x']).flatten()
    beta  = alpha[:n] - alpha[n:]
    return beta


def predict_qsvr(X_train, X_pred, gamma, beta):
    return rbf_kernel(X_pred, X_train, gamma) @ beta


def run_qsvr_tuning(X, y, satellite, out_dir):
    mask = y != 50
    X, y = X[mask], y[mask]
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_80_10_10(X, y)
    sc = MinMaxScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_v_s  = sc.transform(X_v)
    X_te_s = sc.transform(X_te)

    all_results = []
    plot_dir = out_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    for gamma in GAMMA_VALUES:
        exp = int(round(np.log2(gamma)))
        print(f"  [{satellite}] γ=2^{exp:3d} …", end=' ', flush=True)
        try:
            beta_lo = fit_qsvr(X_tr_s, y_tr, gamma, C_VALUE, 0.025)
            beta_hi = fit_qsvr(X_tr_s, y_tr, gamma, C_VALUE, 0.975)

            lo_te = predict_qsvr(X_tr_s, X_te_s, gamma, beta_lo)
            hi_te = predict_qsvr(X_tr_s, X_te_s, gamma, beta_hi)
            lo_v  = predict_qsvr(X_tr_s, X_v_s,  gamma, beta_lo)
            hi_v  = predict_qsvr(X_tr_s, X_v_s,  gamma, beta_hi)

            m_te = pi_metrics(y_te, lo_te, hi_te)
            m_v  = pi_metrics(y_v,  lo_v,  hi_v)
            print(f"PICP={m_te['PICP']:.4f}  MPIW={m_te['MPIW']:.4f}")

            save_pi_plot(y_te, lo_te, hi_te,
                         f'{satellite} QSVM C={C_VALUE} γ=2^{exp}',
                         plot_dir / f"{satellite}_C={C_VALUE}_gamma={gamma}.png")

            all_results.append({'params': {'C': C_VALUE, 'gamma': gamma},
                                  'val':  m_v, 'test': m_te})
        except Exception as e:
            print(f"FAILED ({e})")
            all_results.append({'params': {'C': C_VALUE, 'gamma': gamma},
                                  'val': None, 'test': None, 'error': str(e)})

    df = pd.json_normalize(all_results)
    df.to_csv(out_dir / f"{satellite.lower().replace('-','')}_tuning_summary_C={C_VALUE}.csv", index=False)

    # print best
    valid = [r for r in all_results if r['test'] is not None
             and 0.90 <= r['test']['PICP'] <= 0.97]
    if valid:
        best = min(valid, key=lambda r: r['test']['MPIW'])
        print(f"  [{satellite}] Best: γ={best['params']['gamma']}  "
              f"PICP={best['test']['PICP']:.4f}  MPIW={best['test']['MPIW']:.4f}")

    return all_results


for sat, X, y in [('EOS-04', X_eos, y_eos), ('Sentinel-1', X_sen, y_sen)]:
    run_qsvr_tuning(X, y, sat, OUTPUT_PATH / 'quantile_svr_uncensored')

print("  Phase 8 complete.")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  ALL PHASES COMPLETE")
print("=" * 70)

# load and print key metrics
def load_best(path, key='R2'):
    try:
        d = json.load(open(path))
        best = max(d.items(), key=lambda x: x[1].get(key, -999) if isinstance(x[1], dict) else -999)
        return best
    except:
        return None, {}

print("\n  REGRESSION (uncensored) — best R²:")
for sat in ['EOS-04', 'Sentinel-1']:
    r = load_best(OUTPUT_PATH / 'ml_experiment_uncensored' / f'metrics_{sat}.json')
    if r[0]:
        print(f"  {sat:12s}: {r[0]:15s} R²={r[1]['R2']:.4f}  RMSE={r[1]['RMSE']:.4f}  MAE={r[1]['MAE']:.4f}")

print("\n  PI ESTIMATION (uncensored) — best PICP@95%:")
for sat in ['EOS-04', 'Sentinel-1']:
    try:
        d = json.load(open(OUTPUT_PATH / 'pi_estimation_uncensored' / f'{sat}_metrics.json'))
        best = min(d.items(), key=lambda x: abs(x[1]['test']['PICP'] - 0.95))
        print(f"  {sat:12s}: {best[0]:30s} PICP={best[1]['test']['PICP']:.4f}  MPIW={best[1]['test']['MPIW']:.4f}")
    except: pass

print("\n  CONFORMAL CQR (conformal_results) — all methods:")
for sat in ['EOS-04', 'Sentinel-1']:
    try:
        d = json.load(open(OUTPUT_PATH / 'conformal_results' / f'{sat}_conformal_metrics.json'))
        for name, m in d.items():
            print(f"  {sat:12s}: {name:25s} PICP={m['PICP']:.4f}  MPIW={m['MPIW']:.4f}")
    except: pass

print(f"\n  Output folder : {OUTPUT_PATH}")
print(f"  Backup folder : {OUTPUT_PATH.parent / 'output_baseline_backup'}")
print("\nDone.")
