"""
Phases 4 / 6 / 7 — Improved CQR pipeline.

CHANGES FROM PREVIOUS VERSION
  Seeds    : TF + numpy + random + PYTHONHASHSEED fixed; TF_DETERMINISTIC_OPS=1.
             tf.random.set_seed called inside every training function.
  Split    : 80/10/10 (train/val/test) → 70/10/10/10 (train/val/cal/test).
             val  — early stopping only, never seen by conformal calibration.
             cal  — conformal scores only, never seen by model training.
  Phase 6  : Linear CQR → GBM CQR (non-linear quantile base, sort crossing guard).
             Added ANN Split Conformal (MSE point predictor + residual band).
             All methods now use the separate cal set for conformal calibration.
  Phase 7  : Cal-based tau selection: smallest RawCal_MPIW where RawCal_PICP ≥ 0.95
             (fallback: ≥ 0.90).  Previously selected on test set — data snooping.

PREMORTEM RISKS NOTED
  - 70% train is 10% less data → may slightly widen raw intervals.
  - GBM trains two separate models; sort guard fixes crossings post-hoc.
  - ANN Split Conformal MPIW is driven by point-predictor RMSE — not guaranteed
    to beat SVM Split.
  - Cal PICP ≥ 0.95 criterion may not be met for any tau; fallback to 0.90.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']   = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS']  = '0'
os.environ['TF_DETERMINISTIC_OPS']   = '1'
os.environ['PYTHONHASHSEED']         = '42'

import random
random.seed(42)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda: None

import json, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
np.random.seed(42)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.callbacks import EarlyStopping
tf.get_logger().setLevel('ERROR')

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / 'data'
OUTPUT_PATH = ROOT / 'output'

# ── hyperparameters ────────────────────────────────────────────────────────────
ANN_LR         = 1e-3
ANN_PATIENCE   = 30
ANN_EPOCHS_PI  = 1000
ANN_EPOCHS_CQR = 500
BATCH_SIZE     = 32
RANDOM_SEED    = 42
INVERSION_TOL  = 0.05

X_COLS_EOS = ['HH-pol', 'HV-pol', 'cross_pol_ratio', 'month_sin', 'month_cos', 'crop_encoded']
X_COLS_SEN = ['VH-pol', 'VV-pol', 'cross_pol_ratio', 'month_sin', 'month_cos', 'crop_encoded']
Y_COL      = 'SM1 (%)'

print("=" * 70)
print("  PHASES 4 / 6 / 7 — IMPROVED CQR PIPELINE")
print(f"  lr={ANN_LR}  patience={ANN_PATIENCE}  "
      f"epochs_PI={ANN_EPOCHS_PI}  epochs_CQR/tau={ANN_EPOCHS_CQR}")
print("=" * 70)

# ── cleanup ────────────────────────────────────────────────────────────────────
print("\n[Cleanup] Removing stale Phase 6 / 7 outputs …")
PI_DIR  = OUTPUT_PATH / 'pi_estimation_uncensored'
CQR_DIR = OUTPUT_PATH / 'conformal_results'

for f in ['EOS-04_conformal_metrics.json', 'Sentinel-1_conformal_metrics.json',
          'EOS-04_tau_tuning_metrics.json',  'Sentinel-1_tau_tuning_metrics.json',
          'EOS-04_tau_tuning_comparison.png', 'Sentinel-1_tau_tuning_comparison.png']:
    (CQR_DIR / f).unlink(missing_ok=True)

plots_dir = CQR_DIR / 'plots'
if plots_dir.exists():
    for png in plots_dir.glob('*.png'):
        png.unlink(missing_ok=True)
print("  Done.")

# ── load data ──────────────────────────────────────────────────────────────────
print("\n[Data] Loading enhanced CSVs …")
eos_df = pd.read_csv(DATA_PATH / 'eos-04-enhanced.csv')
sen_df = pd.read_csv(DATA_PATH / 'sentinel-1-enhanced.csv')

X_eos = eos_df[X_COLS_EOS].values;  y_eos = eos_df[Y_COL].values
X_sen = sen_df[X_COLS_SEN].values;  y_sen = sen_df[Y_COL].values
print(f"  EOS-04: {X_eos.shape[0]} samples  |  Sentinel-1: {X_sen.shape[0]} samples")

# ── helpers ────────────────────────────────────────────────────────────────────

def split_70_10_10_10(X, y):
    """70% train | 10% val (early stopping) | 10% cal (conformal) | 10% test."""
    X_tr, X_tmp, y_tr, y_tmp   = train_test_split(X, y, train_size=0.7, random_state=RANDOM_SEED)
    X_v,  X_tmp2, y_v, y_tmp2  = train_test_split(X_tmp, y_tmp, train_size=1/3, random_state=RANDOM_SEED)
    X_cal, X_te, y_cal, y_te   = train_test_split(X_tmp2, y_tmp2, test_size=0.5, random_state=RANDOM_SEED)
    return X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te


def scale_X(X_tr, X_v, X_cal, X_te):
    sc = MinMaxScaler()
    return sc.fit_transform(X_tr), sc.transform(X_v), sc.transform(X_cal), sc.transform(X_te), sc


def pi_metrics(y_true, lo, hi):
    covered = np.sum((y_true >= lo) & (y_true <= hi))
    return {'PICP': round(float(covered / len(y_true)), 6),
            'MPIW': round(float(np.mean(hi - lo)), 6)}


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)


def save_pi_plot(y_true, lo, hi, title, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = np.arange(len(y_true))
    m   = pi_metrics(y_true, lo, hi)
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


def report_crossing(lo, hi, label):
    rate = float((lo > hi).mean())
    if rate > INVERSION_TOL:
        print(f"  ℹ  crossing {rate*100:.1f}% lo>hi — correction applied")
    return rate


def apply_split_conformal(y_cal, pred_cal, pred_te, alpha=0.05):
    q = np.quantile(np.abs(y_cal - pred_cal), 1 - alpha, method='higher')
    return pred_te - q, pred_te + q


def apply_cqr(y_cal, lo_cal, hi_cal, lo_te, hi_te, alpha=0.05):
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    q = np.quantile(scores, 1 - alpha, method='higher')
    return lo_te - q, hi_te + q


def pinball(tau):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(tau * e, (tau - 1) * e))
    return loss


def build_dual_ann(input_dim):
    """Shared backbone → two quantile heads. Eliminates independent-model crossing."""
    inp    = tf.keras.Input(shape=(input_dim,))
    x      = tf.keras.layers.Dense(16, activation='relu')(inp)
    x      = tf.keras.layers.Dropout(0.09)(x)
    x      = tf.keras.layers.Dense(8,  activation='relu')(x)
    x      = tf.keras.layers.Dropout(0.09)(x)
    lo_out = tf.keras.layers.Dense(1, name='lo')(x)
    hi_out = tf.keras.layers.Dense(1, name='hi')(x)
    return tf.keras.Model(inputs=inp, outputs=[lo_out, hi_out])


def train_dual_quantile(X_tr, X_v, y_tr, y_v, lo_tau, hi_tau, epochs, input_dim):
    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)
    model = build_dual_ann(input_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(ANN_LR),
                  loss=[pinball(lo_tau), pinball(hi_tau)],
                  loss_weights=[1.0, 1.0])
    cb = [EarlyStopping(monitor='val_loss', patience=ANN_PATIENCE, restore_best_weights=True)]
    model.fit(X_tr, [y_tr, y_tr], validation_data=(X_v, [y_v, y_v]),
              epochs=epochs, batch_size=BATCH_SIZE, verbose=0, callbacks=cb)
    return model


def predict_dual(model, X):
    """Return (lo, hi) with sample-wise crossing correction."""
    preds = model.predict(X, verbose=0)
    lo = preds[0].flatten(); hi = preds[1].flatten()
    crossed = lo > hi
    return np.where(crossed, hi, lo), np.where(crossed, lo, hi)


def build_ann_regressor(input_dim):
    inp = tf.keras.Input(shape=(input_dim,))
    x   = tf.keras.layers.Dense(16, activation='relu')(inp)
    x   = tf.keras.layers.Dropout(0.09)(x)
    x   = tf.keras.layers.Dense(8,  activation='relu')(x)
    x   = tf.keras.layers.Dropout(0.09)(x)
    out = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=inp, outputs=out)


def train_ann_regressor(X_tr, X_v, y_tr, y_v, epochs, input_dim):
    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)
    model = build_ann_regressor(input_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(ANN_LR), loss='mse')
    cb = [EarlyStopping(monitor='val_loss', patience=ANN_PATIENCE, restore_best_weights=True)]
    model.fit(X_tr, y_tr, validation_data=(X_v, y_v),
              epochs=epochs, batch_size=BATCH_SIZE, verbose=0, callbacks=cb)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — skipped (results already valid from previous run)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Phase 4] Skipped — PI results already valid in output folder.")
PI_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — CQR Suite (4 methods)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Phase 6] Conformalized Quantile Regression (4 methods) …")
print(f"  ANN hyperparams: lr={ANN_LR}  patience={ANN_PATIENCE}  epochs={ANN_EPOCHS_CQR}")
print("  Split: 70/10/10/10 — val=early-stop only  cal=conformal only  test=eval")

CQR_DIR.mkdir(parents=True, exist_ok=True)
(CQR_DIR / 'plots').mkdir(exist_ok=True)

for sat, X, y in [('EOS-04', X_eos, y_eos), ('Sentinel-1', X_sen, y_sen)]:
    mask = y != 50
    Xm, ym = X[mask], y[mask]
    X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te = split_70_10_10_10(Xm, ym)
    X_tr_s, X_v_s, X_cal_s, X_te_s, _ = scale_X(X_tr, X_v, X_cal, X_te)
    results = {}

    # --- SVM Split Conformal ---
    print(f"  [{sat}] SVM Split …", end=' ', flush=True)
    svr = SVR(kernel='rbf')
    svr.fit(X_tr_s, y_tr)
    lo, hi = apply_split_conformal(y_cal, svr.predict(X_cal_s), svr.predict(X_te_s))
    m = pi_metrics(y_te, lo, hi)
    results['SVM_Split_Conformal'] = m
    print(f"PICP={m['PICP']:.4f}  MPIW={m['MPIW']:.4f}")
    save_pi_plot(y_te, lo, hi, f'{sat} SVM Split',
                 CQR_DIR / 'plots' / f"{sat}_SVM_Split_Conformal_plot.png")

    # --- GBM CQR (replaces Linear CQR) ---
    print(f"  [{sat}] GBM CQR …", end=' ', flush=True)
    gbm_lo = GradientBoostingRegressor(loss='quantile', alpha=0.025, n_estimators=200,
                                        max_depth=4, learning_rate=0.05,
                                        random_state=RANDOM_SEED)
    gbm_hi = GradientBoostingRegressor(loss='quantile', alpha=0.975, n_estimators=200,
                                        max_depth=4, learning_rate=0.05,
                                        random_state=RANDOM_SEED)
    gbm_lo.fit(X_tr_s, y_tr); gbm_hi.fit(X_tr_s, y_tr)
    p_lo_cal = gbm_lo.predict(X_cal_s); p_hi_cal = gbm_hi.predict(X_cal_s)
    p_lo_te  = gbm_lo.predict(X_te_s);  p_hi_te  = gbm_hi.predict(X_te_s)
    # sort guard: lo ← min, hi ← max (same contract as predict_dual)
    cross_gbm = report_crossing(p_lo_te, p_hi_te, f"{sat}/GBM_CQR")
    p_lo_cal, p_hi_cal = np.minimum(p_lo_cal, p_hi_cal), np.maximum(p_lo_cal, p_hi_cal)
    p_lo_te,  p_hi_te  = np.minimum(p_lo_te,  p_hi_te),  np.maximum(p_lo_te,  p_hi_te)
    lo, hi = apply_cqr(y_cal, p_lo_cal, p_hi_cal, p_lo_te, p_hi_te)
    m = pi_metrics(y_te, lo, hi)
    m['crossing_rate'] = round(cross_gbm, 4)
    results['GBM_CQR'] = m
    print(f"PICP={m['PICP']:.4f}  MPIW={m['MPIW']:.4f}  crossing={cross_gbm*100:.1f}%")
    save_pi_plot(y_te, lo, hi, f'{sat} GBM CQR',
                 CQR_DIR / 'plots' / f"{sat}_GBM_CQR_plot.png")

    # --- ANN Split Conformal ---
    print(f"  [{sat}] ANN Split Conformal …", end=' ', flush=True)
    ann_reg = train_ann_regressor(X_tr_s, X_v_s, y_tr, y_v, ANN_EPOCHS_CQR, Xm.shape[1])
    pred_cal_ann = ann_reg.predict(X_cal_s, verbose=0).flatten()
    pred_te_ann  = ann_reg.predict(X_te_s,  verbose=0).flatten()
    lo, hi = apply_split_conformal(y_cal, pred_cal_ann, pred_te_ann)
    m = pi_metrics(y_te, lo, hi)
    results['ANN_Split_Conformal'] = m
    print(f"PICP={m['PICP']:.4f}  MPIW={m['MPIW']:.4f}")
    save_pi_plot(y_te, lo, hi, f'{sat} ANN Split Conformal',
                 CQR_DIR / 'plots' / f"{sat}_ANN_Split_Conformal_plot.png")

    # --- ANN CQR (dual-output shared backbone, tau=0.025/0.975) ---
    print(f"  [{sat}] ANN CQR …", end=' ', flush=True)
    dual_m = train_dual_quantile(X_tr_s, X_v_s, y_tr, y_v, 0.025, 0.975,
                                  epochs=ANN_EPOCHS_CQR, input_dim=Xm.shape[1])
    lo_cal_d, hi_cal_d = predict_dual(dual_m, X_cal_s)
    lo_te_d,  hi_te_d  = predict_dual(dual_m, X_te_s)
    cross_rate = report_crossing(lo_te_d, hi_te_d, f"{sat}/ANN_CQR")
    lo, hi = apply_cqr(y_cal, lo_cal_d, hi_cal_d, lo_te_d, hi_te_d)
    m = pi_metrics(y_te, lo, hi)
    m['crossing_rate'] = round(cross_rate, 4)
    results['ANN_CQR'] = m
    print(f"PICP={m['PICP']:.4f}  MPIW={m['MPIW']:.4f}  crossing={cross_rate*100:.1f}%")
    save_pi_plot(y_te, lo, hi, f'{sat} ANN CQR (dual)',
                 CQR_DIR / 'plots' / f"{sat}_ANN_CQR_plot.png")

    save_json(results, CQR_DIR / f"{sat}_conformal_metrics.json")

print("  Phase 6 complete.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — Tau Tuning with cal-based selection
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Phase 7] Tau Tuning (ANN CQR) — cal-based selection …")
print(f"  ANN hyperparams: lr={ANN_LR}  patience={ANN_PATIENCE}  epochs={ANN_EPOCHS_CQR}")
print("  Selection: min RawCal_MPIW where RawCal_PICP ≥ 0.95 (fallback: ≥ 0.90)")

TAU_LOWERS = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04]


def _select_best_tau(tuning_results):
    for threshold in [0.95, 0.90]:
        valid = [r for r in tuning_results if r['RawCal_PICP'] >= threshold]
        if valid:
            return min(valid, key=lambda r: r['RawCal_MPIW']), threshold
    return min(tuning_results, key=lambda r: r['CQR_MPIW']), None


for sat, X, y in [('EOS-04', X_eos, y_eos), ('Sentinel-1', X_sen, y_sen)]:
    mask = y != 50
    Xm, ym = X[mask], y[mask]
    X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te = split_70_10_10_10(Xm, ym)
    X_tr_s, X_v_s, X_cal_s, X_te_s, _ = scale_X(X_tr, X_v, X_cal, X_te)
    tuning_results = []

    for lo_tau in TAU_LOWERS:
        hi_tau = round(lo_tau + 0.95, 3)
        pair   = f"Low:{lo_tau} - High:{hi_tau}"
        print(f"  [{sat}] tau {pair} …", end=' ', flush=True)

        dual_m = train_dual_quantile(
            X_tr_s, X_v_s, y_tr, y_v, lo_tau, hi_tau,
            epochs=ANN_EPOCHS_CQR, input_dim=Xm.shape[1]
        )
        lo_cal_d, hi_cal_d = predict_dual(dual_m, X_cal_s)
        lo_te_d,  hi_te_d  = predict_dual(dual_m, X_te_s)
        cross_rate = report_crossing(lo_te_d, hi_te_d, f"{sat}/tau={pair}")

        raw_cal_m = pi_metrics(y_cal, lo_cal_d, hi_cal_d)
        raw_te_m  = pi_metrics(y_te,  lo_te_d,  hi_te_d)
        clo, chi  = apply_cqr(y_cal, lo_cal_d, hi_cal_d, lo_te_d, hi_te_d)
        cqr_m     = pi_metrics(y_te, clo, chi)

        tuning_results.append({
            'Tau_Pair':     pair,
            'RawCal_PICP':  raw_cal_m['PICP'],
            'RawCal_MPIW':  raw_cal_m['MPIW'],
            'Raw_PICP':     raw_te_m['PICP'],
            'Raw_MPIW':     raw_te_m['MPIW'],
            'CQR_PICP':     cqr_m['PICP'],
            'CQR_MPIW':     cqr_m['MPIW'],
            'crossing_pct': round(cross_rate * 100, 2),
        })
        print(f"RawCal PICP={raw_cal_m['PICP']:.4f}  "
              f"CQR PICP={cqr_m['PICP']:.4f}  MPIW={cqr_m['MPIW']:.4f}  "
              f"crossing={cross_rate*100:.1f}%")

    save_json(tuning_results, CQR_DIR / f"{sat}_tau_tuning_metrics.json")

    best, threshold = _select_best_tau(tuning_results)
    if threshold:
        print(f"  [{sat}] Best tau (RawCal_PICP ≥ {threshold}): {best['Tau_Pair']}"
              f"  CQR PICP={best['CQR_PICP']:.4f}  CQR MPIW={best['CQR_MPIW']:.4f}")
    else:
        print(f"  [{sat}] ⚠ No tau met RawCal_PICP ≥ 0.90 — fallback: {best['Tau_Pair']}"
              f"  CQR PICP={best['CQR_PICP']:.4f}  CQR MPIW={best['CQR_MPIW']:.4f}")

print("  Phase 7 complete.")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASES 4 / 6 / 7 COMPLETE — SUMMARY")
print("=" * 70)

for sat in ['EOS-04', 'Sentinel-1']:
    print(f"\n  {sat}")
    pi_path = PI_DIR / f"{sat}_metrics.json"
    if pi_path.exists():
        pi = json.loads(pi_path.read_text())
        valid_pi = {k: v for k, v in pi.items() if isinstance(v, dict) and v.get('test')}
        if valid_pi:
            best_pi = min(valid_pi.items(), key=lambda x: x[1]['test']['MPIW'])
            print(f"    PI best arch  : {best_pi[0]}  "
                  f"PICP={best_pi[1]['test']['PICP']:.4f}  "
                  f"MPIW={best_pi[1]['test']['MPIW']:.4f}")
    cqr = json.loads((CQR_DIR / f"{sat}_conformal_metrics.json").read_text())
    for method, v in cqr.items():
        cross = f"  crossing={v['crossing_rate']*100:.1f}%" if 'crossing_rate' in v else ""
        if v.get('MPIW') is not None:
            print(f"    {method:28s}: PICP={v['PICP']:.4f}  MPIW={v['MPIW']:.4f}{cross}")
        else:
            print(f"    {method:28s}: INVERTED (null)")

    tau_path = CQR_DIR / f"{sat}_tau_tuning_metrics.json"
    if tau_path.exists():
        tau_res = json.loads(tau_path.read_text())
        best, threshold = _select_best_tau(tau_res)
        label = f"cal-criterion ≥{threshold}" if threshold else "fallback"
        print(f"    ANN_CQR best tau             : "
              f"PICP={best['CQR_PICP']:.4f}  MPIW={best['CQR_MPIW']:.4f}"
              f"  [{best['Tau_Pair']}  {label}]")

print(f"\n  Output → {OUTPUT_PATH}")
print("Done.")
