"""
run_is_recompute.py
Recomputes IS (Winkler 1972) for Phases 4-10 and 16-20 by retraining best
configs and generating per-sample [lo, hi, y] predictions.

Saves results to output/is_results.json and prints a summary table.

IS formula: mean[(hi-lo) + (2/alpha) * (max(0, lo-y) + max(0, y-hi))]
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']   = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS']  = '0'
os.environ['TF_DETERMINISTIC_OPS']   = '1'
os.environ['PYTHONHASHSEED']         = '42'
import random; random.seed(42)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; plt.show = lambda: None

import json, warnings, itertools
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np; np.random.seed(42)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from scipy.spatial.distance import cdist
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.callbacks import EarlyStopping
tf.get_logger().setLevel('ERROR')

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / 'data'
OUT_PATH  = ROOT / 'output'

SEED = 42

X_COLS_EOS6  = ['HH-pol', 'HV-pol', 'cross_pol_ratio', 'month_sin', 'month_cos', 'crop_encoded']
X_COLS_SEN6  = ['VH-pol', 'VV-pol', 'cross_pol_ratio', 'month_sin', 'month_cos', 'crop_encoded']
X_COLS_EOS7  = X_COLS_EOS6 + ['NDVI']
X_COLS_SEN7  = X_COLS_SEN6 + ['NDVI']
X_COLS_EOS2  = ['HH-pol', 'HV-pol']   # Phase 5 original 2-feature setup
X_COLS_SEN2  = ['VH-pol', 'VV-pol']
Y_COL        = 'SM1 (%)'

print("=" * 70)
print("  IS RECOMPUTE — Phases 4-10 and 16-20")
print("=" * 70)

# ── helpers ─────────────────────────────────────────────────────────────────────

Y_RANGE = {'EOS-04': 50.0667, 'Sentinel-1': 55.10}  # fixed normalization range

def interval_score(y, lo, hi, alpha):
    width = hi - lo
    penalty = (2 / alpha) * (np.maximum(0, lo - y) + np.maximum(0, y - hi))
    return float(np.mean(width + penalty))

def cwc_score(picp, mpiw, sat, mu_c, eta=50):
    pinaw = mpiw / Y_RANGE[sat]
    gamma = 0 if picp >= mu_c else 1
    return float(pinaw * (1 + gamma * np.exp(-eta * (picp - mu_c))))

def pi_metrics(y, lo, hi):
    cov = np.sum((y >= lo) & (y <= hi))
    return round(float(cov / len(y)), 6), round(float(np.mean(hi - lo)), 6)

def split_70_10_10_10(X, y):
    X_tr, X_tmp, y_tr, y_tmp   = train_test_split(X, y, train_size=0.7,  random_state=SEED)
    X_v,  X_tmp2, y_v,  y_tmp2 = train_test_split(X_tmp, y_tmp, train_size=1/3, random_state=SEED)
    X_cal, X_te, y_cal, y_te   = train_test_split(X_tmp2, y_tmp2, test_size=0.5, random_state=SEED)
    return X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te

def split_80_10_10(X, y):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, train_size=0.8, random_state=SEED)
    X_v,  X_te,  y_v,  y_te  = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED)
    return X_tr, X_v, X_te, y_tr, y_v, y_te

def scale_X(*splits):
    sc = MinMaxScaler()
    out = [sc.fit_transform(splits[0])] + [sc.transform(s) for s in splits[1:]]
    return out

def apply_cqr(y_cal, lo_cal, hi_cal, lo_te, hi_te, alpha=0.05):
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    q = np.quantile(scores, 1 - alpha, method='higher')
    return lo_te - q, hi_te + q

def pinball(tau):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(tau * e, (tau - 1) * e))
    return loss

def build_single_ann(input_dim, hidden):
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    return tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(hidden, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

def build_dual_ann(input_dim):
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    inp    = tf.keras.Input(shape=(input_dim,))
    x      = tf.keras.layers.Dense(16, activation='relu')(inp)
    x      = tf.keras.layers.Dropout(0.09)(x)
    x      = tf.keras.layers.Dense(8,  activation='relu')(x)
    x      = tf.keras.layers.Dropout(0.09)(x)
    lo_out = tf.keras.layers.Dense(1, name='lo')(x)
    hi_out = tf.keras.layers.Dense(1, name='hi')(x)
    return tf.keras.Model(inputs=inp, outputs=[lo_out, hi_out])

def build_ann_regressor(input_dim):
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    inp = tf.keras.Input(shape=(input_dim,))
    x   = tf.keras.layers.Dense(16, activation='relu')(inp)
    x   = tf.keras.layers.Dropout(0.09)(x)
    x   = tf.keras.layers.Dense(8,  activation='relu')(x)
    x   = tf.keras.layers.Dropout(0.09)(x)
    out = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=inp, outputs=out)

def train_ann(model, X_tr, X_v, y_tr, y_v, loss_fn, epochs=500, lr=1e-3):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss_fn)
    cb = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_v, y_v),
              epochs=epochs, batch_size=32, verbose=0, callbacks=[cb])
    return model

def train_dual_ann(X_tr, X_v, y_tr, y_v, lo_tau, hi_tau, input_dim, epochs=500):
    model = build_dual_ann(input_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=[pinball(lo_tau), pinball(hi_tau)],
                  loss_weights=[1.0, 1.0])
    cb = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    model.fit(X_tr, [y_tr, y_tr], validation_data=(X_v, [y_v, y_v]),
              epochs=epochs, batch_size=32, verbose=0, callbacks=[cb])
    preds = model.predict(X_tr, verbose=0)   # dummy
    return model

def predict_dual(model, X):
    preds = model.predict(X, verbose=0)
    lo = preds[0].flatten(); hi = preds[1].flatten()
    crossed = lo > hi
    return np.where(crossed, hi, lo), np.where(crossed, lo, hi)

# QSVR helpers (from run_phase8b_qsvr_cgrid.py)
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
    return alpha[:n] - alpha[n:]

def predict_qsvr(X_train, X_pred, gamma, beta):
    return rbf_kernel(X_pred, X_train, gamma) @ beta

def check(picp_got, mpiw_got, picp_exp, mpiw_exp, label, tol_picp=0.003, tol_mpiw=1.0):
    ok_p = abs(picp_got - picp_exp) <= tol_picp
    ok_m = abs(mpiw_got - mpiw_exp) <= tol_mpiw
    status = "✓" if (ok_p and ok_m) else "⚠"
    print(f"  {status} {label}: PICP={picp_got*100:.2f}% (exp {picp_exp*100:.2f}%)  "
          f"MPIW={mpiw_got:.2f} (exp {mpiw_exp:.2f})")
    return ok_p and ok_m

# ── load data ────────────────────────────────────────────────────────────────────
print("\n[Data] Loading CSVs …")
eos6_df  = pd.read_csv(DATA_PATH / 'eos-04-enhanced.csv')
sen6_df  = pd.read_csv(DATA_PATH / 'sentinel-1-enhanced.csv')
eos7_df  = pd.read_csv(DATA_PATH / 'eos-04-enhanced-ndvi.csv')
sen7_df  = pd.read_csv(DATA_PATH / 'sentinel-1-enhanced-ndvi.csv')

results = {}  # {phase_label: {sensor: {PICP, MPIW, IS, matched}}}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — ANN-QR raw pinball  (80/10/10, 7-feature NDVI, no CQR)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  PHASE 4 — ANN-QR raw pinball")
print("─"*70)
alpha4 = 0.05

BEST_ARCH = {'EOS-04': 16, 'Sentinel-1': 8}
EXP_P4 = {'EOS-04': (0.9684, 39.64), 'Sentinel-1': (0.9670, 44.56)}

for sat, df, xcols in [('EOS-04', eos7_df, X_COLS_EOS7), ('Sentinel-1', sen7_df, X_COLS_SEN7)]:
    mask = df[Y_COL] != 50
    X = df.loc[mask, xcols].values; y = df.loc[mask, Y_COL].values
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_80_10_10(X, y)
    X_tr_s, X_v_s, X_te_s = scale_X(X_tr, X_v, X_te)

    hidden = BEST_ARCH[sat]
    lo_model = build_single_ann(X.shape[1], hidden)
    hi_model = build_single_ann(X.shape[1], hidden)
    lo_model = train_ann(lo_model, X_tr_s, X_v_s, y_tr, y_v, pinball(0.025), epochs=1000)
    hi_model = train_ann(hi_model, X_tr_s, X_v_s, y_tr, y_v, pinball(0.975), epochs=1000)

    lo_te = lo_model.predict(X_te_s, verbose=0).flatten()
    hi_te = hi_model.predict(X_te_s, verbose=0).flatten()
    # crossing correction
    lo_f = np.minimum(lo_te, hi_te); hi_f = np.maximum(lo_te, hi_te)

    picp, mpiw = pi_metrics(y_te, lo_f, hi_f)
    ep, em = EXP_P4[sat]
    matched = check(picp, mpiw, ep, em, f"Phase4 {sat}")
    is_val  = interval_score(y_te, lo_f, hi_f, alpha4)
    cwc_val = cwc_score(picp, mpiw, sat, mu_c=0.95)
    print(f"    IS={is_val:.2f}  CWC={cwc_val:.4f}")
    results.setdefault('Ph4', {})[sat] = {'PICP': picp, 'MPIW': mpiw, 'CWC': round(cwc_val,4), 'IS': round(is_val, 2), 'matched': matched}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — MAPIE GBR conformal  (80/10/10, 2-feature, GBR CQR via MAPIE)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  PHASE 5 — MAPIE GBR conformal (80/10/10, 2-feature baseline)")
print("─"*70)
alpha5 = 0.05

EXP_P5 = {'EOS-04': (0.9685, 39.05), 'Sentinel-1': (0.9672, 43.54)}

for sat, df, xcols in [('EOS-04', eos7_df, X_COLS_EOS2), ('Sentinel-1', sen7_df, X_COLS_SEN2)]:
    mask = df[Y_COL] != 50
    X = df.loc[mask, xcols].values; y = df.loc[mask, Y_COL].values
    # MAPIE uses train_conformalize_test_split ≈ two sequential train_test_splits
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, train_size=0.8, random_state=42)
    X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

    sc = MinMaxScaler()
    X_tr_s  = sc.fit_transform(X_tr)
    X_cal_s = sc.transform(X_cal)
    X_te_s  = sc.transform(X_te)

    gbm_lo = GradientBoostingRegressor(loss='quantile', alpha=0.025, random_state=SEED)
    gbm_hi = GradientBoostingRegressor(loss='quantile', alpha=0.975, random_state=SEED)
    gbm_lo.fit(X_tr_s, y_tr); gbm_hi.fit(X_tr_s, y_tr)

    p_lo_cal = np.minimum(gbm_lo.predict(X_cal_s), gbm_hi.predict(X_cal_s))
    p_hi_cal = np.maximum(gbm_lo.predict(X_cal_s), gbm_hi.predict(X_cal_s))
    p_lo_te  = np.minimum(gbm_lo.predict(X_te_s),  gbm_hi.predict(X_te_s))
    p_hi_te  = np.maximum(gbm_lo.predict(X_te_s),  gbm_hi.predict(X_te_s))

    lo_te, hi_te = apply_cqr(y_cal, p_lo_cal, p_hi_cal, p_lo_te, p_hi_te, alpha5)

    picp, mpiw = pi_metrics(y_te, lo_te, hi_te)
    ep, em = EXP_P5[sat]
    matched = check(picp, mpiw, ep, em, f"Phase5 {sat}")
    is_val  = interval_score(y_te, lo_te, hi_te, alpha5)
    cwc_val = cwc_score(picp, mpiw, sat, mu_c=0.95)
    print(f"    IS={is_val:.2f}  CWC={cwc_val:.4f}")
    results.setdefault('Ph5', {})[sat] = {'PICP': picp, 'MPIW': mpiw, 'CWC': round(cwc_val,4), 'IS': round(is_val, 2), 'matched': matched}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — CQR suite  (70/10/10/10, 6-feature, multiple methods)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  PHASE 6 — CQR suite (70/10/10/10, 6-feature)")
print("─"*70)
alpha6 = 0.05

EXP_P6 = {
    'GBM_CQR':          {'EOS-04': (0.9659, 35.70), 'Sentinel-1': (0.9542, 35.75)},
    'SVM_Split':         {'EOS-04': (0.9317, 38.38), 'Sentinel-1': (0.9608, 40.81)},
    'ANN_Split':         {'EOS-04': (0.9610, 39.32), 'Sentinel-1': (0.9542, 39.39)},
    'ANN_CQR':           {'EOS-04': (0.9220, 41.18), 'Sentinel-1': (0.9804, 41.77)},
}

for sat, df, xcols in [('EOS-04', eos6_df, X_COLS_EOS6), ('Sentinel-1', sen6_df, X_COLS_SEN6)]:
    mask = df[Y_COL] != 50
    X = df.loc[mask, xcols].values; y = df.loc[mask, Y_COL].values
    X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te = split_70_10_10_10(X, y)
    X_tr_s, X_v_s, X_cal_s, X_te_s = scale_X(X_tr, X_v, X_cal, X_te)

    # --- GBM CQR ---
    gbm_lo = GradientBoostingRegressor(loss='quantile', alpha=0.025, n_estimators=200,
                                        max_depth=4, learning_rate=0.05, random_state=SEED)
    gbm_hi = GradientBoostingRegressor(loss='quantile', alpha=0.975, n_estimators=200,
                                        max_depth=4, learning_rate=0.05, random_state=SEED)
    gbm_lo.fit(X_tr_s, y_tr); gbm_hi.fit(X_tr_s, y_tr)
    p_lo_cal = np.minimum(gbm_lo.predict(X_cal_s), gbm_hi.predict(X_cal_s))
    p_hi_cal = np.maximum(gbm_lo.predict(X_cal_s), gbm_hi.predict(X_cal_s))
    p_lo_te  = np.minimum(gbm_lo.predict(X_te_s),  gbm_hi.predict(X_te_s))
    p_hi_te  = np.maximum(gbm_lo.predict(X_te_s),  gbm_hi.predict(X_te_s))
    lo6_gbm, hi6_gbm = apply_cqr(y_cal, p_lo_cal, p_hi_cal, p_lo_te, p_hi_te, alpha6)
    picp, mpiw = pi_metrics(y_te, lo6_gbm, hi6_gbm)
    ep, em = EXP_P6['GBM_CQR'][sat]
    matched = check(picp, mpiw, ep, em, f"Phase6 GBM_CQR {sat}")
    is_val  = interval_score(y_te, lo6_gbm, hi6_gbm, alpha6)
    cwc_val = cwc_score(picp, mpiw, sat, mu_c=0.95)
    print(f"    IS={is_val:.2f}  CWC={cwc_val:.4f}")
    results.setdefault('Ph6_GBM_CQR', {})[sat] = {'PICP': picp, 'MPIW': mpiw, 'CWC': round(cwc_val,4), 'IS': round(is_val, 2), 'matched': matched}

    # --- SVM Split Conformal ---
    svr = SVR(kernel='rbf'); svr.fit(X_tr_s, y_tr)
    q_svm = np.quantile(np.abs(y_cal - svr.predict(X_cal_s)), 1 - alpha6, method='higher')
    pred_svm = svr.predict(X_te_s)
    lo_svm, hi_svm = pred_svm - q_svm, pred_svm + q_svm
    picp, mpiw = pi_metrics(y_te, lo_svm, hi_svm)
    ep, em = EXP_P6['SVM_Split'][sat]
    matched = check(picp, mpiw, ep, em, f"Phase6 SVM_Split {sat}")
    is_val  = interval_score(y_te, lo_svm, hi_svm, alpha6)
    cwc_val = cwc_score(picp, mpiw, sat, mu_c=0.95)
    results.setdefault('Ph6_SVM_Split', {})[sat] = {'PICP': picp, 'MPIW': mpiw, 'CWC': round(cwc_val,4), 'IS': round(is_val, 2), 'matched': matched}

    # --- ANN Split Conformal ---
    ann_reg = build_ann_regressor(X.shape[1])
    ann_reg = train_ann(ann_reg, X_tr_s, X_v_s, y_tr, y_v, 'mse', epochs=500)
    pred_cal_ann = ann_reg.predict(X_cal_s, verbose=0).flatten()
    pred_te_ann  = ann_reg.predict(X_te_s,  verbose=0).flatten()
    q_ann = np.quantile(np.abs(y_cal - pred_cal_ann), 1 - alpha6, method='higher')
    lo_ans, hi_ans = pred_te_ann - q_ann, pred_te_ann + q_ann
    picp, mpiw = pi_metrics(y_te, lo_ans, hi_ans)
    ep, em = EXP_P6['ANN_Split'][sat]
    matched = check(picp, mpiw, ep, em, f"Phase6 ANN_Split {sat}")
    is_val  = interval_score(y_te, lo_ans, hi_ans, alpha6)
    cwc_val = cwc_score(picp, mpiw, sat, mu_c=0.95)
    results.setdefault('Ph6_ANN_Split', {})[sat] = {'PICP': picp, 'MPIW': mpiw, 'CWC': round(cwc_val,4), 'IS': round(is_val, 2), 'matched': matched}

    # --- ANN CQR (dual output) ---
    dual_m = train_dual_ann(X_tr_s, X_v_s, y_tr, y_v, 0.025, 0.975, X.shape[1])
    lo_cal_d, hi_cal_d = predict_dual(dual_m, X_cal_s)
    lo_te_d,  hi_te_d  = predict_dual(dual_m, X_te_s)
    lo6_acqr, hi6_acqr = apply_cqr(y_cal, lo_cal_d, hi_cal_d, lo_te_d, hi_te_d, alpha6)
    picp, mpiw = pi_metrics(y_te, lo6_acqr, hi6_acqr)
    ep, em = EXP_P6['ANN_CQR'][sat]
    matched = check(picp, mpiw, ep, em, f"Phase6 ANN_CQR {sat}")
    is_val  = interval_score(y_te, lo6_acqr, hi6_acqr, alpha6)
    cwc_val = cwc_score(picp, mpiw, sat, mu_c=0.95)
    results.setdefault('Ph6_ANN_CQR', {})[sat] = {'PICP': picp, 'MPIW': mpiw, 'CWC': round(cwc_val,4), 'IS': round(is_val, 2), 'matched': matched}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — Tau tuning best pair (70/10/10/10, 6-feature)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  PHASE 7 — Tau tuning (70/10/10/10, 6-feature)")
print("─"*70)
alpha7 = 0.05
EXP_P7 = {'EOS-04': (0.8878, 34.52), 'Sentinel-1': (0.9542, 37.40)}

for sat, df, xcols in [('EOS-04', eos6_df, X_COLS_EOS6), ('Sentinel-1', sen6_df, X_COLS_SEN6)]:
    mask = df[Y_COL] != 50
    X = df.loc[mask, xcols].values; y = df.loc[mask, Y_COL].values
    X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te = split_70_10_10_10(X, y)
    X_tr_s, X_v_s, X_cal_s, X_te_s = scale_X(X_tr, X_v, X_cal, X_te)

    dual_m7 = train_dual_ann(X_tr_s, X_v_s, y_tr, y_v, 0.01, 0.96, X.shape[1])
    lo_cal7, hi_cal7 = predict_dual(dual_m7, X_cal_s)
    lo_te7,  hi_te7  = predict_dual(dual_m7, X_te_s)
    lo7, hi7 = apply_cqr(y_cal, lo_cal7, hi_cal7, lo_te7, hi_te7, alpha7)
    picp, mpiw = pi_metrics(y_te, lo7, hi7)
    ep, em = EXP_P7[sat]
    matched = check(picp, mpiw, ep, em, f"Phase7 {sat}")
    is_val  = interval_score(y_te, lo7, hi7, alpha7)
    cwc_val = cwc_score(picp, mpiw, sat, mu_c=0.95)
    print(f"    IS={is_val:.2f}  CWC={cwc_val:.4f}")
    results.setdefault('Ph7', {})[sat] = {'PICP': picp, 'MPIW': mpiw, 'CWC': round(cwc_val,4), 'IS': round(is_val, 2), 'matched': matched}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 8 — QSVR γ grid best  (80/10/10, 7-feature, no CQR)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  PHASE 8 — QSVR γ grid (80/10/10, 7-feature)")
print("─"*70)
alpha8 = 0.05
# Phase 8 best: EOS C=64 γ=2^1, S1 C=64 γ=2^-4
BEST8 = {'EOS-04': (64, 2.0), 'Sentinel-1': (64, 0.0625)}
EXP_P8 = {'EOS-04': (0.9512, 30.91), 'Sentinel-1': (0.9542, 37.56)}

for sat, df, xcols in [('EOS-04', eos7_df, X_COLS_EOS7), ('Sentinel-1', sen7_df, X_COLS_SEN7)]:
    mask = df[Y_COL] != 50
    X = df.loc[mask, xcols].values; y = df.loc[mask, Y_COL].values
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_80_10_10(X, y)
    sc = MinMaxScaler()
    X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

    C, gamma = BEST8[sat]
    print(f"  [{sat}] C={int(C)} γ={gamma} …", end=' ', flush=True)
    try:
        beta_lo = fit_qsvr(X_tr_s, y_tr, gamma, C, 0.025)
        beta_hi = fit_qsvr(X_tr_s, y_tr, gamma, C, 0.975)
        lo_te8 = predict_qsvr(X_tr_s, X_te_s, gamma, beta_lo)
        hi_te8 = predict_qsvr(X_tr_s, X_te_s, gamma, beta_hi)
        picp, mpiw = pi_metrics(y_te, lo_te8, hi_te8)
        ep, em = EXP_P8[sat]
        matched = check(picp, mpiw, ep, em, f"Phase8 {sat}")
        is_val  = interval_score(y_te, lo_te8, hi_te8, alpha8)
        cwc_val = cwc_score(picp, mpiw, sat, mu_c=0.95)
        print(f"    IS={is_val:.2f}  CWC={cwc_val:.4f}")
        results.setdefault('Ph8', {})[sat] = {'PICP': picp, 'MPIW': mpiw, 'CWC': round(cwc_val,4), 'IS': round(is_val, 2), 'matched': matched}
    except Exception as e:
        print(f"FAILED: {e}")
        results.setdefault('Ph8', {})[sat] = {'error': str(e)}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 8b — QSVR C×γ best  (80/10/10, 7-feature, no CQR)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  PHASE 8b — QSVR C×γ (80/10/10, 7-feature)")
print("─"*70)
alpha8b = 0.05
# Phase 8b best: EOS C=256 γ=1, S1 C=64 γ=0.0625 (same as Phase 8)
BEST8b = {'EOS-04': (256, 1.0), 'Sentinel-1': (64, 0.0625)}
EXP_P8b = {'EOS-04': (0.9561, 30.77), 'Sentinel-1': (0.9542, 37.56)}

for sat, df, xcols in [('EOS-04', eos7_df, X_COLS_EOS7), ('Sentinel-1', sen7_df, X_COLS_SEN7)]:
    mask = df[Y_COL] != 50
    X = df.loc[mask, xcols].values; y = df.loc[mask, Y_COL].values
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_80_10_10(X, y)
    sc = MinMaxScaler()
    X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

    C, gamma = BEST8b[sat]
    print(f"  [{sat}] C={int(C)} γ={gamma} …", end=' ', flush=True)
    try:
        beta_lo = fit_qsvr(X_tr_s, y_tr, gamma, C, 0.025)
        beta_hi = fit_qsvr(X_tr_s, y_tr, gamma, C, 0.975)
        lo_te8b = predict_qsvr(X_tr_s, X_te_s, gamma, beta_lo)
        hi_te8b = predict_qsvr(X_tr_s, X_te_s, gamma, beta_hi)
        picp, mpiw = pi_metrics(y_te, lo_te8b, hi_te8b)
        ep, em = EXP_P8b[sat]
        matched = check(picp, mpiw, ep, em, f"Phase8b {sat}")
        is_val  = interval_score(y_te, lo_te8b, hi_te8b, alpha8b)
        cwc_val = cwc_score(picp, mpiw, sat, mu_c=0.95)
        print(f"    IS={is_val:.2f}  CWC={cwc_val:.4f}")
        results.setdefault('Ph8b', {})[sat] = {'PICP': picp, 'MPIW': mpiw, 'CWC': round(cwc_val,4), 'IS': round(is_val, 2), 'matched': matched}
    except Exception as e:
        print(f"FAILED: {e}")
        results.setdefault('Ph8b', {})[sat] = {'error': str(e)}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 9a — CQR-d  (70/10/10/10, 7-feature)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  PHASE 9a — CQR-d (70/10/10/10, 7-feature)")
print("─"*70)
alpha9a = 0.05
EXP_P9a = {'EOS-04': (0.9610, 35.34), 'Sentinel-1': (0.9412, 35.28)}

for sat, df, xcols in [('EOS-04', eos7_df, X_COLS_EOS7), ('Sentinel-1', sen7_df, X_COLS_SEN7)]:
    mask = df[Y_COL] != 50
    X = df.loc[mask, xcols].values; y = df.loc[mask, Y_COL].values
    X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te = split_70_10_10_10(X, y)
    X_tr_s, X_v_s, X_cal_s, X_te_s = scale_X(X_tr, X_v, X_cal, X_te)

    gbm_lo9a = GradientBoostingRegressor(loss='quantile', alpha=0.025, n_estimators=200,
                                          max_depth=4, learning_rate=0.05, random_state=SEED)
    gbm_hi9a = GradientBoostingRegressor(loss='quantile', alpha=0.975, n_estimators=200,
                                          max_depth=4, learning_rate=0.05, random_state=SEED)
    gbm_lo9a.fit(X_tr_s, y_tr); gbm_hi9a.fit(X_tr_s, y_tr)
    p_lo_cal9a = gbm_lo9a.predict(X_cal_s); p_hi_cal9a = gbm_hi9a.predict(X_cal_s)
    p_lo_cal9a, p_hi_cal9a = np.minimum(p_lo_cal9a, p_hi_cal9a), np.maximum(p_lo_cal9a, p_hi_cal9a)

    raw_width_cal = p_hi_cal9a - p_lo_cal9a
    width_floor   = max(np.percentile(raw_width_cal, 1), 0.5)
    sigma_cal     = np.maximum(raw_width_cal, width_floor)
    scores_cal9a  = np.maximum(p_lo_cal9a - y_cal, y_cal - p_hi_cal9a)
    scores_norm   = scores_cal9a / sigma_cal
    q_hat9a       = np.quantile(scores_norm, 1 - alpha9a, method='higher')

    p_lo_te9a = gbm_lo9a.predict(X_te_s); p_hi_te9a = gbm_hi9a.predict(X_te_s)
    p_lo_te9a, p_hi_te9a = np.minimum(p_lo_te9a, p_hi_te9a), np.maximum(p_lo_te9a, p_hi_te9a)
    raw_width_te = p_hi_te9a - p_lo_te9a
    sigma_te = np.maximum(raw_width_te, width_floor)
    lo9a = p_lo_te9a - q_hat9a * sigma_te
    hi9a = p_hi_te9a + q_hat9a * sigma_te

    picp, mpiw = pi_metrics(y_te, lo9a, hi9a)
    ep, em = EXP_P9a[sat]
    matched = check(picp, mpiw, ep, em, f"Phase9a {sat}")
    is_val  = interval_score(y_te, lo9a, hi9a, alpha9a)
    cwc_val = cwc_score(picp, mpiw, sat, mu_c=0.95)
    print(f"    IS={is_val:.2f}  CWC={cwc_val:.4f}")
    results.setdefault('Ph9a', {})[sat] = {'PICP': picp, 'MPIW': mpiw, 'CWC': round(cwc_val,4), 'IS': round(is_val, 2), 'matched': matched}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 9c — Mondrian CQR  (70/10/10/10, 7-feature)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  PHASE 9c — Mondrian CQR (70/10/10/10, 7-feature)")
print("─"*70)
alpha9c = 0.05
MONDRIAN_GROUPS = {'EOS-04': [5, 26, 18], 'Sentinel-1': [2, 19]}
CROP_IDX = 5  # crop_encoded is always column index 5 in 7-feature set
EXP_P9c = {'EOS-04': (0.9317, 36.80), 'Sentinel-1': (0.9216, 33.87)}

for sat, df, xcols in [('EOS-04', eos7_df, X_COLS_EOS7), ('Sentinel-1', sen7_df, X_COLS_SEN7)]:
    mask = df[Y_COL] != 50
    X = df.loc[mask, xcols].values; y = df.loc[mask, Y_COL].values
    X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te = split_70_10_10_10(X, y)
    X_tr_s, X_v_s, X_cal_s, X_te_s = scale_X(X_tr, X_v, X_cal, X_te)

    gbm_lo9c = GradientBoostingRegressor(loss='quantile', alpha=0.025, n_estimators=200,
                                          max_depth=4, learning_rate=0.05, random_state=SEED)
    gbm_hi9c = GradientBoostingRegressor(loss='quantile', alpha=0.975, n_estimators=200,
                                          max_depth=4, learning_rate=0.05, random_state=SEED)
    gbm_lo9c.fit(X_tr_s, y_tr); gbm_hi9c.fit(X_tr_s, y_tr)
    p_lo_cal9c = gbm_lo9c.predict(X_cal_s); p_hi_cal9c = gbm_hi9c.predict(X_cal_s)
    p_lo_te9c  = gbm_lo9c.predict(X_te_s);  p_hi_te9c  = gbm_hi9c.predict(X_te_s)
    p_lo_cal9c, p_hi_cal9c = np.minimum(p_lo_cal9c, p_hi_cal9c), np.maximum(p_lo_cal9c, p_hi_cal9c)
    p_lo_te9c,  p_hi_te9c  = np.minimum(p_lo_te9c,  p_hi_te9c),  np.maximum(p_lo_te9c,  p_hi_te9c)
    scores_cal9c = np.maximum(p_lo_cal9c - y_cal, y_cal - p_hi_cal9c)

    major = MONDRIAN_GROUPS[sat]
    crops_cal9c = X_cal[:, CROP_IDX].astype(int)
    crops_te9c  = X_te[:,  CROP_IDX].astype(int)
    def get_group(c): return c if c in major else -1
    group_cal9c = np.array([get_group(c) for c in crops_cal9c])
    group_te9c  = np.array([get_group(c) for c in crops_te9c])

    q_hat_per_group9c = {}
    for g in sorted(set(major) | {-1}):
        mask_g = group_cal9c == g
        if mask_g.sum() > 0:
            q_hat_per_group9c[g] = np.quantile(scores_cal9c[mask_g], 1-alpha9c, method='higher')
        else:
            q_hat_per_group9c[g] = np.inf
    q_std9c = np.quantile(scores_cal9c, 1-alpha9c, method='higher')

    lo9c = np.zeros(len(y_te)); hi9c = np.zeros(len(y_te))
    for j in range(len(y_te)):
        g = group_te9c[j]
        q = q_hat_per_group9c.get(g, q_std9c)
        lo9c[j] = p_lo_te9c[j] - q
        hi9c[j] = p_hi_te9c[j] + q

    picp, mpiw = pi_metrics(y_te, lo9c, hi9c)
    ep, em = EXP_P9c[sat]
    matched = check(picp, mpiw, ep, em, f"Phase9c {sat}")
    is_val  = interval_score(y_te, lo9c, hi9c, alpha9c)
    cwc_val = cwc_score(picp, mpiw, sat, mu_c=0.95)
    print(f"    IS={is_val:.2f}  CWC={cwc_val:.4f}")
    results.setdefault('Ph9c', {})[sat] = {'PICP': picp, 'MPIW': mpiw, 'CWC': round(cwc_val,4), 'IS': round(is_val, 2), 'matched': matched}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 10 — Tuned GBM CQR  (70/10/10/10, 7-feature)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  PHASE 10 — Tuned GBM CQR (70/10/10/10, 7-feature)")
print("─"*70)
alpha10 = 0.05
BEST10 = {
    'EOS-04':     {'msl': 1,  'depth': 4, 'n': 300, 'sub': 0.8, 'tau_lo': 0.025, 'tau_hi': 0.975},
    'Sentinel-1': {'msl': 5,  'depth': 4, 'n': 300, 'sub': 0.8, 'tau_lo': 0.1,   'tau_hi': 0.9},
}
EXP_P10 = {'EOS-04': (0.9561, 33.40), 'Sentinel-1': (0.9608, 30.61)}

for sat, df, xcols in [('EOS-04', eos7_df, X_COLS_EOS7), ('Sentinel-1', sen7_df, X_COLS_SEN7)]:
    mask = df[Y_COL] != 50
    X = df.loc[mask, xcols].values; y = df.loc[mask, Y_COL].values
    X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te = split_70_10_10_10(X, y)
    X_tr_s, X_v_s, X_cal_s, X_te_s = scale_X(X_tr, X_v, X_cal, X_te)

    p = BEST10[sat]
    gbm_lo10 = GradientBoostingRegressor(loss='quantile', alpha=p['tau_lo'], n_estimators=p['n'],
                                          max_depth=p['depth'], learning_rate=0.05,
                                          min_samples_leaf=p['msl'], subsample=p['sub'],
                                          random_state=SEED)
    gbm_hi10 = GradientBoostingRegressor(loss='quantile', alpha=p['tau_hi'], n_estimators=p['n'],
                                          max_depth=p['depth'], learning_rate=0.05,
                                          min_samples_leaf=p['msl'], subsample=p['sub'],
                                          random_state=SEED)
    gbm_lo10.fit(X_tr_s, y_tr); gbm_hi10.fit(X_tr_s, y_tr)
    p_lo_cal10 = np.minimum(gbm_lo10.predict(X_cal_s), gbm_hi10.predict(X_cal_s))
    p_hi_cal10 = np.maximum(gbm_lo10.predict(X_cal_s), gbm_hi10.predict(X_cal_s))
    p_lo_te10  = np.minimum(gbm_lo10.predict(X_te_s),  gbm_hi10.predict(X_te_s))
    p_hi_te10  = np.maximum(gbm_lo10.predict(X_te_s),  gbm_hi10.predict(X_te_s))
    lo10, hi10 = apply_cqr(y_cal, p_lo_cal10, p_hi_cal10, p_lo_te10, p_hi_te10, alpha10)
    picp, mpiw = pi_metrics(y_te, lo10, hi10)
    ep, em = EXP_P10[sat]
    matched = check(picp, mpiw, ep, em, f"Phase10 {sat}")
    is_val  = interval_score(y_te, lo10, hi10, alpha10)
    cwc_val = cwc_score(picp, mpiw, sat, mu_c=0.95)
    print(f"    IS={is_val:.2f}  CWC={cwc_val:.4f}")
    results.setdefault('Ph10', {})[sat] = {'PICP': picp, 'MPIW': mpiw, 'CWC': round(cwc_val,4), 'IS': round(is_val, 2), 'matched': matched}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 20 (val-criterion winner)  (70/10/10/10, 7-feature, α=0.10)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  PHASE 20 — GBM CQR finetune val-criterion winner (α=0.10)")
print("─"*70)
alpha20 = 0.10
BEST20 = {
    'EOS-04':     {'lr': 0.03,  'n': 400, 'tau_lo': 0.15, 'tau_hi': 0.85, 'msl': 15, 'depth': 5, 'sub': 0.9},
    'Sentinel-1': {'lr': 0.03,  'n': 500, 'tau_lo': 0.15, 'tau_hi': 0.85, 'msl': 20, 'depth': 5, 'sub': 1.0},
}
EXP_P20 = {'EOS-04': (0.8683, 26.37), 'Sentinel-1': (0.8627, 24.67)}

for sat, df, xcols in [('EOS-04', eos7_df, X_COLS_EOS7), ('Sentinel-1', sen7_df, X_COLS_SEN7)]:
    mask = df[Y_COL] != 50
    X = df.loc[mask, xcols].values; y = df.loc[mask, Y_COL].values
    X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te = split_70_10_10_10(X, y)
    X_tr_s, X_v_s, X_cal_s, X_te_s = scale_X(X_tr, X_v, X_cal, X_te)

    p = BEST20[sat]
    gbm_lo20 = GradientBoostingRegressor(loss='quantile', alpha=p['tau_lo'], n_estimators=p['n'],
                                          max_depth=p['depth'], learning_rate=p['lr'],
                                          min_samples_leaf=p['msl'], subsample=p['sub'],
                                          random_state=SEED)
    gbm_hi20 = GradientBoostingRegressor(loss='quantile', alpha=p['tau_hi'], n_estimators=p['n'],
                                          max_depth=p['depth'], learning_rate=p['lr'],
                                          min_samples_leaf=p['msl'], subsample=p['sub'],
                                          random_state=SEED)
    gbm_lo20.fit(X_tr_s, y_tr); gbm_hi20.fit(X_tr_s, y_tr)
    p_lo_cal20 = np.minimum(gbm_lo20.predict(X_cal_s), gbm_hi20.predict(X_cal_s))
    p_hi_cal20 = np.maximum(gbm_lo20.predict(X_cal_s), gbm_hi20.predict(X_cal_s))
    p_lo_te20  = np.minimum(gbm_lo20.predict(X_te_s),  gbm_hi20.predict(X_te_s))
    p_hi_te20  = np.maximum(gbm_lo20.predict(X_te_s),  gbm_hi20.predict(X_te_s))
    lo20, hi20 = apply_cqr(y_cal, p_lo_cal20, p_hi_cal20, p_lo_te20, p_hi_te20, alpha20)
    picp, mpiw = pi_metrics(y_te, lo20, hi20)
    ep, em = EXP_P20[sat]
    matched = check(picp, mpiw, ep, em, f"Phase20 {sat}")
    is_val  = interval_score(y_te, lo20, hi20, alpha20)
    cwc_val = cwc_score(picp, mpiw, sat, mu_c=0.90)   # α=0.10 → μ_c=0.90
    print(f"    IS={is_val:.2f}  CWC={cwc_val:.4f}")
    results.setdefault('Ph20', {})[sat] = {'PICP': picp, 'MPIW': mpiw, 'CWC': round(cwc_val,4), 'IS': round(is_val, 2), 'matched': matched}

# ── Save results ─────────────────────────────────────────────────────────────────
out_json = OUT_PATH / 'is_results.json'
with open(out_json, 'w') as f:
    json.dump(results, f, indent=4)

# ── Summary table ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  IS RECOMPUTE SUMMARY")
print("=" * 80)
print(f"{'Phase':<14} {'Sensor':<14} {'PICP':>8} {'MPIW':>8} {'CWC':>8} {'IS':>8} {'Match':>6}")
print("-" * 68)
for phase in ['Ph4','Ph5','Ph6_GBM_CQR','Ph6_SVM_Split','Ph6_ANN_Split','Ph6_ANN_CQR',
              'Ph7','Ph8','Ph8b','Ph9a','Ph9c','Ph10','Ph20']:
    if phase not in results: continue
    for sat in ['EOS-04', 'Sentinel-1']:
        if sat not in results[phase]: continue
        r = results[phase][sat]
        if 'error' in r:
            print(f"{phase:<14} {sat:<14} {'ERROR':>8}")
        else:
            m = "✓" if r.get('matched') else "⚠"
            print(f"{phase:<14} {sat:<14} {r['PICP']*100:>7.2f}% {r['MPIW']:>8.2f} {r.get('CWC',0):>8.4f} {r['IS']:>8.2f} {m:>6}")

print(f"\nResults saved to {out_json}")
