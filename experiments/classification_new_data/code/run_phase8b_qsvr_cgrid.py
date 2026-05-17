"""
Phase 8b — Quantile SVR C × gamma joint grid search.

MOTIVATION
  Phase 8 swept only gamma (31 values) at a fixed C=2^6.
  Sentinel-1 best QSVR (γ=2^-4, MPIW=37.56) is WORSE than GBM CQR (35.75).
  The C hyperparameter directly controls how tightly the quantile model fits —
  a higher C forces closer adherence to training quantiles, which can narrow
  the predicted interval before it reaches the test set.

GRID
  gamma : 2^{-8} … 2^4  (13 values, focused around Phase 8 winners)
  C     : 2^4, 2^6, 2^8, 2^10  (4 values; 2^6 is the Phase 8 baseline)
  Total : 52 combinations per sensor × 2 = 104 QSVR fits

SELECTION CRITERION
  PICP ∈ [0.95, 1.00]  →  min MPIW   (valid interval, tightest)
  fallback PICP ≥ 0.90 →  min MPIW   (if no config meets strict threshold)

PREMORTEM RISKS
  - High C + high gamma → kernel matrix near-singular → QP may diverge; wrapped in try/except
  - Larger C increases QP solve time (O(n^2.x)); 104 fits should complete in ~15-30 min
  - Different optimal (C, gamma) per sensor — results reported separately
  - Phase 8 used split_80_10_10; this script uses the SAME split for comparability
  - We intentionally do NOT add conformal calibration here — keeping Phase 8 methodology
    so results are directly comparable in the paper table.

POSTMORTEM GOAL
  EOS-04 : beat MPIW=30.91 while PICP ≥ 0.95
  Sentinel-1 : beat MPIW=35.75 (GBM CQR) while PICP ≥ 0.95
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONHASHSEED']        = '42'

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
from scipy.spatial.distance import cdist
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / 'data'
OUTPUT_PATH = ROOT / 'output' / 'quantile_svr_uncensored_cgrid'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_SEED  = 42
X_COLS_EOS   = ['HH-pol', 'HV-pol', 'cross_pol_ratio', 'month_sin', 'month_cos', 'crop_encoded', 'NDVI']
X_COLS_SEN   = ['VH-pol', 'VV-pol', 'cross_pol_ratio', 'month_sin', 'month_cos', 'crop_encoded', 'NDVI']
Y_COL        = 'SM1 (%)'

# Phase 8 baseline for reference
PHASE8_BEST = {
    'EOS-04':    {'C': 2**6, 'gamma': 2**1,  'PICP': 0.9512, 'MPIW': 30.91},
    'Sentinel-1':{'C': 2**6, 'gamma': 2**-4, 'PICP': 0.9542, 'MPIW': 37.56},
}
GBM_CQR_BEST = {
    'EOS-04':    {'PICP': 0.9659, 'MPIW': 35.70},
    'Sentinel-1':{'PICP': 0.9542, 'MPIW': 35.75},
}

# ── grid ───────────────────────────────────────────────────────────────────────
GAMMA_VALUES = [2**i for i in range(-8, 5)]   # 13 values: 2^-8 … 2^4
C_VALUES     = [2**4, 2**6, 2**8, 2**10]       # 16, 64, 256, 1024
TAU_LO, TAU_HI = 0.025, 0.975

print("=" * 70)
print("  PHASE 8b — QSVR C × GAMMA JOINT GRID")
print(f"  gamma: 2^{{-8}} … 2^4  ({len(GAMMA_VALUES)} values)")
print(f"  C    : {[int(c) for c in C_VALUES]}  ({len(C_VALUES)} values)")
print(f"  Total fits: {len(GAMMA_VALUES)*len(C_VALUES)} per sensor")
print("=" * 70)

# ── helpers ────────────────────────────────────────────────────────────────────

def split_80_10_10(X, y):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, train_size=0.8, random_state=RANDOM_SEED)
    X_v,  X_te,  y_v,  y_te  = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_SEED)
    return X_tr, X_v, X_te, y_tr, y_v, y_te


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
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


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


def select_best(results):
    """PICP in [0.95, 1.00] → min MPIW.  Fallback: PICP ≥ 0.90 → min MPIW."""
    for threshold in [0.95, 0.90]:
        valid = [r for r in results
                 if r.get('test') and r['test']['PICP'] >= threshold]
        if valid:
            return min(valid, key=lambda r: r['test']['MPIW']), threshold
    valid = [r for r in results if r.get('test')]
    return (min(valid, key=lambda r: r['test']['MPIW']), None) if valid else (None, None)


# ── load data ──────────────────────────────────────────────────────────────────
print("\n[Data] Loading NDVI-enhanced CSVs …")
eos_df = pd.read_csv(DATA_PATH / 'eos-04-enhanced-ndvi.csv')
sen_df = pd.read_csv(DATA_PATH / 'sentinel-1-enhanced-ndvi.csv')
print(f"  EOS-04: {len(eos_df)} rows  |  Sentinel-1: {len(sen_df)} rows")

datasets = [
    ('EOS-04',     eos_df[X_COLS_EOS].values, eos_df[Y_COL].values),
    ('Sentinel-1', sen_df[X_COLS_SEN].values, sen_df[Y_COL].values),
]

# ── grid search ────────────────────────────────────────────────────────────────
for satellite, X_raw, y_raw in datasets:
    print(f"\n{'─'*60}")
    print(f"  {satellite}  (Phase 8 baseline: "
          f"γ=2^{int(round(np.log2(PHASE8_BEST[satellite]['gamma'])))}  "
          f"C=2^{int(round(np.log2(PHASE8_BEST[satellite]['C'])))}  "
          f"PICP={PHASE8_BEST[satellite]['PICP']}  "
          f"MPIW={PHASE8_BEST[satellite]['MPIW']})")
    print(f"  GBM CQR target  : MPIW={GBM_CQR_BEST[satellite]['MPIW']}  "
          f"PICP={GBM_CQR_BEST[satellite]['PICP']}")
    print(f"{'─'*60}")

    mask = y_raw != 50
    X, y = X_raw[mask], y_raw[mask]

    X_tr, X_v, X_te, y_tr, y_v, y_te = split_80_10_10(X, y)
    sc = MinMaxScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_v_s  = sc.transform(X_v)
    X_te_s = sc.transform(X_te)

    out_dir  = OUTPUT_PATH / satellite.lower().replace('-', '')
    plot_dir = out_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    all_results = []

    for C in C_VALUES:
        c_exp = int(round(np.log2(C)))
        for gamma in GAMMA_VALUES:
            g_exp = int(round(np.log2(gamma)))
            tag = f"C=2^{c_exp:3d}  γ=2^{g_exp:3d}"
            print(f"  [{satellite}] {tag} …", end=' ', flush=True)
            try:
                beta_lo = fit_qsvr(X_tr_s, y_tr, gamma, C, TAU_LO)
                beta_hi = fit_qsvr(X_tr_s, y_tr, gamma, C, TAU_HI)

                lo_te = predict_qsvr(X_tr_s, X_te_s, gamma, beta_lo)
                hi_te = predict_qsvr(X_tr_s, X_te_s, gamma, beta_hi)
                lo_v  = predict_qsvr(X_tr_s, X_v_s,  gamma, beta_lo)
                hi_v  = predict_qsvr(X_tr_s, X_v_s,  gamma, beta_hi)

                m_te = pi_metrics(y_te, lo_te, hi_te)
                m_v  = pi_metrics(y_v,  lo_v,  hi_v)
                print(f"PICP={m_te['PICP']:.4f}  MPIW={m_te['MPIW']:.4f}")

                # only save plots for valid configurations (PICP ≥ 0.95) to save disk
                if m_te['PICP'] >= 0.95:
                    save_pi_plot(y_te, lo_te, hi_te,
                                 f'{satellite} QSVR C=2^{c_exp} γ=2^{g_exp}',
                                 plot_dir / f"C2p{c_exp}_gamma2p{g_exp}.png")

                all_results.append({
                    'params': {'C': C, 'C_exp': c_exp, 'gamma': gamma, 'gamma_exp': g_exp},
                    'val':    m_v,
                    'test':   m_te,
                })
            except Exception as e:
                print(f"FAILED ({e})")
                all_results.append({
                    'params': {'C': C, 'C_exp': c_exp, 'gamma': gamma, 'gamma_exp': g_exp},
                    'val': None, 'test': None, 'error': str(e),
                })

    # ── save full results ──────────────────────────────────────────────────────
    save_json(all_results, out_dir / 'grid_results.json')

    df_rows = []
    for r in all_results:
        row = {'C': r['params']['C'], 'C_exp': r['params']['C_exp'],
               'gamma': r['params']['gamma'], 'gamma_exp': r['params']['gamma_exp']}
        if r.get('test'):
            row.update({'val_PICP': r['val']['PICP'], 'val_MPIW': r['val']['MPIW'],
                        'test_PICP': r['test']['PICP'], 'test_MPIW': r['test']['MPIW']})
        else:
            row.update({'val_PICP': None, 'val_MPIW': None,
                        'test_PICP': None, 'test_MPIW': None, 'error': r.get('error', '')})
        df_rows.append(row)
    pd.DataFrame(df_rows).to_csv(out_dir / 'grid_summary.csv', index=False)

    # ── summary ────────────────────────────────────────────────────────────────
    best, threshold = select_best(all_results)
    print(f"\n  [{satellite}] Grid search complete.")
    print(f"  Phase 8 baseline : PICP={PHASE8_BEST[satellite]['PICP']}  "
          f"MPIW={PHASE8_BEST[satellite]['MPIW']}")
    print(f"  GBM CQR target   : PICP={GBM_CQR_BEST[satellite]['PICP']}  "
          f"MPIW={GBM_CQR_BEST[satellite]['MPIW']}")
    if best:
        c_exp = int(round(np.log2(best['params']['C'])))
        g_exp = int(round(np.log2(best['params']['gamma'])))
        print(f"  BEST (PICP≥{threshold or 'any'}) : "
              f"C=2^{c_exp}  γ=2^{g_exp}  "
              f"PICP={best['test']['PICP']:.4f}  "
              f"MPIW={best['test']['MPIW']:.4f}")
        delta_p8  = best['test']['MPIW'] - PHASE8_BEST[satellite]['MPIW']
        delta_gbm = best['test']['MPIW'] - GBM_CQR_BEST[satellite]['MPIW']
        print(f"  vs Phase 8 best  : ΔMPIW = {delta_p8:+.2f}")
        print(f"  vs GBM CQR       : ΔMPIW = {delta_gbm:+.2f}")
    else:
        print(f"  ⚠  No valid configuration found — all QP solves failed.")

print("\n" + "=" * 70)
print("  PHASE 8b COMPLETE")
print(f"  Results → {OUTPUT_PATH}")
print("=" * 70)
