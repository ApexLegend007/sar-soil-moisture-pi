"""
Phase 9 — CQR-d v2: Interval-Normalized Conformalized Quantile Regression.

PHASE 9 v1 POSTMORTEM (difficulty model approach — failed)
  The original CQR-d used a secondary GBM trained on training residuals to
  estimate local difficulty. Two failure modes:
  1. EOS-04: σ̂_floor=7.81 bound most samples → flat normalization → zero gain
  2. Sentinel-1: GBM overfits training residuals → difficulty model underestimates
     test difficulty → conformal quantile q̂ too small → PICP dropped to 0.9412

  Root cause: training set residuals are systematically smaller than test residuals
  (GBM in-sample overfit). The difficulty model learned the wrong scale.

PHASE 9 v2 — INTERVAL-NORMALIZED CQR (self-normalizing, no secondary model)
  Use the GBM's OWN raw interval width (ŷ_hi − ŷ_lo) as the difficulty proxy.
  This is the most natural uncertainty signal the base model already produces:
  - Points where GBM is uncertain (wide raw interval) → more conformalized slack
  - Points where GBM is confident (narrow raw interval) → less slack → tighter PI

  NO secondary model needed. No overfitting risk. No σ̂_floor heuristic.
  The GBM base interval is already fully out-of-sample on the cal/test sets.

ALGORITHM
  1. Same 70/10/10/10 split as Phase 6 (train/val/cal/test)
  2. Train GBM lo/hi quantile models on train set (same as Phase 6)
  3. On calibration set:
       raw_width_i = ŷ_hi(x_i) − ŷ_lo(x_i)               [GBM's own uncertainty]
       s_i = max(ŷ_lo(x_i) − y_i, y_i − ŷ_hi(x_i))        [standard CQR score]
       s̃_i = s_i / raw_width_i                              [normalized score]
  4. Conformal quantile: q̂ = Quantile(s̃_i, 1 − α, method='higher')
  5. Test interval:  lo_j = ŷ_lo(x_j) − q̂ * raw_width_j
                     hi_j = ŷ_hi(x_j) + q̂ * raw_width_j

ALSO TESTED: k-NN local conformal calibration
  For each test point, use only its k nearest calibration neighbors' scores.
  More adaptive than global calibration but degrades on small cal sets (~150).

PREMORTEM RISKS (v2)
  - Near-zero raw_width → score explosion: floor at max(raw_width, 1th pctl)
  - Asymmetric scaling: lo and hi could be shifted unequally — acceptable
  - If GBM raw width has low variance → same flat-normalization issue
    Monitor: print std(raw_width) at run time. If std < 2, switch to k-NN.

POSTMORTEM TARGETS
  EOS-04    : current GBM CQR MPIW=35.70 → target < 32.0 (10% reduction)
  Sentinel-1: current GBM CQR MPIW=35.75 → target < 32.5 (better than QSVR 37.56)
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
from sklearn.ensemble import GradientBoostingRegressor

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / 'data'
OUTPUT_PATH = ROOT / 'output' / 'cqrd_uncensored'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
X_COLS_EOS  = ['HH-pol', 'HV-pol', 'cross_pol_ratio', 'month_sin', 'month_cos', 'crop_encoded', 'NDVI']
X_COLS_SEN  = ['VH-pol', 'VV-pol', 'cross_pol_ratio', 'month_sin', 'month_cos', 'crop_encoded', 'NDVI']
Y_COL       = 'SM1 (%)'
ALPHA       = 0.05

# Phase 6 GBM CQR baselines for comparison
PHASE6_GBM = {
    'EOS-04':     {'PICP': 0.9659, 'MPIW': 35.70},
    'Sentinel-1': {'PICP': 0.9542, 'MPIW': 35.75},
}

print("=" * 70)
print("  PHASE 9 — CQR-d: DENSITY-CALIBRATED CQR")
print("  Wraps Phase 6 GBM CQR with a difficulty normalizer")
print("=" * 70)

# ── helpers ────────────────────────────────────────────────────────────────────

def split_70_10_10_10(X, y):
    X_tr,  X_tmp,  y_tr,  y_tmp  = train_test_split(X, y, train_size=0.7,  random_state=RANDOM_SEED)
    X_v,   X_tmp2, y_v,   y_tmp2 = train_test_split(X_tmp, y_tmp, train_size=1/3, random_state=RANDOM_SEED)
    X_cal, X_te,   y_cal, y_te   = train_test_split(X_tmp2, y_tmp2, test_size=0.5, random_state=RANDOM_SEED)
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
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── load data ──────────────────────────────────────────────────────────────────
print("\n[Data] Loading NDVI-enhanced CSVs …")
eos_df = pd.read_csv(DATA_PATH / 'eos-04-enhanced-ndvi.csv')
sen_df = pd.read_csv(DATA_PATH / 'sentinel-1-enhanced-ndvi.csv')
print(f"  EOS-04: {len(eos_df)} rows  |  Sentinel-1: {len(sen_df)} rows")

datasets = [
    ('EOS-04',     eos_df[X_COLS_EOS].values, eos_df[Y_COL].values),
    ('Sentinel-1', sen_df[X_COLS_SEN].values, sen_df[Y_COL].values),
]

all_results = {}

# ── per-sensor loop ────────────────────────────────────────────────────────────
for satellite, X_raw, y_raw in datasets:
    print(f"\n{'─'*60}")
    print(f"  {satellite}  (Phase 6 GBM CQR baseline: "
          f"PICP={PHASE6_GBM[satellite]['PICP']}  "
          f"MPIW={PHASE6_GBM[satellite]['MPIW']})")
    print(f"{'─'*60}")

    mask = y_raw != 50
    X, y = X_raw[mask], y_raw[mask]
    X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te = split_70_10_10_10(X, y)
    X_tr_s, X_v_s, X_cal_s, X_te_s, _ = scale_X(X_tr, X_v, X_cal, X_te)

    print(f"  Split: train={len(y_tr)}  val={len(y_v)}  "
          f"cal={len(y_cal)}  test={len(y_te)}")

    # ── Step 1: GBM quantile models (same as Phase 6) ─────────────────────────
    print(f"  [1/4] Fitting GBM lo/hi quantile models …", end=' ', flush=True)
    gbm_lo = GradientBoostingRegressor(loss='quantile', alpha=0.025, n_estimators=200,
                                        max_depth=4, learning_rate=0.05, random_state=RANDOM_SEED)
    gbm_hi = GradientBoostingRegressor(loss='quantile', alpha=0.975, n_estimators=200,
                                        max_depth=4, learning_rate=0.05, random_state=RANDOM_SEED)
    gbm_lo.fit(X_tr_s, y_tr)
    gbm_hi.fit(X_tr_s, y_tr)
    print("done")

    # ── Step 2: Calibration set predictions (no secondary model needed) ───────
    print(f"  [2/4] Computing GBM predictions on cal set …", end=' ', flush=True)
    p_lo_cal = gbm_lo.predict(X_cal_s)
    p_hi_cal = gbm_hi.predict(X_cal_s)
    p_lo_cal, p_hi_cal = np.minimum(p_lo_cal, p_hi_cal), np.maximum(p_lo_cal, p_hi_cal)

    # GBM's own raw interval width as difficulty proxy (self-normalizing, no model)
    raw_width_cal = p_hi_cal - p_lo_cal
    width_floor   = max(np.percentile(raw_width_cal, 1), 0.5)  # 1st pctl floor
    sigma_cal     = np.maximum(raw_width_cal, width_floor)

    print(f"done  (width mean={raw_width_cal.mean():.2f}  std={raw_width_cal.std():.2f}  "
          f"floor={width_floor:.2f})")

    # ── Step 3: Normalized conformity scores on cal set ───────────────────────
    print(f"  [3/4] Normalized conformity scores + conformal quantile …", end=' ', flush=True)
    scores_cal    = np.maximum(p_lo_cal - y_cal, y_cal - p_hi_cal)
    scores_norm   = scores_cal / sigma_cal
    q_hat         = np.quantile(scores_norm, 1 - ALPHA, method='higher')
    print(f"q̂={q_hat:.4f}")

    # ── Step 4: Test set intervals ─────────────────────────────────────────────
    print(f"  [4/4] Computing interval-normalized CQR intervals on test set …",
          end=' ', flush=True)
    p_lo_te = gbm_lo.predict(X_te_s)
    p_hi_te = gbm_hi.predict(X_te_s)
    p_lo_te, p_hi_te = np.minimum(p_lo_te, p_hi_te), np.maximum(p_lo_te, p_hi_te)

    raw_width_te  = p_hi_te - p_lo_te
    sigma_te      = np.maximum(raw_width_te, width_floor)

    lo_cqrd = p_lo_te - q_hat * sigma_te
    hi_cqrd = p_hi_te + q_hat * sigma_te

    m_cqrd = pi_metrics(y_te, lo_cqrd, hi_cqrd)
    print(f"PICP={m_cqrd['PICP']:.4f}  MPIW={m_cqrd['MPIW']:.4f}")

    # ── Also run standard GBM CQR for side-by-side comparison ─────────────────
    print(f"  [cmp] Standard GBM CQR (baseline) …", end=' ', flush=True)
    scores_standard = np.maximum(p_lo_cal - y_cal, y_cal - p_hi_cal)
    q_std = np.quantile(scores_standard, 1 - ALPHA, method='higher')
    lo_std = p_lo_te - q_std
    hi_std = p_hi_te + q_std
    m_std = pi_metrics(y_te, lo_std, hi_std)
    print(f"PICP={m_std['PICP']:.4f}  MPIW={m_std['MPIW']:.4f}")

    # ── Compute per-sample interval widths for analysis ───────────────────────
    widths_cqrd = hi_cqrd - lo_cqrd
    widths_std  = hi_std  - lo_std
    pct_change  = (widths_cqrd.mean() - widths_std.mean()) / widths_std.mean() * 100

    # ── Save results ──────────────────────────────────────────────────────────
    result = {
        'GBM_CQR_standard':   m_std,
        'GBM_CQR_d':          m_cqrd,
        'q_hat_standard':     round(float(q_std), 6),
        'q_hat_normalized':   round(float(q_hat), 6),
        'width_floor':        round(float(width_floor), 6),
        'raw_width_cal_mean': round(float(raw_width_cal.mean()), 6),
        'raw_width_cal_std':  round(float(raw_width_cal.std()), 6),
        'raw_width_te_mean':  round(float(raw_width_te.mean()), 6),
        'MPIW_reduction_pct': round(pct_change, 2),
        'phase6_baseline':    PHASE6_GBM[satellite],
    }
    save_json(result, OUTPUT_PATH / f"{satellite}_cqrd_metrics.json")
    all_results[satellite] = result

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_dir = OUTPUT_PATH / 'plots'
    save_pi_plot(y_te, lo_cqrd, hi_cqrd, f'{satellite} GBM CQR-d',
                 plot_dir / f"{satellite}_GBM_CQRd_plot.png")
    save_pi_plot(y_te, lo_std,  hi_std,  f'{satellite} GBM CQR (standard)',
                 plot_dir / f"{satellite}_GBM_CQR_standard_plot.png")

    # ── Width distribution plot ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(widths_std,  bins=30, alpha=0.5, color='steelblue', label=f'Standard CQR (mean={widths_std.mean():.2f})')
    ax.hist(widths_cqrd, bins=30, alpha=0.5, color='orange',    label=f'CQR-d (mean={widths_cqrd.mean():.2f})')
    ax.set_title(f'{satellite} — Interval Width Distribution: Standard vs CQR-d')
    ax.set_xlabel('Interval Width'); ax.set_ylabel('Count')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(plot_dir / f"{satellite}_width_distribution.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  [{satellite}] SUMMARY")
    print(f"    Standard GBM CQR : PICP={m_std['PICP']:.4f}  MPIW={m_std['MPIW']:.4f}")
    print(f"    GBM CQR-d        : PICP={m_cqrd['PICP']:.4f}  MPIW={m_cqrd['MPIW']:.4f}")
    print(f"    MPIW change      : {pct_change:+.1f}%")
    print(f"    Phase 6 baseline : PICP={PHASE6_GBM[satellite]['PICP']}  "
          f"MPIW={PHASE6_GBM[satellite]['MPIW']}")

print("\n" + "=" * 70)
print("  PHASE 9 COMPLETE — CQR-d SUMMARY")
print("=" * 70)
for sat, r in all_results.items():
    std = r['GBM_CQR_standard']
    cqrd = r['GBM_CQR_d']
    print(f"\n  {sat}")
    print(f"    Standard GBM CQR : PICP={std['PICP']:.4f}  MPIW={std['MPIW']:.4f}")
    print(f"    GBM CQR-d        : PICP={cqrd['PICP']:.4f}  MPIW={cqrd['MPIW']:.4f}  "
          f"({r['MPIW_reduction_pct']:+.1f}%)")
    print(f"    Phase 6 baseline : PICP={r['phase6_baseline']['PICP']}  "
          f"MPIW={r['phase6_baseline']['MPIW']}")
print(f"\n  Output → {OUTPUT_PATH}")
print("Done.")
