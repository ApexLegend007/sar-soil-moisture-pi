"""
Phase 9c — Mondrian CQR: Crop-Stratified Conformalized Quantile Regression.

MOTIVATION (Phase 9/9b postmortem)
  CQR-d (difficulty model) and interval-normalized CQR both failed for Sentinel-1:
  PICP dropped to 0.9412 < 0.95. Root cause: adaptive normalization introduces
  estimation error that overwhelms the small calibration set (~153 samples).
  Interval-width variance across sensors: EOS-04 std/mean=16%, S1 std/mean=26%.

  Alternative: Mondrian CQR. Instead of normalizing, STRATIFY the calibration step
  by the most informative categorical feature — crop type.
  Each crop class gets its own conformal quantile from its own residuals.
  No estimation error, no normalization — just group-specific calibration.

  Theoretical guarantee: marginal coverage ≥ 1-α holds WITHIN each Mondrian cell.
  Aggregate coverage is also ≥ 1-α (convex combination of valid cells).

GROUP DESIGN (from actual calibration split)
  EOS-04  (cal=205): crop 5 → 91 cal | crop 26 → 36 cal | crop 18 → 22 cal | Other → 56
  Sentinel-1 (cal=153): crop 2 → 47 cal | crop 19 → 44 cal | Other → 62
  Min group size: 22 (EOS-04 crop 18) — sufficient for α=0.05 (need ≥20)

PREMORTEM RISKS
  - If test set crop distribution differs from cal → group "Other" may under-cover
    Monitor: per-group PICP at test time
  - crop 18 (EOS-04, 22 cal) is at the borderline; q̂ = max(scores) → conservative
  - Mondrian only helps if different crops genuinely have different conformity score
    distributions. If they're similar → same result as standard CQR.
  - "Other" group pools heterogeneous crops → effectively standard CQR for that group

POSTMORTEM TARGETS
  EOS-04    : beat MPIW=35.70 while PICP ≥ 0.95 for all groups
  Sentinel-1: beat MPIW=35.75 while PICP ≥ 0.95 for all groups
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['PYTHONHASHSEED']        = '42'

import random; random.seed(42)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; plt.show = lambda: None

import json, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np; np.random.seed(42)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / 'data'
OUTPUT_PATH = ROOT / 'output' / 'mondrian_cqr_uncensored'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_SEED  = 42
Y_COL        = 'SM1 (%)'
ALPHA        = 0.05

PHASE6_GBM = {
    'EOS-04':     {'PICP': 0.9659, 'MPIW': 35.70},
    'Sentinel-1': {'PICP': 0.9542, 'MPIW': 35.75},
}

# Mondrian group definitions — crops with ≥22 cal samples get own group, rest → Other
MONDRIAN_GROUPS = {
    'EOS-04':     [5, 26, 18],   # Other = everything else
    'Sentinel-1': [2, 19],       # Other = everything else (crops 16,11 pooled in)
}

print("=" * 70)
print("  PHASE 9c — MONDRIAN CQR (crop-stratified conformal calibration)")
print("=" * 70)

# ── helpers ────────────────────────────────────────────────────────────────────

def split_70_10_10_10(X, y):
    X_tr,  X_tmp,  y_tr,  y_tmp  = train_test_split(X, y, train_size=0.7,  random_state=RANDOM_SEED)
    X_v,   X_tmp2, y_v,   y_tmp2 = train_test_split(X_tmp, y_tmp, train_size=1/3, random_state=RANDOM_SEED)
    X_cal, X_te,   y_cal, y_te   = train_test_split(X_tmp2, y_tmp2, test_size=0.5, random_state=RANDOM_SEED)
    return X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te


def pi_metrics(y_true, lo, hi):
    covered = np.sum((y_true >= lo) & (y_true <= hi))
    return {'PICP': round(float(covered / len(y_true)), 6),
            'MPIW': round(float(np.mean(hi - lo)), 6)}


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f: json.dump(obj, f, indent=4)


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
    ax.set_title(title, fontsize=14); ax.set_xlabel('Sample Index'); ax.set_ylabel('SM1 (%)')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


# ── load data ──────────────────────────────────────────────────────────────────
print("\n[Data] Loading NDVI-enhanced CSVs …")
eos_df = pd.read_csv(DATA_PATH / 'eos-04-enhanced-ndvi.csv')
sen_df = pd.read_csv(DATA_PATH / 'sentinel-1-enhanced-ndvi.csv')

X_COLS = {
    'EOS-04':     ['HH-pol', 'HV-pol', 'cross_pol_ratio', 'month_sin', 'month_cos', 'crop_encoded', 'NDVI'],
    'Sentinel-1': ['VH-pol', 'VV-pol', 'cross_pol_ratio', 'month_sin', 'month_cos', 'crop_encoded', 'NDVI'],
}
CROP_IDX = {'EOS-04': 5, 'Sentinel-1': 5}  # crop_encoded is always index 5

datasets = [
    ('EOS-04',     eos_df[X_COLS['EOS-04']].values,     eos_df[Y_COL].values),
    ('Sentinel-1', sen_df[X_COLS['Sentinel-1']].values, sen_df[Y_COL].values),
]

all_results = {}

for satellite, X_raw, y_raw in datasets:
    print(f"\n{'─'*60}")
    major_crops = MONDRIAN_GROUPS[satellite]
    print(f"  {satellite}  |  Mondrian groups: {major_crops} + Other")
    print(f"  Phase 6 GBM CQR: PICP={PHASE6_GBM[satellite]['PICP']}  "
          f"MPIW={PHASE6_GBM[satellite]['MPIW']}")
    print(f"{'─'*60}")

    mask = y_raw != 50
    X, y = X_raw[mask], y_raw[mask]
    X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te = split_70_10_10_10(X, y)

    sc = MinMaxScaler()
    X_tr_s  = sc.fit_transform(X_tr)
    X_v_s   = sc.transform(X_v)
    X_cal_s = sc.transform(X_cal)
    X_te_s  = sc.transform(X_te)

    ci = CROP_IDX[satellite]
    crops_cal = X_cal[:, ci].astype(int)
    crops_te  = X_te[:,  ci].astype(int)

    # ── GBM quantile models (same as Phase 6) ─────────────────────────────────
    print(f"  Fitting GBM lo/hi quantile models …", end=' ', flush=True)
    gbm_lo = GradientBoostingRegressor(loss='quantile', alpha=0.025, n_estimators=200,
                                        max_depth=4, learning_rate=0.05, random_state=RANDOM_SEED)
    gbm_hi = GradientBoostingRegressor(loss='quantile', alpha=0.975, n_estimators=200,
                                        max_depth=4, learning_rate=0.05, random_state=RANDOM_SEED)
    gbm_lo.fit(X_tr_s, y_tr); gbm_hi.fit(X_tr_s, y_tr)
    print("done")

    # ── GBM predictions on cal and test ───────────────────────────────────────
    p_lo_cal = gbm_lo.predict(X_cal_s); p_hi_cal = gbm_hi.predict(X_cal_s)
    p_lo_te  = gbm_lo.predict(X_te_s);  p_hi_te  = gbm_hi.predict(X_te_s)
    p_lo_cal, p_hi_cal = np.minimum(p_lo_cal, p_hi_cal), np.maximum(p_lo_cal, p_hi_cal)
    p_lo_te,  p_hi_te  = np.minimum(p_lo_te,  p_hi_te),  np.maximum(p_lo_te,  p_hi_te)

    scores_cal = np.maximum(p_lo_cal - y_cal, y_cal - p_hi_cal)

    # ── Mondrian calibration: one q̂ per group ─────────────────────────────────
    def get_group(crop_val):
        return crop_val if crop_val in major_crops else -1  # -1 = Other

    group_cal = np.array([get_group(c) for c in crops_cal])
    group_te  = np.array([get_group(c) for c in crops_te])

    group_ids = sorted(set(major_crops) | {-1})
    q_hat_per_group = {}
    print(f"  Mondrian calibration (α={ALPHA}):")
    for g in group_ids:
        mask_g = group_cal == g
        n_g = mask_g.sum()
        label = f"crop {g}" if g != -1 else "Other"
        if n_g == 0:
            q_hat_per_group[g] = np.inf
            print(f"    {label:10s}: 0 cal samples → q̂=∞ (fallback)")
        else:
            q_g = np.quantile(scores_cal[mask_g], 1 - ALPHA, method='higher')
            q_hat_per_group[g] = q_g
            n_te_g = (group_te == g).sum()
            print(f"    {label:10s}: {n_g:3d} cal → q̂={q_g:.4f}  (test={n_te_g})")

    # ── Standard CQR q̂ for comparison ─────────────────────────────────────────
    q_std = np.quantile(scores_cal, 1 - ALPHA, method='higher')
    print(f"    {'Global':10s}: {len(scores_cal)} cal → q̂={q_std:.4f}  (standard CQR)")

    # ── Test set intervals ─────────────────────────────────────────────────────
    lo_mondrian = np.zeros(len(y_te))
    hi_mondrian = np.zeros(len(y_te))
    for j in range(len(y_te)):
        g = group_te[j]
        q = q_hat_per_group.get(g, q_std)
        lo_mondrian[j] = p_lo_te[j] - q
        hi_mondrian[j] = p_hi_te[j] + q

    lo_std = p_lo_te - q_std
    hi_std = p_hi_te + q_std

    m_mondrian = pi_metrics(y_te, lo_mondrian, hi_mondrian)
    m_std      = pi_metrics(y_te, lo_std,      hi_std)

    # ── Per-group breakdown ────────────────────────────────────────────────────
    print(f"\n  Per-group test coverage:")
    group_details = {}
    for g in group_ids:
        mask_g = group_te == g
        if mask_g.sum() == 0: continue
        m_g = pi_metrics(y_te[mask_g], lo_mondrian[mask_g], hi_mondrian[mask_g])
        label = f"crop {g}" if g != -1 else "Other"
        print(f"    {label:10s}: n={mask_g.sum():3d}  PICP={m_g['PICP']:.4f}  MPIW={m_g['MPIW']:.4f}")
        group_details[label] = {'n_test': int(mask_g.sum()), **m_g, 'q_hat': round(float(q_hat_per_group[g]), 6)}

    pct_change = (m_mondrian['MPIW'] - m_std['MPIW']) / m_std['MPIW'] * 100

    print(f"\n  AGGREGATE:")
    print(f"    Standard GBM CQR : PICP={m_std['PICP']:.4f}  MPIW={m_std['MPIW']:.4f}")
    print(f"    Mondrian CQR     : PICP={m_mondrian['PICP']:.4f}  MPIW={m_mondrian['MPIW']:.4f}  "
          f"({pct_change:+.1f}%)")
    print(f"    Phase 6 baseline : PICP={PHASE6_GBM[satellite]['PICP']}  "
          f"MPIW={PHASE6_GBM[satellite]['MPIW']}")

    result = {
        'GBM_CQR_standard': m_std,
        'Mondrian_CQR':     m_mondrian,
        'q_hat_global':     round(float(q_std), 6),
        'q_hat_per_group':  {str(k): round(float(v), 6) for k, v in q_hat_per_group.items()},
        'group_details':    group_details,
        'MPIW_change_pct':  round(pct_change, 2),
        'phase6_baseline':  PHASE6_GBM[satellite],
    }
    save_json(result, OUTPUT_PATH / f"{satellite}_mondrian_metrics.json")
    all_results[satellite] = result

    plot_dir = OUTPUT_PATH / 'plots'
    save_pi_plot(y_te, lo_mondrian, hi_mondrian, f'{satellite} Mondrian CQR',
                 plot_dir / f"{satellite}_Mondrian_CQR_plot.png")

print("\n" + "=" * 70)
print("  PHASE 9c COMPLETE — MONDRIAN CQR SUMMARY")
print("=" * 70)
for sat, r in all_results.items():
    std = r['GBM_CQR_standard']; mon = r['Mondrian_CQR']
    print(f"\n  {sat}")
    print(f"    Standard GBM CQR : PICP={std['PICP']:.4f}  MPIW={std['MPIW']:.4f}")
    print(f"    Mondrian CQR     : PICP={mon['PICP']:.4f}  MPIW={mon['MPIW']:.4f}  "
          f"({r['MPIW_change_pct']:+.1f}%)")
    print(f"    Phase 6 baseline : PICP={r['phase6_baseline']['PICP']}  "
          f"MPIW={r['phase6_baseline']['MPIW']}")
print(f"\n  Output → {OUTPUT_PATH}")
print("Done.")
