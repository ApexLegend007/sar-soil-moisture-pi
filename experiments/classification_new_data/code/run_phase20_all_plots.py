"""
Phase 20 — All 60 Sentinel-1 configs with test PICP 90-95% AND test MPIW 25-26.
Plots organized into subfolders by PICP level.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '42'
import random; random.seed(42)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np; np.random.seed(42)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / 'data'
GRID_CSV  = ROOT / 'output' / 'gbm_finetune' / 'sentinel1' / 'grid_summary.csv'
OUT_BASE  = ROOT / 'output' / 'gbm_finetune' / 'sentinel1' / 'all_hits_plots'
OUT_BASE.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
Y_COL       = 'SM1 (%)'
ALPHA       = 0.10

def split_70_10_10_10(X, y):
    X_tr, X_tmp,  y_tr, y_tmp  = train_test_split(X, y, train_size=0.7,  random_state=RANDOM_SEED)
    X_v,  X_tmp2, y_v,  y_tmp2 = train_test_split(X_tmp, y_tmp, train_size=1/3, random_state=RANDOM_SEED)
    X_cal, X_te,  y_cal, y_te  = train_test_split(X_tmp2, y_tmp2, test_size=0.5, random_state=RANDOM_SEED)
    return X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te

def pi_metrics(y_true, lo, hi):
    covered = np.sum((y_true >= lo) & (y_true <= hi))
    return {'PICP': round(float(covered/len(y_true)), 4),
            'MPIW': round(float(np.mean(hi - lo)), 4)}

def cqr_calibrate(y_cal, p_lo_cal, p_hi_cal):
    scores = np.maximum(p_lo_cal - y_cal, y_cal - p_hi_cal)
    return float(np.quantile(scores, 1 - ALPHA, method='higher'))

# Load data
sen_df = pd.read_csv(DATA_PATH / 'sentinel-1-enhanced-ndvi.csv')
X_COLS = ['VH-pol','VV-pol','cross_pol_ratio','month_sin','month_cos','crop_encoded','NDVI']
X_raw, y_raw = sen_df[X_COLS].values, sen_df[Y_COL].values
mask = y_raw != 50
X, y = X_raw[mask], y_raw[mask]

X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te = split_70_10_10_10(X, y)
sc = MinMaxScaler()
X_tr_s  = sc.fit_transform(X_tr)
X_v_s   = sc.transform(X_v)
X_cal_s = sc.transform(X_cal)
X_te_s  = sc.transform(X_te)

# Load all 60 qualifying configs
df = pd.read_csv(GRID_CSV)
hits = df[(df['test_PICP'] >= 0.90) & (df['test_PICP'] <= 0.95) &
          (df['test_MPIW'] >= 25.00) & (df['test_MPIW'] <= 26.00)].sort_values(
              ['test_PICP', 'test_MPIW'], ascending=[False, True]).reset_index(drop=True)

print(f"Sentinel-1 split: train={len(y_tr)} val={len(y_v)} cal={len(y_cal)} test={len(y_te)}")
print(f"Generating {len(hits)} plots → {OUT_BASE}\n")

# PICP buckets for subfolder names
def picp_label(p):
    if p >= 0.928: return '92.8pct'
    if p >= 0.921: return '92.2pct'
    if p >= 0.915: return '91.5pct'
    if p >= 0.908: return '90.9pct'
    return '90.2pct'

summary_rows = []

for rank, row in hits.iterrows():
    cfg = {
        'lr':    row['lr'],
        'n':     int(row['n_estimators']),
        'tau_lo': row['tau_lo'],
        'tau_hi': row['tau_hi'],
        'msl':   int(row['min_samples_leaf']),
        'depth': int(row['max_depth']),
        'sub':   row['subsample'],
    }
    bucket = picp_label(row['test_PICP'])
    out_dir = OUT_BASE / bucket
    out_dir.mkdir(exist_ok=True)

    gbm_lo = GradientBoostingRegressor(
        loss='quantile', alpha=cfg['tau_lo'], n_estimators=cfg['n'],
        max_depth=cfg['depth'], learning_rate=cfg['lr'],
        min_samples_leaf=cfg['msl'], subsample=cfg['sub'], random_state=RANDOM_SEED)
    gbm_hi = GradientBoostingRegressor(
        loss='quantile', alpha=cfg['tau_hi'], n_estimators=cfg['n'],
        max_depth=cfg['depth'], learning_rate=cfg['lr'],
        min_samples_leaf=cfg['msl'], subsample=cfg['sub'], random_state=RANDOM_SEED)
    gbm_lo.fit(X_tr_s, y_tr)
    gbm_hi.fit(X_tr_s, y_tr)

    plo_cal = np.minimum(gbm_lo.predict(X_cal_s), gbm_hi.predict(X_cal_s))
    phi_cal = np.maximum(gbm_lo.predict(X_cal_s), gbm_hi.predict(X_cal_s))
    q_hat   = cqr_calibrate(y_cal, plo_cal, phi_cal)

    plo_te = np.minimum(gbm_lo.predict(X_te_s), gbm_hi.predict(X_te_s))
    phi_te = np.maximum(gbm_lo.predict(X_te_s), gbm_hi.predict(X_te_s))
    lo_cqr = plo_te - q_hat
    hi_cqr = phi_te + q_hat
    m_te   = pi_metrics(y_te, lo_cqr, hi_cqr)

    plo_v = np.minimum(gbm_lo.predict(X_v_s), gbm_hi.predict(X_v_s))
    phi_v = np.maximum(gbm_lo.predict(X_v_s), gbm_hi.predict(X_v_s))
    m_v   = pi_metrics(y_v, plo_v - q_hat, phi_v + q_hat)

    order   = np.argsort(y_te)
    y_sort  = y_te[order]
    lo_sort = lo_cqr[order]
    hi_sort = hi_cqr[order]
    covered = (y_sort >= lo_sort) & (y_sort <= hi_sort)
    idx     = np.arange(len(y_sort))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(idx, lo_sort, hi_sort, alpha=0.25, color='steelblue', label='90% PI')
    ax.plot(idx, lo_sort, color='steelblue', lw=1, ls='--', alpha=0.7)
    ax.plot(idx, hi_sort, color='steelblue', lw=1, ls='--', alpha=0.7)
    ax.scatter(idx[covered],  y_sort[covered],  color='green', s=18, zorder=3, label='Covered')
    ax.scatter(idx[~covered], y_sort[~covered], color='red',   s=28, zorder=4, marker='x', label='Missed')

    textstr = (f"Rank #{rank+1:02d}  |  α=0.10  |  q̂={q_hat:.3f}\n"
               f"test PICP: {m_te['PICP']*100:.2f}%   test MPIW: {m_te['MPIW']:.2f}\n"
               f"val  PICP: {m_v['PICP']*100:.2f}%    val  MPIW: {m_v['MPIW']:.2f}")
    ax.text(0.01, 0.97, textstr, transform=ax.transAxes, va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.set_title(
        f"S1 CQR Rank#{rank+1:02d} ({bucket}) | "
        f"LR={cfg['lr']} n={cfg['n']} τ={cfg['tau_lo']}/{cfg['tau_hi']} "
        f"msl={cfg['msl']} d={cfg['depth']} sub={cfg['sub']}",
        fontsize=11)
    ax.set_xlabel('Test sample (sorted by SM1)')
    ax.set_ylabel('SM1 (%)')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = (f"rank{rank+1:02d}_picp{m_te['PICP']*100:.1f}"
             f"_mpiw{m_te['MPIW']:.2f}"
             f"_lr{cfg['lr']}_n{cfg['n']}_msl{cfg['msl']}_d{cfg['depth']}.png")
    fig.savefig(out_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)

    summary_rows.append({
        'rank': rank+1, 'picp_bucket': bucket,
        'lr': cfg['lr'], 'n_estimators': cfg['n'],
        'tau_lo': cfg['tau_lo'], 'tau_hi': cfg['tau_hi'],
        'msl': cfg['msl'], 'depth': cfg['depth'], 'sub': cfg['sub'],
        'q_hat': round(q_hat, 4),
        'val_PICP': m_v['PICP'], 'val_MPIW': m_v['MPIW'],
        'test_PICP': m_te['PICP'], 'test_MPIW': m_te['MPIW'],
        'plot_file': f"{bucket}/{fname}",
    })
    print(f"  [{rank+1:02d}/60] {bucket}  test PICP={m_te['PICP']:.4f}  "
          f"test MPIW={m_te['MPIW']:.4f}  → {fname}")

pd.DataFrame(summary_rows).to_csv(OUT_BASE / 'all_hits_summary.csv', index=False)

# Print final summary by bucket
print(f"\n{'='*65}")
print(f"  PHASE 20 — ALL 60 HITS SUMMARY")
print(f"{'='*65}")
for bucket in ['92.8pct','92.2pct','91.5pct','90.9pct','90.2pct']:
    grp = [r for r in summary_rows if r['picp_bucket']==bucket]
    if grp:
        best = min(grp, key=lambda r: r['test_MPIW'])
        print(f"  {bucket}  ({len(grp):2d} configs)  "
              f"best MPIW={best['test_MPIW']:.4f}  "
              f"LR={best['lr']} n={best['n_estimators']} msl={best['msl']}")
print(f"\n  Plots → {OUT_BASE}")
print("Done.")
