"""
Phase 20 — Highest-PICP Sentinel-1 configs (test PICP>=90%, test MPIW 25-26)
Plots the two configs with the highest test PICP in the target window:
  - 92.8%: LR=0.032, n=450, msl=22, d=4, sub=1.0  → test MPIW=25.85
  - 92.2%: LR=0.025, n=550, msl=25, d=4, sub=1.0  → test MPIW=25.79
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
OUT_DIR   = ROOT / 'output' / 'gbm_finetune' / 'sentinel1' / 'best_picp_plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
Y_COL       = 'SM1 (%)'
ALPHA       = 0.10

# Two highest-PICP configs in the test PICP 90-95% & test MPIW 25-26 window
CONFIGS = [
    {'label': 'PICP_92.8pct', 'lr':0.032,'n':450,'tau_lo':0.15,'tau_hi':0.85,'msl':22,'depth':4,'sub':1.0},
    {'label': 'PICP_92.2pct', 'lr':0.025,'n':550,'tau_lo':0.15,'tau_hi':0.85,'msl':25,'depth':4,'sub':1.0},
]


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

print(f"Sentinel-1 split: train={len(y_tr)} val={len(y_v)} cal={len(y_cal)} test={len(y_te)}")
print(f"Generating highest-PICP PI plots → {OUT_DIR}\n")

summary_rows = []

for cfg in CONFIGS:
    print(f"  [{cfg['label']}] LR={cfg['lr']} n={cfg['n']} τ={cfg['tau_lo']}/{cfg['tau_hi']} "
          f"msl={cfg['msl']} d={cfg['depth']} sub={cfg['sub']}")

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

    print(f"       test PICP={m_te['PICP']:.4f}  test MPIW={m_te['MPIW']:.4f}  q̂={q_hat:.4f}")

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

    textstr = (f"α=0.10  |  q̂={q_hat:.3f}\n"
               f"test PICP: {m_te['PICP']*100:.2f}%   test MPIW: {m_te['MPIW']:.2f}\n"
               f"val  PICP: {m_v['PICP']*100:.2f}%    val  MPIW: {m_v['MPIW']:.2f}")
    ax.text(0.01, 0.97, textstr, transform=ax.transAxes, va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.set_title(
        f"Sentinel-1 CQR ({cfg['label'].replace('_',' ')}) | "
        f"LR={cfg['lr']} n={cfg['n']} τ={cfg['tau_lo']}/{cfg['tau_hi']} "
        f"msl={cfg['msl']} d={cfg['depth']} sub={cfg['sub']}",
        fontsize=12)
    ax.set_xlabel('Test sample (sorted by SM1)')
    ax.set_ylabel('SM1 (%)')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = OUT_DIR / f"{cfg['label']}_lr{cfg['lr']}_n{cfg['n']}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"       saved → {plot_path.name}")

    summary_rows.append({
        'label': cfg['label'], 'lr': cfg['lr'], 'n_estimators': cfg['n'],
        'tau_lo': cfg['tau_lo'], 'tau_hi': cfg['tau_hi'],
        'msl': cfg['msl'], 'depth': cfg['depth'], 'sub': cfg['sub'],
        'q_hat': round(q_hat, 4),
        'val_PICP': m_v['PICP'], 'val_MPIW': m_v['MPIW'],
        'test_PICP': m_te['PICP'], 'test_MPIW': m_te['MPIW'],
    })

pd.DataFrame(summary_rows).to_csv(OUT_DIR / 'best_picp_summary.csv', index=False)

print(f"\n{'='*60}")
print(f"  HIGHEST-PICP CONFIGS (test PICP 90-95%, test MPIW 25-26)")
print(f"{'='*60}")
for r in summary_rows:
    print(f"  {r['label']:20s}  test PICP={r['test_PICP']*100:.2f}%  test MPIW={r['test_MPIW']:.4f}  "
          f"val PICP={r['val_PICP']*100:.2f}%  val MPIW={r['val_MPIW']:.4f}")
print(f"\n  Plots → {OUT_DIR}")
print("Done.")
