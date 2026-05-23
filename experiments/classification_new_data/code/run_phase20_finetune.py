"""
Phase 20 — Fine-tune GBM CQR around Phase 19 best config.

GOAL
  val PICP >= 0.900  AND  val MPIW in [25.00, 25.99]
  (user requirement: validated on validation set, not just test)

PHASE 19 ANCHOR (S1)
  LR=0.03  n=500  τ=0.15/0.85  msl=20  depth=5  sub=1.0
  val PICP=0.9013  val MPIW=25.82
  test PICP=0.8627  test MPIW=24.67

PREMORTEM
  val PICP=0.9013 = 137/152 covered. One flip = 0.8947 (fail).
  The 25.82 result sits 1.04 units below next-best (26.86) — a steep cliff.
  Fine-tuning near the anchor may find more configs in the [25,26) window,
  or may show the cliff is a val split artefact specific to exact params.

  Key risk: val MPIW [25,26) requires covering EXACTLY 137 of 152 val
  samples at minimum interval width. Small param changes shift WHICH
  137 are covered — MPIW can jump discontinuously.

GRID (tight around Phase 19 anchor)
  learning_rate    : [0.02, 0.025, 0.028, 0.03, 0.032, 0.035, 0.04]
  n_estimators     : [400, 450, 500, 550, 600, 700, 800]
  tau pairs        : [(0.15,0.85), (0.2,0.8)]
  min_samples_leaf : [15, 18, 20, 22, 25, 30]
  max_depth        : [4, 5]
  subsample        : [0.8, 0.9, 1.0]
  α                : 0.10  (90% coverage)
  Total: 7×7×2×6×2×3 = 3528 configs per sensor

  Note: large grid is intentional — the val MPIW landscape is sensitive
  to exact param values. Dense search maximises chance of finding the basin.

SELECTION
  Primary: val PICP >= 0.90 AND val MPIW in [25.00, 25.99] → sort by val MPIW
  Report : test PICP and test MPIW for each hit

POSTMORTEM TARGETS
  Find ≥1 config: val PICP>=0.90 AND val MPIW<25.82  → better than anchor
  Find ≥1 config: val PICP>=0.90 AND val MPIW in [25.00,25.99] AND test PICP>=0.90
  Zero configs in [25,26): cliff is artefact → anchor was lucky split
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '42'

import random; random.seed(42)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; plt.show = lambda: None

import json, itertools, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np; np.random.seed(42)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor

ROOT        = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / 'data'
OUTPUT_PATH = ROOT / 'output' / 'gbm_finetune'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
Y_COL       = 'SM1 (%)'
ALPHA       = 0.10

ANCHOR = {
    'Sentinel-1': {'lr':0.03,'n':500,'tau_lo':0.15,'tau_hi':0.85,
                   'msl':20,'depth':5,'sub':1.0,
                   'val_PICP':0.9013,'val_MPIW':25.82,
                   'test_PICP':0.8627,'test_MPIW':24.67},
    'EOS-04':     {'val_MPIW':26.55,'test_MPIW':26.55},
}

GRID = list(itertools.product(
    [0.02, 0.025, 0.028, 0.03, 0.032, 0.035, 0.04],
    [400, 450, 500, 550, 600, 700, 800],
    [(0.15, 0.85), (0.2, 0.8)],
    [15, 18, 20, 22, 25, 30],
    [4, 5],
    [0.8, 0.9, 1.0],
))

print("=" * 72)
print("  PHASE 20 — FINE-TUNE GBM CQR  (val PICP≥0.90, val MPIW 25-26)")
print("=" * 72)
print(f"  Grid : 7×LR × 7×n × 2×τ × 6×msl × 2×depth × 3×sub = {len(GRID)} configs/sensor")
print(f"  Goal : val PICP >= 0.900  AND  val MPIW in [25.00, 25.99]")
print(f"  S1 anchor: val PICP={ANCHOR['Sentinel-1']['val_PICP']}  val MPIW={ANCHOR['Sentinel-1']['val_MPIW']}")


def split_70_10_10_10(X, y):
    X_tr, X_tmp,  y_tr, y_tmp  = train_test_split(X, y, train_size=0.7,  random_state=RANDOM_SEED)
    X_v,  X_tmp2, y_v,  y_tmp2 = train_test_split(X_tmp, y_tmp, train_size=1/3, random_state=RANDOM_SEED)
    X_cal, X_te,  y_cal, y_te  = train_test_split(X_tmp2, y_tmp2, test_size=0.5, random_state=RANDOM_SEED)
    return X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te


def pi_metrics(y_true, lo, hi):
    covered = np.sum((y_true >= lo) & (y_true <= hi))
    return {'PICP': round(float(covered / len(y_true)), 6),
            'MPIW': round(float(np.mean(hi - lo)), 6)}


def cqr_calibrate(y_cal, p_lo_cal, p_hi_cal):
    scores = np.maximum(p_lo_cal - y_cal, y_cal - p_hi_cal)
    return float(np.quantile(scores, 1 - ALPHA, method='higher'))


def apply_cqr(p_lo, p_hi, q_hat):
    return p_lo - q_hat, p_hi + q_hat


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
    ax.plot(idx, lo, 'r--', lw=1, label='Lower'); ax.plot(idx, hi, color='orange', ls='--', lw=1, label='Upper')
    ax.fill_between(idx, lo, hi, alpha=0.15, color='gray')
    ax.text(0.02, 0.97, f"val_PICP≥0.90\ntest PICP:{m['PICP']*100:.2f}%\nMPIW:{m['MPIW']:.2f}",
            transform=ax.transAxes, va='top', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title(title, fontsize=13); ax.set_xlabel('Sample'); ax.set_ylabel('SM1 (%)')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


print("\n[Data] Loading CSVs …")
eos_df = pd.read_csv(DATA_PATH / 'eos-04-enhanced-ndvi.csv')
sen_df = pd.read_csv(DATA_PATH / 'sentinel-1-enhanced-ndvi.csv')

X_COLS = {
    'EOS-04':     ['HH-pol','HV-pol','cross_pol_ratio','month_sin','month_cos','crop_encoded','NDVI'],
    'Sentinel-1': ['VH-pol','VV-pol','cross_pol_ratio','month_sin','month_cos','crop_encoded','NDVI'],
}

datasets = [
    ('EOS-04',     eos_df[X_COLS['EOS-04']].values,     eos_df[Y_COL].values),
    ('Sentinel-1', sen_df[X_COLS['Sentinel-1']].values, sen_df[Y_COL].values),
]

all_sensor_results = {}

for satellite, X_raw, y_raw in datasets:
    anc = ANCHOR[satellite]
    print(f"\n{'─'*72}")
    print(f"  {satellite}  |  {len(GRID)} configs  |  α=0.10")
    print(f"  Target: val PICP>=0.90 AND val MPIW in [25.00, 25.99]")
    print(f"{'─'*72}")

    mask = y_raw != 50
    X, y = X_raw[mask], y_raw[mask]
    X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te = split_70_10_10_10(X, y)

    sc = MinMaxScaler()
    X_tr_s  = sc.fit_transform(X_tr)
    X_v_s   = sc.transform(X_v)
    X_cal_s = sc.transform(X_cal)
    X_te_s  = sc.transform(X_te)

    print(f"  Split: train={len(y_tr)} val={len(y_v)} cal={len(y_cal)} test={len(y_te)}")
    print(f"  val PICP step size = 1/{len(y_v)} = {1/len(y_v):.4f}  "
          f"(0.90 threshold = {int(0.90*len(y_v))}/{len(y_v)} covered)")

    all_rows   = []
    hits_target = []   # val PICP>=0.90 AND val MPIW in [25,26)
    best_val    = None

    for i, (lr, n_est, (tau_lo, tau_hi), msl, depth, sub) in enumerate(GRID):
        try:
            gbm_lo = GradientBoostingRegressor(
                loss='quantile', alpha=tau_lo, n_estimators=n_est,
                max_depth=depth, learning_rate=lr,
                min_samples_leaf=msl, subsample=sub, random_state=RANDOM_SEED)
            gbm_hi = GradientBoostingRegressor(
                loss='quantile', alpha=tau_hi, n_estimators=n_est,
                max_depth=depth, learning_rate=lr,
                min_samples_leaf=msl, subsample=sub, random_state=RANDOM_SEED)
            gbm_lo.fit(X_tr_s, y_tr); gbm_hi.fit(X_tr_s, y_tr)

            p_lo_cal = np.minimum(gbm_lo.predict(X_cal_s), gbm_hi.predict(X_cal_s))
            p_hi_cal = np.maximum(gbm_lo.predict(X_cal_s), gbm_hi.predict(X_cal_s))
            q_hat    = cqr_calibrate(y_cal, p_lo_cal, p_hi_cal)

            p_lo_v = np.minimum(gbm_lo.predict(X_v_s), gbm_hi.predict(X_v_s))
            p_hi_v = np.maximum(gbm_lo.predict(X_v_s), gbm_hi.predict(X_v_s))
            m_v    = pi_metrics(y_v, *apply_cqr(p_lo_v, p_hi_v, q_hat))

            p_lo_te = np.minimum(gbm_lo.predict(X_te_s), gbm_hi.predict(X_te_s))
            p_hi_te = np.maximum(gbm_lo.predict(X_te_s), gbm_hi.predict(X_te_s))
            m_te    = pi_metrics(y_te, *apply_cqr(p_lo_te, p_hi_te, q_hat))

            base_w_te = round(float(np.mean(p_hi_te - p_lo_te)), 4)

            row = {'params': {'lr':lr,'n_estimators':n_est,'tau_lo':tau_lo,'tau_hi':tau_hi,
                              'min_samples_leaf':msl,'max_depth':depth,'subsample':sub},
                   'q_hat': round(q_hat,6), 'base_width_te': base_w_te,
                   'val': m_v, 'test': m_te}
            all_rows.append(row)

            is_target = m_v['PICP'] >= 0.90 and 25.00 <= m_v['MPIW'] <= 25.99
            if is_target:
                hits_target.append(row)

            if best_val is None or (m_v['PICP'] >= 0.90 and m_v['MPIW'] < best_val['val']['MPIW']):
                if m_v['PICP'] >= 0.90:
                    best_val = row

            if (i + 1) % 500 == 0 or i == len(GRID) - 1:
                n_hits = len(hits_target)
                bv = best_val['val']['MPIW'] if best_val else float('nan')
                print(f"  [{satellite}] {i+1:4d}/{len(GRID)}  "
                      f"target_hits={n_hits}  best_val_MPIW={bv:.4f}")

        except Exception as e:
            all_rows.append({'params':{'lr':lr,'n':n_est},'error':str(e),'val':None,'test':None})

    # ── save grid CSV ────────────────────────────────────────────────────
    out_dir = OUTPUT_PATH / satellite.lower().replace('-','')
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    for r in all_rows:
        if not r.get('val'): continue
        p = r['params']
        csv_rows.append({**p,'q_hat':r['q_hat'],'base_w_te':r['base_width_te'],
                         'val_PICP':r['val']['PICP'],'val_MPIW':r['val']['MPIW'],
                         'test_PICP':r['test']['PICP'],'test_MPIW':r['test']['MPIW']})
    pd.DataFrame(csv_rows).to_csv(out_dir / 'grid_summary.csv', index=False)

    # ── report ─────────────────────────────────────────────────────────
    hits_target.sort(key=lambda r: r['val']['MPIW'])
    also_test_90 = [r for r in hits_target if r['test']['PICP'] >= 0.90]

    print(f"\n  [{satellite}] Grid complete — {len(all_rows)} configs")
    print(f"  val PICP>=0.90 AND val MPIW [25.00,25.99] : {len(hits_target)} configs")
    print(f"  ... of which also test PICP>=0.90          : {len(also_test_90)} configs")

    if hits_target:
        print(f"\n  ALL target hits (sorted by val MPIW):")
        print(f"  {'val_MPIW':>9}  {'val_PICP':>9}  {'te_MPIW':>8}  {'te_PICP':>8}  "
              f"{'q̂':>5}  {'bw_te':>5}  lr     n    τ    msl  d  sub")
        for r in hits_target:
            p   = r['params']
            tag = ' *** test≥0.90' if r['test']['PICP'] >= 0.90 else ''
            print(f"  {r['val']['MPIW']:>9.4f}  {r['val']['PICP']:>9.4f}  "
                  f"{r['test']['MPIW']:>8.4f}  {r['test']['PICP']:>8.4f}  "
                  f"{r['q_hat']:>5.3f}  {r['base_width_te']:>5.3f}  "
                  f"{p['lr']:.3f}  {p['n_estimators']:>4}  {p['tau_lo']:.2f}  "
                  f"{p['min_samples_leaf']:>3}  {p['max_depth']}  {p['subsample']}{tag}")
    else:
        print(f"\n  NO configs found in target range.")
        print(f"  Best val MPIW at val PICP>=0.90: {best_val['val']['MPIW']:.4f}" if best_val else "  no valid configs")
        print(f"  Postmortem: val cliff artefact confirmed — anchor was a lucky split.")

    best_to_save = hits_target[0] if hits_target else best_val
    if best_to_save:
        p = best_to_save['params']
        m = best_to_save['test']
        in_range = bool(25.00 <= best_to_save['val']['MPIW'] <= 25.99
                        and best_to_save['val']['PICP'] >= 0.90)
        save_json({
            'alpha': ALPHA,
            'goal_achieved': in_range,
            'params': p,
            'q_hat': best_to_save['q_hat'],
            'base_width_te': best_to_save['base_width_te'],
            'val':  best_to_save['val'],
            'test': m,
            'n_target_hits': len(hits_target),
            'n_also_test_90': len(also_test_90),
            'anchor': anc,
            'all_target_hits': [
                {'val_PICP': r['val']['PICP'], 'val_MPIW': r['val']['MPIW'],
                 'test_PICP': r['test']['PICP'], 'test_MPIW': r['test']['MPIW'],
                 'q_hat': r['q_hat'], 'base_width_te': r['base_width_te'],
                 'params': r['params']}
                for r in hits_target
            ],
        }, out_dir / 'best_config.json')

        if hits_target:
            bl = GradientBoostingRegressor(loss='quantile', alpha=p['tau_lo'],
                n_estimators=p['n_estimators'], max_depth=p['max_depth'],
                learning_rate=p['lr'], min_samples_leaf=p['min_samples_leaf'],
                subsample=p['subsample'], random_state=RANDOM_SEED)
            bh = GradientBoostingRegressor(loss='quantile', alpha=p['tau_hi'],
                n_estimators=p['n_estimators'], max_depth=p['max_depth'],
                learning_rate=p['lr'], min_samples_leaf=p['min_samples_leaf'],
                subsample=p['subsample'], random_state=RANDOM_SEED)
            bl.fit(X_tr_s, y_tr); bh.fit(X_tr_s, y_tr)
            plo_c = np.minimum(bl.predict(X_cal_s), bh.predict(X_cal_s))
            phi_c = np.maximum(bl.predict(X_cal_s), bh.predict(X_cal_s))
            q_b   = cqr_calibrate(y_cal, plo_c, phi_c)
            plo_t = np.minimum(bl.predict(X_te_s), bh.predict(X_te_s))
            phi_t = np.maximum(bl.predict(X_te_s), bh.predict(X_te_s))
            lo_pl, hi_pl = apply_cqr(plo_t, phi_t, q_b)
            title = (f"{satellite} Phase20 (α=0.10) — LR={p['lr']} n={p['n_estimators']} "
                     f"τ={p['tau_lo']}/{p['tau_hi']} msl={p['min_samples_leaf']} d={p['max_depth']}")
            save_pi_plot(y_te, lo_pl, hi_pl, title, out_dir/'plots'/'best_plot.png')

    all_sensor_results[satellite] = {'hits': hits_target, 'best': best_to_save}

# ── final summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  PHASE 20 COMPLETE — FINE-TUNE SUMMARY")
print("=" * 72)
print(f"\n  {'Sensor':<14}  {'Anchor val MPIW':>16}  {'Target hits':>12}  "
      f"{'Best val MPIW':>14}  {'Best val PICP':>14}  {'Goal?':>6}")
for sat, res in all_sensor_results.items():
    anc  = ANCHOR[sat]
    hits = res['hits']
    best = res['best']
    bvm  = best['val']['MPIW']  if best else float('nan')
    bvp  = best['val']['PICP']  if best else float('nan')
    goal = bool(hits and hits[0]['val']['PICP'] >= 0.90 and 25.00 <= hits[0]['val']['MPIW'] <= 25.99)
    print(f"  {sat:<14}  {anc.get('val_MPIW',0):>16.3f}  {len(hits):>12}  "
          f"{bvm:>14.4f}  {bvp:>14.4f}  {'YES' if goal else 'no':>6}")

print(f"\n  Output → {OUTPUT_PATH}")
print("Done.")
