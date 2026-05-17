"""
Phase 10 — Tuned GBM CQR: Hyperparameter Grid for Base Quantile Model.

POSTMORTEM (Phase 9 series)
  All adaptive CQR variants (interval-normalized, Mondrian) failed because they
  introduce estimation error on ~150 calibration samples. Mondrian produced negative
  q̂ values — exchangeability violated per-crop due to random split.

DIAGNOSIS
  MPIW decomposition (from Phase 9 logs):
    EOS-04  : base_raw_width=32.86 (92.1%) + 2×q̂=2.84 (7.9%)  = 35.70 MPIW
    Sentinel-1: base_raw_width=32.25 (90.2%) + 2×q̂=3.50 (9.8%) = 35.75 MPIW

  The CQR correction is only 8-10% of final MPIW. The base GBM quantile model
  contributes 90-92%. Phase 6 GBM hyperparameters were never tuned for interval
  tightness. THIS is the primary lever.

WHY TUNING SHOULD HELP
  min_samples_leaf=1 (GBM default): extreme quantiles (2.5th/97.5th percentile)
    are estimated from single-sample leaves → high variance → overwide intervals.
  min_samples_leaf ≥ 10: each leaf averages ≥10 samples → smoother, more
    conservative quantile surface → potentially narrower intervals on test data.
  base_tau=0.1/0.9 instead of 0.025/0.975: GBM fits a less extreme quantile
    (8/10 coverage vs 19/20), avoiding noisy leaf-level extreme estimates.
    CQR calibration handles the remaining gap. Total = base_width + 2×q̂ may
    be smaller if the base estimate is tighter.

PREMORTEM RISKS
  1. Val MPIW std ≈ 0.5 units on 170 samples — only trust Δ > 1 unit
  2. min_samples_leaf too large → coarse trees → wider intervals (non-monotonic)
  3. base_tau=0.1/0.9 may produce larger q̂ that exceeds base width gain
  4. max_depth=3 may underfit crop×pol interactions (7 features, 20+ crops)
  5. All results driven off val set — test never seen during selection

GRID
  min_samples_leaf : [1, 5, 10, 20]           (leaf stability)
  max_depth        : [3, 4, 5]                 (expressivity vs smoothness)
  n_estimators     : [200, 300]                (ensemble depth)
  subsample        : [1.0, 0.8]               (stochastic vs deterministic)
  base_tau         : [(0.025, 0.975),
                      (0.1, 0.9)]              (extreme vs moderate base quantile)
  Total: 4 × 3 × 2 × 2 × 2 = 96 configs per sensor

SELECTION (val set only — test set never touched)
  PICP_val ≥ 0.95  →  minimize MPIW_val
  Fallback: PICP_val ≥ 0.93 → minimize MPIW_val

POSTMORTEM TARGETS
  EOS-04    : beat MPIW=35.70 (GBM CQR Phase 6) and ideally approach 30.77 (QSVR)
  Sentinel-1: beat MPIW=35.75 (GBM CQR Phase 6) while PICP ≥ 0.95
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['PYTHONHASHSEED']        = '42'

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

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / 'data'
OUTPUT_PATH = ROOT / 'output' / 'gbm_tuned_cqr'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
Y_COL       = 'SM1 (%)'
ALPHA       = 0.05

BASELINES = {
    'EOS-04':     {'GBM_CQR': {'PICP': 0.9659, 'MPIW': 35.70},
                   'QSVR':    {'PICP': 0.9561, 'MPIW': 30.77}},
    'Sentinel-1': {'GBM_CQR': {'PICP': 0.9542, 'MPIW': 35.75},
                   'QSVR':    {'PICP': 0.9542, 'MPIW': 37.56}},
}

# ── grid ───────────────────────────────────────────────────────────────────────
MIN_SAMPLES_LEAF = [1, 5, 10, 20]
MAX_DEPTH        = [3, 4, 5]
N_ESTIMATORS     = [200, 300]
SUBSAMPLE        = [1.0, 0.8]
BASE_TAUS        = [(0.025, 0.975), (0.1, 0.9)]
LR               = 0.05  # fixed — co-linear with n_estimators; current baseline

GRID = list(itertools.product(MIN_SAMPLES_LEAF, MAX_DEPTH, N_ESTIMATORS, SUBSAMPLE, BASE_TAUS))
print("=" * 70)
print("  PHASE 10 — TUNED GBM CQR (hyperparameter grid for MPIW reduction)")
print("=" * 70)
print(f"  Grid: {len(MIN_SAMPLES_LEAF)}×msl × {len(MAX_DEPTH)}×depth × "
      f"{len(N_ESTIMATORS)}×n_est × {len(SUBSAMPLE)}×sub × {len(BASE_TAUS)}×tau "
      f"= {len(GRID)} configs per sensor")

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


def cqr_calibrate(y_cal, p_lo_cal, p_hi_cal):
    scores = np.maximum(p_lo_cal - y_cal, y_cal - p_hi_cal)
    return float(np.quantile(scores, 1 - ALPHA, method='higher'))


def apply_cqr(p_lo, p_hi, q_hat):
    return p_lo - q_hat, p_hi + q_hat


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f: json.dump(obj, f, indent=4)


def save_pi_plot(y_true, lo, hi, title, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = np.arange(len(y_true))
    m   = pi_metrics(y_true, lo, hi)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(idx, y_true, 'o', color='steelblue', ms=3, alpha=0.6, label='Actual')
    ax.plot(idx, lo,  'r--', lw=1, label='Lower bound')
    ax.plot(idx, hi,  color='orange', ls='--', lw=1, label='Upper bound')
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

datasets = [
    ('EOS-04',     eos_df[X_COLS['EOS-04']].values,     eos_df[Y_COL].values),
    ('Sentinel-1', sen_df[X_COLS['Sentinel-1']].values, sen_df[Y_COL].values),
]

all_sensor_results = {}

for satellite, X_raw, y_raw in datasets:
    print(f"\n{'─'*70}")
    print(f"  {satellite}  —  {len(X_raw)} rows  |  grid: {len(GRID)} configs")
    bl = BASELINES[satellite]
    print(f"  Targets: beat GBM CQR MPIW={bl['GBM_CQR']['MPIW']}  "
          f"(QSVR best={bl['QSVR']['MPIW']})")
    print(f"{'─'*70}")

    mask = y_raw != 50
    X, y = X_raw[mask], y_raw[mask]
    X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te = split_70_10_10_10(X, y)

    sc = MinMaxScaler()
    X_tr_s  = sc.fit_transform(X_tr)
    X_v_s   = sc.transform(X_v)
    X_cal_s = sc.transform(X_cal)
    X_te_s  = sc.transform(X_te)

    print(f"  Split: train={len(y_tr)}  val={len(y_v)}  cal={len(y_cal)}  test={len(y_te)}")

    all_results = []
    best_val_result = None

    for i, (msl, depth, n_est, sub, (tau_lo, tau_hi)) in enumerate(GRID):
        tag = (f"msl={msl:2d} depth={depth} n={n_est} sub={sub:.1f} "
               f"τ=({tau_lo:.3f},{tau_hi:.3f})")
        try:
            gbm_lo = GradientBoostingRegressor(
                loss='quantile', alpha=tau_lo, n_estimators=n_est,
                max_depth=depth, learning_rate=LR,
                min_samples_leaf=msl, subsample=sub,
                random_state=RANDOM_SEED)
            gbm_hi = GradientBoostingRegressor(
                loss='quantile', alpha=tau_hi, n_estimators=n_est,
                max_depth=depth, learning_rate=LR,
                min_samples_leaf=msl, subsample=sub,
                random_state=RANDOM_SEED)
            gbm_lo.fit(X_tr_s, y_tr)
            gbm_hi.fit(X_tr_s, y_tr)

            # CQR calibration on cal set
            p_lo_cal = np.minimum(gbm_lo.predict(X_cal_s), gbm_hi.predict(X_cal_s))
            p_hi_cal = np.maximum(gbm_lo.predict(X_cal_s), gbm_hi.predict(X_cal_s))
            q_hat = cqr_calibrate(y_cal, p_lo_cal, p_hi_cal)

            # Evaluate on val (for selection) and test (for reporting)
            p_lo_v = np.minimum(gbm_lo.predict(X_v_s), gbm_hi.predict(X_v_s))
            p_hi_v = np.maximum(gbm_lo.predict(X_v_s), gbm_hi.predict(X_v_s))
            lo_v, hi_v = apply_cqr(p_lo_v, p_hi_v, q_hat)
            m_v = pi_metrics(y_v, lo_v, hi_v)

            p_lo_te = np.minimum(gbm_lo.predict(X_te_s), gbm_hi.predict(X_te_s))
            p_hi_te = np.maximum(gbm_lo.predict(X_te_s), gbm_hi.predict(X_te_s))
            lo_te, hi_te = apply_cqr(p_lo_te, p_hi_te, q_hat)
            m_te = pi_metrics(y_te, lo_te, hi_te)

            base_w_cal = float(np.mean(p_hi_cal - p_lo_cal))
            base_w_te  = float(np.mean(p_hi_te - p_lo_te))

            row = {
                'params': {'min_samples_leaf': msl, 'max_depth': depth,
                           'n_estimators': n_est, 'subsample': sub,
                           'tau_lo': tau_lo, 'tau_hi': tau_hi},
                'q_hat':    round(q_hat, 6),
                'base_width_cal': round(base_w_cal, 4),
                'base_width_te':  round(base_w_te, 4),
                'val':  m_v,
                'test': m_te,
            }
            all_results.append(row)

            # Track best by val (PICP ≥ 0.95 → min MPIW)
            if m_v['PICP'] >= 0.95:
                if best_val_result is None or m_v['MPIW'] < best_val_result['val']['MPIW']:
                    best_val_result = row

            if (i + 1) % 16 == 0 or i == len(GRID) - 1:
                n_valid = sum(1 for r in all_results if r['val']['PICP'] >= 0.95)
                if all_results:
                    best_so_far = min(
                        (r for r in all_results if r['val']['PICP'] >= 0.95),
                        key=lambda r: r['val']['MPIW'], default=None)
                    bs = f"best_val_MPIW={best_so_far['val']['MPIW']:.4f}" if best_so_far else "none valid"
                    print(f"  [{satellite}] {i+1:3d}/{len(GRID)}  valid={n_valid}  {bs}")

        except Exception as e:
            all_results.append({'params': {'min_samples_leaf': msl, 'max_depth': depth,
                                           'n_estimators': n_est, 'subsample': sub,
                                           'tau_lo': tau_lo, 'tau_hi': tau_hi},
                                 'error': str(e), 'val': None, 'test': None})

    # ── fallback: PICP ≥ 0.93 if nothing at 0.95 ─────────────────────────────
    selection_threshold = 0.95
    if best_val_result is None:
        selection_threshold = 0.93
        candidates = [r for r in all_results if r.get('val') and r['val']['PICP'] >= 0.93]
        if candidates:
            best_val_result = min(candidates, key=lambda r: r['val']['MPIW'])

    # ── save full grid ─────────────────────────────────────────────────────────
    out_dir = OUTPUT_PATH / satellite.lower().replace('-', '')
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(all_results, out_dir / 'grid_results.json')

    df_rows = []
    for r in all_results:
        row = {**r['params']}
        if r.get('val'):
            row.update({'q_hat': r['q_hat'],
                        'base_w_cal': r['base_width_cal'],
                        'base_w_te':  r['base_width_te'],
                        'val_PICP': r['val']['PICP'], 'val_MPIW': r['val']['MPIW'],
                        'test_PICP': r['test']['PICP'], 'test_MPIW': r['test']['MPIW']})
        else:
            row['error'] = r.get('error', '')
        df_rows.append(row)
    pd.DataFrame(df_rows).to_csv(out_dir / 'grid_summary.csv', index=False)

    # ── report ─────────────────────────────────────────────────────────────────
    print(f"\n  [{satellite}] Grid complete — {len(all_results)} configs run")
    valid_results = [r for r in all_results if r.get('val') and r['val']['PICP'] >= selection_threshold]
    print(f"  Valid at PICP≥{selection_threshold}: {len(valid_results)}")

    # Top 5 by val MPIW
    top5 = sorted(valid_results, key=lambda r: r['val']['MPIW'])[:5]
    print(f"\n  Top 5 by val MPIW (selection set, PICP≥{selection_threshold}):")
    print(f"  {'msl':>4} {'depth':>5} {'n_est':>5} {'sub':>4} {'tau_lo':>6}  "
          f"{'q̂':>6}  {'base_w_te':>9}  {'val_PICP':>8}  {'val_MPIW':>8}  "
          f"{'test_PICP':>9}  {'test_MPIW':>9}")
    for r in top5:
        p = r['params']
        print(f"  {p['min_samples_leaf']:>4} {p['max_depth']:>5} {p['n_estimators']:>5} "
              f"{p['subsample']:>4.1f} {p['tau_lo']:>6.3f}  "
              f"{r['q_hat']:>6.4f}  {r['base_width_te']:>9.4f}  "
              f"{r['val']['PICP']:>8.4f}  {r['val']['MPIW']:>8.4f}  "
              f"{r['test']['PICP']:>9.4f}  {r['test']['MPIW']:>9.4f}")

    if best_val_result:
        bp  = best_val_result['params']
        bm  = best_val_result['test']
        bl  = BASELINES[satellite]
        delta_gbm  = bm['MPIW'] - bl['GBM_CQR']['MPIW']
        delta_qsvr = bm['MPIW'] - bl['QSVR']['MPIW']

        print(f"\n  SELECTED (val PICP≥{selection_threshold}, min val MPIW):")
        print(f"    min_samples_leaf={bp['min_samples_leaf']}  max_depth={bp['max_depth']}  "
              f"n_estimators={bp['n_estimators']}  subsample={bp['subsample']}  "
              f"tau=({bp['tau_lo']},{bp['tau_hi']})")
        print(f"    val : PICP={best_val_result['val']['PICP']:.4f}  MPIW={best_val_result['val']['MPIW']:.4f}")
        print(f"    test: PICP={bm['PICP']:.4f}  MPIW={bm['MPIW']:.4f}")
        print(f"    vs GBM CQR baseline : ΔMPIW = {delta_gbm:+.4f} "
              f"({'✓ improved' if delta_gbm < 0 else '✗ degraded'})")
        print(f"    vs QSVR best        : ΔMPIW = {delta_qsvr:+.4f} "
              f"({'✓ improved' if delta_qsvr < 0 else '✗ still wider'})")

        # Save best config result
        save_json({
            'selected_params': bp,
            'q_hat': best_val_result['q_hat'],
            'base_width_te': best_val_result['base_width_te'],
            'val':  best_val_result['val'],
            'test': best_val_result['test'],
            'delta_vs_gbm_cqr':  round(delta_gbm, 6),
            'delta_vs_qsvr':     round(delta_qsvr, 6),
            'baselines':          bl,
            'selection_threshold': selection_threshold,
        }, out_dir / 'best_config.json')

        # Plot for best config
        gbm_lo_best = GradientBoostingRegressor(
            loss='quantile', alpha=bp['tau_lo'], n_estimators=bp['n_estimators'],
            max_depth=bp['max_depth'], learning_rate=LR,
            min_samples_leaf=bp['min_samples_leaf'], subsample=bp['subsample'],
            random_state=RANDOM_SEED)
        gbm_hi_best = GradientBoostingRegressor(
            loss='quantile', alpha=bp['tau_hi'], n_estimators=bp['n_estimators'],
            max_depth=bp['max_depth'], learning_rate=LR,
            min_samples_leaf=bp['min_samples_leaf'], subsample=bp['subsample'],
            random_state=RANDOM_SEED)
        gbm_lo_best.fit(X_tr_s, y_tr); gbm_hi_best.fit(X_tr_s, y_tr)
        p_lo_c = np.minimum(gbm_lo_best.predict(X_cal_s), gbm_hi_best.predict(X_cal_s))
        p_hi_c = np.maximum(gbm_lo_best.predict(X_cal_s), gbm_hi_best.predict(X_cal_s))
        q_best = cqr_calibrate(y_cal, p_lo_c, p_hi_c)
        p_lo_t = np.minimum(gbm_lo_best.predict(X_te_s), gbm_hi_best.predict(X_te_s))
        p_hi_t = np.maximum(gbm_lo_best.predict(X_te_s), gbm_hi_best.predict(X_te_s))
        lo_plot, hi_plot = apply_cqr(p_lo_t, p_hi_t, q_best)
        title = (f"{satellite} Tuned GBM CQR — msl={bp['min_samples_leaf']} "
                 f"depth={bp['max_depth']} τ=({bp['tau_lo']},{bp['tau_hi']})")
        save_pi_plot(y_te, lo_plot, hi_plot, title,
                     out_dir / 'plots' / 'best_config_plot.png')
        all_sensor_results[satellite] = best_val_result
    else:
        print(f"\n  ⚠  No valid config found at PICP≥{selection_threshold}.")
        all_sensor_results[satellite] = None

# ── final summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 10 COMPLETE — TUNED GBM CQR SUMMARY")
print("=" * 70)
for sat, result in all_sensor_results.items():
    bl = BASELINES[sat]
    print(f"\n  {sat}")
    print(f"    GBM CQR baseline : PICP={bl['GBM_CQR']['PICP']}  MPIW={bl['GBM_CQR']['MPIW']}")
    print(f"    QSVR best        : PICP={bl['QSVR']['PICP']}   MPIW={bl['QSVR']['MPIW']}")
    if result:
        m = result['test']
        delta = m['MPIW'] - bl['GBM_CQR']['MPIW']
        print(f"    Phase 10 tuned   : PICP={m['PICP']:.4f}  MPIW={m['MPIW']:.4f}  "
              f"ΔMPIW={delta:+.4f} ({'✓' if delta < 0 else '✗'})")
    else:
        print(f"    Phase 10 tuned   : no valid config found")
print(f"\n  Output → {OUTPUT_PATH}")
print("Done.")
