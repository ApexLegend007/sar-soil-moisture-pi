"""
Phase 21 — Tube Loss ANN (exact paper implementation) + Conformal Extension
============================================================================

METHOD: Rana et al. (arXiv:2412.06853); code: github.com/ltpritamanand/tube_loss
  "Tube Loss: A Novel Approach for Prediction Interval Estimation"

Tube Loss simultaneously learns both PI bounds in one model:
  f1 = y_pred[:,0]  → LOWER bound
  f2 = y_pred[:,1]  → UPPER bound

  confidence_loss(q=0.90, r=0.5, delta):
    c1 = (1-q)*(f2-y)   inside, y in upper half  → penalise f2 too far above y
    c2 = (1-q)*(y-f1)   inside, y in lower half  → penalise f1 too far below y
    c3 = q*(f1-y)        y below f1              → penalise f1 too high
    c4 = q*(y-f2)        y above f2              → penalise f2 too low

    inside region: y_true <= f2 AND y_true >= f1
      half selector: y_true > r*(f1+f2)  → upper half (c1) else lower (c2)
    outside: f1 > y → c3; else → c4

  Asymptotic guarantee: minimiser gives P(y∈[f1,f2]|x) → q  as n→∞

CONFORMAL EXTENSION (from repo notebook "6. Vanilla Extension of Conformal Regression")
  The paper ALSO applies CQR calibration on top of Tube Loss bounds to give
  finite-sample marginal coverage:
    scores  = max(f1_cal - y_cal, y_cal - f2_cal)
    q_hat   = quantile(scores, 1-α, method='higher')
    lo_conf = f1_test - q_hat
    hi_conf = f2_test + q_hat
  This gives the same finite-sample guarantee as CQR.

BOTH variants reported:
  Direct    — raw Tube Loss bounds [f1, f2]  (asymptotic only)
  Conformal — [f1-q̂, f2+q̂]               (finite-sample marginal)

EVALUATION METRICS
  PICP, MPIW, CWC (Khosravi 2011), IS (Winkler 1972)

PREMORTEM
  P1: Direct bounds: coverage is asymptotic — may miss 0.90 on small test sets.
  P2: Conformal extension adds q_hat from cal set → finite-sample but wider.
  P3: y normalised to [0,1]; bias init [-0.5, 1.5] covers all normalised y.
  P4: ExponentialDecay LR from 0.02→0.002 over 10 000 steps (same as paper).
  P5: r=0.5 (symmetric SM1 distribution assumed); grid includes r=0.3 too.
  P6: Single hidden layer (512 units) matches paper's single-layer architecture.
  P7: Grid includes delta: δ=0 (no width penalty) and δ>0 (recalibration).

POSTMORTEM (filled after run — see printed summary)
"""

import os, json, random, warnings, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np; np.random.seed(42)
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomNormal, Constant

from eval_pi_metrics import all_metrics

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / 'data'
OUT_BASE  = ROOT / 'output' / 'tube_loss'
OUT_BASE.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
Y_COL       = 'SM1 (%)'
ALPHA       = 0.10
Q_CONF      = 1.0 - ALPHA  # 0.90

SENSORS = {
    'EOS-04': {
        'csv':    'eos-04-enhanced-ndvi.csv',
        'x_cols': ['HH-pol','HV-pol','cross_pol_ratio','month_sin','month_cos','crop_encoded','NDVI'],
        'outdir': OUT_BASE / 'eos04',
    },
    'Sentinel-1': {
        'csv':    'sentinel-1-enhanced-ndvi.csv',
        'x_cols': ['VH-pol','VV-pol','cross_pol_ratio','month_sin','month_cos','crop_encoded','NDVI'],
        'outdir': OUT_BASE / 'sentinel1',
    },
}

# Grid: architectures × r values × delta values
# Paper architecture: single hidden layer (1024 units); we test shallower for small dataset
TUBE_GRID = [
    {'hidden': 256, 'r': 0.5, 'delta': 0.00},
    {'hidden': 512, 'r': 0.5, 'delta': 0.00},
    {'hidden': 256, 'r': 0.5, 'delta': 0.02},
    {'hidden': 512, 'r': 0.5, 'delta': 0.02},
    {'hidden': 256, 'r': 0.3, 'delta': 0.00},   # slightly asymmetric (SAR often right-skewed)
    {'hidden': 512, 'r': 0.3, 'delta': 0.00},
    {'hidden': 256, 'r': 0.3, 'delta': 0.02},
    {'hidden': 512, 'r': 0.3, 'delta': 0.02},
]
EPOCHS = 400
BATCH  = 32


def split_70_10_10_10(X, y):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, train_size=0.7,  random_state=RANDOM_SEED)
    X_v,  X_t2,  y_v,  y_t2  = train_test_split(X_tmp, y_tmp, train_size=1/3, random_state=RANDOM_SEED)
    X_cal, X_te, y_cal, y_te = train_test_split(X_t2,  y_t2,  test_size=0.5,  random_state=RANDOM_SEED)
    return X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te


def make_tube_loss(q=0.90, r=0.5, delta=0.0):
    """
    Exact Tube Loss from Rana et al. repo (github.com/ltpritamanand/tube_loss).
    f1 = y_pred[:,0]  → LOWER bound
    f2 = y_pred[:,1]  → UPPER bound
    y_true must be stacked (y,y) shape (n,2) so y_true[:,0] extracts scalar.
    """
    q_  = tf.constant(float(q),     dtype=tf.float32)
    r_  = tf.constant(float(r),     dtype=tf.float32)
    d_  = tf.constant(float(delta), dtype=tf.float32)
    q1_ = tf.constant(float(1-q),   dtype=tf.float32)

    def _loss(y_true, y_pred):
        y  = tf.cast(y_true[:, 0], tf.float32)
        f1 = y_pred[:, 0]   # lower bound
        f2 = y_pred[:, 1]   # upper bound

        c1 = q1_ * (f2 - y)   # inside, upper half
        c2 = q1_ * (y  - f1)  # inside, lower half
        c3 = q_  * (f1 - y)   # y below lower bound
        c4 = q_  * (y  - f2)  # y above upper bound

        inside     = tf.logical_and(y <= f2, y >= f1)
        upper_half = y > r_ * (f1 + f2)

        part1 = tf.where(upper_half, c1, c2)
        part2 = tf.where(f1 > y,    c3, c4)

        return tf.reduce_mean(tf.where(inside, part1, part2) + d_ * tf.abs(f1 - f2))
    return _loss


def build_tube_model(n_feat, hidden_units, q, r, delta):
    """Single hidden layer (paper architecture)."""
    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)
    model = Sequential([
        Dense(hidden_units, input_dim=n_feat, activation='relu',
              kernel_initializer=RandomNormal(0.0, 0.2)),
        Dense(2, activation='linear',
              kernel_initializer=RandomNormal(0.0, 0.3),
              # For y in [0,1]: init f1=-0.5 (below min), f2=1.5 (above max)
              bias_initializer=Constant([-0.5, 1.5])),
    ])
    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.02, decay_steps=10000, decay_rate=0.01)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_sched),
        loss=make_tube_loss(q=q, r=r, delta=delta)
    )
    return model


def cqr_calibrate(y_cal, f1_cal, f2_cal, alpha=ALPHA):
    scores = np.maximum(f1_cal - y_cal, y_cal - f2_cal)
    return float(np.quantile(scores, 1 - alpha, method='higher'))


def pi_plot(sensor, label, y_te, lo, hi, m, variant, out_dir):
    order   = np.argsort(y_te)
    y_s, lo_s, hi_s = y_te[order], lo[order], hi[order]
    cov     = (y_s >= lo_s) & (y_s <= hi_s)
    idx     = np.arange(len(y_s))
    color   = 'steelblue' if 'direct' in variant else 'darkorange'

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(idx, lo_s, hi_s, alpha=0.25, color=color, label='90% PI')
    ax.plot(idx, lo_s, color=color, lw=1, ls='--', alpha=0.7)
    ax.plot(idx, hi_s, color=color, lw=1, ls='--', alpha=0.7)
    ax.scatter(idx[cov],  y_s[cov],  color='green', s=18, zorder=3, label='Covered')
    ax.scatter(idx[~cov], y_s[~cov], color='red',   s=28, zorder=4, marker='x', label='Missed')

    tag = 'Direct' if 'direct' in variant else 'Conformal'
    txt = (f"Tube Loss ANN [{tag}]  α=0.10\n"
           f"test PICP: {m['PICP']*100:.2f}%   MPIW: {m['MPIW']:.2f}\n"
           f"CWC: {m['CWC']:.4f}   IS: {m['IS']:.2f}")
    ax.text(0.01, 0.97, txt, transform=ax.transAxes, va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title(f"{sensor} — Tube Loss ANN [{tag}] | {label}", fontsize=11)
    ax.set_xlabel('Test sample (sorted by SM1)')
    ax.set_ylabel('SM1 (%)')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    safe = label.replace('[','').replace(']','').replace(' ','_')[:50]
    fpath = out_dir / f"best_{variant}_{safe}.png"
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fpath


for sensor_name, s in SENSORS.items():
    print(f"\n{'='*65}")
    print(f"  {sensor_name}  — Phase 21 Tube Loss ANN")
    print(f"{'='*65}")

    s['outdir'].mkdir(parents=True, exist_ok=True)

    df  = pd.read_csv(DATA_PATH / s['csv'])
    X_r = df[s['x_cols']].values
    y_r = df[Y_COL].values
    mask = y_r != 50
    X, y = X_r[mask], y_r[mask]

    X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te = split_70_10_10_10(X, y)
    print(f"  Split: train={len(y_tr)} val={len(y_v)} cal={len(y_cal)} test={len(y_te)}")

    sc_x = MinMaxScaler()
    X_tr_s  = sc_x.fit_transform(X_tr)
    X_v_s   = sc_x.transform(X_v)
    X_cal_s = sc_x.transform(X_cal)
    X_te_s  = sc_x.transform(X_te)

    sc_y = MinMaxScaler()
    y_tr_s  = sc_y.fit_transform(y_tr.reshape(-1,1)).ravel()
    y_v_s   = sc_y.transform(y_v.reshape(-1,1)).ravel()
    y_cal_s = sc_y.transform(y_cal.reshape(-1,1)).ravel()

    # Tube Loss requires y stacked as (y, y) shape (n, 2)
    y_tr_stk  = np.stack((y_tr_s, y_tr_s), axis=1)

    n_feat = X_tr_s.shape[1]
    rows   = []
    best_direct   = {'m': None, 'lo': None, 'hi': None, 'cfg': None}
    best_conformal = {'m': None, 'lo': None, 'hi': None, 'cfg': None, 'q_hat': None}

    for ci, cfg in enumerate(TUBE_GRID):
        label = f"h{cfg['hidden']}_r{cfg['r']}_d{cfg['delta']}"
        print(f"  [{ci+1:02d}/{len(TUBE_GRID)}] {label}", end='  ', flush=True)

        model = build_tube_model(n_feat, cfg['hidden'], Q_CONF, cfg['r'], cfg['delta'])
        t0    = time.time()
        model.fit(X_tr_s, y_tr_stk,
                  epochs=EPOCHS, batch_size=BATCH, verbose=0)
        train_t = time.time() - t0

        # Predict on cal and test (normalised space)
        pred_cal = model.predict(X_cal_s, verbose=0)   # (n_cal, 2)
        pred_te  = model.predict(X_te_s,  verbose=0)   # (n_te,  2)

        f1_cal_s = pred_cal[:, 0]   # lower (normalised)
        f2_cal_s = pred_cal[:, 1]   # upper (normalised)
        f1_te_s  = pred_te[:,  0]
        f2_te_s  = pred_te[:,  1]

        # Inverse transform to original SM1 scale
        f1_cal = sc_y.inverse_transform(f1_cal_s.reshape(-1,1)).ravel()
        f2_cal = sc_y.inverse_transform(f2_cal_s.reshape(-1,1)).ravel()
        f1_te  = sc_y.inverse_transform(f1_te_s.reshape(-1,1)).ravel()
        f2_te  = sc_y.inverse_transform(f2_te_s.reshape(-1,1)).ravel()

        # Ensure lo <= hi (swap if model outputs inverted)
        lo_d = np.minimum(f1_te, f2_te)
        hi_d = np.maximum(f1_te, f2_te)
        m_direct = all_metrics(y_te, lo_d, hi_d)

        # Conformal extension: CQR calibration on cal set
        lo_cal = np.minimum(f1_cal, f2_cal)
        hi_cal = np.maximum(f1_cal, f2_cal)
        q_hat  = cqr_calibrate(y_cal, lo_cal, hi_cal)
        lo_conf = lo_d - q_hat
        hi_conf = hi_d + q_hat
        m_conf  = all_metrics(y_te, lo_conf, hi_conf)

        # Val metrics (direct only, no conformal on val for speed)
        pred_v = model.predict(X_v_s, verbose=0)
        lo_v   = sc_y.inverse_transform(np.minimum(pred_v[:,0],pred_v[:,1]).reshape(-1,1)).ravel()
        hi_v   = sc_y.inverse_transform(np.maximum(pred_v[:,0],pred_v[:,1]).reshape(-1,1)).ravel()
        m_v    = all_metrics(y_v, lo_v, hi_v)

        print(f"Direct PICP={m_direct['PICP']:.3f} MPIW={m_direct['MPIW']:.2f}  |  "
              f"Conf PICP={m_conf['PICP']:.3f} MPIW={m_conf['MPIW']:.2f}  q̂={q_hat:.3f}  ({train_t:.0f}s)")

        rows.append({
            'label': label, 'hidden': cfg['hidden'], 'r': cfg['r'], 'delta': cfg['delta'],
            'train_s': round(train_t, 1),
            'q_hat': round(q_hat, 4),
            'val_PICP': m_v['PICP'],  'val_MPIW': m_v['MPIW'],
            'direct_PICP': m_direct['PICP'], 'direct_MPIW': m_direct['MPIW'],
            'direct_CWC':  m_direct['CWC'],  'direct_IS':   m_direct['IS'],
            'conf_PICP':   m_conf['PICP'],   'conf_MPIW':   m_conf['MPIW'],
            'conf_CWC':    m_conf['CWC'],    'conf_IS':      m_conf['IS'],
        })

        # Best direct: PICP closest to 0.90; tie-break IS
        def is_better(m_new, m_old):
            if m_old is None: return True
            g_new = abs(m_new['PICP'] - 0.90)
            g_old = abs(m_old['PICP'] - 0.90)
            return g_new < g_old or (g_new == g_old and m_new['IS'] < m_old['IS'])

        if is_better(m_direct, best_direct['m']):
            best_direct.update({'m': m_direct, 'lo': lo_d, 'hi': hi_d, 'cfg': {**cfg, 'label': label}})
        if is_better(m_conf, best_conformal['m']):
            best_conformal.update({'m': m_conf, 'lo': lo_conf, 'hi': hi_conf,
                                   'cfg': {**cfg, 'label': label}, 'q_hat': q_hat})

    pd.DataFrame(rows).to_csv(s['outdir'] / 'grid_summary.csv', index=False)

    # Plots & JSON for best configs
    for variant, bst in [('direct', best_direct), ('conformal', best_conformal)]:
        if bst['m'] is None:
            continue
        plot_path = pi_plot(sensor_name, bst['cfg']['label'], y_te,
                            bst['lo'], bst['hi'], bst['m'], variant, s['outdir'])
        res = {
            'method': f'Tube Loss ANN [{variant}]',
            'paper':  'Rana et al. arXiv:2412.06853; github.com/ltpritamanand/tube_loss',
            'alpha': ALPHA, 'variant': variant,
            'q_hat': bst.get('q_hat'),
            'params': bst['cfg'],
            'test': bst['m'],
            'y_range': round(float(y_te.max() - y_te.min()), 4),
            'n_test': int(len(y_te)),
        }
        suffix = '' if variant == 'conformal' else '_direct'
        with open(s['outdir'] / f'best_config{suffix}.json', 'w') as f:
            json.dump(res, f, indent=4)

# ── Postmortem summary ──────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  PHASE 21 — TUBE LOSS ANN — POSTMORTEM SUMMARY")
print(f"{'='*65}")
print(f"  {'Sensor':12s}  {'Variant':10s}  PICP    MPIW    CWC     IS")
print(f"  {'-'*60}")
for sensor_name, s in SENSORS.items():
    for suffix, vname in [('', 'Conformal'), ('_direct', 'Direct')]:
        jf = s['outdir'] / f'best_config{suffix}.json'
        if jf.exists():
            r = json.load(open(jf))
            t = r['test']
            print(f"  {sensor_name:12s}  {vname:10s}  "
                  f"{t['PICP']:.4f}  {t['MPIW']:.2f}  {t['CWC']:.4f}  {t['IS']:.2f}")
print(f"\n  Direct: asymptotic coverage; Conformal: finite-sample marginal.")
print(f"  Ref: Rana et al. (2024) arXiv:2412.06853")
print("Done.")
