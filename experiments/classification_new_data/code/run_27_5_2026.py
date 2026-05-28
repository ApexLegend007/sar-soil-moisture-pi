# -*- coding: utf-8 -*-
"""
New Model Experiments — 27 May 2026
=====================================
TARGET  : 90% PICP (alpha = 0.10) with tight MPIW
New models added to the 25-May-2026 benchmark:

  QR-NN     : Quantile Regression Neural Network (pinball loss, dual-output ANN)
               WITHOUT conformal calibration — a direct QR baseline.
               tau_lo = alpha/2 = 0.05 ;  tau_hi = 1 - alpha/2 = 0.95

  CQR_tube  : Exact Tube Loss ANN (Rana et al. arXiv:2412.06853,
               github.com/ltpritamanand/tube_loss) + CQR conformal extension.
               Uses the paper's confidence_loss with q=0.90, r=0.5, delta=0.
               Backed by finite-sample marginal coverage guarantee via CQR.

Reference for Tube Loss:
  Nand, Pritma (2024). "Tube Loss: A Novel Approach for Prediction Interval
  Estimation and Deep Probabilistic Forecasting." arXiv:2412.06853.
  Code: https://github.com/ltpritamanand/Tube_loss

Metrics : PICP | MPIW | CWC | IS (Winkler / Interval Score)
Output  : experiments/classification_new_data/output/27_5_2026_output/
"""

# ---- stdlib / env -------------------------------------------------------
import os, sys, warnings, json, random
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONHASHSEED']        = '42'
os.environ['PYTHONUTF8']            = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

random.seed(42)
np.random.seed(42)

# ---- plotting (Agg backend - no GUI needed) ------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        150,
    'savefig.dpi':       200,
    'savefig.bbox':      'tight',
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
})

plt.show = lambda: None   # block any accidental show() calls

# ---- ML / DL libraries --------------------------------------------------
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.callbacks import EarlyStopping   # type: ignore
from tensorflow.keras.initializers import RandomNormal, Constant

from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import MinMaxScaler

# =========================================================================
# PATHS & CONSTANTS
# =========================================================================
ROOT        = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / 'data'
OUTPUT_PATH = ROOT / 'output' / '27_5_2026_output'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_SEED  = 42
ALPHA        = 0.10           # <-- 90% coverage target
MU_COVERAGE  = 1 - ALPHA      # 0.90
ETA          = 50.0           # CWC penalty exponent (Khosravi 2011)

X_COLS_EOS = ['HH-pol','HV-pol','cross_pol_ratio','month_sin','month_cos','crop_encoded','NDVI']
X_COLS_SEN = ['VH-pol','VV-pol','cross_pol_ratio','month_sin','month_cos','crop_encoded','NDVI']
Y_COL      = 'SM1 (%)'

# Colour palette
PALETTE = {
    'QR-NN':    '#E91E63',   # pink
    'CQR_tube': '#00897B',   # teal
}

# =========================================================================
# METRICS
# =========================================================================

def picp(y_true, lo, hi):
    return float(np.mean((y_true >= lo) & (y_true <= hi)))

def mpiw(lo, hi):
    return float(np.mean(hi - lo))

def cwc(y_true, lo, hi, mu=MU_COVERAGE, y_range=1.0):
    """
    Coverage Width Criterion — Khosravi (2011), same formula as eval_comparison.
      PINAW = MPIW / R          (R = test-set y range, normalises to [0,1] scale)
      gamma = 0  if PICP >= mu  (no penalty)
              1  otherwise      (binary penalty flag)
      CWC   = PINAW * (1 + gamma * exp(-50 * (PICP - mu)))
    Lower is better; comparable across sensors/datasets.
    """
    cov   = picp(y_true, lo, hi)
    pinaw = mpiw(lo, hi) / max(float(y_range), 1e-9)
    gamma = 0.0 if cov >= mu else 1.0
    return float(pinaw * (1.0 + gamma * np.exp(-ETA * (cov - mu))))

def interval_score(y_true, lo, hi, alpha=ALPHA):
    """
    Winkler Interval Score (IS):
      IS = (hi-lo) + (2/alpha)*max(lo-y, 0) + (2/alpha)*max(y-hi, 0)
    Lower is better.
    """
    width = hi - lo
    under = np.maximum(lo - y_true, 0.0)
    over  = np.maximum(y_true - hi, 0.0)
    return float(np.mean(width + (2.0 / alpha) * (under + over)))

def all_metrics(y_true, lo, hi, label='', alpha=ALPHA, y_range=1.0):
    p  = picp(y_true, lo, hi)
    m  = mpiw(lo, hi)
    c  = cwc(y_true, lo, hi, mu=1-alpha, y_range=y_range)
    s  = interval_score(y_true, lo, hi, alpha=alpha)
    status = 'OK' if p >= (1 - alpha) else 'LOW'
    print(f'  [{label:14s}] PICP={p*100:5.2f}% [{status}]  MPIW={m:7.3f}  '
          f'CWC={c:8.6f}  IS={s:8.3f}')
    return {'PICP': round(p,6), 'MPIW': round(m,6),
            'CWC': round(c,6),  'IS':   round(s,6)}

# =========================================================================
# DATA HELPERS
# =========================================================================

def split_70_10_10_10(X, y):
    """70% train | 10% val (early-stop) | 10% cal (conformal) | 10% test."""
    X_tr,  X_tmp,  y_tr,  y_tmp  = train_test_split(X, y, train_size=0.70,
                                                      random_state=RANDOM_SEED)
    X_v,   X_tmp2, y_v,   y_tmp2 = train_test_split(X_tmp,  y_tmp,  train_size=1/3,
                                                      random_state=RANDOM_SEED)
    X_cal, X_te,   y_cal, y_te   = train_test_split(X_tmp2, y_tmp2, test_size=0.5,
                                                      random_state=RANDOM_SEED)
    return X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te

def scale_splits(X_tr, X_v, X_cal, X_te):
    sc = MinMaxScaler()
    return (sc.fit_transform(X_tr), sc.transform(X_v),
            sc.transform(X_cal), sc.transform(X_te), sc)

def scale_y(y_tr, y_v, y_cal):
    """MinMax scale for tube model y target (tube loss expects normalised y)."""
    sc_y = MinMaxScaler()
    y_tr_s  = sc_y.fit_transform(y_tr.reshape(-1,1)).ravel()
    y_v_s   = sc_y.transform(y_v.reshape(-1,1)).ravel()
    y_cal_s = sc_y.transform(y_cal.reshape(-1,1)).ravel()
    return y_tr_s, y_v_s, y_cal_s, sc_y

def cqr_calibrate(y_cal, lo_cal, hi_cal, alpha=ALPHA):
    """CQR finite-sample correction: level = (1-alpha)*(1+1/n)."""
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    n      = len(scores)
    level  = min(1.0, (1 - alpha) * (1 + 1.0 / n))
    return float(np.quantile(scores, level, method='higher'))

def apply_cqr(lo, hi, q_hat):
    return lo - q_hat, hi + q_hat

def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=4)

# =========================================================================
# LOSS FUNCTIONS
# =========================================================================

def pinball_loss(tau):
    """Quantile (pinball) loss for a single quantile tau."""
    def loss(y_true, y_pred):
        e = tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)
        return tf.reduce_mean(tf.maximum(tau * e, (tau - 1) * e))
    return loss


def make_tube_loss(q=0.90, r=0.5, delta=0.0):
    """
    Exact Tube Loss from Rana et al. (arXiv:2412.06853).
    Reference implementation: github.com/ltpritamanand/tube_loss

    f1 = y_pred[:,0]  → LOWER bound
    f2 = y_pred[:,1]  → UPPER bound
    y_true must be stacked (y,y) so y_true[:,0] gives the scalar.

    confidence_loss(q, r, delta):
      Inside PI:
        upper half (y > r*(f1+f2)): c1 = (1-q)*(f2-y)
        lower half:                 c2 = (1-q)*(y-f1)
      Outside PI:
        f1 > y  : c3 = q*(f1-y)
        else    : c4 = q*(y-f2)
      width penalty: delta * |f1 - f2|
    """
    q_   = tf.constant(float(q),     dtype=tf.float32)
    r_   = tf.constant(float(r),     dtype=tf.float32)
    d_   = tf.constant(float(delta), dtype=tf.float32)
    q1_  = tf.constant(float(1 - q), dtype=tf.float32)

    def _loss(y_true, y_pred):
        y  = tf.cast(y_true[:, 0], tf.float32)
        f1 = y_pred[:, 0]   # lower
        f2 = y_pred[:, 1]   # upper

        c1 = q1_ * (f2 - y)   # inside, upper half
        c2 = q1_ * (y  - f1)  # inside, lower half
        c3 = q_  * (f1 - y)   # y below lower
        c4 = q_  * (y  - f2)  # y above upper

        inside     = tf.logical_and(y <= f2, y >= f1)
        upper_half = y > r_ * (f1 + f2)

        part1 = tf.where(upper_half, c1, c2)
        part2 = tf.where(f1 > y, c3, c4)

        return tf.reduce_mean(tf.where(inside, part1, part2) + d_ * tf.abs(f1 - f2))
    return _loss

# =========================================================================
# ANN ARCHITECTURES
# =========================================================================

def build_dual_pinball_ann(input_dim):
    """Shared-backbone dual-output ANN: [lo_output, hi_output]."""
    inp    = tf.keras.Input(shape=(input_dim,))
    x      = tf.keras.layers.Dense(128, activation='relu')(inp)
    x      = tf.keras.layers.BatchNormalization()(x)
    x      = tf.keras.layers.Dropout(0.15)(x)
    x      = tf.keras.layers.Dense(64,  activation='relu')(x)
    x      = tf.keras.layers.BatchNormalization()(x)
    x      = tf.keras.layers.Dropout(0.10)(x)
    x      = tf.keras.layers.Dense(32,  activation='relu')(x)
    lo_out = tf.keras.layers.Dense(1, name='lo')(x)
    hi_out = tf.keras.layers.Dense(1, name='hi')(x)
    return tf.keras.Model(inputs=inp, outputs=[lo_out, hi_out])


def build_tube_ann_exact(input_dim, hidden_units=512):
    """
    Tube Loss ANN matching the paper architecture:
    single hidden layer, bias-init at [-0.5, 1.5] to span normalised y ∈ [0,1].
    Outputs [f1 (lower), f2 (upper)] concatenated.
    """
    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    model = Sequential([
        Dense(hidden_units, input_dim=input_dim, activation='relu',
              kernel_initializer=RandomNormal(0.0, 0.2)),
        Dense(2, activation='linear',
              kernel_initializer=RandomNormal(0.0, 0.3),
              bias_initializer=Constant([-0.5, 1.5])),
    ])
    return model

# =========================================================================
# MODEL RUNNERS
# =========================================================================

# ---------- 1. QR-NN (Quantile Regression Neural Network, no CQR) ---------
def run_qr_nn(X_tr_s, X_v_s, X_cal_s, X_te_s,
              y_tr, y_v, y_cal, y_te, alpha=ALPHA):
    """
    Dual-output pinball-loss ANN producing tau=alpha/2 and tau=1-alpha/2
    quantile predictions directly.  NO conformal calibration step.
    This is the plain QR-NN baseline without distribution-free coverage guarantee.

    References:
      Koenker & Bassett (1978) — Quantile Regression
      Cannon (2011) — Quantile regression neural networks
    """
    lo_tau, hi_tau = alpha / 2, 1 - alpha / 2
    print(f'  [QR-NN      ] Training (tau={lo_tau:.2f}/{hi_tau:.2f}, no CQR)...',
          end=' ', flush=True)
    input_dim = X_tr_s.shape[1]

    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)
    model = build_dual_pinball_ann(input_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
                  loss=[pinball_loss(lo_tau), pinball_loss(hi_tau)],
                  loss_weights=[1.0, 1.0])
    cb = [EarlyStopping(monitor='val_loss', patience=30,
                        restore_best_weights=True, verbose=0)]
    model.fit(X_tr_s, [y_tr, y_tr],
              validation_data=(X_v_s, [y_v, y_v]),
              epochs=400, batch_size=32, verbose=0, callbacks=cb)

    pt    = model.predict(X_te_s, verbose=0)
    lo_te = np.minimum(pt[0].flatten(), pt[1].flatten())
    hi_te = np.maximum(pt[0].flatten(), pt[1].flatten())
    print('done (no q_hat)')
    return lo_te, hi_te


# ---------- 2. CQR_tube (Exact Tube Loss ANN + CQR conformal extension) ---
def run_cqr_tube(X_tr_s, X_v_s, X_cal_s, X_te_s,
                 y_tr, y_v, y_cal, y_te, alpha=ALPHA):
    """
    Exact Tube Loss ANN (Rana et al. arXiv:2412.06853) with CQR conformal
    calibration extension (as in paper repo notebook 6):
      scores  = max(f1_cal - y_cal, y_cal - f2_cal)
      q_hat   = quantile(scores, (1-alpha)*(1+1/n), method='higher')
      lo_conf = f1_test - q_hat
      hi_conf = f2_test + q_hat

    Model is trained on normalised y ∈ [0,1]; bounds are inverse-transformed
    back to SM1 (%) before CQR calibration and reporting.

    Paper architecture: single hidden layer (512 units), ExponentialDecay LR.
    """
    print(f'  [CQR_tube   ] Training (Tube Loss q={1-alpha:.2f}, +CQR)...',
          end=' ', flush=True)
    input_dim = X_tr_s.shape[1]

    # Normalise y for tube model (requires y ≈ [0,1] for bias init to work)
    y_tr_s, y_v_s, y_cal_s, sc_y = scale_y(y_tr, y_v, y_cal)
    y_tr_stk = np.stack((y_tr_s, y_tr_s), axis=1)   # shape (n, 2)

    # LR schedule matching paper: 0.02 → 0.002 over 10 000 steps
    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.02, decay_steps=10000, decay_rate=0.1)

    model = build_tube_ann_exact(input_dim, hidden_units=512)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_sched),
                  loss=make_tube_loss(q=1 - alpha, r=0.5, delta=0.0))
    cb = [EarlyStopping(monitor='val_loss', patience=30,
                        restore_best_weights=True, verbose=0)]
    # val target also needs stacking
    y_v_stk = np.stack((y_v_s, y_v_s), axis=1)
    model.fit(X_tr_s, y_tr_stk,
              validation_data=(X_v_s, y_v_stk),
              epochs=400, batch_size=32, verbose=0, callbacks=cb)

    # Calibrate on cal set (inverse-transform to original SM1 scale)
    pred_cal = model.predict(X_cal_s, verbose=0)
    f1_cal_s = pred_cal[:, 0];  f2_cal_s = pred_cal[:, 1]
    f1_cal   = sc_y.inverse_transform(f1_cal_s.reshape(-1,1)).ravel()
    f2_cal   = sc_y.inverse_transform(f2_cal_s.reshape(-1,1)).ravel()
    lo_cal   = np.minimum(f1_cal, f2_cal)
    hi_cal   = np.maximum(f1_cal, f2_cal)

    q_hat = cqr_calibrate(y_cal, lo_cal, hi_cal, alpha)
    print(f'q_hat={q_hat:.3f}')

    # Test predictions
    pred_te = model.predict(X_te_s, verbose=0)
    f1_te_s = pred_te[:, 0];  f2_te_s = pred_te[:, 1]
    f1_te   = sc_y.inverse_transform(f1_te_s.reshape(-1,1)).ravel()
    f2_te   = sc_y.inverse_transform(f2_te_s.reshape(-1,1)).ravel()
    lo_te   = np.minimum(f1_te, f2_te)
    hi_te   = np.maximum(f1_te, f2_te)
    return apply_cqr(lo_te, hi_te, q_hat)

# =========================================================================
# PLOTTING  -  publication-quality
# =========================================================================

def plot_pi_series(y_true, lo, hi, model_name, satellite, out_path, alpha=ALPHA, y_range=1.0):
    """Full-width prediction interval series plot, sorted by actual value."""
    colour = PALETTE.get(model_name, '#607D8B')
    target = 1 - alpha
    n      = len(y_true)
    idx    = np.arange(n)

    order   = np.argsort(y_true)
    ys      = y_true[order]; los = lo[order]; his = hi[order]
    covered = (ys >= los) & (ys <= his)

    p  = picp(y_true, lo, hi)
    m  = mpiw(lo, hi)
    c  = cwc(y_true, lo, hi, mu=target, y_range=y_range)
    s  = interval_score(y_true, lo, hi, alpha=alpha)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(idx, los, his, alpha=0.20, color=colour, label='PI band')
    ax.plot(idx, los, color=colour, lw=0.8, ls='--', alpha=0.7, label='Lower / Upper')
    ax.plot(idx, his, color=colour, lw=0.8, ls='--', alpha=0.7)
    ax.scatter(idx[covered],  ys[covered],  s=10, color='#1B5E20',
               alpha=0.75, zorder=3, label='Covered')
    ax.scatter(idx[~covered], ys[~covered], s=14, color='#B71C1C',
               marker='x', zorder=4, label='Missed')

    ax_title = (f'{satellite}  --  {model_name}   '
                f'(target: {target*100:.0f}% PICP)')
    ax.set_title(ax_title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Sample Index (sorted by actual SM1)', fontsize=11)
    ax.set_ylabel('SM1 (%)', fontsize=11)

    ok  = p >= target
    fc  = '#E8F5E9' if ok else '#FFEBEE'
    ec  = '#388E3C' if ok else '#D32F2F'
    txt = (f'PICP : {p*100:.2f}%  ({"OK" if ok else "LOW"})\n'
           f'MPIW : {m:.3f}\n'
           f'CWC  : {c:.3f}\n'
           f'IS   : {s:.3f}')
    ax.text(0.015, 0.97, txt, transform=ax.transAxes,
            va='top', ha='left', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=fc,
                      edgecolor=ec, linewidth=1.5, alpha=0.92))

    legend_elems = [
        Patch(facecolor=colour, alpha=0.25, label=f'{target*100:.0f}% PI band'),
        Line2D([0],[0], color=colour, lw=1, ls='--', label='Bounds'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#1B5E20',
               markersize=7, label='Covered'),
        Line2D([0],[0], marker='x', color='#B71C1C',
               markersize=8, label='Missed'),
    ]
    ax.legend(handles=legend_elems, loc='upper right', fontsize=9,
              framealpha=0.9, edgecolor='#BDBDBD')

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_comparison_dashboard(results_dict, satellite, out_path, alpha=ALPHA):
    """2x2 dashboard comparing models on PICP, MPIW, CWC, IS."""
    target  = 1 - alpha
    valid   = {k: v for k,v in results_dict.items() if 'error' not in v}
    if not valid:
        return

    models   = list(valid.keys())
    colours  = [PALETTE.get(m, '#607D8B') for m in models]
    picps    = [valid[m]['PICP']*100 for m in models]
    mpiws    = [valid[m]['MPIW']     for m in models]
    cwcs_v   = [valid[m]['CWC']      for m in models]
    iss      = [valid[m]['IS']       for m in models]
    x        = np.arange(len(models))

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'{satellite}  --  QR-NN vs CQR_tube  (Target: {target*100:.0f}% PICP)',
                 fontsize=16, fontweight='bold', y=1.01)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    specs = [
        (gs[0,0], picps,  'PICP (%)',     'Coverage Probability',                   False, target*100),
        (gs[0,1], mpiws,  'MPIW',         'Mean Interval Width (lower is better)',  True,  None),
        (gs[1,0], cwcs_v, 'CWC',          'Coverage Width Criterion (lower=better)',True,  None),
        (gs[1,1], iss,    'IS (Winkler)', 'Interval Score (lower is better)',        True,  None),
    ]

    for spec, vals, ylabel, title, lower_better, hline in specs:
        ax = fig.add_subplot(spec)
        bar_colours = []
        for i, (m, v) in enumerate(zip(models, vals)):
            if ylabel == 'PICP (%)':
                bc = '#388E3C' if v >= target*100 else '#C62828'
            else:
                bc = colours[i]
            bar_colours.append(bc)

        bars = ax.bar(x, vals, color=bar_colours, edgecolor='white',
                      linewidth=1.2, alpha=0.88, width=0.6)

        if hline is not None:
            ax.axhline(hline, color='#E53935', ls='--', lw=1.8,
                       label=f'Target {hline:.0f}%')
            ax.legend(fontsize=9, framealpha=0.85)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005*max(vals),
                    f'{val:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha='right', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='semibold')

        vmin, vmax = min(vals), max(vals)
        pad = (vmax - vmin) * 0.18 if (vmax - vmin) > 0 else 1.0
        ax.set_ylim(max(0, vmin - pad), vmax + pad * 1.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path.name}')


def plot_summary_heatmap(all_results, out_path, alpha=ALPHA):
    """Heatmap of PICP/MPIW/CWC/IS across all models x satellites."""
    target  = 1 - alpha
    metrics = ['PICP', 'MPIW', 'CWC', 'IS']
    rows    = []
    row_labels = []

    for sat, res in all_results.items():
        for model, m in res.items():
            if 'error' in m:
                continue
            rows.append([m['PICP']*100, m['MPIW'], m['CWC'], m['IS']])
            row_labels.append(f'{sat[:4]}  {model}')

    if not rows:
        return

    data = np.array(rows)   # shape: (n_rows, 4)

    norm_data = np.zeros_like(data)
    for col in range(data.shape[1]):
        col_min, col_max = data[:,col].min(), data[:,col].max()
        rng = col_max - col_min if col_max > col_min else 1.0
        if col == 0:  # PICP: higher is better
            norm_data[:,col] = (data[:,col] - col_min) / rng
        else:          # rest: lower is better
            norm_data[:,col] = 1 - (data[:,col] - col_min) / rng

    fig, ax = plt.subplots(figsize=(11, max(5, len(rows)*0.55 + 2)))
    im = ax.imshow(norm_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(4))
    ax.set_xticklabels(['PICP (%)', 'MPIW', 'CWC', 'IS (Winkler)'],
                       fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = data[r,c]
            fmt = f'{val:.2f}' if c > 0 else f'{val:.1f}%'
            if c == 0 and (val/100) < target:
                fmt += '*'
            ax.text(c, r, fmt, ha='center', va='center',
                    fontsize=9, color='black' if 0.25 < norm_data[r,c] < 0.85 else 'white',
                    fontweight='bold')

    ax.set_title(f'Model Performance Heatmap  (target PICP >= {target*100:.0f}%)\n'
                 f'Green = better | * = below target',
                 fontsize=12, fontweight='bold', pad=12)
    plt.colorbar(im, ax=ax, label='Relative performance (higher = better)',
                 shrink=0.7, pad=0.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path.name}')


# =========================================================================
# MAIN LOOP
# =========================================================================

print('=' * 68)
print('  NEW MODEL EXPERIMENTS  --  27 May 2026')
print(f'  TARGET : {MU_COVERAGE*100:.0f}% PICP (alpha={ALPHA})')
print('  Models : QR-NN | CQR_tube')
print('  Metrics: PICP | MPIW | CWC | IS (Winkler Interval Score)')
print(f'  Output : {OUTPUT_PATH}')
print('=' * 68)

print('\n[Data] Loading CSVs ...')
eos_df = pd.read_csv(DATA_PATH / 'eos-04-enhanced-ndvi.csv')
sen_df = pd.read_csv(DATA_PATH / 'sentinel-1-enhanced-ndvi.csv')
print(f'  EOS-04: {len(eos_df)} rows  |  Sentinel-1: {len(sen_df)} rows')

datasets = [
    ('EOS-04',     eos_df[X_COLS_EOS].values, eos_df[Y_COL].values),
    ('Sentinel-1', sen_df[X_COLS_SEN].values, sen_df[Y_COL].values),
]

RUNNERS = [
    ('QR-NN',    run_qr_nn),
    ('CQR_tube', run_cqr_tube),
]

all_results = {}

for satellite, X_raw, y_raw in datasets:
    print(f'\n{"="*68}')
    print(f'  SATELLITE : {satellite}')
    print(f'{"="*68}')

    mask = y_raw != 50
    X, y = X_raw[mask], y_raw[mask]

    X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te = split_70_10_10_10(X, y)
    X_tr_s, X_v_s, X_cal_s, X_te_s, _ = scale_splits(X_tr, X_v, X_cal, X_te)

    # y_range of test set — used for PINAW normalisation in CWC (Khosravi 2011)
    y_range = float(y_te.max() - y_te.min())

    print(f'  Split: train={len(y_tr)} | val={len(y_v)} | '
          f'cal={len(y_cal)} | test={len(y_te)}  y_range={y_range:.3f}')
    print(f'  {"-"*62}')

    sat_dir  = OUTPUT_PATH / satellite.replace('-', '_')
    plot_dir = sat_dir / 'plots'
    sat_results = {}

    for model_name, runner in RUNNERS:
        try:
            lo, hi = runner(X_tr_s, X_v_s, X_cal_s, X_te_s,
                            y_tr,   y_v,   y_cal,   y_te)
            m = all_metrics(y_te, lo, hi, label=model_name, y_range=y_range)
            sat_results[model_name] = m

            plot_pi_series(
                y_te, lo, hi,
                model_name=model_name,
                satellite=satellite,
                out_path=plot_dir / f'{model_name}_pi_series.png',
                y_range=y_range,
            )
        except Exception as e:
            import traceback
            print(f'  [ERROR] {model_name}: {e}')
            traceback.print_exc()
            sat_results[model_name] = {'error': str(e)}

    all_results[satellite] = sat_results

    # per-satellite JSON
    sat_json = sat_dir / f'{satellite}_results.json'
    save_json(sat_results, sat_json)
    print(f'\n  Results JSON: {sat_json.name}')

    # per-satellite comparison dashboard
    plot_comparison_dashboard(
        sat_results, satellite,
        out_path=sat_dir / f'{satellite}_comparison_dashboard.png',
    )

# ---- combined JSON -------------------------------------------------------
save_json(all_results, OUTPUT_PATH / 'all_results.json')

# ---- cross-satellite heatmap --------------------------------------------
plot_summary_heatmap(all_results,
                     out_path=OUTPUT_PATH / 'summary_heatmap.png')

# =========================================================================
# FINAL SUMMARY TABLE
# =========================================================================
print(f'\n{"="*68}')
print(f'  FINAL RESULTS  (target PICP >= {MU_COVERAGE*100:.0f}%)')
print(f'{"="*68}')
hdr = f'  {"Satellite":<12}  {"Model":<14}  {"PICP%":>7}  {"MPIW":>8}  {"CWC":>8}  {"IS":>9}  Status'
print(hdr)
print('  ' + '-'*(len(hdr)-2))

for sat, res in all_results.items():
    for mname, m in res.items():
        if 'error' in m:
            print(f'  {sat:<12}  {mname:<14}  ERROR: {m["error"][:30]}')
            continue
        ok = 'OK ' if m['PICP'] >= MU_COVERAGE else 'LOW'
        print(f'  {sat:<12}  {mname:<14}  '
              f'{m["PICP"]*100:>7.2f}  {m["MPIW"]:>8.3f}  '
              f'{m["CWC"]:>8.3f}  {m["IS"]:>9.3f}  {ok}')

print(f'\n  All outputs -> {OUTPUT_PATH}')
print('Done.')
