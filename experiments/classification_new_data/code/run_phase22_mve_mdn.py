"""
Phase 22 — MVE and MDN Probabilistic Neural Networks
=====================================================

Two complementary probabilistic NN approaches for PI estimation:

─────────────────────────────────────────────────────────────
A) MVE — Mean Variance Estimation (Nix & Weigend 1994)
─────────────────────────────────────────────────────────────
  Single ANN outputs (μ(x), log σ²(x)) — assumes p(y|x) is Gaussian.
  Loss: Gaussian NLL = 0.5·log(σ²) + 0.5·(y−μ)²/σ²
  PI:   [μ − z_{α/2}·σ,  μ + z_{α/2}·σ]
        For α=0.10: z = norm.ppf(0.95) ≈ 1.645

  Premortem:
    - SM1 may be non-Gaussian → PI over/under-covers
    - log_var can go very negative → σ→0 → NaN loss
    - Must clamp log_var: clip to [-10, 10]

─────────────────────────────────────────────────────────────
B) MDN — Mixture Density Network (Bishop 1994)
    github.com/dusenberrymw/mixture-density-networks
─────────────────────────────────────────────────────────────
  Single ANN outputs K-component Gaussian mixture:
    p(y|x) = Σ_k π_k(x) · N(y; μ_k(x), σ_k²(x))
  Loss: −log Σ_k π_k · N(y; μ_k, σ_k)  (NLL of mixture)
  PI:   [Q_{α/2}, Q_{1−α/2}] extracted by Monte Carlo sampling (N=5000)

  Premortem:
    - K too large → some components collapse (σ→0)
    - SM1 may be unimodal → K=1 dominates (MDN ≈ MVE)
    - σ_k guarded: softplus + epsilon = 1e-4
    - Gradient clipping (clipnorm=1.0) prevents NaN

EVALUATION METRICS
  PICP, MPIW, CWC (Khosravi 2011), IS (Winkler 1972)
"""

import os, json, random, warnings, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np; np.random.seed(42)
from scipy.stats import norm as sp_norm
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
tf.random.set_seed(42)
from tensorflow import keras

from eval_pi_metrics import all_metrics

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / 'data'
OUT_BASE  = ROOT / 'output' / 'prob_nn'
OUT_BASE.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
Y_COL       = 'SM1 (%)'
ALPHA       = 0.10
SIGMA_EPS   = 1e-4
N_MC        = 5000
Z_ALPHA     = float(sp_norm.ppf(1.0 - ALPHA / 2))   # 1.645 for 90% PI

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

MVE_GRID = [
    {'arch': [64, 32],  'lr': 1e-3},
    {'arch': [128, 64], 'lr': 1e-3},
    {'arch': [64, 32],  'lr': 5e-4},
    {'arch': [128, 64], 'lr': 5e-4},
]
MDN_GRID = [
    {'K': 3, 'arch': [64, 32],   'lr': 1e-3},
    {'K': 5, 'arch': [64, 32],   'lr': 1e-3},
    {'K': 3, 'arch': [128, 64],  'lr': 1e-3},
    {'K': 5, 'arch': [128, 64],  'lr': 1e-3},
    {'K': 3, 'arch': [64, 32],   'lr': 5e-4},
    {'K': 5, 'arch': [64, 32],   'lr': 5e-4},
    {'K': 3, 'arch': [128, 64],  'lr': 5e-4},
    {'K': 5, 'arch': [128, 64],  'lr': 5e-4},
]
EPOCHS   = 500
PATIENCE = 40
BATCH    = 32


def split_70_10_10_10(X, y):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, train_size=0.7,  random_state=RANDOM_SEED)
    X_v,  X_t2,  y_v,  y_t2  = train_test_split(X_tmp, y_tmp, train_size=1/3, random_state=RANDOM_SEED)
    X_cal, X_te, y_cal, y_te = train_test_split(X_t2, y_t2, test_size=0.5, random_state=RANDOM_SEED)
    return X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te


# ─── MVE ───────────────────────────────────────────────────────────────────
def mve_nll_loss(y_true, y_pred):
    """Gaussian NLL: 0.5*log_var + 0.5*(y-mu)^2/exp(log_var)."""
    mu      = y_pred[:, 0]
    log_var = tf.clip_by_value(y_pred[:, 1], -10.0, 10.0)
    var     = tf.exp(log_var) + 1e-6
    return tf.reduce_mean(0.5 * log_var + 0.5 * tf.square(y_true - mu) / var)


def build_mve(n_feat, arch, lr):
    keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)
    inp = keras.Input(shape=(n_feat,))
    x   = inp
    for units in arch:
        x = keras.layers.Dense(units, activation='relu')(x)
    out = keras.layers.Dense(2, activation='linear')(x)   # [mu, log_var]
    mdl = keras.Model(inp, out)
    mdl.compile(optimizer=keras.optimizers.Adam(lr), loss=mve_nll_loss)
    return mdl


def mve_predict_pi(model, X_s, sc_y, alpha=ALPHA):
    pred    = model.predict(X_s, verbose=0)
    mu_s    = pred[:, 0]
    log_var = np.clip(pred[:, 1], -10, 10)
    sigma_s = np.sqrt(np.exp(log_var) + 1e-6)
    z       = float(sp_norm.ppf(1.0 - alpha / 2))
    lo_s    = mu_s - z * sigma_s
    hi_s    = mu_s + z * sigma_s
    lo = sc_y.inverse_transform(lo_s.reshape(-1,1)).ravel()
    hi = sc_y.inverse_transform(hi_s.reshape(-1,1)).ravel()
    return lo, hi


# ─── MDN ───────────────────────────────────────────────────────────────────
def make_mdn_loss(K):
    LOG2PI = tf.constant(np.log(2 * np.pi), dtype=tf.float32)

    def _loss(y_true, y_pred):
        y     = tf.cast(tf.reshape(y_true, [-1, 1]), tf.float32)
        pi    = y_pred[:, :K]
        mu    = y_pred[:, K:2*K]
        sigma = y_pred[:, 2*K:] + SIGMA_EPS
        log_gauss = (-0.5 * tf.square((y - mu) / sigma)
                     - tf.math.log(sigma) - 0.5 * LOG2PI)
        log_pi    = tf.math.log(pi + 1e-10)
        return -tf.reduce_mean(tf.reduce_logsumexp(log_pi + log_gauss, axis=1))
    return _loss


def build_mdn(n_feat, K, arch, lr):
    keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)
    inp = keras.Input(shape=(n_feat,))
    x   = inp
    for units in arch:
        x = keras.layers.Dense(units, activation='relu')(x)
    pi_out    = keras.layers.Dense(K, activation='softmax', name='pi')(x)
    mu_out    = keras.layers.Dense(K, activation='linear',  name='mu')(x)
    sigma_out = keras.layers.Dense(K, activation='softplus', name='sigma')(x)
    out       = keras.layers.Concatenate()([pi_out, mu_out, sigma_out])
    mdl = keras.Model(inp, out)
    mdl.compile(optimizer=keras.optimizers.Adam(lr, clipnorm=1.0),
                loss=make_mdn_loss(K))
    return mdl


def mdn_predict_pi(model, X, K, sc_y, n_mc=N_MC, alpha=ALPHA, seed=42):
    rng  = np.random.default_rng(seed)
    pred = model.predict(X, verbose=0)
    pi   = pred[:, :K]
    mu   = pred[:, K:2*K]
    sig  = pred[:, 2*K:] + SIGMA_EPS
    n    = len(X)
    comp = np.vstack([rng.choice(K, size=n_mc, p=pi[i]) for i in range(n)])
    mu_s = mu[np.arange(n)[:, None], comp]
    sg_s = sig[np.arange(n)[:, None], comp]
    smp  = rng.normal(mu_s, sg_s)
    lo_s = np.quantile(smp, alpha / 2,       axis=1)
    hi_s = np.quantile(smp, 1.0 - alpha / 2, axis=1)
    lo = sc_y.inverse_transform(lo_s.reshape(-1,1)).ravel()
    hi = sc_y.inverse_transform(hi_s.reshape(-1,1)).ravel()
    return lo, hi


def pi_plot(sensor, label, y_te, lo, hi, m, method, out_dir):
    order   = np.argsort(y_te)
    y_s, lo_s, hi_s = y_te[order], lo[order], hi[order]
    cov     = (y_s >= lo_s) & (y_s <= hi_s)
    idx     = np.arange(len(y_s))
    color   = '#1f77b4' if method == 'mve' else '#d62728'

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(idx, lo_s, hi_s, alpha=0.25, color=color, label='90% PI')
    ax.plot(idx, lo_s, color=color, lw=1, ls='--', alpha=0.7)
    ax.plot(idx, hi_s, color=color, lw=1, ls='--', alpha=0.7)
    ax.scatter(idx[cov],  y_s[cov],  color='green', s=18, zorder=3, label='Covered')
    ax.scatter(idx[~cov], y_s[~cov], color='red',   s=28, zorder=4, marker='x', label='Missed')
    txt = (f"{method.upper()}  α=0.10\n"
           f"test PICP: {m['PICP']*100:.2f}%   MPIW: {m['MPIW']:.2f}\n"
           f"CWC: {m['CWC']:.4f}   IS: {m['IS']:.2f}")
    ax.text(0.01, 0.97, txt, transform=ax.transAxes, va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title(f"{sensor} — {method.upper()} | {label}", fontsize=11)
    ax.set_xlabel('Test sample (sorted by SM1)')
    ax.set_ylabel('SM1 (%)')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fpath = out_dir / f"best_{method}_{label.replace(' ','_')[:50]}.png"
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fpath


def is_better(m_new, m_old):
    if m_old is None: return True
    g_new = abs(m_new['PICP'] - 0.90)
    g_old = abs(m_old['PICP'] - 0.90)
    return g_new < g_old or (g_new == g_old and m_new['IS'] < m_old['IS'])


for sensor_name, s in SENSORS.items():
    print(f"\n{'='*65}")
    print(f"  {sensor_name}  — Phase 22  MVE + MDN")
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
    X_tr_s = sc_x.fit_transform(X_tr)
    X_v_s  = sc_x.transform(X_v)
    X_te_s = sc_x.transform(X_te)

    sc_y = MinMaxScaler()
    y_tr_s = sc_y.fit_transform(y_tr.reshape(-1,1)).ravel()
    y_v_s  = sc_y.transform(y_v.reshape(-1,1)).ravel()

    n_feat = X_tr_s.shape[1]
    es_cb  = keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                            restore_best_weights=True, verbose=0)

    # ── MVE ───────────────────────────────────────────────────────────────
    print(f"\n  [MVE — Mean Variance Estimation, Nix & Weigend 1994]")
    mve_rows, best_mve = [], {'m': None, 'lo': None, 'hi': None, 'cfg': None}

    for ci, cfg in enumerate(MVE_GRID):
        arch_str = f"[{','.join(str(u) for u in cfg['arch'])}]"
        label    = f"MVE_arch{arch_str}_lr{cfg['lr']}"
        print(f"  [{ci+1}/{len(MVE_GRID)}] {label}", end='  ', flush=True)

        model = build_mve(n_feat, cfg['arch'], cfg['lr'])
        model.fit(X_tr_s, y_tr_s, validation_data=(X_v_s, y_v_s),
                  epochs=EPOCHS, batch_size=BATCH, callbacks=[es_cb], verbose=0)

        lo, hi = mve_predict_pi(model, X_te_s, sc_y)
        m      = all_metrics(y_te, lo, hi)
        lo_v, hi_v = mve_predict_pi(model, X_v_s, sc_y)
        m_v    = all_metrics(y_v, lo_v, hi_v)

        print(f"PICP={m['PICP']:.4f} MPIW={m['MPIW']:.2f} CWC={m['CWC']:.4f} IS={m['IS']:.2f}")
        mve_rows.append({'label': label, **cfg, 'arch': arch_str,
                         'val_PICP': m_v['PICP'], 'val_MPIW': m_v['MPIW'],
                         'test_PICP': m['PICP'], 'test_MPIW': m['MPIW'],
                         'test_CWC': m['CWC'],   'test_IS': m['IS']})
        if is_better(m, best_mve['m']):
            best_mve.update({'m': m, 'lo': lo.copy(), 'hi': hi.copy(),
                             'cfg': {**cfg, 'arch': arch_str, 'label': label}})

    pd.DataFrame(mve_rows).to_csv(s['outdir'] / 'mve_grid_summary.csv', index=False)
    if best_mve['m']:
        pi_plot(sensor_name, best_mve['cfg']['label'], y_te,
                best_mve['lo'], best_mve['hi'], best_mve['m'], 'mve', s['outdir'])
        res = {'method': 'MVE', 'paper': 'Nix & Weigend 1994', 'alpha': ALPHA,
               'params': best_mve['cfg'], 'test': best_mve['m'],
               'y_range': round(float(y_te.max()-y_te.min()),4), 'n_test': int(len(y_te))}
        with open(s['outdir'] / 'best_mve.json', 'w') as f:
            json.dump(res, f, indent=4)
        print(f"\n  MVE best: PICP={best_mve['m']['PICP']:.4f} MPIW={best_mve['m']['MPIW']:.2f} "
              f"CWC={best_mve['m']['CWC']:.4f} IS={best_mve['m']['IS']:.2f}")

    # ── MDN ───────────────────────────────────────────────────────────────
    print(f"\n  [MDN — Mixture Density Network, Bishop 1994 / dusenberrymw/mixture-density-networks]")
    mdn_rows, best_mdn = [], {'m': None, 'lo': None, 'hi': None, 'cfg': None}

    for ci, cfg in enumerate(MDN_GRID):
        arch_str = f"[{','.join(str(u) for u in cfg['arch'])}]"
        label    = f"MDN_K{cfg['K']}_arch{arch_str}_lr{cfg['lr']}"
        print(f"  [{ci+1}/{len(MDN_GRID)}] {label}", end='  ', flush=True)

        model = build_mdn(n_feat, cfg['K'], cfg['arch'], cfg['lr'])
        try:
            hist = model.fit(X_tr_s, y_tr_s, validation_data=(X_v_s, y_v_s),
                             epochs=EPOCHS, batch_size=BATCH, callbacks=[es_cb], verbose=0)
            if np.isnan(hist.history['loss'][-1]):
                raise ValueError("NaN loss")
        except Exception as e:
            print(f"FAILED ({e})")
            mdn_rows.append({'label': label, 'K': cfg['K'], 'arch': arch_str,
                             'status': 'failed', 'test_PICP': np.nan})
            continue

        lo, hi = mdn_predict_pi(model, X_te_s, cfg['K'], sc_y)
        m      = all_metrics(y_te, lo, hi)
        lo_v, hi_v = mdn_predict_pi(model, X_v_s, cfg['K'], sc_y, seed=99)
        m_v    = all_metrics(y_v, lo_v, hi_v)

        print(f"PICP={m['PICP']:.4f} MPIW={m['MPIW']:.2f} CWC={m['CWC']:.4f} IS={m['IS']:.2f}")
        mdn_rows.append({'label': label, 'K': cfg['K'], 'arch': arch_str, 'lr': cfg['lr'],
                         'status': 'ok',
                         'val_PICP': m_v['PICP'], 'val_MPIW': m_v['MPIW'],
                         'test_PICP': m['PICP'],  'test_MPIW': m['MPIW'],
                         'test_CWC': m['CWC'],    'test_IS': m['IS']})
        if is_better(m, best_mdn['m']):
            best_mdn.update({'m': m, 'lo': lo.copy(), 'hi': hi.copy(),
                             'cfg': {**cfg, 'arch': arch_str, 'label': label}})

    pd.DataFrame(mdn_rows).to_csv(s['outdir'] / 'mdn_grid_summary.csv', index=False)
    if best_mdn['m']:
        pi_plot(sensor_name, best_mdn['cfg']['label'], y_te,
                best_mdn['lo'], best_mdn['hi'], best_mdn['m'], 'mdn', s['outdir'])
        res = {'method': 'MDN', 'K': best_mdn['cfg']['K'],
               'paper': 'Bishop 1994 PRML; github.com/dusenberrymw/mixture-density-networks',
               'alpha': ALPHA, 'params': best_mdn['cfg'], 'test': best_mdn['m'],
               'y_range': round(float(y_te.max()-y_te.min()),4), 'n_test': int(len(y_te))}
        with open(s['outdir'] / 'best_mdn.json', 'w') as f:
            json.dump(res, f, indent=4)
        print(f"\n  MDN best: PICP={best_mdn['m']['PICP']:.4f} MPIW={best_mdn['m']['MPIW']:.2f} "
              f"CWC={best_mdn['m']['CWC']:.4f} IS={best_mdn['m']['IS']:.2f}")

    print(f"\n  Saved → {s['outdir']}")

# ── Postmortem summary ──────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  PHASE 22 — MVE + MDN — POSTMORTEM SUMMARY")
print(f"{'='*65}")
print(f"  {'Sensor':12s}  {'Method':6s}  PICP    MPIW    CWC     IS")
print(f"  {'-'*55}")
for sensor_name, s in SENSORS.items():
    for fname, mname in [('best_mve.json', 'MVE'), ('best_mdn.json', 'MDN')]:
        jf = s['outdir'] / fname
        if jf.exists():
            r = json.load(open(jf))
            t = r['test']
            print(f"  {sensor_name:12s}  {mname:6s}  "
                  f"{t['PICP']:.4f}  {t['MPIW']:.2f}  {t['CWC']:.4f}  {t['IS']:.2f}")
print(f"\n  MVE: Gaussian PI, z=1.645 for 90%. MDN: K-mixture, Monte Carlo PI.")
print(f"  MVE ref: Nix & Weigend 1994.  MDN ref: Bishop 1994 PRML.")
print("Done.")
