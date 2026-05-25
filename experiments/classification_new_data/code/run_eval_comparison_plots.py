"""
Evaluation Comparison Plots — All Methods (Phases 16-23)
=========================================================

Compares all PI methods on both sensors (EOS-04, Sentinel-1):
  Phase 16-20 : CQR-GBM      (GBM + CQR; Romano NeurIPS 2019)
  Phase 21    : Tube Loss (D) (direct bounds; Rana arXiv:2412.06853)
  Phase 21    : Tube Loss (C) (Tube Loss + CQR conformal extension)
  Phase 22    : MVE           (Mean Variance Estimation; Nix & Weigend 1994)
  Phase 22    : MDN           (Mixture Density Network; Bishop 1994)
  Phase 23    : CQR-ANN       (ANN quantile + CQR)
  Phase 23    : CQR-RF        (Quantile Forest + CQR; Meinshausen 2006)

Metrics:
  PICP  — Prediction Interval Coverage Probability
  MPIW  — Mean Prediction Interval Width
  CWC   — Coverage Width Criterion  (Khosravi 2011)
          CWC = PINAW × (1 + γ × exp(−η × (PICP − μ_c)))
          γ=0 if PICP ≥ μ_c, else 1;  μ_c=0.90;  η=50
          PINAW = MPIW / R  where R = y_test range
  IS    — Interval Score / Winkler Score (Winkler 1972)
          IS = MPIW + (2/α) × mean_miss_penalty

CWC for phases without stored CWC (CQR-GBM):
  Computed from stored PICP + MPIW + y_range (re-split data for R).
IS for CQR-GBM:
  Not stored per-sample → shown as N/A.

Run AFTER all phases (21, 22, 23) have completed.
"""

import os, json, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from eval_pi_metrics import cwc as _cwc

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / 'data'
OUT_DIR   = ROOT / 'output' / 'eval_comparison'
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
Y_COL       = 'SM1 (%)'
ALPHA       = 0.10

SENSORS_META = {
    'EOS-04': {
        'csv':    'eos-04-enhanced-ndvi.csv',
        'x_cols': ['HH-pol','HV-pol','cross_pol_ratio','month_sin','month_cos','crop_encoded','NDVI'],
    },
    'Sentinel-1': {
        'csv':    'sentinel-1-enhanced-ndvi.csv',
        'x_cols': ['VH-pol','VV-pol','cross_pol_ratio','month_sin','month_cos','crop_encoded','NDVI'],
    },
}

# Method registry: label → (eos04_json_path, sentinel_json_path, color, has_IS)
OUTPUT_ROOT = ROOT / 'output'
METHODS = [
    ('CQR-GBM',
     OUTPUT_ROOT/'gbm_finetune'/'eos04'/'best_config.json',
     OUTPUT_ROOT/'gbm_finetune'/'sentinel1'/'best_config.json',
     '#4C72B0', False),
    ('CQR-ANN',
     OUTPUT_ROOT/'cqr_ann_rf'/'eos04'/'best_cqr_ann.json',
     OUTPUT_ROOT/'cqr_ann_rf'/'sentinel1'/'best_cqr_ann.json',
     '#9467bd', True),
    ('CQR-RF',
     OUTPUT_ROOT/'cqr_ann_rf'/'eos04'/'best_cqr_rf.json',
     OUTPUT_ROOT/'cqr_ann_rf'/'sentinel1'/'best_cqr_rf.json',
     '#8c564b', True),
    ('Tube(D)',
     OUTPUT_ROOT/'tube_loss'/'eos04'/'best_config_direct.json',
     OUTPUT_ROOT/'tube_loss'/'sentinel1'/'best_config_direct.json',
     '#e377c2', True),
    ('Tube(C)',
     OUTPUT_ROOT/'tube_loss'/'eos04'/'best_config.json',
     OUTPUT_ROOT/'tube_loss'/'sentinel1'/'best_config.json',
     '#DD8452', True),
    ('MVE',
     OUTPUT_ROOT/'prob_nn'/'eos04'/'best_mve.json',
     OUTPUT_ROOT/'prob_nn'/'sentinel1'/'best_mve.json',
     '#55A868', True),
    ('MDN',
     OUTPUT_ROOT/'prob_nn'/'eos04'/'best_mdn.json',
     OUTPUT_ROOT/'prob_nn'/'sentinel1'/'best_mdn.json',
     '#d62728', True),
]


def split_70_10_10_10(X, y):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, train_size=0.7,  random_state=RANDOM_SEED)
    X_v,  X_t2,  y_v,  y_t2  = train_test_split(X_tmp, y_tmp, train_size=1/3, random_state=RANDOM_SEED)
    X_cal, X_te, y_cal, y_te = train_test_split(X_t2, y_t2, test_size=0.5, random_state=RANDOM_SEED)
    return X_tr, X_v, X_cal, X_te, y_tr, y_v, y_cal, y_te


def get_y_range(sensor_key):
    s   = SENSORS_META[sensor_key]
    df  = pd.read_csv(DATA_PATH / s['csv'])
    y_r = df[Y_COL].values
    mask = y_r != 50
    y   = y_r[mask]
    X   = df[s['x_cols']].values[mask]
    *_, y_te = split_70_10_10_10(X, y)
    return float(y_te.max() - y_te.min())


def load_result(jf, y_range, has_IS):
    """Load a JSON result file and extract PICP, MPIW, CWC, IS."""
    if not Path(jf).exists():
        return None
    r = json.load(open(jf))

    # Support multiple JSON layouts
    test = r.get('test', r)
    picp  = test.get('PICP', test.get('test_PICP'))
    mpiw_ = test.get('MPIW', test.get('test_MPIW'))
    if picp is None or mpiw_ is None:
        return None

    # CWC: use stored value if present; else compute
    if 'CWC' in test:
        cwc_val = test['CWC']
    else:
        pinaw   = mpiw_ / max(y_range, 1e-9)
        mu_c    = 1.0 - ALPHA
        gamma   = 0.0 if picp >= mu_c else 1.0
        cwc_val = pinaw * (1.0 + gamma * np.exp(-50.0 * (picp - mu_c)))

    is_val = test.get('IS', None) if has_IS else None
    return {'PICP': picp, 'MPIW': mpiw_, 'CWC': cwc_val, 'IS': is_val}


# ── Build data matrix ────────────────────────────────────────────────────────
print("Loading results...")
y_ranges = {k: get_y_range(k) for k in SENSORS_META}

# data[sensor][method_label] = {'PICP': ..., 'MPIW': ..., 'CWC': ..., 'IS': ...} | None
data = {s: {} for s in SENSORS_META}
for label, eos_jf, s1_jf, color, has_IS in METHODS:
    for sensor, jf in [('EOS-04', eos_jf), ('Sentinel-1', s1_jf)]:
        r = load_result(jf, y_ranges[sensor], has_IS)
        data[sensor][label] = r
        status = (f"PICP={r['PICP']:.4f} MPIW={r['MPIW']:.2f} "
                  f"CWC={r['CWC']:.4f} IS={r['IS']}"
                  if r else "not found")
        print(f"  {sensor:12s} {label:10s}: {status}")

method_labels = [m[0] for m in METHODS]
method_colors = {m[0]: m[3] for m in METHODS}


def bar_chart(metric, title, fname, is_lower_better):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    for ai, sensor in enumerate(SENSORS_META):
        ax   = axes[ai]
        vals = [(label, data[sensor].get(label)) for label in method_labels]
        xs   = np.arange(len(vals))

        for xi, (label, v) in enumerate(vals):
            if v is None or v.get(metric) is None:
                ax.bar(xi, 0.001, color='lightgrey', edgecolor='grey',
                       linewidth=1, hatch='///', label='_nolegend_' if xi > 0 else 'N/A')
            else:
                val = v[metric]
                ax.bar(xi, val, color=method_colors[label],
                       edgecolor='black', linewidth=0.7, alpha=0.85)
                ax.text(xi, val * 1.01, f'{val:.3f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

        ax.axhline(0, color='black', lw=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(method_labels, rotation=30, ha='right', fontsize=9)
        direction = '↓ lower=better' if is_lower_better else '↑ higher=better'
        ax.set_title(f'{sensor}  |  {metric}  ({direction})', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.grid(axis='y', alpha=0.3)

    # Legend
    patches = [mpatches.Patch(color=method_colors[lb], label=lb) for lb in method_labels]
    patches.append(mpatches.Patch(color='lightgrey', hatch='///', label='N/A'))
    fig.legend(handles=patches, loc='upper center', ncol=len(method_labels)+1,
               bbox_to_anchor=(0.5, 1.03), fontsize=9)
    fig.suptitle(title, fontsize=13, y=1.06, fontweight='bold')
    fig.tight_layout()
    fpath = OUT_DIR / fname
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


def scatter_picp_mpiw():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    marker_map = {'CQR-GBM': 'o', 'CQR-ANN': 's', 'CQR-RF': 'D',
                  'Tube(D)': '^', 'Tube(C)': 'v', 'MVE': 'P', 'MDN': '*'}
    for ai, sensor in enumerate(SENSORS_META):
        ax = axes[ai]
        for label in method_labels:
            r = data[sensor].get(label)
            if r is None: continue
            ax.scatter(r['PICP'] * 100, r['MPIW'],
                       color=method_colors[label],
                       marker=marker_map.get(label, 'o'), s=150,
                       edgecolors='black', linewidths=0.8, zorder=4, label=label)
            ax.annotate(f'  {label}', (r['PICP']*100, r['MPIW']), fontsize=8)
        ax.axvline(90, color='red', ls='--', lw=1.5, alpha=0.7, label='90% target')
        ax.set_xlabel('Test PICP (%)')
        ax.set_ylabel('Test MPIW')
        ax.set_title(f'{sensor} — PICP vs MPIW\n(ideal: right of dashed, low MPIW)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    fig.suptitle('All Methods: PICP vs MPIW Tradeoff', fontsize=13)
    fig.tight_layout()
    fpath = OUT_DIR / 'picp_vs_mpiw_scatter.png'
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved picp_vs_mpiw_scatter.png")


def summary_table():
    rows = []
    for sensor in SENSORS_META:
        for label in method_labels:
            r = data[sensor].get(label)
            rows.append({
                'Sensor': sensor, 'Method': label,
                'PICP': f"{r['PICP']*100:.2f}%" if r else '—',
                'MPIW': f"{r['MPIW']:.2f}"       if r else '—',
                'CWC':  f"{r['CWC']:.4f}"         if r else '—',
                'IS':   (f"{r['IS']:.2f}" if r and r['IS'] is not None else 'N/A') if r else '—',
                'Note': 'CWC exact,IS exact' if (r and r['IS'] is not None) else
                        'CWC approx,IS N/A' if r else 'pending',
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'summary_table.csv', index=False)

    fig, ax = plt.subplots(figsize=(14, max(4, len(rows)*0.45 + 1.5)))
    ax.axis('off')
    tbl = ax.table(cellText=df[['Sensor','Method','PICP','MPIW','CWC','IS']].values,
                   colLabels=['Sensor','Method','PICP','MPIW','CWC','IS'],
                   cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2d6a9f')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f0f4f8')
    fig.suptitle('PI Metric Summary — All Methods (Phases 16-23)', fontsize=12,
                 fontweight='bold', y=0.99)
    fig.tight_layout()
    fpath = OUT_DIR / 'summary_table.png'
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved summary_table.png + summary_table.csv")


print("\nGenerating comparison plots...")
bar_chart('CWC',  'CWC — Coverage Width Criterion (Khosravi 2011, lower=better)',
          'cwc_comparison.png', True)
bar_chart('IS',   'Interval Score — Winkler Score (Winkler 1972, lower=better)',
          'is_comparison.png', True)
bar_chart('PICP', 'PICP — Prediction Interval Coverage Probability (higher=better)',
          'picp_comparison.png', False)
bar_chart('MPIW', 'MPIW — Mean Prediction Interval Width (lower=better)',
          'mpiw_comparison.png', True)
scatter_picp_mpiw()
summary_table()

print(f"\n  All comparison plots → {OUT_DIR}")
print("Done.")
