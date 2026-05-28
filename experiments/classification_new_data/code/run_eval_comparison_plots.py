"""
Evaluation Comparison Plots — All Methods (Phases 16-23)
=========================================================

Compares all PI methods on both sensors (EOS-04, Sentinel-1).
Valid coverage window: PICP ∈ [90%, 95%]  (α=0.10 target ±5 pp tolerance)

Methods:
  Phase 16-20 : CQR-GBM      (GBM + CQR; Romano NeurIPS 2019)
  Phase 21    : Tube Loss (D) (direct bounds; Rana arXiv:2412.06853)
  Phase 21    : Tube Loss (C) (Tube Loss + CQR conformal extension)
  Phase 22    : MVE           (Mean Variance Estimation; Nix & Weigend 1994)
  Phase 22    : MDN           (Mixture Density Network; Bishop 1994)
  Phase 23    : CQR-ANN       (ANN quantile + CQR; Romano NeurIPS 2019)
  Phase 23    : CQR-RF        (Quantile Forest + CQR; Meinshausen 2006)

Metrics:
  PICP  — Prediction Interval Coverage Probability (target: 90-95%)
  MPIW  — Mean Prediction Interval Width (lower = better)
  CWC   — Coverage Width Criterion (Khosravi 2011, lower = better)
          CWC = PINAW × (1 + γ × exp(−50 × (PICP − 0.90)))
          γ=0 if PICP ≥ 0.90, else 1;  PINAW = MPIW / R
  IS    — Interval Score / Winkler Score (Winkler 1972, lower = better)
          IS = MPIW + (2/α) × mean_miss_penalty

Valid window [90%, 95%]:
  Methods outside this range are shown greyed/hatched.
  Best in-range method per metric is annotated with ★.
"""

import os, json, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / 'data'
OUT_DIR   = ROOT / 'output' / 'eval_comparison'
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED  = 42
Y_COL        = 'SM1 (%)'
ALPHA        = 0.10
PICP_LO      = 0.90   # valid coverage window lower bound
PICP_HI      = 0.95   # valid coverage window upper bound

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

OUTPUT_ROOT = ROOT / 'output'
METHODS = [
    # (label, eos04_json, sentinel_json, color, has_IS)
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

# ── New 27-May models: loaded from flat per-sensor JSON ───────────────────────
# Format: {"QR-NN": {PICP, MPIW, CWC, IS}, "CQR_tube": {...}}
METHODS_27MAY = [
    # (label_in_json,  display_label,  color)
    ('QR-NN',    'QR-NN',     '#E91E63'),   # pink
    ('CQR_tube', 'CQR-Tube',  '#00897B'),   # teal
]
JSON_27MAY = {
    'EOS-04':     OUTPUT_ROOT / '27_5_2026_output' / 'EOS_04'     / 'EOS-04_results.json',
    'Sentinel-1': OUTPUT_ROOT / '27_5_2026_output' / 'Sentinel_1' / 'Sentinel-1_results.json',
}


def split_70_10_10_10(X, y):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, train_size=0.7, random_state=RANDOM_SEED)
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
    if not Path(jf).exists():
        return None
    r = json.load(open(jf))
    test  = r.get('test', r)
    picp  = test.get('PICP', test.get('test_PICP'))
    mpiw_ = test.get('MPIW', test.get('test_MPIW'))
    if picp is None or mpiw_ is None:
        return None
    if 'CWC' in test:
        cwc_val = test['CWC']
    else:
        pinaw   = mpiw_ / max(y_range, 1e-9)
        mu_c    = 1.0 - ALPHA
        gamma   = 0.0 if picp >= mu_c else 1.0
        cwc_val = pinaw * (1.0 + gamma * np.exp(-50.0 * (picp - mu_c)))
    is_val = test.get('IS', None) if has_IS else None
    return {'PICP': picp, 'MPIW': mpiw_, 'CWC': cwc_val, 'IS': is_val}


def in_range(picp):
    return PICP_LO <= picp <= PICP_HI


# ── Load phase 16-23 results ──────────────────────────────────────────────────
print("Loading results...")
y_ranges = {k: get_y_range(k) for k in SENSORS_META}

data = {s: {} for s in SENSORS_META}
for label, eos_jf, s1_jf, color, has_IS in METHODS:
    for sensor, jf in [('EOS-04', eos_jf), ('Sentinel-1', s1_jf)]:
        r = load_result(jf, y_ranges[sensor], has_IS)
        data[sensor][label] = r
        valid_tag = 'OK' if (r and in_range(r['PICP'])) else 'LOW'
        status = (f"PICP={r['PICP']:.4f} MPIW={r['MPIW']:.2f} "
                  f"CWC={r['CWC']:.4f} IS={r['IS']}  [{valid_tag}]"
                  if r else "not found")
        print(f"  {sensor:12s} {label:10s}: {status}")

# ── Load 27-May models (QR-NN, CQR-Tube) from flat JSON ──────────────────────
print("\nLoading 27-May-2026 models (QR-NN, CQR-Tube)...")
for sensor, jf in JSON_27MAY.items():
    if not Path(jf).exists():
        print(f"  WARNING: {jf} not found — skipping")
        continue
    raw = json.load(open(jf))
    for json_key, display_label, _ in METHODS_27MAY:
        m = raw.get(json_key)
        if m is None:
            print(f"  WARNING: key '{json_key}' missing in {jf}")
            continue
        data[sensor][display_label] = {
            'PICP': m['PICP'],
            'MPIW': m['MPIW'],
            'CWC':  m['CWC'],    # already PINAW-normalised
            'IS':   m.get('IS'),
        }
        valid_tag = 'OK' if in_range(m['PICP']) else 'LOW'
        print(f"  {sensor:12s} {display_label:10s}: "
              f"PICP={m['PICP']:.4f} MPIW={m['MPIW']:.2f} "
              f"CWC={m['CWC']:.4f} IS={m.get('IS')}  [{valid_tag}]")

method_labels = [m[0] for m in METHODS] + [m[1] for m in METHODS_27MAY]
method_colors = {m[0]: m[3] for m in METHODS}
method_colors.update({m[1]: m[2] for m in METHODS_27MAY})


def _find_best_in_range(sensor, metric, lower_is_better=True):
    """Return (label, value) of best in-range method for a metric."""
    candidates = [
        (lb, data[sensor][lb][metric])
        for lb in method_labels
        if data[sensor].get(lb) and data[sensor][lb].get(metric) is not None
        and in_range(data[sensor][lb]['PICP'])
    ]
    if not candidates:
        return None, None
    return min(candidates, key=lambda x: x[1]) if lower_is_better \
           else max(candidates, key=lambda x: x[1])


# ── 1. Bar charts with valid-range shading ────────────────────────────────────
def bar_chart(metric, title, fname, is_lower_better):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)
    for ai, sensor in enumerate(SENSORS_META):
        ax   = axes[ai]
        vals = [(lb, data[sensor].get(lb)) for lb in method_labels]
        xs   = np.arange(len(vals))
        best_label, _ = _find_best_in_range(sensor, metric, is_lower_better)

        for xi, (lb, v) in enumerate(vals):
            if v is None or v.get(metric) is None:
                ax.bar(xi, 0.001, color='#cccccc', edgecolor='grey',
                       linewidth=1, hatch='///', zorder=2)
                ax.text(xi, 0.005, 'N/A', ha='center', va='bottom',
                        fontsize=7, color='grey', style='italic')
                continue

            val      = v[metric]
            valid    = in_range(v['PICP'])
            color    = method_colors[lb]
            alpha_v  = 0.88 if valid else 0.35
            edge_col = 'black' if valid else '#888888'
            edge_lw  = 1.2 if valid else 0.7

            bar = ax.bar(xi, val, color=color, alpha=alpha_v,
                         edgecolor=edge_col, linewidth=edge_lw, zorder=2)

            # Cross-hatch invalid bars
            if not valid:
                ax.bar(xi, val, fill=False, edgecolor='#555555',
                       linewidth=0.5, hatch='xxx', zorder=3)

            # Value label
            txt_color = 'black' if valid else '#666666'
            ax.text(xi, val * 1.012, f'{val:.3f}', ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold' if valid else 'normal',
                    color=txt_color)

            # Star annotation for best in-range
            if lb == best_label:
                ax.text(xi, val * 1.07, '★ BEST', ha='center', va='bottom',
                        fontsize=7.5, color='#c8880a', fontweight='bold')

        # 90-95% validity label below x-axis (no axis band for non-PICP charts)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(method_labels, rotation=30, ha='right', fontsize=9)

        direction = '↓ lower=better' if is_lower_better else '↑ higher=better'
        valid_str = 'bright = PICP 90-95% ✓ | faded = outside range ✗'
        ax.set_title(f'{sensor}  |  {metric}  ({direction})\n{valid_str}',
                     fontsize=10)
        ax.set_ylabel(metric, fontsize=11)
        ax.grid(axis='y', alpha=0.3, zorder=0)

    # Legend
    patches = [mpatches.Patch(color=method_colors[lb], label=lb, alpha=0.85)
               for lb in method_labels]
    patches.append(mpatches.Patch(color='#cccccc', hatch='xxx',
                                  label='Outside 90-95% PICP'))
    patches.append(mpatches.Patch(color='#cccccc', hatch='///', label='N/A'))
    fig.legend(handles=patches, loc='upper center',
               ncol=min(5, len(patches)), bbox_to_anchor=(0.5, 1.03), fontsize=8.5)
    fig.suptitle(title, fontsize=12, y=1.07, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


# ── 2. PICP bar chart with 90-95% band ────────────────────────────────────────
def picp_bar_chart():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)
    for ai, sensor in enumerate(SENSORS_META):
        ax   = axes[ai]
        vals = [(lb, data[sensor].get(lb)) for lb in method_labels]
        xs   = np.arange(len(vals))

        # Green shaded valid band
        ax.axhspan(PICP_LO * 100, PICP_HI * 100,
                   color='#c8f5c8', alpha=0.45, zorder=0, label='Valid 90-95%')
        ax.axhline(PICP_LO * 100, color='#2ca02c', lw=1.5, ls='--', zorder=1,
                   label='90% lower bound')
        ax.axhline(PICP_HI * 100, color='#ff7f0e', lw=1.5, ls='--', zorder=1,
                   label='95% upper bound')

        best_label, _ = _find_best_in_range(sensor, 'CWC', lower_is_better=True)

        for xi, (lb, v) in enumerate(vals):
            if v is None:
                ax.bar(xi, 0, color='#cccccc', edgecolor='grey',
                       linewidth=1, hatch='///', zorder=2, height=85)
                continue
            pct   = v['PICP'] * 100
            valid = in_range(v['PICP'])
            color = method_colors[lb]
            alpha_v = 0.88 if valid else 0.35
            ax.bar(xi, pct, color=color, alpha=alpha_v,
                   edgecolor='black' if valid else '#888888',
                   linewidth=1.2 if valid else 0.7, zorder=2, bottom=0)
            if not valid:
                ax.bar(xi, pct, fill=False, edgecolor='#555555',
                       linewidth=0.5, hatch='xxx', zorder=3, bottom=0)
            # Value label
            offset = 0.3
            ax.text(xi, pct + offset, f'{pct:.1f}%',
                    ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold' if valid else 'normal',
                    color='black' if valid else '#666666')
            if lb == best_label and valid:
                ax.text(xi, pct + 1.8, '★ Best CWC', ha='center', va='bottom',
                        fontsize=7, color='#c8880a', fontweight='bold')

        ax.set_ylim(75, 102)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
        ax.set_xticks(xs)
        ax.set_xticklabels(method_labels, rotation=30, ha='right', fontsize=9)
        ax.set_title(f'{sensor}  |  PICP  (target: green band 90-95%)', fontsize=10)
        ax.set_ylabel('PICP (%)', fontsize=11)
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.legend(fontsize=8, loc='lower right')

    patches = [mpatches.Patch(color=method_colors[lb], label=lb, alpha=0.85)
               for lb in method_labels]
    patches.append(mpatches.Patch(color='#cccccc', hatch='xxx',
                                  label='Outside 90-95% PICP'))
    fig.legend(handles=patches, loc='upper center',
               ncol=min(5, len(patches)), bbox_to_anchor=(0.5, 1.03), fontsize=8.5)
    fig.suptitle('PICP — Prediction Interval Coverage Probability\n'
                 'Green band = valid 90-95% window', fontsize=12, y=1.07,
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'picp_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved picp_comparison.png")


# ── 3. Focused "valid-range only" comparison ─────────────────────────────────
def valid_range_chart():
    """Bar chart showing ONLY methods with PICP in [90%, 95%], ranked by CWC."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    for ai, sensor in enumerate(SENSORS_META):
        ax = axes[ai]
        valid_methods = [
            lb for lb in method_labels
            if data[sensor].get(lb) and in_range(data[sensor][lb]['PICP'])
        ]
        if not valid_methods:
            ax.text(0.5, 0.5, 'No methods in\n90-95% PICP range',
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title(f'{sensor}', fontsize=11)
            continue

        # Sort by CWC ascending
        valid_methods.sort(key=lambda lb: data[sensor][lb]['CWC'])
        xs = np.arange(len(valid_methods))

        for xi, lb in enumerate(valid_methods):
            v    = data[sensor][lb]
            cwc  = v['CWC']
            is_  = v['IS'] if v['IS'] is not None else 0
            mpiw = v['MPIW']
            picp = v['PICP'] * 100

            color = method_colors[lb]
            ax.bar(xi - 0.25, cwc,  width=0.22, color=color, alpha=0.90,
                   edgecolor='black', linewidth=0.8, label='_CWC' if xi else 'CWC')
            ax.bar(xi,        mpiw / 10, width=0.22, color=color, alpha=0.55,
                   edgecolor='black', linewidth=0.8, hatch='...',
                   label='_MPIW/10' if xi else 'MPIW÷10')
            if v['IS'] is not None:
                ax.bar(xi + 0.25, is_ / 10, width=0.22, color=color, alpha=0.35,
                       edgecolor='black', linewidth=0.8, hatch='////',
                       label='_IS/10' if xi else 'IS÷10')

            # Rank badge
            rank_color = ['#FFD700', '#C0C0C0', '#CD7F32'] + ['#aaaaaa'] * 10
            ax.text(xi, -0.07, f'#{xi+1}\n{picp:.1f}%',
                    ha='center', va='top', fontsize=7.5,
                    color=rank_color[xi], fontweight='bold',
                    transform=ax.get_xaxis_transform())

            ax.text(xi - 0.25, cwc + 0.01, f'{cwc:.3f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
            ax.text(xi, mpiw / 10 + 0.01, f'{mpiw:.1f}',
                    ha='center', va='bottom', fontsize=7, color='#444444')
            if v['IS'] is not None:
                ax.text(xi + 0.25, is_ / 10 + 0.01, f'{is_:.1f}',
                        ha='center', va='bottom', fontsize=7, color='#666666')

        ax.set_xticks(xs)
        ax.set_xticklabels(valid_methods, rotation=20, ha='right', fontsize=9)
        ax.set_title(f'{sensor}  |  Valid PICP 90-95% — sorted by CWC ↑best',
                     fontsize=10)
        ax.set_ylabel('Metric value', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)

        # Custom legend
        from matplotlib.patches import Patch
        legend_els = [
            Patch(facecolor='grey', alpha=0.9,  label='CWC (raw scale)'),
            Patch(facecolor='grey', alpha=0.55, hatch='...', label='MPIW÷10'),
            Patch(facecolor='grey', alpha=0.35, hatch='////', label='IS÷10'),
        ]
        ax.legend(handles=legend_els, fontsize=8, loc='upper right')

    fig.suptitle('Best Methods — PICP Valid Range 90-95%\n'
                 'Ranked by CWC  (★ #1 = lowest CWC)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUT_DIR / 'valid_range_ranking.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved valid_range_ranking.png")


# ── 4. PICP vs MPIW scatter with valid zone ───────────────────────────────────
def scatter_picp_mpiw():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    marker_map = {'CQR-GBM': 'o', 'CQR-ANN': 's', 'CQR-RF': 'D',
                  'Tube(D)': '^', 'Tube(C)': 'v', 'MVE': 'P', 'MDN': '*'}
    for ai, sensor in enumerate(SENSORS_META):
        ax = axes[ai]

        # Collect y range for ideal zone upper bound
        mpiw_vals = [data[sensor][lb]['MPIW'] for lb in method_labels
                     if data[sensor].get(lb) and in_range(data[sensor][lb]['PICP'])]
        zone_top = max(mpiw_vals) * 1.15 if mpiw_vals else 50

        # Valid zone shading
        ax.axvspan(PICP_LO * 100, PICP_HI * 100, ymin=0, ymax=1,
                   color='#c8f5c8', alpha=0.35, zorder=0, label='Valid 90-95%')
        ax.axvline(PICP_LO * 100, color='#2ca02c', lw=1.5, ls='--', zorder=1)
        ax.axvline(PICP_HI * 100, color='#ff7f0e', lw=1.5, ls='--', zorder=1)

        for lb in method_labels:
            r = data[sensor].get(lb)
            if r is None:
                continue
            pct   = r['PICP'] * 100
            valid = in_range(r['PICP'])
            alpha_v = 0.95 if valid else 0.35
            size    = 220 if valid else 110
            ec      = 'black' if valid else '#aaaaaa'
            ax.scatter(pct, r['MPIW'],
                       color=method_colors[lb], marker=marker_map.get(lb, 'o'),
                       s=size, alpha=alpha_v, edgecolors=ec, linewidths=1.0,
                       zorder=4 if valid else 3, label=lb)
            offset_x = 0.25
            offset_y = 0.4 if valid else 0.2
            ax.annotate(f'  {lb}', (pct, r['MPIW']),
                        fontsize=8 if valid else 7,
                        color='black' if valid else '#888888',
                        fontweight='bold' if valid else 'normal')

        # Best CWC in range
        best_lb, _ = _find_best_in_range(sensor, 'CWC')
        if best_lb:
            r = data[sensor][best_lb]
            ax.annotate('★ Best CWC',
                        xy=(r['PICP']*100, r['MPIW']),
                        xytext=(r['PICP']*100 - 2.5, r['MPIW'] + 2.5),
                        arrowprops=dict(arrowstyle='->', color='#c8880a', lw=1.5),
                        fontsize=8.5, color='#c8880a', fontweight='bold')

        ax.set_xlabel('Test PICP (%)', fontsize=11)
        ax.set_ylabel('Test MPIW', fontsize=11)
        ax.set_title(f'{sensor} — PICP vs MPIW\n'
                     f'Green band = valid 90-95% | ideal: in band + low MPIW',
                     fontsize=10)
        ax.grid(True, alpha=0.3)

        # Legend for zone lines
        from matplotlib.lines import Line2D
        extra = [Line2D([0], [0], color='#2ca02c', ls='--', lw=1.5, label='90% lower'),
                 Line2D([0], [0], color='#ff7f0e', ls='--', lw=1.5, label='95% upper'),
                 mpatches.Patch(color='#c8f5c8', alpha=0.6, label='Valid zone')]
        ax.legend(handles=extra, fontsize=8, loc='upper right')

    patches = [mpatches.Patch(color=method_colors[lb], alpha=0.85, label=lb)
               for lb in method_labels]
    fig.legend(handles=patches, loc='upper center',
               ncol=len(patches), bbox_to_anchor=(0.5, 1.03), fontsize=9)
    fig.suptitle('All Methods: PICP vs MPIW Tradeoff\n'
                 'Faded = outside 90-95% valid window',
                 fontsize=12, y=1.07, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'picp_vs_mpiw_scatter.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved picp_vs_mpiw_scatter.png")


# ── 5. Summary table with valid-range highlights ──────────────────────────────
def summary_table():
    rows = []
    for sensor in SENSORS_META:
        valid_in_sensor = {
            m: data[sensor][m]
            for m in method_labels
            if data[sensor].get(m) and in_range(data[sensor][m]['PICP'])
        }
        best_cwc_lb  = min(valid_in_sensor, key=lambda m: valid_in_sensor[m]['CWC'])  \
                       if valid_in_sensor else None
        best_is_lb   = min((m for m in valid_in_sensor
                            if valid_in_sensor[m]['IS'] is not None),
                           key=lambda m: valid_in_sensor[m]['IS'],
                           default=None)
        best_mpiw_lb = min(valid_in_sensor, key=lambda m: valid_in_sensor[m]['MPIW']) \
                       if valid_in_sensor else None

        for lb in method_labels:
            r     = data[sensor].get(lb)
            valid = r and in_range(r['PICP'])
            tags  = []
            if r and lb == best_cwc_lb:  tags.append('★CWC')
            if r and lb == best_is_lb:   tags.append('★IS')
            if r and lb == best_mpiw_lb: tags.append('★MPIW')
            rows.append({
                'Sensor':   sensor,
                'Method':   lb,
                'PICP':     f"{r['PICP']*100:.2f}%" if r else '—',
                'In Range': '✓' if valid else ('✗' if r else '—'),
                'MPIW':     f"{r['MPIW']:.2f}"      if r else '—',
                'CWC':      f"{r['CWC']:.4f}"        if r else '—',
                'IS':       (f"{r['IS']:.2f}" if r and r['IS'] is not None else 'N/A') if r else '—',
                'Best':     '  '.join(tags) if tags else '',
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'summary_table.csv', index=False)

    cols = ['Sensor', 'Method', 'PICP', 'In Range', 'MPIW', 'CWC', 'IS', 'Best']
    fig, ax = plt.subplots(figsize=(16, max(5, len(rows) * 0.52 + 1.8)))
    ax.axis('off')
    tbl = ax.table(cellText=df[cols].values,
                   colLabels=cols,
                   cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    n_sensors = len(SENSORS_META)
    n_methods = len(method_labels)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1a4f7a')
            cell.set_text_props(color='white', fontweight='bold')
            continue
        data_row = row - 1
        label    = df.iloc[data_row]['Method']
        sensor_v = df.iloc[data_row]['Sensor']
        r        = data[sensor_v].get(label)
        valid    = r and in_range(r['PICP'])
        best_tag = df.iloc[data_row]['Best']

        if valid and best_tag:
            cell.set_facecolor('#d4edda')        # green — best in range
            cell.set_text_props(fontweight='bold')
        elif valid:
            cell.set_facecolor('#eaf4ea')        # light green — valid
        elif row % 2 == 0:
            cell.set_facecolor('#f5f5f5')        # light grey — invalid alternating
        else:
            cell.set_facecolor('#eeeeee')

        if not valid and r:
            cell.set_text_props(color='#888888')

    fig.suptitle(
        'PI Metric Summary — All Methods (Phases 16-23)\n'
        '✓ Green = PICP in valid 90-95% window  |  ★ = Best in-range per metric',
        fontsize=11, fontweight='bold', y=0.99)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'summary_table.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved summary_table.png + summary_table.csv")

    # Print console summary
    print("\n  -- Best in 90-95% PICP range --")
    for sensor in SENSORS_META:
        valid_m = {m: data[sensor][m] for m in method_labels
                   if data[sensor].get(m) and in_range(data[sensor][m]['PICP'])}
        if not valid_m:
            print(f"  {sensor}: no methods in range")
            continue
        best_cwc  = min(valid_m, key=lambda m: valid_m[m]['CWC'])
        best_is   = min((m for m in valid_m if valid_m[m]['IS'] is not None),
                        key=lambda m: valid_m[m]['IS'], default='N/A')
        best_mpiw = min(valid_m, key=lambda m: valid_m[m]['MPIW'])
        print(f"  {sensor}:")
        print(f"    Best CWC  -> {best_cwc:10s}  CWC={valid_m[best_cwc]['CWC']:.4f}"
              f"  PICP={valid_m[best_cwc]['PICP']*100:.1f}%"
              f"  MPIW={valid_m[best_cwc]['MPIW']:.2f}")
        if best_is != 'N/A':
            print(f"    Best IS   -> {best_is:10s}  IS={valid_m[best_is]['IS']:.2f}"
                  f"   PICP={valid_m[best_is]['PICP']*100:.1f}%"
                  f"  MPIW={valid_m[best_is]['MPIW']:.2f}")
        print(f"    Best MPIW -> {best_mpiw:10s}  MPIW={valid_m[best_mpiw]['MPIW']:.2f}"
              f"  PICP={valid_m[best_mpiw]['PICP']*100:.1f}%")


# -- Run all plots ------------------------------------------------------------───
print("\nGenerating comparison plots...")
bar_chart('CWC',  'CWC — Coverage Width Criterion (Khosravi 2011, lower=better)',
          'cwc_comparison.png', True)
bar_chart('IS',   'Interval Score — Winkler Score (Winkler 1972, lower=better)',
          'is_comparison.png', True)
bar_chart('MPIW', 'MPIW — Mean Prediction Interval Width (lower=better)',
          'mpiw_comparison.png', True)
picp_bar_chart()
valid_range_chart()
scatter_picp_mpiw()
summary_table()

print(f"\n  All comparison plots -> {OUT_DIR}")
print("Done.")
