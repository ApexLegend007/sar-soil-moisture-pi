# -*- coding: utf-8 -*-
"""
draw_ann_architecture.py
========================
Publication-quality ANN architecture diagrams based on actual experiment data.

ARCHITECTURE (from run_27_5_2026.py  build_dual_pinball_ann):
  Input    : 7 features  (sensor-specific)
  Hidden 1 : 128 neurons   ReLU -> BatchNorm -> Dropout(0.15)
  Hidden 2 :  64 neurons   ReLU -> BatchNorm -> Dropout(0.10)
  Hidden 3 :  32 neurons   ReLU
  Output   :   2 neurons   Linear -> [lo, hi] Prediction Interval

EOS-04 inputs  : HH-pol | HV-pol | cross_pol_ratio | month_sin | month_cos | crop_encoded | NDVI
Sentinel-1     : VH-pol | VV-pol | cross_pol_ratio | month_sin | month_cos | crop_encoded | NDVI

Data stats (actual):
  EOS-04    : 2528 samples, 27 crops, SM1 range 1.2-54.1 %
  Sentinel-1: 1816 samples, 20 crops, SM1 range 1.2-56.4 %

Run:  python draw_ann_architecture.py
Outputs -> experiments/classification_new_data/output/figures/
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch, Circle
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  — edit all values here
# ──────────────────────────────────────────────────────────────────────────────

# Network topology
EOS04_INPUTS = ['HH-pol', 'HV-pol', 'cross_pol_ratio',
                'month_sin', 'month_cos', 'crop_encoded', 'NDVI']
SEN1_INPUTS  = ['VH-pol', 'VV-pol', 'cross_pol_ratio',
                'month_sin', 'month_cos', 'crop_encoded', 'NDVI']

# (true_neuron_count, label, activation, batch_norm, dropout_rate)
HIDDEN = [
    (128, 'H1', 'ReLU', True,  0.15),
    ( 64, 'H2', 'ReLU', True,  0.10),
    ( 32, 'H3', 'ReLU', False, 0.00),
]
OUTPUT_N   = 2
OUTPUT_ACT = 'Linear'

MAX_DRAW = [6, 5, 4]    # max circles to draw per hidden layer

# X positions in inches from left (we use a 20×8 figure)
FIG_W = 20.0
FIG_H =  8.0
DPI   = 160

# Column x-positions (in axes fraction)
XC_IN_BOX  = 0.065
XC_IN_CIRC = 0.140
XC_H       = [0.285, 0.440, 0.570]
XC_OUT     = 0.690
XC_ACT     = 0.755
XC_SM      = 0.840

# Row: neurons span Y_TOP to Y_BOT (axes fraction)
Y_TOP = 0.88     # top of neuron column
Y_BOT = 0.32     # bottom — labels + sub-boxes go below this

# Box sizes (axes fraction)
IN_W  = 0.095;  IN_H  = 0.052
H_R   = 0.022   # hidden neuron circle radius
OR    = 0.026   # output neuron radius
SB_W  = 0.07;   SB_H  = 0.040   # sub-box (act/bn/drop) width/height

# Gap between sub-boxes (axes fraction)
SB_GAP = 0.006

# Colors
C_IN_FC  = '#D6EAF8'; C_IN_EC  = '#2471A3'
C_INn_FC = '#F1948A'; C_INn_EC = '#C0392B'   # input neurons
C_Hn_FC  = '#C39BD3'; C_Hn_EC  = '#6C3483'   # hidden neurons
C_On_FC  = '#76D7C4'; C_On_EC  = '#117A65'   # output neuron
C_RELU   = '#A9DFBF'; C_RELU_E = '#1E8449'
C_LIN    = '#F9E79F'; C_LIN_E  = '#B7950B'
C_BN     = '#AED6F1'; C_BN_E   = '#1A5276'
C_DR     = '#F5EEF8'; C_DR_E   = '#7D3C98'
C_SM_FC  = '#FDEBD0'; C_SM_EC  = '#CA6F1E'
C_CONN   = '#BBBBBB'
C_BIAS   = '#FDFEFE'; C_BIAS_E = '#AAAAAA'
C_ARROW  = '#444444'
C_WT_LBL = '#888888'

# Output dir
OUT_DIR = Path(__file__).resolve().parent.parent / 'output' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def col_ys(n, top=Y_TOP, bot=Y_BOT):
    """n y-positions evenly spaced from top to bot."""
    if n == 1:
        return [(top + bot) / 2]
    return [top - i * (top - bot) / (n - 1) for i in range(n)]


def add_circle(ax, cx, cy, r, fc, ec, lw=1.2, z=5):
    ax.add_patch(Circle((cx, cy), r, fc=fc, ec=ec, lw=lw,
                         transform=ax.transAxes, zorder=z, clip_on=False))


def add_rect(ax, cx, cy, w, h, fc, ec, lw=1.0, z=4, label='', fs=8, fw='normal', lc='black'):
    """Simple rectangle (no fancy padding) centred at (cx,cy)."""
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.005',
        fc=fc, ec=ec, lw=lw,
        transform=ax.transAxes, zorder=z, clip_on=False))
    if label:
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=fs, fontweight=fw, color=lc,
                transform=ax.transAxes, zorder=z+1, clip_on=False)


def add_arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=0.9, ms=8, z=3):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle='->',
        color=color, lw=lw, mutation_scale=ms,
        transform=ax.transAxes, zorder=z, clip_on=False))


def weight_label(ax, xmid, ytop, label='Weights'):
    ax.text(xmid, ytop + 0.03, label,
            ha='center', va='bottom', fontsize=7, color=C_WT_LBL,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='#DDDDDD', lw=0.5))


def draw_connections(ax, ys_src, ys_dst, x_src, x_dst):
    for y0 in ys_src:
        for y1 in ys_dst:
            ax.plot([x_src, x_dst], [y0, y1],
                    transform=ax.transAxes, color=C_CONN,
                    lw=0.28, alpha=0.5, zorder=1, clip_on=False)


def dashed_box(ax, x0, y0, x1, y1, ec='#888888', lw=0.8, ls='--'):
    ax.add_patch(plt.Polygon(
        [[x0,y0],[x1,y0],[x1,y1],[x0,y1]],
        fill=False, ec=ec, lw=lw, ls=ls,
        transform=ax.transAxes, zorder=0, clip_on=False))


# ──────────────────────────────────────────────────────────────────────────────
# Main diagram drawing function
# ──────────────────────────────────────────────────────────────────────────────

def draw(ax, inputs, sensor, info):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    N_IN = len(inputs)
    IN_YS = col_ys(N_IN)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(0.5, 0.99, f'ANN Architecture  —  {sensor}',
            ha='center', va='top', fontsize=11, fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.5, 0.95, info,
            ha='center', va='top', fontsize=7, color='#444',
            transform=ax.transAxes, clip_on=True)

    # ── Bias bar ──────────────────────────────────────────────────────────────
    BIAS_Y = 0.92
    bias_xs = XC_H + [XC_OUT]
    bar_x0  = min(bias_xs) - 0.03
    bar_x1  = max(bias_xs) + 0.03
    bar_cx  = (bar_x0 + bar_x1) / 2
    bar_w   = bar_x1 - bar_x0
    add_rect(ax, bar_cx, BIAS_Y, bar_w, 0.038, C_BIAS, C_BIAS_E,
             label='Biases', fs=9)

    for i, (bx, bl) in enumerate(zip(bias_xs, [f'b{j+1}' for j in range(len(bias_xs))])):
        y_top_neuron = Y_TOP + 0.03
        ax.plot([bx, bx], [BIAS_Y - 0.019, y_top_neuron + 0.005],
                transform=ax.transAxes, color=C_BIAS_E, lw=0.8, zorder=2, clip_on=False)
        add_arrow(ax, bx, y_top_neuron + 0.005, bx, y_top_neuron,
                  color=C_BIAS_E, lw=0.7, ms=6)
        ax.text(bx + 0.01, BIAS_Y - 0.04, bl,
                ha='left', va='top', fontsize=8, color='#666',
                transform=ax.transAxes, zorder=6)

    # ── Input feature boxes ───────────────────────────────────────────────────
    ax.text(XC_IN_BOX - 0.032, (Y_TOP + Y_BOT) / 2,
            'Input\nFeatures', ha='center', va='center', fontsize=7.5,
            fontweight='bold', color='#1A5276', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.2', fc='#EBF5FB', ec='#2E86C1', lw=0.8))

    for lbl, iy in zip(inputs, IN_YS):
        add_rect(ax, XC_IN_BOX, iy, IN_W, IN_H, C_IN_FC, C_IN_EC,
                 label=lbl, fs=8, fw='bold')
        add_circle(ax, XC_IN_CIRC, iy, H_R, C_INn_FC, C_INn_EC)
        add_arrow(ax, XC_IN_BOX + IN_W/2, iy, XC_IN_CIRC - H_R, iy,
                  color=C_IN_EC, lw=0.7, ms=6)

    ax.text(XC_IN_CIRC, Y_BOT - 0.05, 'Input Layer',
            ha='center', va='top', fontsize=7, color='#666', transform=ax.transAxes)

    dashed_box(ax, XC_IN_BOX - IN_W/2 - 0.01,  Y_BOT - 0.02,
               XC_IN_CIRC + H_R + 0.01, Y_TOP + 0.02, ec=C_IN_EC)

    # ── Hidden layers ─────────────────────────────────────────────────────────
    prev_ys = IN_YS
    prev_x  = XC_IN_CIRC

    for i, (n_true, hid_lbl, act_nm, has_bn, drop_r) in enumerate(HIDDEN):
        xh     = XC_H[i]
        n_draw = min(MAX_DRAW[i], n_true)
        h_ys   = col_ys(n_draw)

        # Weights label
        weight_label(ax, (prev_x + xh) / 2, Y_TOP)

        # Connections
        draw_connections(ax, prev_ys, h_ys, prev_x + H_R, xh - H_R)

        # Neurons
        for j, hy in enumerate(h_ys):
            add_circle(ax, xh, hy, H_R, C_Hn_FC, C_Hn_EC)
            txt = '...' if (j == n_draw - 1 and n_true > n_draw) else str(j + 1)
            ax.text(xh, hy, txt, ha='center', va='center', fontsize=6,
                    color='#4A235A', transform=ax.transAxes, zorder=7, clip_on=False)

        dashed_box(ax, xh - H_R - 0.012, min(h_ys) - H_R - 0.008,
                   xh + H_R + 0.012, max(h_ys) + H_R + 0.008, ec=C_Hn_EC)

        ax.text(xh, Y_BOT - 0.05, f'Hidden {i+1}\n({n_true} neurons)',
                ha='center', va='top', fontsize=7, color='#555', transform=ax.transAxes)

        # Sub-boxes: ReLU / BatchNorm / Dropout — stacked below neuron column
        sy = Y_BOT - 0.005 - SB_H / 2   # top sub-box centre

        # ReLU
        add_rect(ax, xh, sy, SB_W, SB_H, C_RELU, C_RELU_E,
                 label=act_nm, fs=7.5, fw='bold')
        add_arrow(ax, xh, min(h_ys) - H_R, xh, sy + SB_H/2,
                  color=C_RELU_E, lw=0.8, ms=7)
        sy -= SB_H + SB_GAP

        # BatchNorm
        if has_bn:
            add_rect(ax, xh, sy, SB_W, SB_H, C_BN, C_BN_E,
                     label='BatchNorm', fs=6.5)
            sy -= SB_H + SB_GAP

        # Dropout
        if drop_r > 0:
            add_rect(ax, xh, sy, SB_W + 0.01, SB_H, C_DR, C_DR_E,
                     label=f'Drop {drop_r}', fs=7)

        prev_ys = h_ys
        prev_x  = xh

    # ── Output neuron ─────────────────────────────────────────────────────────
    weight_label(ax, (prev_x + XC_OUT) / 2, Y_TOP)
    draw_connections(ax, prev_ys, [(Y_TOP + Y_BOT) / 2],
                     prev_x + H_R, XC_OUT - OR)

    oy = (Y_TOP + Y_BOT) / 2
    add_circle(ax, XC_OUT, oy, OR, C_On_FC, C_On_EC, lw=1.5)
    ax.text(XC_OUT, Y_BOT - 0.05, f'Output\n({OUTPUT_N} neurons)',
            ha='center', va='top', fontsize=7, color='#555', transform=ax.transAxes)

    # Horizontal flow: output neuron -> Linear label -> Soil Moisture box
    add_arrow(ax, XC_OUT + OR, oy, XC_ACT - SB_W/2, oy,
              color=C_LIN_E, lw=0.9, ms=8)
    add_rect(ax, XC_ACT, oy, SB_W, SB_H, C_LIN, C_LIN_E,
             label=OUTPUT_ACT, fs=7.5, fw='bold')
    add_arrow(ax, XC_ACT + SB_W/2, oy, XC_SM - 0.050, oy,
              color=C_SM_EC, lw=1.0, ms=8)

    SM_W = 0.095; SM_H = 0.065
    add_rect(ax, XC_SM, oy, SM_W, SM_H, C_SM_FC, C_SM_EC,
             label='Soil\nMoisture (%)', fs=8.5, fw='bold', lc='#6E2C00')

    # ── Legend ────────────────────────────────────────────────────────────────
    leg = [
        mpatches.Patch(fc=C_INn_FC, ec=C_INn_EC, label='Input neurons'),
        mpatches.Patch(fc=C_Hn_FC,  ec=C_Hn_EC,  label='Hidden neurons'),
        mpatches.Patch(fc=C_On_FC,  ec=C_On_EC,  label='Output neurons'),
        mpatches.Patch(fc=C_RELU,   ec=C_RELU_E, label='ReLU'),
        mpatches.Patch(fc=C_LIN,    ec=C_LIN_E,  label='Linear'),
        mpatches.Patch(fc=C_BN,     ec=C_BN_E,   label='BatchNorm'),
        mpatches.Patch(fc=C_DR,     ec=C_DR_E,   label='Dropout'),
        mpatches.Patch(fc=C_IN_FC,  ec=C_IN_EC,  label='Input features'),
        mpatches.Patch(fc=C_SM_FC,  ec=C_SM_EC,  label='Output (SM%)'),
    ]
    ax.legend(handles=leg, loc='lower left', bbox_to_anchor=(0.0, 0.0),
              fontsize=7, ncol=3, framealpha=0.95, edgecolor='#CCC',
              title='Legend', title_fontsize=7.5)

    ax.text(0.5, 0.005,
            f'Fig. ANN: {sensor}  (7 inputs -> 128 -> 64 -> 32 -> 2 outputs [lo, hi])',
            ha='center', va='bottom', fontsize=7.5, style='italic',
            color='#555', transform=ax.transAxes)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset info strings (from actual data inspection)
# ──────────────────────────────────────────────────────────────────────────────
EOS_INFO = ('EOS-04  |  2528 samples  |  27 crops  |  SM1: 1.2-54.1%  |  '
            '70/10/10/10 split  |  MinMaxScaler  |  Target: 90% PICP (alpha=0.10)')
SEN_INFO = ('Sentinel-1  |  1816 samples  |  20 crops  |  SM1: 1.2-56.4%  |  '
            '70/10/10/10 split  |  MinMaxScaler  |  Target: 90% PICP (alpha=0.10)')

# ──────────────────────────────────────────────────────────────────────────────
# Generate all diagrams
# ──────────────────────────────────────────────────────────────────────────────
CONFIGS = [
    ('EOS-04',     EOS04_INPUTS, EOS_INFO, 'ann_architecture_eos04'),
    ('Sentinel-1', SEN1_INPUTS,  SEN_INFO, 'ann_architecture_sentinel1'),
]

# Individual
for sensor, inputs, info, fname in CONFIGS:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor('white')
    draw(ax, inputs, sensor, info)
    fig.savefig(OUT_DIR / f'{fname}.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / f'{fname}.svg', format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [OK]  {fname}.png / .svg')

# Combined side-by-side
fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 2, FIG_H), dpi=DPI)
fig.patch.set_facecolor('white')
for ax, (sensor, inputs, info, _) in zip(axes, CONFIGS):
    draw(ax, inputs, sensor, info)
fig.suptitle('ANN Architecture — SAR Soil Moisture Prediction'
             '  (Dual-Output Prediction Interval: [lo, hi])',
             fontsize=14, fontweight='bold', y=1.005)
fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, wspace=0.04)
fig.savefig(OUT_DIR / 'ann_architecture_combined.png',
            dpi=DPI, bbox_inches='tight', facecolor='white')
fig.savefig(OUT_DIR / 'ann_architecture_combined.svg',
            format='svg', bbox_inches='tight', facecolor='white')
plt.close(fig)
print('  [OK]  ann_architecture_combined.png / .svg')

print(f'\nAll outputs -> {OUT_DIR}')
print('Done.')
