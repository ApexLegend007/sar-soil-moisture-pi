"""
Prediction Interval evaluation metrics — Phases 21+

PICP  — Prediction Interval Coverage Probability
MPIW  — Mean Prediction Interval Width
CWC   — Coverage Width-based Criterion  (Khosravi et al. 2011)
IS    — Interval Score / Winkler Score  (Winkler 1972)

Usage:
    from eval_pi_metrics import all_metrics
    m = all_metrics(y_te, lo, hi)
"""
import numpy as np


def picp(y_true, lo, hi):
    return round(float(np.mean((lo <= y_true) & (y_true <= hi))), 4)


def mpiw(lo, hi):
    return round(float(np.mean(hi - lo)), 4)


def cwc(y_true, lo, hi, alpha=0.10, eta=50.0):
    """
    Coverage Width-based Criterion (Khosravi et al. 2011).

    CWC = PINAW * (1 + γ * exp(−η * (PICP − μ_c)))
      γ = 0  if  PICP >= μ_c  (coverage met, only width penalised)
      γ = 1  if  PICP <  μ_c  (heavy penalty for under-coverage)
      PINAW = MPIW / R,  R = range of y_true in test set

    Lower CWC = better interval quality.
    η=50 is standard (Khosravi 2011); large η makes under-coverage
    catastrophically expensive.
    """
    p    = picp(y_true, lo, hi)
    mu_c = 1.0 - alpha
    R    = max(float(y_true.max() - y_true.min()), 1e-9)
    pinaw = mpiw(lo, hi) / R
    gamma = 0.0 if p >= mu_c else 1.0
    return round(float(pinaw * (1.0 + gamma * np.exp(-eta * (p - mu_c)))), 6)


def interval_score(y_true, lo, hi, alpha=0.10):
    """
    Winkler Interval Score (1972).

    IS = mean[ (hi - lo) + (2/α) * max(0, lo-y) + (2/α) * max(0, y-hi) ]

    The penalty term (2/α) heavily punishes misses: at α=0.10, each
    missed sample adds 20 × its distance outside the interval.
    Lower IS = better.
    """
    pen = np.maximum(0.0, lo - y_true) + np.maximum(0.0, y_true - hi)
    return round(float(np.mean((hi - lo) + (2.0 / alpha) * pen)), 4)


def all_metrics(y_true, lo, hi, alpha=0.10):
    return {
        'PICP': picp(y_true, lo, hi),
        'MPIW': mpiw(lo, hi),
        'CWC':  cwc(y_true, lo, hi, alpha),
        'IS':   interval_score(y_true, lo, hi, alpha),
    }
