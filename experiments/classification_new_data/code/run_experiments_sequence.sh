#!/usr/bin/env bash
# run_experiments_sequence.sh
# Executes all experiment notebooks in the correct dependency order.
#
# USAGE
#   cd experiments/classification_new_data/code
#   bash run_experiments_sequence.sh            # run all notebooks
#   bash run_experiments_sequence.sh --skip-ml  # skip slow GridSearch phases
#
# ALTERNATIVE (headless Python, covers all 8 phases at once):
#   uv run python run_all_enhanced.py
#
# NOTE
#   Notebooks write results to ../output/.
#   For a fresh run from scratch, run run_all_enhanced.py — it is faster
#   because it avoids Jupyter kernel startup overhead.
#   Use this script when you need cell-level output or want to inspect
#   intermediate results interactively.

set -euo pipefail

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$CODE_DIR"

# ── helper ────────────────────────────────────────────────────────────────────
run_nb() {
    local nb="$1"
    echo ""
    echo "=================================================================="
    echo "  Running: $nb"
    echo "=================================================================="
    uv run jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout=3600 \
        --ExecutePreprocessor.kernel_name=ml-experiments \
        "$nb"
    echo "  Done: $nb"
}

SKIP_ML=false
for arg in "$@"; do
    [[ "$arg" == "--skip-ml" ]] && SKIP_ML=true
done

echo "=================================================================="
echo "  EXPERIMENT NOTEBOOK SEQUENCE RUNNER"
echo "  Working dir: $CODE_DIR"
echo "=================================================================="

# Phase 0 — Exploration / data understanding (optional, read-only)
 run_nb exploration_eos.ipynb
 run_nb exploration_sentinel.ipynb

# Phase 1 — Classical ML Regression
if [[ "$SKIP_ML" == false ]]; then
    run_nb classical_ml_uncensored.ipynb
    run_nb classical_ml_censored.ipynb
fi

# Phase 2 — Classification
run_nb classification_uncensored.ipynb
run_nb classification_censored.ipynb

# Phase 3 — ANN Point Estimation
run_nb ann_uncensored.ipynb
run_nb ann_censored.ipynb

# Phase 4 — Prediction Interval (ANN Quantile)
run_nb pi_estimation_uncensored.ipynb
run_nb pi_estimation_censored.ipynb

# Phase 5 — Conformal Regression (MAPIE)
run_nb conformal_regression_uncensored.ipynb
run_nb conformal_regression_censored.ipynb

# Phase 6 — Conformalized Quantile Regression (CQR)
run_nb conformalized_quantile_regression_uncensored.ipynb

# Phase 7 — Tau Hyperparameter Tuning (CQR)
run_nb quantile_regression_tau_tuning_uncensored.ipynb

# Phase 8 — Quantile SVR Gamma Tuning
run_nb quantile_svr_HP_tuning.ipynb

echo ""
echo "=================================================================="
echo "  ALL NOTEBOOKS COMPLETE"
echo "  Results: $(cd .. && pwd)/output/"
echo "=================================================================="
