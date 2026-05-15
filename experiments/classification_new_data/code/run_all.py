"""
run_all.py — single entry point to execute every experiment.

Usage:
    uv run python run_all.py                  # both censored + uncensored
    uv run python run_all.py --mode censored
    uv run python run_all.py --mode uncensored
    uv run python run_all.py --skip qsvr      # skip slow Q-SVR HP sweep
"""
import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf

from constants import (
    OUTPUT_PATH,
    CLASS_LABELS,
    X_cols_eos, X_cols_sentinel,
    X_cols_eos_no_ndvi, X_cols_sentinel_no_ndvi,
    y_col, label_col,
    load_data,
)
from model_experiments import (
    ClassificationExperiment,
    RegressionExperiment,
    ANNExperiment,
    PredictionIntervalEstimation,
    ConformalRegression,
    ConformalizedQuantileExperiment,
    QuantileSVRExperiment,
    KFoldRegressionExperiment,
    KFoldClassificationExperiment,
)

# ── reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ── ANN architecture (input shape set dynamically from n_features) ────────────
def build_model(n_features: int) -> dict:
    """Return the two ANN architectures used across all experiments."""
    return {
        "16_D_8_D": tf.keras.Sequential([
            tf.keras.Input(shape=(n_features,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.09),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dropout(0.09),
            tf.keras.layers.Dense(1),
        ]),
        "16_D": tf.keras.Sequential([
            tf.keras.Input(shape=(n_features,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1),
        ]),
    }


def save_json(data: dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Individual experiment runners
# ─────────────────────────────────────────────────────────────────────────────

def run_classification(X_eos, y_eos_label, X_sentinel, y_sentinel_label, mode: str):
    print(f"\n{'='*60}")
    print(f"  CLASSIFICATION  [{mode}]")
    print(f"{'='*60}")
    for X, y, satellite in [
        (X_eos,      y_eos_label,      "EOS-04"),
        (X_sentinel, y_sentinel_label, "Sentinel-1"),
    ]:
        ce = ClassificationExperiment(
            X, y,
            satellite_name=satellite,
            labels=CLASS_LABELS,
            print_stats=True,
            split_type='train-test',
            train_size=0.8, test_size=0.2,
            type=mode,
        )
        ce.run_experiment()


def run_classical_ml(X_eos, y_eos, X_sentinel, y_sentinel, mode: str):
    print(f"\n{'='*60}")
    print(f"  CLASSICAL ML REGRESSION  [{mode}]")
    print(f"{'='*60}")
    for X, y, satellite in [
        (X_eos,      y_eos,      "EOS-04"),
        (X_sentinel, y_sentinel, "Sentinel-1"),
    ]:
        re = RegressionExperiment(
            X, y,
            satellite=satellite,
            train_size=0.8, test_size=0.2,
            split_type='train-test',
            type=mode,
        )
        re.run_experiment()


def run_ann(X_eos, y_eos, X_sentinel, y_sentinel, mode: str):
    print(f"\n{'='*60}")
    print(f"  ANN REGRESSION  [{mode}]")
    print(f"{'='*60}")
    n_features = X_eos.shape[1]

    for X, y, satellite in [
        (X_eos,      y_eos,      "EOS-04"),
        (X_sentinel, y_sentinel, "Sentinel-1"),
    ]:
        models = build_model(n_features)
        all_results = {}
        for name, model in models.items():
            tf.keras.backend.clear_session()
            ae = ANNExperiment(X, y, satellite=satellite, type=mode)
            all_results[name] = ae.run_experiment(model, model_param_string=name)

        out = OUTPUT_PATH / f"ann_experiments_{mode}" / f"{satellite}_metrics.json"
        save_json(all_results, out)


def run_pi_estimation(X_eos, y_eos, X_sentinel, y_sentinel, mode: str):
    print(f"\n{'='*60}")
    print(f"  PREDICTION INTERVAL ESTIMATION  [{mode}]")
    print(f"{'='*60}")
    n_features = X_eos.shape[1]

    for X, y, satellite in [
        (X_eos,      y_eos,      "EOS-04"),
        (X_sentinel, y_sentinel, "Sentinel-1"),
    ]:
        models = build_model(n_features)
        all_results = {}
        for name, model in models.items():
            tf.keras.backend.clear_session()
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            exp = PredictionIntervalEstimation(X, y, satellite=satellite, type=mode)
            all_results[name] = exp.run_experiment(
                model, model_param_string=name, optimizer=optimizer, epochs=1000
            )

        out = OUTPUT_PATH / f"pi_estimation_{mode}" / f"{satellite}_metrics.json"
        save_json(all_results, out)


def run_conformal_regression(X_eos, y_eos, X_sentinel, y_sentinel, mode: str):
    print(f"\n{'='*60}")
    print(f"  CONFORMAL REGRESSION  [{mode}]")
    print(f"{'='*60}")
    for X, y, satellite in [
        (X_eos,      y_eos,      "EOS-04"),
        (X_sentinel, y_sentinel, "Sentinel-1"),
    ]:
        cr = ConformalRegression(X, y, satellite=satellite, type=mode)
        cr.run_experiment()


def run_cqr(X_eos, y_eos, X_sentinel, y_sentinel, mode: str):
    print(f"\n{'='*60}")
    print(f"  CONFORMALIZED QUANTILE REGRESSION  [{mode}]")
    print(f"{'='*60}")
    n_features = X_eos.shape[1]
    model_template = build_model(n_features)["16_D_8_D"]

    for X, y, satellite in [
        (X_eos,      y_eos,      "EOS-04"),
        (X_sentinel, y_sentinel, "Sentinel-1"),
    ]:
        tf.keras.backend.clear_session()
        exp = ConformalizedQuantileExperiment(X, y, satellite=satellite)
        exp.run_experiment(model_template)


def run_tau_tuning(X_eos, y_eos, X_sentinel, y_sentinel):
    print(f"\n{'='*60}")
    print(f"  QUANTILE REGRESSION TAU TUNING  [uncensored]")
    print(f"{'='*60}")
    n_features = X_eos.shape[1]
    model_template = build_model(n_features)["16_D_8_D"]

    for X, y, satellite in [
        (X_eos,      y_eos,      "EOS-04"),
        (X_sentinel, y_sentinel, "Sentinel-1"),
    ]:
        tf.keras.backend.clear_session()
        exp = ConformalizedQuantileExperiment(X, y, satellite=satellite)
        exp.run_ann_tuning_experiment(model_template)


def run_kfold_regression(X_eos, y_eos, X_sentinel, y_sentinel, mode: str):
    print(f"\n{'='*60}")
    print(f"  K-FOLD REGRESSION (k=5, stratified)  [{mode}]")
    print(f"  Output → ml_experiment_{mode}_kfold/  (original results unchanged)")
    print(f"{'='*60}")
    for X, y, satellite in [
        (X_eos,      y_eos,      "EOS-04"),
        (X_sentinel, y_sentinel, "Sentinel-1"),
    ]:
        exp = KFoldRegressionExperiment(X, y, satellite=satellite, type=mode)
        exp.run_experiment()


def run_kfold_classification(X_eos, y_eos_label, X_sentinel, y_sentinel_label, mode: str):
    print(f"\n{'='*60}")
    print(f"  K-FOLD CLASSIFICATION (k=5, stratified)  [{mode}]")
    print(f"  Output → classification_{mode}_kfold/  (original results unchanged)")
    print(f"{'='*60}")
    from constants import CLASS_LABELS
    for X, y, satellite in [
        (X_eos,      y_eos_label,      "EOS-04"),
        (X_sentinel, y_sentinel_label, "Sentinel-1"),
    ]:
        exp = KFoldClassificationExperiment(X, y, satellite=satellite,
                                            labels=CLASS_LABELS, type=mode)
        exp.run_experiment()


def run_qsvr(X_eos, y_eos, X_sentinel, y_sentinel, mode: str):
    print(f"\n{'='*60}")
    print(f"  QUANTILE SVR HP TUNING  [{mode}]")
    print(f"{'='*60}")
    C_VALUE     = 2 ** 6
    GAMMA_VALUES = [2 ** i for i in range(-15, 16)]

    for X, y, satellite in [
        (X_eos,      y_eos,      "EOS-04"),
        (X_sentinel, y_sentinel, "Sentinel-1"),
    ]:
        exp = QuantileSVRExperiment(X, y, satellite=satellite, type=mode)
        all_results = []
        for gamma in GAMMA_VALUES:
            try:
                result = exp.run_experiment(C=C_VALUE, gamma=gamma)
                all_results.append(result)
            except Exception as e:
                print(f"  [WARN] gamma={gamma} failed: {e}")
                all_results.append({
                    "params": {"C": C_VALUE, "gamma": gamma},
                    "val": {"PICP": None, "MPIW": None},
                    "test": {"PICP": None, "MPIW": None},
                    "error": str(e),
                })

        out = OUTPUT_PATH / f"qsvr_pi_estimation_{mode}" / f"{satellite}_tuning_C={C_VALUE}.json"
        save_json(all_results, out)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_mode(mode: str, skip: list):
    print(f"\n{'#'*60}")
    print(f"#  MODE: {mode.upper()}")
    print(f"{'#'*60}")

    eos, sentinel = load_data(mode)

    # ── regression features ──────────────────────────────────────────────────
    X_eos      = eos[X_cols_eos].values
    X_sentinel = sentinel[X_cols_sentinel].values
    y_eos      = eos[y_col].values
    y_sentinel = sentinel[y_col].values

    # ── classification features (no NDVI — label target not SM) ─────────────
    X_eos_cls      = eos[X_cols_eos_no_ndvi].values
    X_sentinel_cls = sentinel[X_cols_sentinel_no_ndvi].values
    y_eos_label      = eos[[label_col]].values
    y_sentinel_label = sentinel[[label_col]].values

    if 'classification' not in skip:
        run_classification(X_eos_cls, y_eos_label, X_sentinel_cls, y_sentinel_label, mode)

    if 'kfold' not in skip:
        run_kfold_regression(X_eos, y_eos, X_sentinel, y_sentinel, mode)
        run_kfold_classification(X_eos_cls, y_eos_label, X_sentinel_cls, y_sentinel_label, mode)

    if 'classical_ml' not in skip:
        run_classical_ml(X_eos, y_eos, X_sentinel, y_sentinel, mode)

    if 'ann' not in skip:
        run_ann(X_eos, y_eos, X_sentinel, y_sentinel, mode)

    if 'pi' not in skip:
        run_pi_estimation(X_eos, y_eos, X_sentinel, y_sentinel, mode)

    if 'conformal' not in skip:
        run_conformal_regression(X_eos, y_eos, X_sentinel, y_sentinel, mode)

    if 'cqr' not in skip:
        run_cqr(X_eos, y_eos, X_sentinel, y_sentinel, mode)

    # tau tuning and qsvr are uncensored-only in original experiments
    if mode == 'uncensored':
        if 'tau' not in skip:
            run_tau_tuning(X_eos, y_eos, X_sentinel, y_sentinel)
        if 'qsvr' not in skip:
            run_qsvr(X_eos, y_eos, X_sentinel, y_sentinel, mode)


def main():
    parser = argparse.ArgumentParser(description="Run all soil moisture experiments.")
    parser.add_argument(
        '--mode', choices=['censored', 'uncensored', 'both'], default='both',
        help="Which data mode to run (default: both)"
    )
    parser.add_argument(
        '--skip', nargs='*', default=[],
        choices=['classification', 'classical_ml', 'ann', 'pi', 'conformal', 'cqr', 'tau', 'qsvr', 'kfold'],
        help="Experiment names to skip"
    )
    args = parser.parse_args()

    modes = ['censored', 'uncensored'] if args.mode == 'both' else [args.mode]
    for mode in modes:
        run_mode(mode, args.skip)

    print("\nAll experiments complete.")


if __name__ == '__main__':
    main()
