"""
export_to_excel.py — Export all experiment results to styled Excel workbooks.

Outputs
-------
Per-experiment workbooks (one per output folder):
  output/<folder>/<folder>_results.xlsx
    • Metrics sheet   — flattened JSON metrics in a pivot table
    • Data Source sheet — experiment metadata, features, sample breakdown
    • <plot name>     — one sheet per PNG with caption (satellite, model, test-set info)

Consolidated master report (all experiments in one file):
  output/SAR_Moisture_Final_Report/SAR_Experiments_Summary.xlsx
    • Overview              — project description + metric definitions
    • Regression_Metrics    — MAE / RMSE / R² across all regression experiments
    • Classification_Metrics — accuracy + macro-F1 across all classifiers
    • PI_Metrics            — PICP + MPIW across all PI / conformal experiments
    • All_Metrics_Flat      — every metric from every experiment in one searchable table

Run:
    /snap/bin/astral-uv.uv run python experiments/classification_new_data/code/export_to_excel.py
"""

import json
from pathlib import Path

import openpyxl
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import (Alignment, Border, Font, PatternFill, Side)
from openpyxl.utils import get_column_letter

from constants import OUTPUT_PATH, X_cols_eos, X_cols_sentinel

# ---------------------------------------------------------------------------
# Experiment metadata — maps output folder name → rich description
# ---------------------------------------------------------------------------

FEATURES_EOS      = X_cols_eos       # ['HH-pol','HV-pol','NDVI','DpRVI','Depolarization_Rate']
FEATURES_SENTINEL = X_cols_sentinel   # ['VH-pol','VV-pol','NDVI','DpRVI','Depolarization_Rate']

# Total rows in processed CSVs (header not counted)
SATELLITE_TOTAL = {"EOS-04": 1953, "Sentinel-1": 1575}

# Default split fractions (all experiment classes use 80/10/10)
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10

EXPERIMENT_META: dict[str, dict] = {
    "classification_censored": {
        "label":       "4-class Classification — Censored",
        "stage":       "Stage 2",
        "task":        "Classification",
        "models":      ["RandomForest", "XGBoost", "AdaBoost", "SVC"],
        "description": (
            "Classifies SM1 (%) into 4 ordinal labels: Low / Medium / High / Very High. "
            "Censored rows (SM1 = 50, sensor detection limit) are retained in this variant."
        ),
        "split":       "80% train · 10% val · 10% test",
        "metrics_key": "classification",
    },
    "classification_uncensored": {
        "label":       "4-class Classification — Uncensored",
        "stage":       "Stage 2",
        "task":        "Classification",
        "models":      ["RandomForest", "XGBoost", "AdaBoost", "SVC"],
        "description": (
            "Classifies SM1 (%) into 4 ordinal labels. "
            "Censored rows (SM1 = 50) are removed before training."
        ),
        "split":       "80% train · 10% val · 10% test",
        "metrics_key": "classification",
    },
    "ml_experiment_censored": {
        "label":       "Classical ML Regression — Censored",
        "stage":       "Stage 3",
        "task":        "Regression",
        "models":      ["RandomForest", "XGBoost", "AdaBoost", "SVR"],
        "description": (
            "Point regression of SM1 (%) using classical ML models with RandomizedSearchCV. "
            "Censored rows retained. XGBoost uses device='cuda'."
        ),
        "split":       "80% train · 10% val · 10% test",
        "metrics_key": "regression",
    },
    "ml_experiment_uncensored": {
        "label":       "Classical ML Regression — Uncensored",
        "stage":       "Stage 3",
        "task":        "Regression",
        "models":      ["RandomForest", "XGBoost", "AdaBoost", "SVR"],
        "description": (
            "Point regression of SM1 (%) using classical ML models with RandomizedSearchCV. "
            "Censored rows (SM1 = 50) removed."
        ),
        "split":       "80% train · 10% val · 10% test",
        "metrics_key": "regression",
    },
    "ann_experiments_censored": {
        "label":       "ANN Regression — Censored",
        "stage":       "Stage 4",
        "task":        "Regression",
        "models":      ["MLP (various architectures)"],
        "description": (
            "Keras MLP regression with MSE loss. Multiple architectures tested. "
            "Mixed precision float16, batch_size=256, EarlyStopping. Censored rows retained."
        ),
        "split":       "80% train · 10% val · 10% test",
        "metrics_key": "regression",
    },
    "ann_experiments_uncensored": {
        "label":       "ANN Regression — Uncensored",
        "stage":       "Stage 4",
        "task":        "Regression",
        "models":      ["MLP (various architectures)"],
        "description": (
            "Keras MLP regression with MSE loss. Multiple architectures tested. "
            "Mixed precision float16, batch_size=256, EarlyStopping. Censored rows removed."
        ),
        "split":       "80% train · 10% val · 10% test",
        "metrics_key": "regression",
    },
    "pi_estimation_censored": {
        "label":       "Quantile ANN PI Estimation — Censored",
        "stage":       "Stage 5",
        "task":        "Prediction Interval",
        "models":      ["Quantile ANN (pinball loss)", "Tube Loss ANN"],
        "description": (
            "Prediction intervals via quantile ANN (pinball loss at lower/upper τ) "
            "and tube loss ANN. Reports PICP and MPIW. Censored rows retained."
        ),
        "split":       "80% train · 10% val · 10% test",
        "metrics_key": "pi",
    },
    "pi_estimation_uncensored": {
        "label":       "Quantile ANN PI Estimation — Uncensored",
        "stage":       "Stage 5",
        "task":        "Prediction Interval",
        "models":      ["Quantile ANN (pinball loss)", "Tube Loss ANN"],
        "description": (
            "Prediction intervals via quantile ANN (pinball loss at lower/upper τ) "
            "and tube loss ANN. Censored rows removed."
        ),
        "split":       "80% train · 10% val · 10% test",
        "metrics_key": "pi",
    },
    "conformal_regression_censored": {
        "label":       "Conformal Regression PI — Censored",
        "stage":       "Stage 6",
        "task":        "Prediction Interval",
        "models":      ["QuantileRegressor", "GBR", "HistGBR", "LGBMRegressor"],
        "description": (
            "Distribution-free prediction intervals using MAPIE ConformalizedQuantileRegressor. "
            "Conformalization split used for coverage calibration. Censored rows retained."
        ),
        "split":       "80% train · 10% conformalize · 10% test",
        "metrics_key": "pi",
    },
    "conformal_regression_uncensored": {
        "label":       "Conformal Regression PI — Uncensored",
        "stage":       "Stage 6",
        "task":        "Prediction Interval",
        "models":      ["QuantileRegressor", "GBR", "HistGBR", "LGBMRegressor"],
        "description": (
            "Distribution-free prediction intervals using MAPIE ConformalizedQuantileRegressor. "
            "Censored rows removed."
        ),
        "split":       "80% train · 10% conformalize · 10% test",
        "metrics_key": "pi",
    },
    "conformal_results_uncensored": {
        "label":       "Conformalized Quantile Regression (CQR) — Uncensored",
        "stage":       "Stage 7",
        "task":        "Prediction Interval",
        "models":      ["CQR (Quantile ANN + SVM split-conformal)"],
        "description": (
            "CQR: Quantile ANN trained on lower/upper quantiles, "
            "then calibrated with split-conformal scores on a held-out set."
        ),
        "split":       "80% train · 10% calibration · 10% test",
        "metrics_key": "pi",
    },
    "quantile_tau_tuning_uncensored": {
        "label":       "Quantile ANN τ Tuning — Uncensored",
        "stage":       "Stage 7",
        "task":        "Prediction Interval",
        "models":      ["Quantile ANN (tuned τ)"],
        "description": (
            "Grid search over lower/upper τ pairs to find the τ combination "
            "that minimises MPIW while maintaining PICP ≥ 0.95."
        ),
        "split":       "80% train · 10% val · 10% test",
        "metrics_key": "pi",
    },
    "qsvr_pi_estimation_uncensored": {
        "label":       "Quantile SVR Hyperparameter Tuning — Uncensored",
        "stage":       "Stage 8",
        "task":        "Prediction Interval",
        "models":      ["Quantile SVR (cvxopt QP)"],
        "description": (
            "Support vector quantile regression via cvxopt QP solver. "
            "Grid search over C and gamma with joblib parallelism."
        ),
        "split":       "80% train · 10% val · 10% test",
        "metrics_key": "pi",
    },
}

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

_HEADER_FILL  = PatternFill("solid", fgColor="1F4E79")
_HEADER_FONT  = Font(color="FFFFFF", bold=True, size=10)
_ALT_FILL     = PatternFill("solid", fgColor="D6E4F0")
_TITLE_FONT   = Font(bold=True, size=12, color="1F4E79")
_SECTION_FILL = PatternFill("solid", fgColor="E8F4FD")
_THIN         = Side(style="thin", color="AAAAAA")
_BORDER       = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)


def _hdr(cell, text=None):
    if text is not None:
        cell.value = text
    cell.fill      = _HEADER_FILL
    cell.font      = _HEADER_FONT
    cell.border    = _BORDER
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def _dat(cell, text=None, alt=False):
    if text is not None:
        cell.value = text
    cell.fill      = _ALT_FILL if alt else PatternFill("solid", fgColor="FFFFFF")
    cell.border    = _BORDER
    cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)


def _auto_width(ws, max_width=60):
    for col in ws.columns:
        w = max((len(str(c.value or "")) for c in col), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(w + 4, max_width)


# ---------------------------------------------------------------------------
# Plot filename parser
# ---------------------------------------------------------------------------

def _parse_plot(stem: str, folder_name: str) -> dict:
    """Return metadata dict for a given plot filename stem + experiment folder."""
    satellite = "Unknown"
    for sat in ("EOS-04", "Sentinel-1"):
        if stem.startswith(sat):
            satellite = sat
            break

    model_part = stem[len(satellite):].lstrip("_") if satellite != "Unknown" else stem

    # Strip common suffixes for cleaner model name
    for suffix in ("_actual_vs_predicted", "_prediction_error", "_prediction_interval"):
        if model_part.endswith(suffix):
            model_part = model_part[: -len(suffix)]
            break

    total    = SATELLITE_TOTAL.get(satellite, 0)
    n_test   = round(total * TEST_FRAC)
    n_train  = round(total * TRAIN_FRAC)
    n_val    = total - n_train - n_test

    features = FEATURES_EOS if satellite == "EOS-04" else FEATURES_SENTINEL
    censored = "Censored" if "censored" in folder_name and "uncensored" not in folder_name else "Uncensored"

    meta = EXPERIMENT_META.get(folder_name, {})

    return {
        "satellite":    satellite,
        "model":        model_part,
        "censored":     censored,
        "experiment":   meta.get("label", folder_name),
        "stage":        meta.get("stage", "—"),
        "features":     ", ".join(features),
        "total_rows":   total,
        "train_n":      n_train,
        "val_n":        n_val,
        "test_n":       n_test,
        "split":        meta.get("split", f"80% train · 10% val · 10% test"),
        "description":  meta.get("description", ""),
    }


# ---------------------------------------------------------------------------
# Sheet builders
# ---------------------------------------------------------------------------

def _write_metrics_sheet(wb: openpyxl.Workbook, all_metrics: list[tuple]):
    ws = wb.create_sheet("Metrics")
    ws.freeze_panes = "A2"

    for ci, h in enumerate(("Source File", "Metric Path", "Value"), 1):
        _hdr(ws.cell(row=1, column=ci), h)

    r = 2
    for src_file, rows in all_metrics:
        for key, val in rows:
            alt = r % 2 == 0
            _dat(ws.cell(row=r, column=1, value=src_file), alt=alt)
            _dat(ws.cell(row=r, column=2, value=key),      alt=alt)
            v = round(val, 6) if isinstance(val, float) else val
            _dat(ws.cell(row=r, column=3, value=v),        alt=alt)
            r += 1

    _auto_width(ws)


def _write_data_source_sheet(wb: openpyxl.Workbook, folder_name: str):
    meta = EXPERIMENT_META.get(folder_name)
    if not meta:
        return

    ws = wb.create_sheet("Data Source")
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 70

    rows = [
        ("Experiment",       meta["label"]),
        ("Pipeline Stage",   meta["stage"]),
        ("Task Type",        meta["task"]),
        ("Models Compared",  ", ".join(meta["models"])),
        ("Description",      meta["description"]),
        ("Data Split",       meta["split"]),
        ("", ""),
        ("EOS-04 Features",  ", ".join(FEATURES_EOS)),
        ("EOS-04 Total Rows",   str(SATELLITE_TOTAL["EOS-04"])),
        ("EOS-04 Train / Val / Test",
         f"{round(SATELLITE_TOTAL['EOS-04']*TRAIN_FRAC)} / "
         f"{round(SATELLITE_TOTAL['EOS-04']*VAL_FRAC)} / "
         f"{round(SATELLITE_TOTAL['EOS-04']*TEST_FRAC)} samples"),
        ("", ""),
        ("Sentinel-1 Features", ", ".join(FEATURES_SENTINEL)),
        ("Sentinel-1 Total Rows", str(SATELLITE_TOTAL["Sentinel-1"])),
        ("Sentinel-1 Train / Val / Test",
         f"{round(SATELLITE_TOTAL['Sentinel-1']*TRAIN_FRAC)} / "
         f"{round(SATELLITE_TOTAL['Sentinel-1']*VAL_FRAC)} / "
         f"{round(SATELLITE_TOTAL['Sentinel-1']*TEST_FRAC)} samples"),
        ("", ""),
        ("Target Variable",  "SM1 (%) — Surface soil moisture percentage"),
        ("Censored rows",    "SM1 = 50 (sensor detection limit)" + (
            " — RETAINED in this experiment" if "censored" in folder_name and "uncensored" not in folder_name
            else " — REMOVED in this experiment"
        )),
        ("Random Seed",      "42 (all splits)"),
    ]

    ws.cell(row=1, column=1, value=meta["label"]).font = _TITLE_FONT
    ws.merge_cells("A1:B1")

    for i, (label, value) in enumerate(rows, start=3):
        c1 = ws.cell(row=i, column=1, value=label)
        c2 = ws.cell(row=i, column=2, value=value)
        if label:
            c1.font   = Font(bold=True, size=10)
            c1.fill   = _SECTION_FILL
            c1.border = _BORDER
            c2.border = _BORDER
            c2.alignment = Alignment(wrap_text=True, vertical="top")


def _write_plot_sheet(wb: openpyxl.Workbook, png_path: Path, folder_name: str):
    stem   = png_path.stem
    sname  = stem[:31]
    existing = [s.title for s in wb.worksheets]
    base, n = sname, 1
    while sname in existing:
        sname = f"{base[:28]}_{n}"; n += 1

    ws  = wb.create_sheet(sname)
    pm  = _parse_plot(stem, folder_name)

    # Row 1 — bold title
    ws.cell(row=1, column=1, value=png_path.name).font = Font(bold=True, size=12, color="1F4E79")
    ws.merge_cells("A1:F1")

    # Rows 2–3 — metadata table header + values
    meta_headers = ["Satellite", "Model / Architecture", "Experiment Type",
                    "Data Type", "Input Features", "Test Set (plot data)"]
    meta_values  = [
        pm["satellite"],
        pm["model"],
        pm["experiment"],
        pm["censored"],
        pm["features"],
        (f"~{pm['test_n']} samples  ({int(TEST_FRAC*100)}% held-out split)\n"
         f"Total dataset: {pm['total_rows']} rows\n"
         f"Train {pm['train_n']} · Val {pm['val_n']} · Test {pm['test_n']}"),
    ]
    for ci, (h, v) in enumerate(zip(meta_headers, meta_values), 1):
        _hdr(ws.cell(row=2, column=ci), h)
        c = ws.cell(row=3, column=ci, value=v)
        c.border    = _BORDER
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.column_dimensions[get_column_letter(ci)].width = 22

    ws.row_dimensions[3].height = 52

    # Row 4 — source note
    note = (
        f"Plot generated from the test split of {pm['satellite']} processed CSV  "
        f"({pm['split']}).  "
        f"Each point represents one field measurement."
    )
    ws.cell(row=4, column=1, value=note).font = Font(italic=True, color="555555", size=9)
    ws.merge_cells("A4:F4")

    # Row 6+ — image
    try:
        img = XLImage(str(png_path))
        max_px = 900
        if img.width > max_px:
            scale      = max_px / img.width
            img.width  = int(img.width  * scale)
            img.height = int(img.height * scale)
        ws.add_image(img, "A6")
    except Exception as e:
        ws.cell(row=6, column=1, value=f"[Could not embed image: {e}]")


# ---------------------------------------------------------------------------
# JSON flattener
# ---------------------------------------------------------------------------

def _flatten(obj, prefix="") -> list[tuple]:
    rows = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            label = f"{prefix} › {k}" if prefix else str(k)
            rows.extend(_flatten(v, label) if isinstance(v, dict) else [(label, v)])
    else:
        rows.append((prefix, obj))
    return rows


def _load_all_metrics(folder: Path) -> list[tuple]:
    results = []
    for jf in sorted(folder.glob("*.json")):
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        results.append((jf.name, _flatten(data)))
    return results


# ---------------------------------------------------------------------------
# Per-folder workbook
# ---------------------------------------------------------------------------

def export_folder(folder: Path):
    all_metrics = _load_all_metrics(folder)
    plots_dir   = folder / "plots"
    png_files   = sorted(folder.glob("*.png")) + (
        sorted(plots_dir.glob("*.png")) if plots_dir.exists() else []
    )

    if not all_metrics and not png_files:
        return

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    if all_metrics:
        _write_metrics_sheet(wb, all_metrics)
    _write_data_source_sheet(wb, folder.name)
    for png in png_files:
        _write_plot_sheet(wb, png, folder.name)

    out = folder / f"{folder.name}_results.xlsx"
    wb.save(out)
    n_plots = len(png_files)
    print(f"  {folder.name}_results.xlsx  ({len(all_metrics)} metric file(s), {n_plots} plot(s))")


# ---------------------------------------------------------------------------
# Master consolidated report
# ---------------------------------------------------------------------------

def _collect_all_data(folders: list[Path]) -> list[dict]:
    """Flatten every metric from every folder into a list of dicts."""
    rows = []
    for folder in folders:
        meta     = EXPERIMENT_META.get(folder.name, {})
        exp_label = meta.get("label", folder.name)
        task      = meta.get("task", "")
        mkey      = meta.get("metrics_key", "")

        for jf in sorted(folder.glob("*.json")):
            try:
                data = json.loads(jf.read_text())
            except Exception:
                continue

            # Infer satellite from filename
            sat = "Unknown"
            for s in ("EOS-04", "Sentinel-1"):
                if s in jf.name:
                    sat = s
                    break

            for metric_path, value in _flatten(data):
                rows.append({
                    "experiment": exp_label,
                    "folder":     folder.name,
                    "satellite":  sat,
                    "task":       task,
                    "metrics_key": mkey,
                    "source_file": jf.name,
                    "metric":     metric_path,
                    "value":      value,
                })
    return rows


def _write_overview_sheet(wb: openpyxl.Workbook):
    ws = wb.create_sheet("Overview", 0)
    ws.column_dimensions["A"].width = 25
    ws.column_dimensions["B"].width = 75

    def row(r, label, value, bold_label=False):
        c1 = ws.cell(row=r, column=1, value=label)
        c2 = ws.cell(row=r, column=2, value=value)
        if bold_label and label:
            c1.font = Font(bold=True, size=10, color="1F4E79")
            c1.fill = _SECTION_FILL
        c2.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[r].height = 18

    ws.cell(row=1, column=1,
            value="SAR Soil Moisture — Complete Experiment Report").font = Font(bold=True, size=14, color="1F4E79")
    ws.merge_cells("A1:B1")
    ws.row_dimensions[1].height = 26

    r = 3
    sections = [
        ("Project",         "Soil moisture estimation from SAR satellite data (EOS-04 + Sentinel-1) "
                            "with calibrated prediction interval estimation."),
        ("Satellites",      "EOS-04 (C/X-band, HH+HV polarisation)\nSentinel-1 (C-band, VH+VV polarisation)"),
        ("Target",          "SM1 (%) — surface soil moisture, range ~5–50%"),
        ("EOS-04 dataset",  f"1953 field measurement rows · 5 features"),
        ("Sentinel-1 dataset", "1575 field measurement rows · 5 features"),
        ("Features",        "HH-pol/VH-pol, HV-pol/VV-pol, NDVI (Sentinel-2 via GEE), DpRVI, Depolarization_Rate"),
        ("Data split",      "80% training · 10% validation · 10% test  (random_state=42)"),
        ("Censored rows",   "SM1 = 50 marks the sensor detection ceiling. "
                            "Censored experiments retain these rows; uncensored experiments drop them."),
        ("", ""),
        ("Key metrics", ""),
        ("PICP",            "Prediction Interval Coverage Probability — fraction of true values inside [y_low, y_high]. Target ≥ 0.95."),
        ("MPIW",            "Mean Prediction Interval Width — average interval size. Lower is better (given PICP ≥ 0.95)."),
        ("MAE",             "Mean Absolute Error (% moisture units)"),
        ("RMSE",            "Root Mean Squared Error (% moisture units)"),
        ("R²",              "Coefficient of determination (1 = perfect, 0 = mean-only baseline)"),
        ("MAPE",            "Mean Absolute Percentage Error"),
        ("", ""),
        ("Sheets in this workbook", ""),
        ("Regression_Metrics",      "MAE / RMSE / R² for all regression experiments × satellite × model"),
        ("Classification_Metrics",  "Accuracy + macro-F1 for all classifiers"),
        ("PI_Metrics",              "PICP + MPIW for all prediction interval methods"),
        ("All_Metrics_Flat",        "Every metric from every experiment in a single searchable table"),
    ]
    for label, value in sections:
        bold = label in ("Key metrics", "Sheets in this workbook", "Project", "PICP", "MPIW",
                         "MAE", "RMSE", "R²", "MAPE", "Regression_Metrics",
                         "Classification_Metrics", "PI_Metrics", "All_Metrics_Flat")
        row(r, label, value, bold_label=bold)
        r += 1


def _write_regression_summary(wb: openpyxl.Workbook, all_rows: list[dict]):
    ws = wb.create_sheet("Regression_Metrics")
    ws.freeze_panes = "A2"

    reg_rows = [r for r in all_rows if r["task"] == "Regression"]
    if not reg_rows:
        ws.cell(row=1, column=1, value="No regression results available yet.")
        return

    # Extract model name and metric name from metric path "Model › MetricName"
    def parse(r):
        parts = r["metric"].split(" › ")
        return parts[0] if len(parts) >= 1 else "?", parts[-1] if len(parts) >= 2 else r["metric"]

    headers = ["Experiment", "Satellite", "Model", "MAE", "RMSE", "R²", "MAPE"]
    for ci, h in enumerate(headers, 1):
        _hdr(ws.cell(row=1, column=ci), h)

    # Pivot: group by (experiment, satellite, model) → metrics dict
    pivot: dict[tuple, dict] = {}
    for r in reg_rows:
        model, metric = parse(r)
        key = (r["experiment"], r["satellite"], model)
        pivot.setdefault(key, {})[metric] = r["value"]

    row_num = 2
    prev_exp = None
    for (exp, sat, model), mdict in sorted(pivot.items()):
        alt = row_num % 2 == 0
        cells = [exp if exp != prev_exp else "", sat, model,
                 mdict.get("MAE"),  mdict.get("RMSE"), mdict.get("R2"), mdict.get("MAPE")]
        prev_exp = exp
        for ci, v in enumerate(cells, 1):
            c = ws.cell(row=row_num, column=ci,
                        value=round(v, 4) if isinstance(v, float) else (v or ""))
            _dat(c, alt=alt)
        row_num += 1

    _auto_width(ws)


def _write_classification_summary(wb: openpyxl.Workbook, all_rows: list[dict]):
    ws = wb.create_sheet("Classification_Metrics")
    ws.freeze_panes = "A2"

    cls_rows = [r for r in all_rows if r["task"] == "Classification"]
    if not cls_rows:
        ws.cell(row=1, column=1, value="No classification results available yet.")
        return

    headers = ["Experiment", "Satellite", "Model", "Metric Path", "Value"]
    for ci, h in enumerate(headers, 1):
        _hdr(ws.cell(row=1, column=ci), h)

    row_num = 2
    prev_exp = None
    for r in sorted(cls_rows, key=lambda x: (x["experiment"], x["satellite"], x["metric"])):
        alt = row_num % 2 == 0
        vals = [
            r["experiment"] if r["experiment"] != prev_exp else "",
            r["satellite"], "", r["metric"],
            round(r["value"], 4) if isinstance(r["value"], float) else r["value"],
        ]
        prev_exp = r["experiment"]
        # Try to split model from metric path
        parts = r["metric"].split(" › ")
        if len(parts) >= 2:
            vals[2] = parts[0]
            vals[3] = " › ".join(parts[1:])
        for ci, v in enumerate(vals, 1):
            _dat(ws.cell(row=row_num, column=ci, value=v or ""), alt=alt)
        row_num += 1

    _auto_width(ws)


def _write_pi_summary(wb: openpyxl.Workbook, all_rows: list[dict]):
    ws = wb.create_sheet("PI_Metrics")
    ws.freeze_panes = "A2"

    pi_rows = [r for r in all_rows if r["task"] == "Prediction Interval"]
    if not pi_rows:
        ws.cell(row=1, column=1, value="No PI results available yet.")
        return

    headers = ["Experiment", "Satellite", "Model", "PICP", "MPIW"]
    for ci, h in enumerate(headers, 1):
        _hdr(ws.cell(row=1, column=ci), h)

    pivot: dict[tuple, dict] = {}
    for r in pi_rows:
        parts = r["metric"].split(" › ")
        model  = parts[0] if len(parts) >= 2 else "?"
        metric = parts[-1] if len(parts) >= 2 else r["metric"]
        key    = (r["experiment"], r["satellite"], model)
        pivot.setdefault(key, {})[metric] = r["value"]

    row_num = 2
    prev_exp = None
    for (exp, sat, model), mdict in sorted(pivot.items()):
        alt  = row_num % 2 == 0
        picp = mdict.get("PICP")
        mpiw = mdict.get("MPIW")
        if picp is None and mpiw is None:
            continue
        vals = [exp if exp != prev_exp else "", sat, model,
                round(picp, 4) if isinstance(picp, float) else (picp or ""),
                round(mpiw, 4) if isinstance(mpiw, float) else (mpiw or "")]
        prev_exp = exp
        for ci, v in enumerate(vals, 1):
            c = ws.cell(row=row_num, column=ci, value=v)
            _dat(c, alt=alt)
            # Highlight PICP < 0.95 in red
            if ci == 4 and isinstance(v, float) and v < 0.95:
                c.font = Font(color="CC0000", bold=True)
        row_num += 1

    _auto_width(ws)


def _write_flat_sheet(wb: openpyxl.Workbook, all_rows: list[dict]):
    ws = wb.create_sheet("All_Metrics_Flat")
    ws.freeze_panes = "A2"

    headers = ["Experiment", "Folder", "Satellite", "Task", "Source File", "Metric Path", "Value"]
    for ci, h in enumerate(headers, 1):
        _hdr(ws.cell(row=1, column=ci), h)

    for ri, r in enumerate(all_rows, start=2):
        alt = ri % 2 == 0
        vals = [r["experiment"], r["folder"], r["satellite"], r["task"],
                r["source_file"], r["metric"],
                round(r["value"], 6) if isinstance(r["value"], float) else r["value"]]
        for ci, v in enumerate(vals, 1):
            _dat(ws.cell(row=ri, column=ci, value=v), alt=alt)

    _auto_width(ws)


def export_master(folders: list[Path]):
    report_dir = OUTPUT_PATH / "SAR_Moisture_Final_Report"
    report_dir.mkdir(exist_ok=True)

    all_rows = _collect_all_data(folders)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    _write_overview_sheet(wb)
    _write_regression_summary(wb, all_rows)
    _write_classification_summary(wb, all_rows)
    _write_pi_summary(wb, all_rows)
    _write_flat_sheet(wb, all_rows)

    out = report_dir / "SAR_Experiments_Summary.xlsx"
    wb.save(out)
    total = len(all_rows)
    print(f"\n  SAR_Moisture_Final_Report/SAR_Experiments_Summary.xlsx  ({total} total metric rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Scanning {OUTPUT_PATH} …\n")
    folders = sorted(p for p in OUTPUT_PATH.iterdir() if p.is_dir()
                     and p.name != "SAR_Moisture_Final_Report")

    if not folders:
        print("No output folders found.")
        return

    print("Per-experiment workbooks:")
    for folder in folders:
        export_folder(folder)

    print("\nConsolidated master report:")
    export_master(folders)

    print(f"\nDone.  All results → {OUTPUT_PATH}/")


if __name__ == "__main__":
    main()
