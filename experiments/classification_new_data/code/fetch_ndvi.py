"""
Fetch Sentinel-2 NDVI for every (date, lat, lon) in EOS-04 and Sentinel-1 datasheets.
Batches all 145 points per date into a single GEE reduceRegions call (~34 calls total).
Writes:
  data/eos04_ndvi.csv
  data/sentinel1_ndvi.csv
"""
import time
from pathlib import Path

import ee
import pandas as pd

# ── Auth ──────────────────────────────────────────────────────────────────────
KEY_FILE = Path(__file__).parent.parent / "data" / "ee-key.json"
DATA_DIR  = Path(__file__).parent.parent / "data"

credentials = ee.ServiceAccountCredentials(email=None, key_file=str(KEY_FILE))
ee.Initialize(credentials)
print("GEE authenticated OK")

S2 = "COPERNICUS/S2_SR_HARMONIZED"


def _mask_s2_clouds(image):
    """Per-pixel cloud mask using SCL band (Scene Classification Layer)."""
    scl = image.select("SCL")
    # Keep: 4=vegetation, 5=bare soil, 6=water, 11=snow (everything except clouds/shadow)
    cloud_mask = scl.neq(1).And(scl.neq(2)).And(scl.neq(3)) \
                    .And(scl.neq(7)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return ndvi.updateMask(cloud_mask)


def fetch_ndvi_for_date(date_str: str, points_df: pd.DataFrame,
                        window_days: int = 5) -> pd.Series:
    """
    One GEE call: median NDVI over ±window_days for all points.
    Uses per-pixel SCL cloud masking so partial-cloud images still contribute.
    Returns a Series of NDVI values indexed like points_df. Empty → all NaN.
    """
    center = ee.Date(date_str)
    start  = center.advance(-window_days, "day")
    end    = center.advance(window_days,  "day")

    bbox = ee.Geometry.BBox(
        points_df["lon"].min() - 0.01, points_df["lat"].min() - 0.01,
        points_df["lon"].max() + 0.01, points_df["lat"].max() + 0.01,
    )

    col = (
        ee.ImageCollection(S2)
        .filterDate(start, end)
        .filterBounds(bbox)
        .map(_mask_s2_clouds)
    )

    # Guard: if collection is empty return NaN series
    try:
        n_imgs = col.size().getInfo()
    except Exception:
        return pd.Series(index=points_df.index, dtype=float)

    if n_imgs == 0:
        return pd.Series(index=points_df.index, dtype=float)

    img = col.median()

    features = [
        ee.Feature(ee.Geometry.Point([row["lon"], row["lat"]]),
                   {"row_idx": int(idx)})
        for idx, row in points_df.iterrows()
    ]
    fc = ee.FeatureCollection(features)

    sampled = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10)

    try:
        info = sampled.getInfo()
        result = {f["properties"]["row_idx"]: f["properties"].get("mean")
                  for f in info["features"]}
        return pd.Series(result, dtype=float)
    except Exception as e:
        print(f"    GEE error on {date_str} (±{window_days}d): {e}")
        return pd.Series(index=points_df.index, dtype=float)


def fetch_for_datasheet(xl_path: Path, pol_cols: list, label: str) -> pd.DataFrame:
    xl = pd.ExcelFile(xl_path)
    date_sheets = [s for s in xl.sheet_names if s.strip() != "UniqueCrops"]

    all_rows = []

    for sheet in date_sheets:
        df = xl.parse(sheet)
        df.columns = df.columns.str.strip()
        # Use sheet name as authoritative date (column value can be malformed)
        try:
            date_str = pd.to_datetime(sheet.strip(), dayfirst=True).strftime("%Y-%m-%d")
        except Exception:
            date_str = pd.to_datetime(df["Sample Date & Time"].iloc[0]).strftime("%Y-%m-%d")
        print(f"  [{label}] {sheet} → {date_str} …", end=" ", flush=True)

        df = df.reset_index(drop=True)

        # Normalise lat/lon column names
        pts = df[["Latitude (Centre of grid)", "Longitude (Centre of grid)"]].rename(
            columns={"Latitude (Centre of grid)": "lat",
                     "Longitude (Centre of grid)": "lon"}
        )

        # Try ±5 days first
        ndvi_5  = fetch_ndvi_for_date(date_str, pts, window_days=5)
        missing = ndvi_5[ndvi_5.isna()].index.tolist()

        ndvi_15  = pd.Series(dtype=float)
        fallback = []
        if missing:
            ndvi_15  = fetch_ndvi_for_date(date_str, pts.loc[missing], window_days=15)
            fallback = ndvi_15.dropna().index.tolist()

        ndvi_vals = ndvi_5.combine_first(ndvi_15)
        n_ok = ndvi_vals.notna().sum()
        print(f"NDVI ok={n_ok}/{len(df)}", flush=True)

        for idx, row in df.iterrows():
            rec = {
                "date":          date_str,
                "sample_id":     row["Sample Id (Grid)"],
                "lat":           pts.loc[idx, "lat"],
                "lon":           pts.loc[idx, "lon"],
                "SM1":           float(row["SM1 (%)"]),
                "crop":          row["Crop Name"],
                "NDVI":          ndvi_vals.get(idx),
                "ndvi_fallback": idx in fallback,
            }
            for col in pol_cols:
                rec[col] = float(row[col])
            all_rows.append(rec)

        time.sleep(0.3)

    return pd.DataFrame(all_rows)


# ── EOS-04 ────────────────────────────────────────────────────────────────────
out_eos = DATA_DIR / "eos04_ndvi.csv"
if out_eos.exists():
    print("\n=== EOS-04 (already done — skipping) ===")
    eos_df = pd.read_csv(out_eos)
else:
    print("\n=== EOS-04 ===")
    eos_df = fetch_for_datasheet(
        DATA_DIR / "EOS-04_datasheet.xlsx",
        pol_cols=["HH-pol", "HV-pol"],
        label="EOS-04",
    )
    eos_df.to_csv(out_eos, index=False)
ok = eos_df["NDVI"].notna().sum()
print(f"EOS-04 → {out_eos}")
print(f"  NDVI retrieved : {ok}/{len(eos_df)} ({100*ok/len(eos_df):.1f}%)")
print(f"  Fallback (±15d): {eos_df['ndvi_fallback'].sum()}")
print(f"  NaN (no data)  : {eos_df['NDVI'].isna().sum()}")

# ── Sentinel-1 ───────────────────────────────────────────────────────────────
print("\n=== Sentinel-1 ===")
s1_df = fetch_for_datasheet(
    DATA_DIR / "sentinel-1.xlsx",
    pol_cols=["VH-pol", "VV-pol"],
    label="Sentinel-1",
)
out_s1 = DATA_DIR / "sentinel1_ndvi.csv"
s1_df.to_csv(out_s1, index=False)
ok = s1_df["NDVI"].notna().sum()
print(f"\nSentinel-1 → {out_s1}")
print(f"  NDVI retrieved : {ok}/{len(s1_df)} ({100*ok/len(s1_df):.1f}%)")
print(f"  Fallback (±15d): {s1_df['ndvi_fallback'].sum()}")
print(f"  NaN (no data)  : {s1_df['NDVI'].isna().sum()}")

print("\nAll done.")
