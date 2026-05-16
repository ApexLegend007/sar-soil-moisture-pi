from pathlib import Path

DATA_PATH = Path('/home/lmaosid/Desktop/major_orig/experiments/classification_new_data/data')
OUTPUT_PATH = Path('/home/lmaosid/Desktop/major_orig/experiments/classification_new_data/output')

SENTINEL_FILE = 'sentinel-1-enhanced-ndvi.csv'
EOS_FILE = 'eos-04-enhanced-ndvi.csv'

X_cols_eos = ['HH-pol', 'HV-pol', 'cross_pol_ratio', 'month_sin', 'month_cos', 'crop_encoded', 'NDVI']
X_cols_sentinel = ['VH-pol', 'VV-pol', 'cross_pol_ratio', 'month_sin', 'month_cos', 'crop_encoded', 'NDVI']
# 6-feature set for Sentinel-1 regression/classification only (NDVI excluded — collinear with VH-pol r≈0.60)
y_col = ['SM1 (%)']