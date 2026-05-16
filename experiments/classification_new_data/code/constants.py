from pathlib import Path

DATA_PATH = Path('/home/lmaosid/Desktop/major_orig/experiments/classification_new_data/data')
OUTPUT_PATH = Path('/home/lmaosid/Desktop/major_orig/experiments/classification_new_data/output')

SENTINEL_FILE = 'sentinel-1-enhanced.csv'
EOS_FILE = 'eos-04-enhanced.csv'

X_cols_eos = ['HH-pol', 'HV-pol', 'cross_pol_ratio', 'month_sin', 'month_cos', 'crop_encoded']
X_cols_sentinel = ['VH-pol', 'VV-pol', 'cross_pol_ratio', 'month_sin', 'month_cos', 'crop_encoded']

y_col = ['SM1 (%)']