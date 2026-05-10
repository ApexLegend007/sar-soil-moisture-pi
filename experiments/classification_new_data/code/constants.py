from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = _EXPERIMENT_ROOT / 'data'
OUTPUT_PATH = _EXPERIMENT_ROOT / 'output'

SENTINEL_FILE = 'sentinel-1-processed.csv'
EOS_FILE = 'eos-04-processed.csv'

X_cols_eos = ['HH-pol', 'HV-pol', 'NDVI', 'DpRVI', 'Depolarization_Rate']
X_cols_sentinel = ['VH-pol', 'VV-pol', 'NDVI', 'DpRVI', 'Depolarization_Rate']

y_col = ['SM1 (%)']