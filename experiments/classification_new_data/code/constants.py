from pathlib import Path
import pandas as pd

DATA_PATH   = Path(__file__).parent.parent / 'data'
OUTPUT_PATH = Path(__file__).parent.parent / 'output'

SENTINEL_FILE = 'Sentinel-1_Crop_Analysis.xlsx'
EOS_FILE      = 'EOS-04_Crop_Analysis.xlsx'

CENSOR_VALUE = 50

CLASS_LABELS = ['Low', 'Medium', 'High', 'Very High']

X_cols_eos      = ['HH-pol', 'HV-pol', 'NDVI']
X_cols_sentinel = ['VH-pol', 'VV-pol', 'NDVI']

X_cols_eos_no_ndvi      = ['HH-pol', 'HV-pol']
X_cols_sentinel_no_ndvi = ['VH-pol', 'VV-pol']

y_col      = ['SM1 (%)']
crop_col   = 'Crop Name'
label_col  = 'label'


def load_data(mode: str = 'censored') -> tuple:
    """
    Load and return (eos_df, sentinel_df) from the new Excel files.

    mode='censored'   — full dataset
    mode='uncensored' — removes rows where SM1 (%) == CENSOR_VALUE
    """
    eos = pd.read_excel(DATA_PATH / EOS_FILE, sheet_name='All Data', header=1)
    sentinel = pd.read_excel(DATA_PATH / SENTINEL_FILE, sheet_name='All Data', header=1)

    if mode == 'uncensored':
        eos      = eos[eos['SM1 (%)'] != CENSOR_VALUE].reset_index(drop=True)
        sentinel = sentinel[sentinel['SM1 (%)'] != CENSOR_VALUE].reset_index(drop=True)

    return eos, sentinel
