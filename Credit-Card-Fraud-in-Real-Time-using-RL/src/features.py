import pandas as pd
import numpy as np

def add_derived(X: pd.DataFrame) -> pd.DataFrame:
    X2 = X.copy()
    X2['Amount_log1p'] = np.log1p(np.abs(X2['Amount']))
    return X2
