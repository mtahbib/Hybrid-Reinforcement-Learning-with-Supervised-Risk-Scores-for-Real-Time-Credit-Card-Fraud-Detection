from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import StandardScaler

@dataclass
class Splits:
    X_train: pd.DataFrame; y_train: pd.Series
    X_val:   pd.DataFrame; y_val:   pd.Series
    X_test:  pd.DataFrame; y_test:  pd.Series
    scaler:  StandardScaler

def chrono_split(df: pd.DataFrame, train: float = 0.70, val: float = 0.15) -> Splits:
    df = df.sort_values('Time').reset_index(drop=True)
    X = df.drop(columns=['Class'])
    y = df['Class'].astype(int)
    n = len(df); tr = int(n*train); va = int(n*(train+val))
    X_tr, y_tr = X.iloc[:tr], y.iloc[:tr]
    X_va, y_va = X.iloc[tr:va], y.iloc[tr:va]
    X_te, y_te = X.iloc[va:],  y.iloc[va:]
    # Scale only Time & Amount (V1..V28 are already standardized)
    scaler = StandardScaler().fit(X_tr[['Time','Amount']])
    def scale(X_):
        Xs = X_.copy()
        Xs[['Time','Amount']] = scaler.transform(X_[['Time','Amount']])
        return Xs
    return Splits(scale(X_tr), y_tr, scale(X_va), y_va, scale(X_te), y_te, scaler)
