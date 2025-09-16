# run_baselines.py  — one-file baseline trainer (LR, RF, XGBoost-GPU)
import json, joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

DATA = Path("data/creditcard.csv")
OUT  = Path("models"); OUT.mkdir(parents=True, exist_ok=True)

def chrono_split(df, train=0.70, val=0.15):
    df = df.sort_values('Time').reset_index(drop=True)
    X = df.drop(columns=['Class'])
    y = df['Class'].astype(int)
    n = len(df); tr = int(n*train); va = int(n*(train+val))
    return (X.iloc[:tr], y.iloc[:tr],
            X.iloc[tr:va], y.iloc[tr:va],
            X.iloc[va:],   y.iloc[va:])

def add_derived(X: pd.DataFrame) -> pd.DataFrame:
    X2 = X.copy()
    # scale only Time & Amount (V1..V28 are already standardized in this dataset)
    sc = StandardScaler().fit(X2[['Time','Amount']])
    X2[['Time','Amount']] = sc.transform(X2[['Time','Amount']])
    X2['Amount_log1p'] = np.log1p(np.abs(X2['Amount']))
    joblib.dump(sc, OUT/'scaler.joblib')
    return X2

def eval_scores(y_true, proba):
    return {"AUROC": float(roc_auc_score(y_true, proba)),
            "AUPRC": float(average_precision_score(y_true, proba))}

def main():
    assert DATA.exists(), f"Missing {DATA} — put creditcard.csv into data/"
    df = pd.read_csv(DATA)
    Xtr, ytr, Xva, yva, Xte, yte = chrono_split(df)
    Xtr = add_derived(Xtr); Xva = add_derived(Xva); Xte = add_derived(Xte)

    results = {}

    # 1) Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(Xtr, ytr)
    proba_lr = lr.predict_proba(Xte)[:,1]
    results["LogReg"] = eval_scores(yte, proba_lr)
    joblib.dump(lr, OUT/'lr.joblib')

    # 2) Random Forest
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, random_state=42,
        class_weight='balanced_subsample', n_jobs=-1
    )
    rf.fit(Xtr, ytr)
    proba_rf = rf.predict_proba(Xte)[:,1]
    results["RandomForest"] = eval_scores(yte, proba_rf)
    joblib.dump(rf, OUT/'rf.joblib')

    # 3) XGBoost (GPU)
    pos_w = (ytr.value_counts()[0]/ytr.value_counts()[1])
    xgb = XGBClassifier(
        tree_method='gpu_hist',
        max_depth=6, n_estimators=800, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8, scale_pos_weight=pos_w,
        random_state=42
    )
    xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    proba_xgb = xgb.predict_proba(Xte)[:,1]
    results["XGBoost_GPU"] = eval_scores(yte, proba_xgb)
    xgb.save_model(str(OUT/'xgb_base.json'))

    (OUT/'baselines.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("Saved models in:", str(OUT.resolve()))

if __name__ == "__main__":
    main()
