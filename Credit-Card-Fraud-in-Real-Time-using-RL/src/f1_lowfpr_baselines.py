# f1_lowfpr_baselines.py — baselines @ FPR≈1% with range illustration
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_score, recall_score, f1_score
import joblib
from xgboost import Booster, DMatrix

from src.data_loader import chrono_split
from src.features import add_derived

ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data" / "creditcard.csv"
MODELS = ROOT / "models"
FIGS   = MODELS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

def threshold_at_fpr(y_true, scores, target_fpr=0.01):
    fpr, tpr, thr = roc_curve(y_true, scores)
    i = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(thr[i])

def prf(y_true, y_pred):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    return p, r, f

def main():
    # ---------- Data ----------
    df = pd.read_csv(DATA)
    sp = chrono_split(df)
    Xtr = add_derived(sp.X_train); Xva = add_derived(sp.X_val); Xte = add_derived(sp.X_test)
    yva = sp.y_val.values.astype(int)
    yte = sp.y_test.values.astype(int)

    # ---------- Load trained baselines ----------
    lr = joblib.load(MODELS / "lr.joblib")
    rf = joblib.load(MODELS / "rf.joblib")
    xgb = Booster(); xgb.load_model(str(MODELS / "xgb_base.json"))

    # ---------- Scores ----------
    lr_va  = lr.predict_proba(Xva)[:,1];   lr_te  = lr.predict_proba(Xte)[:,1]
    rf_va  = rf.predict_proba(Xva)[:,1];   rf_te  = rf.predict_proba(Xte)[:,1]
    xg_va  = xgb.predict(DMatrix(Xva));    xg_te  = xgb.predict(DMatrix(Xte))

    # ---------- Thresholds on VAL for ~1% FPR ----------
    thr = {
        "LogReg":       threshold_at_fpr(yva, lr_va, 0.01),
        "RandomForest": threshold_at_fpr(yva, rf_va, 0.01),
        "XGBoost":      threshold_at_fpr(yva, xg_va, 0.01),
    }

    # ---------- Predictions on TEST ----------
    preds = {
        "LogReg":       (lr_te >= thr["LogReg"]).astype(int),
        "RandomForest": (rf_te >= thr["RandomForest"]).astype(int),
        "XGBoost":      (xg_te >= thr["XGBoost"]).astype(int),
    }

    # ---------- Metrics ----------
    metrics = {}
    for name, yhat in preds.items():
        p, r, f = prf(yte, yhat)
        metrics[name] = {"Precision": p, "Recall": r, "F1": f}

    # Save per-model metrics
    MODELS.mkdir(parents=True, exist_ok=True)
    (MODELS / "baselines_lowfpr_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(metrics).T.to_csv(MODELS / "baselines_lowfpr_metrics.csv")

    # ---------- Build RANGES (min–max across baselines) ----------
    rng = {}
    for m in ["Precision", "Recall", "F1"]:
        vals = [metrics[k][m] for k in metrics.keys()]
        rng[m] = (float(np.min(vals)), float(np.max(vals)))

    # ---------- Illustration 1: compact table-style figure ----------
    fig, ax = plt.subplots(figsize=(4.2, 2.2), dpi=200)
    ax.axis("off")
    table_data = [
        ["Metric", "Range"],
        ["Precision", f"{rng['Precision'][0]:.2f}–{rng['Precision'][1]:.2f}"],
        ["Recall",    f"{rng['Recall'][0]:.2f}–{rng['Recall'][1]:.2f}"],
        ["F1-score",  f"{rng['F1'][0]:.2f}–{rng['F1'][1]:.2f}"],
    ]
    the_table = ax.table(cellText=table_data, cellLoc="center", colWidths=[0.48, 0.42], loc="center")
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.1, 1.3)
    ax.set_title("Baseline performance at FPR ≤ 1% (Test)", pad=6)
    out_table = FIGS / "baseline_lowfpr_ranges.png"
    plt.savefig(out_table, bbox_inches="tight")
    plt.close()

    # ---------- Illustration 2: dumbbell range plot (optional, nice for slides) ----------
    fig, ax = plt.subplots(figsize=(5.2, 2.8), dpi=200)
    y_pos = np.arange(3)
    mins = [rng["Precision"][0], rng["Recall"][0], rng["F1"][0]]
    maxs = [rng["Precision"][1], rng["Recall"][1], rng["F1"][1]]
    ax.hlines(y=y_pos, xmin=mins, xmax=maxs)
    ax.plot(mins, y_pos, "o")
    ax.plot(maxs, y_pos, "o")
    ax.set_yticks(y_pos, ["Precision", "Recall", "F1-score"])
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1)
    ax.set_title("Baseline ranges at FPR ≤ 1% (Test)")
    plt.tight_layout()
    out_dumbbell = FIGS / "baseline_lowfpr_ranges_dumbbell.png"
    plt.savefig(out_dumbbell)
    plt.close()

    # ---------- Console summary (for logs) ----------
    print("\n=== Baselines @ FPR≈1% (TEST) ===")
    for k, v in metrics.items():
        print(f"{k:12s}  P={v['Precision']:.3f}  R={v['Recall']:.3f}  F1={v['F1']:.3f}")
    print("\nRanges (min–max across baselines):")
    for k, (lo, hi) in rng.items():
        print(f"{k:9s}  {lo:.2f}–{hi:.2f}")
    print(f"\nSaved: {out_table}")
    print(f"Saved: {out_dumbbell}")
    print("Also wrote CSV ->", MODELS / "baselines_lowfpr_metrics.csv")

if __name__ == "__main__":
    main()
