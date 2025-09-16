# src/plot_compare.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, average_precision_score

from xgboost import Booster, DMatrix
import joblib

# local imports
from src.data_loader import chrono_split
from src.features import add_derived

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DATA   = ROOT / "data" / "creditcard.csv"
OUTDIR = ROOT / "models" / "figs"
OUTDIR.mkdir(parents=True, exist_ok=True)

def load_baselines():
    lr = joblib.load(MODELS/"lr.joblib")
    rf = joblib.load(MODELS/"rf.joblib")
    booster = Booster(); booster.load_model(str(MODELS/"xgb_base.json"))
    return lr, rf, booster

def main():
    # ---------- data ----------
    df = pd.read_csv(DATA)
    sp = chrono_split(df)
    Xtr = add_derived(sp.X_train); Xva = add_derived(sp.X_val); Xte = add_derived(sp.X_test)
    yte = sp.y_test.values

    # ---------- baselines: PR curves ----------
    lr, rf, booster = load_baselines()
    lr_scores  = lr.predict_proba(Xte)[:,1]
    rf_scores  = rf.predict_proba(Xte)[:,1]
    xgb_scores = booster.predict(DMatrix(Xte))

    curves = {}
    for name, s in [("LogReg", lr_scores), ("RandomForest", rf_scores), ("XGBoost", xgb_scores)]:
        p, r, _ = precision_recall_curve(yte, s)
        ap = average_precision_score(yte, s)
        curves[name] = (p, r, ap)

    # ---------- read headline metrics from compare_results.json ----------
    cmp_path = MODELS/"compare_results.json"
    with open(cmp_path, "r") as f:
        cmp = json.load(f)

    # baselines F1 (at val-tuned ~1% FPR)
    b_f1 = {
        "LogReg":  cmp["LogReg"]["ValThr@1%FPR"]["F1"],
        "RandomForest": cmp["RandomForest"]["ValThr@1%FPR"]["F1"],
        "XGBoost": cmp["XGBoost"]["ValThr@1%FPR"]["F1"],
    }
    # RL (BLOCK-only) headline
    rl_f1 = cmp["RL_DQN"]["Alerts_BLOCK_only"]["F1"]
    rl_prec = cmp["RL_DQN"]["Alerts_BLOCK_only"]["Precision"]
    rl_rec  = cmp["RL_DQN"]["Alerts_BLOCK_only"]["Recall"]
    rl_cost = cmp["RL_DQN"]["Cost_per_1k"]

    # ---------- FIGURE 1: Precision–Recall curves (baselines) ----------
    plt.figure(figsize=(6.0, 4.5))
    for name, (p, r, ap) in curves.items():
        plt.plot(r, p, label=f"{name} (AUPRC={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall (Test)")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = OUTDIR/"pr_curves_baselines.png"
    plt.savefig(pr_path, dpi=200)
    plt.close()

    # ---------- FIGURE 2: F1 bars (baselines @1%FPR vs RL BLOCK-only) ----------
    names = ["LogReg","RandomForest","XGBoost","RL (BLOCK)"]
    vals  = [b_f1["LogReg"], b_f1["RandomForest"], b_f1["XGBoost"], rl_f1]
    plt.figure(figsize=(6.0, 4.0))
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), names, rotation=10)
    plt.ylabel("F1 score")
    plt.title("F1 comparison: Baselines (~1% FPR) vs RL (PROPOSED)")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    f1_path = OUTDIR/"f1_bars_baselines_vs_rl.png"
    plt.savefig(f1_path, dpi=200)
    plt.close()

    # ---------- TABLE: Thesis-ready markdown ----------
    md = []
    md.append("| Metric (Test) | LogReg | RandomForest | XGBoost | RL (PROPOSED) |")
    md.append("|---|---:|---:|---:|---:|")
    md.append(f"| **AUROC** | {cmp['LogReg']['AUROC']:.3f} | {cmp['RandomForest']['AUROC']:.3f} | {cmp['XGBoost']['AUROC']:.3f} | — |")
    md.append(f"| **AUPRC** | {cmp['LogReg']['AUPRC']:.3f} | {cmp['RandomForest']['AUPRC']:.3f} | {cmp['XGBoost']['AUPRC']:.3f} | — |")
    md.append(f"| **F1 (≈1% FPR / PROPOSED)** | {b_f1['LogReg']:.3f} | {b_f1['RandomForest']:.3f} | {b_f1['XGBoost']:.3f} | **{rl_f1:.3f}** |")
    md.append(f"| **Cost per 1k tx** | — | — | — | **{rl_cost:.1f}** |")
    table_path = OUTDIR/"results_table.md"
    Path(table_path).write_text("\n".join(md), encoding="utf-8")

    print("Saved:")
    print("  PR curves  ->", pr_path)
    print("  F1 bars    ->", f1_path)
    print("  Markdown   ->", table_path)

if __name__ == "__main__":
    main()
