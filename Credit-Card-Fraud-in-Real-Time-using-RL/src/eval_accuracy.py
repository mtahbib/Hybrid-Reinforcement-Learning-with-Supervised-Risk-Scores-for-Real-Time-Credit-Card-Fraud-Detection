# src/eval_accuracy_multi_rl_margin.py
"""
Evaluate LR/RF/XGB at FPR targets (1%, 0.5%, 0.2%, 0.1%) and
ALSO evaluate RL in two ways:
  (A) RL_native: triage mode (block/review/approve) -> recall decomposition
  (B) RL_margin: block if (Q_block - max(Q_approve,Q_review)) >= tau_target,
      where tau_target is chosen on validation to hit the desired FPR.
Outputs:
  models/metrics/accuracy_multi_rl_{timestamp}.json
  models/metrics/accuracy_multi_rl_{timestamp}.csv
"""

from pathlib import Path
import sys, time, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix, average_precision_score
import joblib, torch
from xgboost import Booster, DMatrix

# ---------- wiring ----------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.data_loader import chrono_split
from src.features import add_derived

MODELS = ROOT / "models"
DATA   = ROOT / "data" / "creditcard.csv"
OUTDIR = MODELS / "metrics"

# ---------- helpers ----------
def proba_xgb(booster, X): return booster.predict(DMatrix(X))

def thr_at_fpr(y_true, scores, target_fpr):
    fpr, tpr, thr = roc_curve(y_true, scores)
    i = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(thr[i])

def cm_stats(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = tp/(tp+fp) if tp+fp>0 else 0.0
    rec  = tp/(tp+fn) if tp+fn>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
    test_fpr = fp/(fp+tn) if (fp+tn)>0 else 0.0
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                precision=float(prec), recall=float(rec), f1=float(f1),
                test_fpr=float(test_fpr))

def now_tag(): return time.strftime("%Y%m%d_%H%M%S")

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tag = now_tag()

    # ----- data -----
    df = pd.read_csv(DATA)
    sp = chrono_split(df)
    Xtr = add_derived(sp.X_train); Xva = add_derived(sp.X_val); Xte = add_derived(sp.X_test)
    yva = sp.y_val.values.astype(int)
    yte = sp.y_test.values.astype(int)

    # ----- models -----
    lr  = joblib.load(MODELS / "lr.joblib")
    rf  = joblib.load(MODELS / "rf.joblib")
    booster = Booster(); booster.load_model(str(MODELS / "xgb_base.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = torch.jit.load(str(MODELS / "dqn_policy.ts"), map_location=device).eval()

    # ----- scores for baselines -----
    s_lr_va  = lr.predict_proba(Xva)[:,1]
    s_rf_va  = rf.predict_proba(Xva)[:,1]
    s_xgb_va = proba_xgb(booster, Xva)
    s_lr_te  = lr.predict_proba(Xte)[:,1]
    s_rf_te  = rf.predict_proba(Xte)[:,1]
    s_xgb_te = proba_xgb(booster, Xte)

    # ----- RL scores (margins) on val & test -----
    # obs = [features, xgb_score, 1.0]
    xgb_va = s_xgb_va.astype(np.float32)
    xgb_te = s_xgb_te.astype(np.float32)
    f_va   = Xva.values.astype(np.float32)
    f_te   = Xte.values.astype(np.float32)
    obs_va = np.concatenate([f_va, xgb_va.reshape(-1,1), np.ones((len(Xva),1), np.float32)], axis=1)
    obs_te = np.concatenate([f_te, xgb_te.reshape(-1,1), np.ones((len(Xte),1), np.float32)], axis=1)

    with torch.inference_mode():
        q_va = policy(torch.as_tensor(obs_va, device=device)).cpu().numpy()
        q_te = policy(torch.as_tensor(obs_te, device=device)).cpu().numpy()
    margin_va = q_va[:,2] - np.max(q_va[:,0:2], axis=1)  # block advantage
    margin_te = q_te[:,2] - np.max(q_te[:,0:2], axis=1)

    # RL native triage (for recall decomposition)
    a_te = np.argmax(q_te, axis=1)  # 0 approve, 1 review, 2 block
    total_fraud = int(yte.sum())
    rl_native = {
        "block_recall": float(((a_te==2) & (yte==1)).sum() / max(1,total_fraud)),
        "review_recall": float(((a_te==1) & (yte==1)).sum() / max(1,total_fraud)),
        "total_recall": 0.0,
        "action_mix": {
            "approve": float((a_te==0).mean()),
            "review":  float((a_te==1).mean()),
            "block":   float((a_te==2).mean()),
        },
    }
    rl_native["total_recall"] = rl_native["block_recall"] + rl_native["review_recall"]

    # AUPRC (baselines)
    auprc = {
        "lr":  float(average_precision_score(yte, s_lr_te)),
        "rf":  float(average_precision_score(yte, s_rf_te)),
        "xgb": float(average_precision_score(yte, s_xgb_te)),
    }

    # ----- evaluate at multiple targets -----
    fpr_targets = [0.01, 0.005, 0.002, 0.001]
    rows = []
    thresholds = {"lr":{}, "rf":{}, "xgb":{}, "rl_margin":{}}

    for fpr_t in fpr_targets:
        # baselines thresholds (picked on val)
        thr_lr  = thr_at_fpr(yva, s_lr_va,  fpr_t)
        thr_rf  = thr_at_fpr(yva, s_rf_va,  fpr_t)
        thr_xgb = thr_at_fpr(yva, s_xgb_va, fpr_t)
        thresholds["lr"][f"{fpr_t:.4f}"]  = thr_lr
        thresholds["rf"][f"{fpr_t:.4f}"]  = thr_rf
        thresholds["xgb"][f"{fpr_t:.4f}"] = thr_xgb

        # RL margin threshold (picked on val)
        tau = thr_at_fpr(yva, margin_va, fpr_t)
        thresholds["rl_margin"][f"{fpr_t:.4f}"] = float(tau)

        # predictions on TEST
        yhat_lr  = (s_lr_te  >= thr_lr ).astype(int)
        yhat_rf  = (s_rf_te  >= thr_rf ).astype(int)
        yhat_xgb = (s_xgb_te >= thr_xgb).astype(int)
        yhat_rlM = (margin_te >= tau    ).astype(int)  # "block if margin≥tau"

        # metrics
        rows.append(dict(target_fpr=fpr_t, model="lr",          **cm_stats(yte, yhat_lr)))
        rows.append(dict(target_fpr=fpr_t, model="rf",          **cm_stats(yte, yhat_rf)))
        rows.append(dict(target_fpr=fpr_t, model="xgb",         **cm_stats(yte, yhat_xgb)))
        rows.append(dict(target_fpr=fpr_t, model="rl_margin",   **cm_stats(yte, yhat_rlM)))

    # also include RL_native (block-only view at its *own* operating point)
    yhat_rl_native_block = (a_te==2).astype(int)
    rl_native_block = cm_stats(yte, yhat_rl_native_block)

    # ----- save -----
    out_json = OUTDIR / f"accuracy_multi_rl_{tag}.json"
    out_csv  = OUTDIR / f"accuracy_multi_rl_{tag}.csv"

    bundle = {
        "fpr_targets": fpr_targets,
        "thresholds_by_target": thresholds,
        "auprc": auprc,
        "rl_native": rl_native,
        "rl_native_block_metrics": rl_native_block,
        "metrics": rows,
    }
    with open(out_json, "w") as f: json.dump(bundle, f, indent=2)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved -> {out_json}")
    print(f"Saved -> {out_csv}")

if __name__ == "__main__":
    main()
