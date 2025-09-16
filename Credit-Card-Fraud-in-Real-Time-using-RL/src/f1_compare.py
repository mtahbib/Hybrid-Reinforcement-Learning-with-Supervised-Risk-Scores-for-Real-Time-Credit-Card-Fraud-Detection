# src/f1_compare.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_score, recall_score, f1_score

from src.data_loader import chrono_split
from src.features import add_derived

import joblib
from xgboost import Booster, DMatrix
import torch

# NEW: import your env
from src.rl_env import CreditFraudEnv

ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data" / "creditcard.csv"
MODELS = ROOT / "models"
FIGS   = MODELS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

def threshold_at_fpr(y_true, scores, target_fpr=0.01):
    fpr, tpr, thr = roc_curve(y_true, scores)
    i = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(thr[i])

def load_artifacts():
    lr = joblib.load(MODELS / "lr.joblib")
    rf = joblib.load(MODELS / "rf.joblib")
    booster = Booster(); booster.load_model(str(MODELS / "xgb_base.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts_path = MODELS / "dqn_policy.ts"
    pt_path = MODELS / "dqn_policy.pt"
    if ts_path.exists():
        policy = torch.jit.load(str(ts_path), map_location=device).eval()
        scripted = True
    elif pt_path.exists():
        from src.rl_train_dqn import DuelingQ
        policy = DuelingQ(1,3).to(device).eval()  # dummy, will be replaced after we know obs_dim
        policy.load_state_dict(torch.load(pt_path, map_location=device))
        scripted = False
    else:
        policy = None; scripted = False
    return lr, rf, booster, policy, scripted, device

def prf(y_true, y_pred):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    return p, r, f

def main():
    # -------- data & split ----------
    df = pd.read_csv(DATA)
    sp = chrono_split(df)
    Xtr = add_derived(sp.X_train); Xva = add_derived(sp.X_val); Xte = add_derived(sp.X_test)
    yva = sp.y_val.values.astype(int)
    yte = sp.y_test.values.astype(int)

    # -------- artifacts -------------
    lr, rf, booster, policy, scripted, device = load_artifacts()
    if policy is None:
        raise FileNotFoundError("Missing RL policy (models/dqn_policy.ts or .pt); train/export first.")

    # -------- baselines @ FPR≈1% ----
    lr_va  = lr.predict_proba(Xva)[:,1];   lr_te  = lr.predict_proba(Xte)[:,1]
    rf_va  = rf.predict_proba(Xva)[:,1];   rf_te  = rf.predict_proba(Xte)[:,1]
    xgb_va = booster.predict(DMatrix(Xva)); xgb_te = booster.predict(DMatrix(Xte))

    thr_lr  = threshold_at_fpr(yva, lr_va, 0.01)
    thr_rf  = threshold_at_fpr(yva, rf_va, 0.01)
    thr_xgb = threshold_at_fpr(yva, xgb_va, 0.01)

    yhat_lr  = (lr_te  >= thr_lr ).astype(int)
    yhat_rf  = (rf_te  >= thr_rf ).astype(int)
    yhat_xgb = (xgb_te >= thr_xgb).astype(int)

    # -------- RL via ENV ROLLOUT (CRITICAL) ----------
    # Build XGB scores for the test set (teacher feature)
    xgb_scores_te = booster.predict(DMatrix(Xte)).astype(np.float32)

    # Create an environment that streams through the entire test set once
    env_te = CreditFraudEnv(
        X=Xte, y=sp.y_test, xgb_scores=xgb_scores_te,
        review_rate=0.05,                 # same as training/validation
        episode_len=len(Xte),
        seed=999, strict_budget=True
    )

    # Ensure policy has correct input dim if we loaded .pt
    obs_dim = env_te.observation_space.shape[0]
    if not scripted and isinstance(policy, torch.nn.Module) and policy.backbone[0].in_features != obs_dim:
        from src.rl_train_dqn import DuelingQ
        policy = DuelingQ(obs_dim, env_te.action_space.n).to(device).eval()
        policy.load_state_dict(torch.load(MODELS / "dqn_policy.pt", map_location=device))

    # Rollout deterministically (eps ~ 0)
    acts = []
    obs,_ = env_te.reset()
    with torch.inference_mode():
        for _ in range(len(Xte)):
            q = policy(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            a = int(q.argmax(1).item())
            acts.append(a)
            obs, r, done, _, _ = env_te.step(a)
            if done:
                break

    acts = np.array(acts, dtype=int)
    yhat_rl_block = (acts == 2).astype(int)                    # BLOCK only
    yhat_rl_block_or_review = np.isin(acts, [1,2]).astype(int) # BLOCK ∪ REVIEW

    # -------- metrics --------------
    results = {
        "LogReg":              dict(zip(["Precision","Recall","F1"], prf(yte, yhat_lr))),
        "RandomForest":        dict(zip(["Precision","Recall","F1"], prf(yte, yhat_rf))),
        "XGBoost":             dict(zip(["Precision","Recall","F1"], prf(yte, yhat_xgb))),
        "RL (BLOCK)":          dict(zip(["Precision","Recall","F1"], prf(yte, yhat_rl_block))),
        "RL (BLOCK|REVIEW)":   dict(zip(["Precision","Recall","F1"], prf(yte, yhat_rl_block_or_review))),
        "thresholds_val_FPR1pct": {"LogReg": thr_lr, "RandomForest": thr_rf, "XGBoost": thr_xgb},
        "rl_rates": {
            "block_rate": float((acts==2).mean()),
            "review_rate": float((acts==1).mean()),
            "approve_rate": float((acts==0).mean())
        }
    }

    print("\n=== Low-FPR (~1%) F1 comparison (TEST) — ENV-CONSISTENT ===")
    for k in ["LogReg","RandomForest","XGBoost","RL (BLOCK)","RL (BLOCK|REVIEW)"]:
        r = results[k]
        print(f"{k:16s}  P={r['Precision']:.3f}  R={r['Recall']:.3f}  F1={r['F1']:.3f}")
    print("\nVAL thresholds @≈1% FPR:", json.dumps(results["thresholds_val_FPR1pct"], indent=2))
    print("RL action rates:", results["rl_rates"])

    (MODELS / "f1_results_env.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame({k: v for k, v in results.items() if "thresholds" not in k and "rl_rates" not in k}).T.to_csv(
        MODELS / "f1_results_env.csv"
    )

    # ------- Visuals -------
    labels = ["LogReg", "RandomForest", "XGBoost", "RL (BLOCK)", "RL (BLOCK|REVIEW)"]
    f1vals = [results[m]["F1"] for m in labels]
    colors = ["#888","#888","#888","#0aa6b7","#32c3cf"]

    plt.figure(figsize=(7.6, 4.4))
    bars = plt.bar(labels, f1vals, color=colors)
    plt.ylabel("F1 score (Test)")
    plt.title("F1 at Low-FPR (~1%) — Baselines vs RL (env-aware)")
    for b, v in zip(bars, f1vals):
        plt.text(b.get_x()+b.get_width()/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    plt.ylim(0, max(0.1, max(f1vals) + 0.15))
    plt.tight_layout()
    out1 = FIGS / "f1_bars_lowFPR_env.png"
    plt.savefig(out1, dpi=200); plt.close()

    print(f"\nSaved figure: {out1}")

if __name__ == "__main__":
    main()
