import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, confusion_matrix

# imports
from src.data_loader import chrono_split
from src.features import add_derived
from src.rl_env import CreditFraudEnv

import joblib
from xgboost import Booster, DMatrix
import torch

ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data" / "creditcard.csv"
MODELS = ROOT / "models"
FIGS   = MODELS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

def threshold_at_fpr(y_true, scores, target_fpr=0.01):
    fpr, tpr, thr = roc_curve(y_true, scores)
    i = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(thr[i])

def main():
    # -------- load data --------
    df = pd.read_csv(DATA)
    sp = chrono_split(df)
    Xtr = add_derived(sp.X_train); Xva = add_derived(sp.X_val); Xte = add_derived(sp.X_test)
    yva = sp.y_val.values.astype(int)
    yte = sp.y_test.values.astype(int)

    # -------- load models --------
    lr = joblib.load(MODELS / "lr.joblib")
    rf = joblib.load(MODELS / "rf.joblib")
    booster = Booster(); booster.load_model(str(MODELS / "xgb_base.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts = MODELS/"dqn_policy.ts"; pt = MODELS/"dqn_policy.pt"
    if ts.exists():
        policy = torch.jit.load(str(ts), map_location=device).eval()
    elif pt.exists():
        from src.rl_train_dqn import DuelingQ
        policy = DuelingQ(1,3).to(device).eval()  # will fix dims later
        policy.load_state_dict(torch.load(pt, map_location=device))
    else:
        raise FileNotFoundError("Missing RL policy (ts/pt)")

    # -------- baselines: scores & thresholds --------
    lr_va = lr.predict_proba(Xva)[:,1]; lr_te = lr.predict_proba(Xte)[:,1]
    rf_va = rf.predict_proba(Xva)[:,1]; rf_te = rf.predict_proba(Xte)[:,1]
    xgb_va= booster.predict(DMatrix(Xva)); xgb_te= booster.predict(DMatrix(Xte))

    thr_lr  = threshold_at_fpr(yva, lr_va, 0.01)
    thr_rf  = threshold_at_fpr(yva, rf_va, 0.01)
    thr_xgb = threshold_at_fpr(yva, xgb_va, 0.01)

    yhat_lr  = (lr_te  >= thr_lr ).astype(int)
    yhat_rf  = (rf_te  >= thr_rf ).astype(int)
    yhat_xgb = (xgb_te >= thr_xgb).astype(int)

    # -------- RL rollout --------
    xgb_scores_te = booster.predict(DMatrix(Xte)).astype(np.float32)
    env = CreditFraudEnv(
        X=Xte, y=sp.y_test, xgb_scores=xgb_scores_te,
        review_rate=0.05, episode_len=len(Xte),
        seed=42, strict_budget=True
    )
    obs,_ = env.reset()
    acts=[]
    with torch.inference_mode():
        for _ in range(len(Xte)):
            q = policy(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            a = int(q.argmax(1).item())
            acts.append(a)
            obs, r, done, _, _ = env.step(a)
            if done: break
    acts = np.array(acts)
    yhat_rl_block = (acts==2).astype(int)

    # -------- confusion matrices --------
    cms = {}
    for name, pred in [
        ("LogReg", yhat_lr),
        ("RandomForest", yhat_rf),
        ("XGBoost", yhat_xgb),
        ("RL (Block)", yhat_rl_block)
    ]:
        tn, fp, fn, tp = confusion_matrix(yte[:len(pred)], pred).ravel()
        cms[name] = {"TP":int(tp),"FP":int(fp),"TN":int(tn),"FN":int(fn)}
        print(f"\n{name} Confusion Matrix:")
        print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    (MODELS/"confusion_matrices.json").write_text(json.dumps(cms, indent=2))
    pd.DataFrame(cms).T.to_csv(MODELS/"confusion_matrices.csv")

    # -------- plot side-by-side heatmaps --------
    fig, axes = plt.subplots(1,4, figsize=(14,4))
    for ax,(name,cm) in zip(axes, cms.items()):
        mat = np.array([[cm["TN"], cm["FP"]],
                        [cm["FN"], cm["TP"]]])
        sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Pred:Legit","Pred:Fraud"],
                    yticklabels=["True:Legit","True:Fraud"], ax=ax)
        ax.set_title(name)
    plt.tight_layout()
    out = FIGS/"confusion_matrices.png"
    plt.savefig(out, dpi=200); plt.close()
    print(f"\nSaved plot -> {out}")

if __name__ == "__main__":
    main()
