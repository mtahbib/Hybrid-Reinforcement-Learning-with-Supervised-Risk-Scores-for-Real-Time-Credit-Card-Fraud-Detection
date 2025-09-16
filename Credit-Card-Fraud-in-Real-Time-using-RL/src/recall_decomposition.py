# src/recall_decomposition.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, recall_score

# Local imports (run with: python -u -m src.recall_decomposition)
from src.data_loader import chrono_split
from src.features import add_derived
from src.rl_env import CreditFraudEnv

import joblib, torch
from xgboost import Booster, DMatrix

ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data" / "creditcard.csv"
MODELS = ROOT / "models"
FIGS   = MODELS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

def thr_at_fpr(y_true, scores, target_fpr=0.01):
    fpr, tpr, thr = roc_curve(y_true, scores)
    i = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(thr[i])

def load_artifacts():
    lr = joblib.load(MODELS/"lr.joblib")
    rf = joblib.load(MODELS/"rf.joblib")
    booster = Booster(); booster.load_model(str(MODELS/"xgb_base.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts = MODELS/"dqn_policy.ts"; pt = MODELS/"dqn_policy.pt"
    if ts.exists():
        policy = torch.jit.load(str(ts), map_location=device).eval()
        scripted = True
    elif pt.exists():
        from src.rl_train_dqn import DuelingQ
        policy = ("pt_fallback", device)  # lazy init with correct obs_dim later
        scripted = False
    else:
        raise FileNotFoundError("Missing RL policy (dqn_policy.ts/.pt)")
    return lr, rf, booster, policy, scripted, device

def ensure_pt(policy_sentinel, obs_dim, device):
    from src.rl_train_dqn import DuelingQ
    net = DuelingQ(obs_dim, 3).to(device).eval()
    net.load_state_dict(torch.load(MODELS/"dqn_policy.pt", map_location=device))
    return net

def main():
    # ---------- data ----------
    df = pd.read_csv(DATA)
    sp = chrono_split(df)
    Xtr = add_derived(sp.X_train); Xva = add_derived(sp.X_val); Xte = add_derived(sp.X_test)
    yva = sp.y_val.values.astype(int)
    yte = sp.y_test.values.astype(int)

    # ---------- models ----------
    lr, rf, booster, policy, scripted, device = load_artifacts()

    # ---------- baselines: best recall at ~1% FPR (choose best of LR/RF/XGB) ----------
    lr_va = lr.predict_proba(Xva)[:,1];   lr_te = lr.predict_proba(Xte)[:,1]
    rf_va = rf.predict_proba(Xva)[:,1];   rf_te = rf.predict_proba(Xte)[:,1]
    xg_va = booster.predict(DMatrix(Xva)); xg_te = booster.predict(DMatrix(Xte))

    thr_lr  = thr_at_fpr(yva, lr_va, 0.01)
    thr_rf  = thr_at_fpr(yva, rf_va, 0.01)
    thr_xgb = thr_at_fpr(yva, xg_va, 0.01)

    yhat_lr  = (lr_te  >= thr_lr ).astype(int)
    yhat_rf  = (rf_te  >= thr_rf ).astype(int)
    yhat_xgb = (xg_te  >= thr_xgb).astype(int)

    rec_lr  = recall_score(yte, yhat_lr)
    rec_rf  = recall_score(yte, yhat_rf)
    rec_xgb = recall_score(yte, yhat_xgb)

    base_best_recall = max(rec_lr, rec_rf, rec_xgb)
    base_name = ["LogReg","RandomForest","XGBoost"][np.argmax([rec_lr,rec_rf,rec_xgb])]

    # ---------- RL rollout (budget-aware) ----------
    xgb_scores_te = booster.predict(DMatrix(Xte)).astype(np.float32)
    env = CreditFraudEnv(
        X=Xte, y=sp.y_test, xgb_scores=xgb_scores_te,
        review_rate=0.05, episode_len=len(Xte), seed=123, strict_budget=True
    )
    obs,_ = env.reset()
    obs_dim = env.observation_space.shape[0]
    if isinstance(policy, tuple):
        policy = ensure_pt(policy, obs_dim, device); scripted=False

    acts = []
    with torch.inference_mode():
        for _ in range(len(Xte)):
            q = policy(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            a = int(q.argmax(1).item())   # 0 approve, 1 review, 2 block
            acts.append(a)
            obs, r, done, _, _ = env.step(a)
            if done: break
    acts = np.array(acts, dtype=int)
    y = yte[:len(acts)]

    # Decompose RL recall
    total_fraud = (y==1).sum()
    tp_block  = int(((acts==2) & (y==1)).sum())
    tp_review = int(((acts==1) & (y==1)).sum())
    rl_block_recall  = tp_block  / total_fraud if total_fraud>0 else 0.0
    rl_review_recall = tp_review / total_fraud if total_fraud>0 else 0.0
    rl_total_recall  = rl_block_recall + rl_review_recall

    results = {
        "baseline_best": {"name": base_name, "recall": base_best_recall},
        "rl": {
            "block_recall": rl_block_recall,
            "review_recall": rl_review_recall,
            "total_recall": rl_total_recall,
            "tp_block": tp_block,
            "tp_review": tp_review,
            "total_fraud": int(total_fraud)
        }
    }
    print(json.dumps(results, indent=2))

    # ---------- Stacked bar figure ----------
    labels = ["Baseline (best @~1% FPR)", "RL"]
    # Baseline has no review stage → all 'block' (single segment)
    base_block = base_best_recall
    rl_block   = rl_block_recall
    rl_review  = rl_review_recall

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.bar(["Baseline (best)"], [base_block], color="#7e7e7e", label="Caught (BLOCK)")
    ax.bar(["RL"], [rl_block], color="#0aa6b7", label="Caught by BLOCK")
    ax.bar(["RL"], [rl_review], bottom=[rl_block], color="orange", label="Caught by REVIEW")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall (fraction of all frauds caught)")
    ax.set_title("Recall Decomposition: RL (BLOCK vs REVIEW) vs Best Baseline")
    # Annotate percentages
    ax.text(0, base_block+0.02, f"{base_block:.2%}", ha="center")
    ax.text(1, rl_block/2, f"{rl_block:.2%}\nBLOCK", ha="center", color="white", fontsize=10)
    ax.text(1, rl_block + rl_review/2 + 0.01, f"{rl_review:.2%}\nREVIEW", ha="center", color="black", fontsize=10)
    ax.legend(loc="upper right")
    plt.tight_layout()
    out = FIGS / "recall_decomposition.png"
    plt.savefig(out, dpi=200); plt.close()
    print(f"Saved figure -> {out}")

if __name__ == "__main__":
    main()
