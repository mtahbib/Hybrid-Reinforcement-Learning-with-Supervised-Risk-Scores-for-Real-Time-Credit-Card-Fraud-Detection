import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve

# local imports (
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

# ---------- business reward schema ----------
RC = dict(
    tp_block=+5.0,   # correct block on fraud
    fp_block=-2.0,   # wrong block on legit
    fn=-10.0,        # approved a fraud
    tn=+0.2,         # approved a legit
    review_cost=-0.2,
    review_catch=+3.0,
    over_budget=-5.0 # not used if strict_budget=True
)

def reward_from_actions(y_true, actions):
    """
    actions: 0=Approve, 1=Review, 2=Block
    returns total reward/utility over the sequence
    """
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(actions, dtype=int)
    reward = 0.0
    for ai, yi in zip(a, y):
        if ai == 2:   # BLOCK
            reward += RC["tp_block"] if yi == 1 else RC["fp_block"]
        elif ai == 1: # REVIEW
            reward += RC["review_cost"] + (RC["review_catch"] if yi == 1 else 0.0)
        else:         # APPROVE
            reward += RC["tn"] if yi == 0 else RC["fn"]
    return float(reward)

def reward_per_1k(y_true, actions):
    tot = reward_from_actions(y_true, actions)
    n = len(y_true)
    return 1000.0 * tot / n

def threshold_at_fpr(y_true, scores, target_fpr=0.01):
    fpr, tpr, thr = roc_curve(y_true, scores)
    i = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(thr[i])

def main():
    # ---------- data & split ----------
    df = pd.read_csv(DATA)
    sp = chrono_split(df)
    Xtr = add_derived(sp.X_train); Xva = add_derived(sp.X_val); Xte = add_derived(sp.X_test)
    yva = sp.y_val.values.astype(int)
    yte = sp.y_test.values.astype(int)

    # ---------- load models ----------
    lr = joblib.load(MODELS / "lr.joblib")
    rf = joblib.load(MODELS / "rf.joblib")
    booster = Booster(); booster.load_model(str(MODELS / "xgb_base.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prefer TorchScript RL; fallback to .pt
    policy = None; scripted = False
    ts = MODELS/"dqn_policy.ts"; pt = MODELS/"dqn_policy.pt"
    if ts.exists():
        policy = torch.jit.load(str(ts), map_location=device).eval()
        scripted = True
    elif pt.exists():
        from src.rl_train_dqn import DuelingQ
        policy = ("pt_fallback", device)
        scripted = False
    else:
        raise FileNotFoundError("Missing RL policy (models/dqn_policy.ts or .pt). Train/export first.")

    # ---------- baseline scores ----------
    lr_va  = lr.predict_proba(Xva)[:,1];   lr_te  = lr.predict_proba(Xte)[:,1]
    rf_va  = rf.predict_proba(Xva)[:,1];   rf_te  = rf.predict_proba(Xte)[:,1]
    xgb_va = booster.predict(DMatrix(Xva)); xgb_te = booster.predict(DMatrix(Xte))

    # thresholds at ~1% FPR (picked on validation)
    thr_lr  = threshold_at_fpr(yva, lr_va, 0.01)
    thr_rf  = threshold_at_fpr(yva, rf_va, 0.01)
    thr_xgb = threshold_at_fpr(yva, xgb_va, 0.01)

    # map baselines to actions (2=BLOCK if ≥ thr, else 0=APPROVE)
    acts_lr   = np.where(lr_te  >= thr_lr ,  2, 0)
    acts_rf   = np.where(rf_te  >= thr_rf ,  2, 0)
    acts_xgb  = np.where(xgb_te >= thr_xgb,  2, 0)

    # ---------- RL via environment (budget-aware rollout) ----------
    xgb_scores_te = booster.predict(DMatrix(Xte)).astype(np.float32)
    env_te = CreditFraudEnv(
        X=Xte, y=sp.y_test, xgb_scores=xgb_scores_te,
        review_rate=0.05, episode_len=len(Xte),
        seed=999, strict_budget=True
    )
    obs_dim = env_te.observation_space.shape[0]
    if isinstance(policy, tuple):
        from src.rl_train_dqn import DuelingQ
        net = DuelingQ(obs_dim, env_te.action_space.n).to(device).eval()
        net.load_state_dict(torch.load(MODELS/"dqn_policy.pt", map_location=device))
    else:
        net = policy

    acts_rl = []
    obs,_ = env_te.reset()
    with torch.inference_mode():
        for _ in range(len(Xte)):
            q = net(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            a = int(q.argmax(1).item())
            acts_rl.append(a)
            obs, r, done, _, _ = env_te.step(a)
            if done: break
    acts_rl = np.asarray(acts_rl, dtype=int)

    # ---------- rewards ----------
    reward_lr   = reward_per_1k(yte, acts_lr)
    reward_rf   = reward_per_1k(yte, acts_rf)
    reward_xgb  = reward_per_1k(yte, acts_xgb)
    reward_rl   = reward_per_1k(yte[:len(acts_rl)], acts_rl)  # align lengths exactly

    results = {
        "reward_per_1k": {
            "LogReg": reward_lr,
            "RandomForest": reward_rf,
            "XGBoost": reward_xgb,
            "RL": reward_rl
        },
        "thresholds_val_FPR1pct": {
            "LogReg": thr_lr, "RandomForest": thr_rf, "XGBoost": thr_xgb
        },
        "rl_action_rates": {
            "approve_rate": float((acts_rl==0).mean()),
            "review_rate":  float((acts_rl==1).mean()),
            "block_rate":   float((acts_rl==2).mean())
        }
    }

    print("\n=== Reward per 1,000 Transactions (TEST) ===")
    for k, v in results["reward_per_1k"].items():
        print(f"{k:12s} : {v:+.2f}")
    print("\nVAL thresholds @≈1% FPR:", json.dumps(results["thresholds_val_FPR1pct"], indent=2))
    print("RL action rates:", results["rl_action_rates"])

    # save numerics
    (MODELS/"reward_results.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame([results["reward_per_1k"]]).to_csv(MODELS/"reward_results.csv", index=False)

    # ---------- bar chart ----------
    labels = ["LogReg", "RandomForest", "XGBoost", "RL"]
    vals   = [reward_lr, reward_rf, reward_xgb, reward_rl]
    colors = ["#888888", "#888888", "#888888", "#0aa6b7"]

    plt.figure(figsize=(6.8, 4.2))
    bars = plt.bar(labels, vals, color=colors)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Reward per 1,000 transactions (higher = better)")
    plt.title("Business Utility — Reward/1k ")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width()/2, v + (2.0 if v>=0 else -4.0), f"{v:+.1f}",
                 ha="center", va="bottom" if v>=0 else "top", fontsize=10)
    pad = max(10.0, 0.15*max(abs(np.max(vals)), abs(np.min(vals))))
    plt.ylim(min(vals)-pad, max(vals)+pad)
    plt.tight_layout()
    out = FIGS/"reward_per_11k.png"
    plt.savefig(out, dpi=200); plt.close()
    print(f"\nSaved figure -> {out}")
    print(f"Saved CSV    -> {MODELS/'reward_results1.csv'}")
    print(f"Saved JSON   -> {MODELS/'reward_results1.json'}")

if __name__ == "__main__":
    main()
