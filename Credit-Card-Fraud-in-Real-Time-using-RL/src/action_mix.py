# src/action_mix.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve

# local imports
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
    # baselines
    lr = joblib.load(MODELS / "lr.joblib")
    rf = joblib.load(MODELS / "rf.joblib")
    booster = Booster(); booster.load_model(str(MODELS / "xgb_base.json"))
    # RL (prefer TorchScript)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts = MODELS/"dqn_policy.ts"; pt = MODELS/"dqn_policy.pt"
    if ts.exists():
        policy = torch.jit.load(str(ts), map_location=device).eval()
        scripted = True
    elif pt.exists():
        policy = ("pt_fallback", device)  # lazy init with proper obs_dim
        scripted = False
    else:
        policy = None; scripted = False
    return lr, rf, booster, policy, scripted, device

def ensure_pt(policy_sentinel, obs_dim, device):
    # instantiate DuelingQ for .pt fallback
    from src.rl_train_dqn import DuelingQ
    net = DuelingQ(obs_dim, 3).to(device).eval()
    net.load_state_dict(torch.load(MODELS/"dqn_policy.pt", map_location=device))
    return net

def main():
    # -------- data --------
    df = pd.read_csv(DATA)
    sp = chrono_split(df)                                # chronological split (train/val/test)
    Xtr = add_derived(sp.X_train); Xva = add_derived(sp.X_val); Xte = add_derived(sp.X_test)
    yva = sp.y_val.values.astype(int); yte = sp.y_test.values.astype(int)

    # -------- models --------
    lr, rf, booster, policy, scripted, device = load_artifacts()
    if policy is None:
        raise FileNotFoundError("Missing RL policy (models/dqn_policy.ts or .pt). Train/export first.")

    # -------- RL rollout (env-aware: budget updates each step) --------
    xgb_scores_te = booster.predict(DMatrix(Xte)).astype(np.float32)
    env = CreditFraudEnv(
        X=Xte, y=sp.y_test, xgb_scores=xgb_scores_te,
        review_rate=0.05, episode_len=len(Xte), seed=999, strict_budget=True
    )
    obs,_ = env.reset()
    obs_dim = env.observation_space.shape[0]
    if isinstance(policy, tuple):                 # .pt fallback — build net with correct input size
        policy = ensure_pt(policy, obs_dim, device); scripted = False

    acts = []
    with torch.inference_mode():
        for _ in range(len(Xte)):
            q = policy(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            a = int(q.argmax(1).item())          # 0=approve, 1=review, 2=block
            acts.append(a)
            obs, r, done, _, _ = env.step(a)
            if done: break
    acts = np.array(acts, dtype=int)
    n = len(acts)
    rl_rates = {
        "approve_rate": float((acts==0).mean()),
        "review_rate":  float((acts==1).mean()),
        "block_rate":   float((acts==2).mean())
    }

    # -------- Baseline alert (block) rates @ ~1% FPR (optional comparison) --------
    lr_va  = lr.predict_proba(Xva)[:,1];   lr_te  = lr.predict_proba(Xte)[:,1]
    rf_va  = rf.predict_proba(Xva)[:,1];   rf_te  = rf.predict_proba(Xte)[:,1]
    xg_va  = booster.predict(DMatrix(Xva)); xg_te  = booster.predict(DMatrix(Xte))

    thr_lr  = thr_at_fpr(yva, lr_va, 0.01)
    thr_rf  = thr_at_fpr(yva, rf_va, 0.01)
    thr_xgb = thr_at_fpr(yva, xg_va, 0.01)

    bl_rates = {
        "LogReg_block_rate": float((lr_te  >= thr_lr ).mean()),
        "RandomForest_block_rate": float((rf_te  >= thr_rf ).mean()),
        "XGBoost_block_rate": float((xg_te  >= thr_xgb).mean())
    }

    # -------- Print + save numerics --------
    out = {"rl_action_rates": rl_rates, "baseline_block_rates": bl_rates,
           "val_thresholds_fpr1pct": {"LogReg": thr_lr, "RandomForest": thr_rf, "XGBoost": thr_xgb}}
    print(json.dumps(out, indent=2))
    (MODELS/"action_mix_results.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame([rl_rates]).to_csv(MODELS/"action_mix_rl.csv", index=False)
    pd.DataFrame([bl_rates]).to_csv(MODELS/"action_mix_baselines.csv", index=False)

    # -------- Plot 1: RL Action Mix (bar with % labels) --------
    labels = ["Approve","Review","Block"]
    vals   = [rl_rates["approve_rate"], rl_rates["review_rate"], rl_rates["block_rate"]]

    plt.figure(figsize=(6.6,4.2))
    bars = plt.bar(labels, vals, color=["#94a3b8","#f59e0b","#0aa6b7"])
    plt.ylim(0,1.0); plt.ylabel("Fraction of transactions")
    plt.title("RL Action Mix")
    for b,v in zip(bars, vals):
        plt.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2%}", ha="center")
    plt.tight_layout()
    out1 = FIGS/"action_mix_rl.png"
    plt.savefig(out1, dpi=200); plt.close()

    # -------- Plot 2: Baseline alert (block) rates vs RL --------
    labels2 = ["LogReg (alert)", "RandomForest (alert)", "XGBoost (alert)", "RL (block)", "RL (review)"]
    vals2   = [bl_rates["LogReg_block_rate"],
               bl_rates["RandomForest_block_rate"],
               bl_rates["XGBoost_block_rate"],
               rl_rates["block_rate"],
               rl_rates["review_rate"]]
    colors2 = ["#888888","#888888","#888888","#0aa6b7","#f59e0b"]

    plt.figure(figsize=(8.2,4.2))
    bars2 = plt.bar(labels2, vals2, color=colors2)
    plt.ylim(0, max(max(vals2)*1.2, 0.05))
    plt.ylabel("Rate")
    plt.title("Alert/Block Rates at ~1% FPR (Baselines) vs RL Triage")
    for b,v in zip(bars2, vals2):
        plt.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2%}", ha="center", fontsize=9)
    plt.tight_layout()
    out2 = FIGS/"action_mix_comparison.png"
    plt.savefig(out2, dpi=200); plt.close()

    print(f"\nSaved:\n  {out1}\n  {out2}\n  {MODELS/'action_mix_results.json'}")

if __name__ == "__main__":
    main()
