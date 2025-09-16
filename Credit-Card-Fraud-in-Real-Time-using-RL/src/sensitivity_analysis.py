# src/reward_breakdown.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import joblib, torch
from xgboost import Booster, DMatrix

from src.data_loader import chrono_split
from src.features import add_derived
from src.rl_env import CreditFraudEnv

ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data" / "creditcard.csv"
MODELS = ROOT / "models"
FIGS   = MODELS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# ---- reward schema ----
RC = dict(tp_block=+5.0, fp_block=-2.0, fn=-10.0, tn=+0.2,
          review_cost=-0.2, review_catch=+3.0)

def thr_at_fpr(y, s, f=0.01):
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.argmin(np.abs(fpr - f)))
    return float(thr[i])

def baseline_actions(y_true, scores, thr):
    # 2=BLOCK (alert), 0=APPROVE
    return np.where(scores >= thr, 2, 0).astype(int)

def reward_components(y, actions, rc):
    """
    Return dict with component sums:
    block_TP, block_FP, approve_TN, approve_FN, review_legit, review_fraud,
    total_reward, reward_per_1k
    """
    y = np.asarray(y, dtype=int)
    a = np.asarray(actions, dtype=int)
    comp = dict(block_TP=0, block_FP=0, approve_TN=0,
                approve_FN=0, review_legit=0, review_fraud=0)
    reward = 0.0
    for ai, yi in zip(a, y):
        if ai==2:   # BLOCK
            if yi==1:
                comp["block_TP"] += 1
                reward += rc["tp_block"]
            else:
                comp["block_FP"] += 1
                reward += rc["fp_block"]
        elif ai==1: # REVIEW
            if yi==1:
                comp["review_fraud"] += 1
                reward += rc["review_cost"] + rc["review_catch"]
            else:
                comp["review_legit"] += 1
                reward += rc["review_cost"]
        else:       # APPROVE
            if yi==1:
                comp["approve_FN"] += 1
                reward += rc["fn"]
            else:
                comp["approve_TN"] += 1
                reward += rc["tn"]
    comp["total_reward"] = float(reward)
    comp["reward_per_1k"] = 1000.0 * reward / len(y)
    return comp

def rollout_rl_actions(Xte, yte, booster, policy):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    xgb_te = booster.predict(DMatrix(Xte)).astype(np.float32)
    env = CreditFraudEnv(Xte, pd.Series(yte), xgb_te,
                         review_rate=0.05, episode_len=len(Xte),
                         seed=42, strict_budget=True)
    obs,_ = env.reset()
    acts=[]
    with torch.inference_mode():
        for _ in range(len(Xte)):
            q = policy(torch.as_tensor(obs, dtype=torch.float32,
                                       device=device).unsqueeze(0))
            a = int(q.argmax(1).item())
            acts.append(a)
            obs, r, done, _, _ = env.step(a)
            if done: break
    return np.array(acts), xgb_te

def load_models():
    lr = joblib.load(MODELS / "lr.joblib")
    rf = joblib.load(MODELS / "rf.joblib")
    booster = Booster(); booster.load_model(str(MODELS / "xgb_base.json"))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ts = MODELS/"dqn_policy.ts"; pt = MODELS/"dqn_policy.pt"
    if ts.exists():
        policy = torch.jit.load(str(ts), map_location=device).eval()
    else:
        from src.rl_train_dqn import DuelingQ
        if not pt.exists():
            raise FileNotFoundError("Missing RL policy (dqn_policy.ts or dqn_policy.pt).")
        # fallback to pt: will rebuild later with obs_dim
        policy = ("pt_fallback", device)
    return lr, rf, booster, policy

def ensure_pt(policy, obs_dim):
    if isinstance(policy, tuple):
        from src.rl_train_dqn import DuelingQ
        device = policy[1]
        net = DuelingQ(obs_dim, 3).to(device).eval()
        net.load_state_dict(torch.load(MODELS/"dqn_policy.pt", map_location=device))
        return net
    return policy

def main():
    # ---- data ----
    df = pd.read_csv(DATA)
    sp = chrono_split(df)
    Xtr = add_derived(sp.X_train); Xva = add_derived(sp.X_val); Xte = add_derived(sp.X_test)
    yva = sp.y_val.values.astype(int); yte = sp.y_test.values.astype(int)

    # ---- models ----
    lr, rf, booster, policy = load_models()

    # ---- baseline scores & thresholds ----
    lr_va = lr.predict_proba(Xva)[:,1];   lr_te = lr.predict_proba(Xte)[:,1]
    rf_va = rf.predict_proba(Xva)[:,1];   rf_te = rf.predict_proba(Xte)[:,1]
    xg_va = booster.predict(DMatrix(Xva)); xg_te = booster.predict(DMatrix(Xte))

    thr_lr  = thr_at_fpr(yva, lr_va, 0.01)
    thr_rf  = thr_at_fpr(yva, rf_va, 0.01)
    thr_xgb = thr_at_fpr(yva, xg_va, 0.01)

    acts_lr  = baseline_actions(yte, lr_te,  thr_lr)
    acts_rf  = baseline_actions(yte, rf_te,  thr_rf)
    acts_xgb = baseline_actions(yte, xg_te,  thr_xgb)

    # ---- RL rollout ----
    from src.rl_env import CreditFraudEnv
    xgb_te_tmp = booster.predict(DMatrix(Xte.iloc[:5]))
    env_tmp = CreditFraudEnv(Xte.iloc[:5], pd.Series(yte[:5]),
                             xgb_te_tmp.astype(np.float32), 0.05, 5, 1, True)
    obs_dim = env_tmp.observation_space.shape[0]
    policy = ensure_pt(policy, obs_dim)
    acts_rl, _ = rollout_rl_actions(Xte, yte, booster, policy)

    # ---- reward components ----
    comp = {}
    comp["LogReg"]        = reward_components(yte, acts_lr, RC)
    comp["RandomForest"]  = reward_components(yte, acts_rf, RC)
    comp["XGBoost"]       = reward_components(yte, acts_xgb, RC)
    comp["RL"]            = reward_components(yte[:len(acts_rl)], acts_rl, RC)

    print("\n=== Reward components (current schema) ===")
    for k,v in comp.items():
        print(f"\n{k}")
        print({kk:int(v[kk]) for kk in ["block_TP","block_FP","approve_TN",
                                        "approve_FN","review_legit","review_fraud"]})
        print(f"reward_per_1k={v['reward_per_1k']:+.2f}")

    # save numerics
    (MODELS/"reward_components.json").write_text(json.dumps(comp, indent=2))
    pd.DataFrame(comp).T.to_csv(MODELS/"reward_components.csv")

    # ---- sensitivity analysis ----
    fp_grid   = [-6,-4,-2,-1]         # harsher FP penalties
    rev_grid  = [-0.30,-0.20,-0.10,0] # cheaper reviews help RL
    table = []
    for fp in fp_grid:
        for rcost in rev_grid:
            rc = dict(RC); rc["fp_block"]=fp; rc["review_cost"]=rcost
            row = {"fp_block": fp, "review_cost": rcost}
            for name, acts in [("LogReg",acts_lr),("RandomForest",acts_rf),
                               ("XGBoost",acts_xgb),("RL",acts_rl)]:
                row[name] = reward_components(yte[:len(acts)], acts, rc)["reward_per_1k"]
            table.append(row)
    df_sens = pd.DataFrame(table)
    df_sens.to_csv(MODELS/"reward_sensitivity.csv", index=False)

    # RL advantage over best baseline
    best_base = df_sens[["LogReg","RandomForest","XGBoost"]].max(axis=1)
    rl_adv = df_sens["RL"] - best_base
    df_sens["RL_minus_bestBaseline"] = rl_adv

    pivot = df_sens.pivot(index="fp_block", columns="review_cost",
                          values="RL_minus_bestBaseline")
    plt.figure(figsize=(6,4))
    im = plt.imshow(pivot.values, cmap="coolwarm", origin="lower",
                    extent=[min(rev_grid)-0.05,max(rev_grid)+0.05,
                            min(fp_grid)-0.5,max(fp_grid)+0.5],
                    aspect="auto")
    plt.colorbar(im,label="RL Reward/1k − Best Baseline (higher = better)")
    plt.xlabel("review_cost")
    plt.ylabel("fp_block")
    plt.title("Sensitivity Analysis: RL vs Baselines (Reward/1k)")
    xt = sorted(rev_grid); yt = sorted(fp_grid)
    plt.xticks(xt); plt.yticks(yt)
    out_heat = FIGS/"reward_sensitivity_heatmap1.png"
    plt.tight_layout(); plt.savefig(out_heat, dpi=200); plt.close()

    print(f"\nSaved: {MODELS/'reward_components.json'}, {MODELS/'reward_components1.csv'}")
    print(f"Saved sensitivity grid: {MODELS/'reward_sensitivity1.csv'}")
    print(f"Saved heatmap: {out_heat}")

if __name__ == "__main__":
    main()
