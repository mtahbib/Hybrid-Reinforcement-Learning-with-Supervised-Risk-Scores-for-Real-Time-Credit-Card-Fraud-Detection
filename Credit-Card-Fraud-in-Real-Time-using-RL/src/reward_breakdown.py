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

# ---- reward schema (current) ----
RC = dict(tp_block=+5.0, fp_block=-2.0, fn=-10.0, tn=+0.2,
          review_cost=-0.2, review_catch=+3.0)

def thr_at_fpr(y, s, f=0.01):
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.argmin(np.abs(fpr - f)))
    return float(thr[i])

def baseline_actions(scores, thr):
    # 2=BLOCK alert, 0=APPROVE (no review in baseline)
    return np.where(scores >= thr, 2, 0).astype(int)

def reward_components(y, actions, rc):
    """
    Return dict with component counts + totals:
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

def load_models_and_policy():
    lr = joblib.load(MODELS / "lr.joblib")
    rf = joblib.load(MODELS / "rf.joblib")
    booster = Booster(); booster.load_model(str(MODELS / "xgb_base.json"))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ts = MODELS/"dqn_policy.ts"; pt = MODELS/"dqn_policy.pt"
    if ts.exists():
        policy = torch.jit.load(str(ts), map_location=device).eval()
        scripted = True
    elif pt.exists():
        from src.rl_train_dqn import DuelingQ
        policy = DuelingQ(1,3).to(device).eval()  # temporary; rebuild later
        policy.load_state_dict(torch.load(pt, map_location=device))
        scripted = False
    else:
        raise FileNotFoundError("Missing RL policy (dqn_policy.ts or dqn_policy.pt).")
    return lr, rf, booster, policy, scripted, device

def rollout_rl_actions(Xte, yte, booster, policy, scripted, device):
    xgb_te = booster.predict(DMatrix(Xte)).astype(np.float32)
    env = CreditFraudEnv(Xte, pd.Series(yte), xgb_te,
                         review_rate=0.05, episode_len=len(Xte), seed=42)
    obs_dim = env.observation_space.shape[0]
    if not scripted and isinstance(policy, torch.nn.Module) and getattr(policy, "in_features", None) != obs_dim:
        from src.rl_train_dqn import DuelingQ
        policy = DuelingQ(obs_dim, env.action_space.n).to(device).eval()
        policy.load_state_dict(torch.load(MODELS/"dqn_policy.pt", map_location=device))

    acts = []
    obs,_ = env.reset()
    with torch.inference_mode():
        for _ in range(len(Xte)):
            q = policy(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            a = int(q.argmax(1).item())
            acts.append(a)
            obs, _, done, _, _ = env.step(a)
            if done: break
    return np.array(acts), xgb_te

def main():
    # ---- data ----
    df = pd.read_csv(DATA)
    sp = chrono_split(df)
    Xtr = add_derived(sp.X_train); Xva = add_derived(sp.X_val); Xte = add_derived(sp.X_test)
    yva = sp.y_val.values.astype(int); yte = sp.y_test.values.astype(int)

    # ---- models & policy ----
    lr, rf, booster, policy, scripted, device = load_models_and_policy()

    # ---- baseline scores & thresholds ----
    lr_va = lr.predict_proba(Xva)[:,1];   lr_te = lr.predict_proba(Xte)[:,1]
    rf_va = rf.predict_proba(Xva)[:,1];   rf_te = rf.predict_proba(Xte)[:,1]
    xg_va = booster.predict(DMatrix(Xva)); xg_te = booster.predict(DMatrix(Xte))

    thr_lr  = thr_at_fpr(yva, lr_va, 0.01)
    thr_rf  = thr_at_fpr(yva, rf_va, 0.01)
    thr_xgb = thr_at_fpr(yva, xg_va, 0.01)

    acts_lr  = baseline_actions(lr_te,  thr_lr)
    acts_rf  = baseline_actions(rf_te,  thr_rf)
    acts_xgb = baseline_actions(xg_te,  thr_xgb)

    # ---- RL rollout ----
    acts_rl, _ = rollout_rl_actions(Xte, yte, booster, policy, scripted, device)

    # ---- components ----
    comp = {
        "LogReg":       reward_components(yte, acts_lr,  RC),
        "RandomForest": reward_components(yte, acts_rf,  RC),
        "XGBoost":      reward_components(yte, acts_xgb, RC),
        "RL":           reward_components(yte[:len(acts_rl)], acts_rl, RC),
    }

    # ---- console output ----
    print("\n=== Reward components (current schema) ===")
    for name in ["LogReg","RandomForest","XGBoost","RL"]:
        v = comp[name]
        counts = {k:int(v[k]) for k in ["block_TP","block_FP","approve_TN","approve_FN","review_legit","review_fraud"]}
        print(f"\n{name}")
        print(counts)
        print(f"reward_per_1k={v['reward_per_1k']:+.2f}")

    # ---- save numerics ----
    (MODELS/"reward_components.json").write_text(json.dumps(comp, indent=2))
    pd.DataFrame(comp).T.to_csv(MODELS/"reward_components.csv")

    # ---------- VISUAL 1: stacked counts ----------
    order_models = ["LogReg","RandomForest","XGBoost","RL"]
    key_components = ["block_TP","block_FP","approve_FN","review_legit","review_fraud"]
    colors = {
        "block_TP":"#4CAF50","block_FP":"#E53935","approve_FN":"#8E24AA",
        "review_legit":"#FB8C00","review_fraud":"#1E88E5"
    }
    counts_mat = np.array([[comp[m][k] for k in key_components] for m in order_models], dtype=float)
    x = np.arange(len(order_models))
    bottom = np.zeros(len(order_models))
    plt.figure(figsize=(9,4.6), dpi=200)
    for j,k in enumerate(key_components):
        plt.bar(x, counts_mat[:,j], bottom=bottom, label=k.replace("_"," "), color=colors[k])
        bottom += counts_mat[:,j]
    plt.xticks(x, order_models, rotation=12)
    plt.ylabel("Count (test set)")
    plt.title("Reward Components — Event Counts by Model")
    plt.legend(ncol=3, frameon=True)
    plt.tight_layout()
    out1 = FIGS/"reward_components_stacked_counts.png"
    plt.savefig(out1); plt.close()

    # ---------- VISUAL 2: reward contribution ----------
    def per1k_contrib(v):
        n = sum(v[k] for k in ["block_TP","block_FP","approve_TN","approve_FN","review_legit","review_fraud"])
        return dict(
            tp_block      = RC["tp_block"]      * v["block_TP"]    * 1000.0 / n,
            fp_block      = RC["fp_block"]      * v["block_FP"]    * 1000.0 / n,
            tn_approve    = RC["tn"]            * v["approve_TN"]  * 1000.0 / n,
            fn_approve    = RC["fn"]            * v["approve_FN"]  * 1000.0 / n,
            review_legit  = RC["review_cost"]   * v["review_legit"]* 1000.0 / n,
            review_fraud  = (RC["review_cost"]+RC["review_catch"]) * v["review_fraud"] * 1000.0 / n
        )

    contrib = {m: per1k_contrib(comp[m]) for m in order_models}
    comp_keys = ["tp_block","fp_block","fn_approve","review_legit","review_fraud","tn_approve"]
    comp_labels = ["TP block (+)","FP block (−)","FN approve (−)","Review legit (−)",
                   "Review fraud (+)","TN approve (+)"]
    comp_colors = ["#4CAF50","#E53935","#8E24AA","#FB8C00","#1E88E5","#9E9E9E"]

    width = 0.12
    cx = np.arange(len(order_models))
    plt.figure(figsize=(10,4.8), dpi=200)
    for j,(k,lab,col) in enumerate(zip(comp_keys, comp_labels, comp_colors)):
        vals = [contrib[m][k] for m in order_models]
        plt.bar(cx + (j-2.5)*width, vals, width=width, label=lab, color=col)
        for xi,vi in zip(cx + (j-2.5)*width, vals):
            plt.text(xi, vi + (1 if vi>=0 else -1.5), f"{vi:+.1f}",
                     ha="center", va="bottom" if vi>=0 else "top", fontsize=8)
    plt.xticks(cx, order_models, rotation=12)
    plt.ylabel("Reward contribution per 1k transactions (higher = better)")
    plt.title("Reward Decomposition — Contribution per 1k by Component")
    plt.axhline(0, color="#444", linewidth=0.8)
    plt.legend(ncol=3, frameon=True)
    plt.tight_layout()
    out2 = FIGS/"reward_contribution_per_11k.png"
    plt.savefig(out2); plt.close()

    print(f"\nSaved figures -> {out1}  and  {out2}")
    print(f"Saved tables  -> {MODELS/'reward_components1.csv'}, {MODELS/'reward_components1.json'}")

if __name__ == "__main__":
    main()
