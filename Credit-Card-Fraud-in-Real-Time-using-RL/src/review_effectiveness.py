import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

from xgboost import Booster, DMatrix
import joblib, torch

from src.data_loader import chrono_split
from src.features import add_derived
from src.rl_env import CreditFraudEnv

ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data" / "creditcard.csv"
MODELS = ROOT / "models"
FIGS   = MODELS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

def load_artifacts():
    booster = Booster(); booster.load_model(str(MODELS/"xgb_base.json"))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ts = MODELS/"dqn_policy.ts"; pt = MODELS/"dqn_policy.pt"
    if ts.exists():
        policy = torch.jit.load(str(ts), map_location=device).eval()
        scripted = True
    else:
        from src.rl_train_dqn import DuelingQ
        # obs_dim is inferred from env later; we'll rebuild the net then
        policy = ("pt_fallback", device); scripted = False
    return booster, policy, scripted, device

def ensure_pt_policy(policy_tuple, obs_dim, device):
    from src.rl_train_dqn import DuelingQ
    net = DuelingQ(obs_dim, 3).to(device).eval()
    net.load_state_dict(torch.load(MODELS/"dqn_policy.pt", map_location=device))
    return net

def main():
    # === data & split ===
    df = pd.read_csv(DATA)
    sp = chrono_split(df)
    Xte = add_derived(sp.X_test); yte = sp.y_test.values.astype(int)

    # === artifacts ===
    booster, policy, scripted, device = load_artifacts()
    xgb_scores_te = booster.predict(DMatrix(Xte)).astype(np.float32)

    # === env rollout (budget-aware) ===
    env = CreditFraudEnv(
        X=Xte, y=sp.y_test, xgb_scores=xgb_scores_te,
        review_rate=0.05, episode_len=len(Xte), seed=123, strict_budget=True
    )
    obs,_ = env.reset()
    obs_dim = env.observation_space.shape[0]

    if isinstance(policy, tuple):  # pt fallback
        policy = ensure_pt_policy(policy, obs_dim, device); scripted = False

    acts = []
    with torch.inference_mode():
        for _ in range(len(Xte)):
            q = policy(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            a = int(q.argmax(1).item())
            acts.append(a)
            obs, r, done, _, _ = env.step(a)
            if done: break
    acts = np.array(acts, dtype=int)

    # === triage metrics ===
    y = yte[:len(acts)]
    is_block  = (acts==2)
    is_review = (acts==1)
    is_approve= (acts==0)

    # Block-only classification metrics
    yhat_block = is_block.astype(int)
    block_prec = precision_score(y, yhat_block, zero_division=0)
    block_rec  = recall_score(y, yhat_block,  zero_division=0)
    block_f1   = (0 if (block_prec+block_rec)==0 else 2*block_prec*block_rec/(block_prec+block_rec))

    # Review effectiveness
    reviewed_idx = np.where(is_review)[0]
    review_count = reviewed_idx.size
    review_fraud = int(y[reviewed_idx].sum()) if review_count>0 else 0
    review_hit_rate = (review_fraud / review_count) if review_count>0 else 0.0

    # Combined recall the business way: frauds caught by BLOCK or REVIEW
    caught = (is_block & (y==1)) | (is_review & (y==1))
    combined_recall = caught.sum() / (y==1).sum()

    # Action mix
    rates = {
        "approve_rate": float(is_approve.mean()),
        "review_rate" : float(is_review.mean()),
        "block_rate"  : float(is_block.mean())
    }

    # Save & print
    out = {
        "block_metrics": {"precision": block_prec, "recall": block_rec, "f1": block_f1},
        "review": {"count": int(review_count), "frauds": int(review_fraud), "hit_rate": review_hit_rate},
        "combined_recall_block_or_review": combined_recall,
        "action_rates": rates
    }
    print(json.dumps(out, indent=2))
    (MODELS/"triage_metrics.json").write_text(json.dumps(out, indent=2))

    # === visuals ===
    # 1) Action mix + review hit-rate
    fig, ax = plt.subplots(1,2, figsize=(9.2,4.2))
    # action mix
    ax[0].bar(["Approve","Review","Block"], [rates["approve_rate"], rates["review_rate"], rates["block_rate"]],
              color=["#9aa","orange","#0aa6b7"])
    ax[0].set_ylim(0,1.0); ax[0].set_title("RL Action Mix (Test)")
    for i,v in enumerate([rates["approve_rate"], rates["review_rate"], rates["block_rate"]]):
        ax[0].text(i, v+0.01, f"{v:.2%}", ha="center")
    # review hit-rate
    ax[1].bar(["Review hit-rate"], [review_hit_rate], color="orange")
    ax[1].set_ylim(0,1.0); ax[1].set_title("Fraud Proportion in Reviewed Items")
    ax[1].text(0, review_hit_rate+0.01, f"{review_hit_rate:.2%}", ha="center")
    plt.tight_layout()
    fig1 = FIGS/"rl_review_effectiveness.png"
    plt.savefig(fig1, dpi=200); plt.close()

    # 2) Recall contributions (stack: review vs block)
    total_fraud = (y==1).sum()
    tp_block = int((is_block & (y==1)).sum())
    tp_review= int((is_review & (y==1)).sum())
    fig, ax = plt.subplots(figsize=(6.0,4.2))
    ax.bar(["RL"], [tp_block/total_fraud], label="Caught by BLOCK", color="#0aa6b7")
    ax.bar(["RL"], [tp_review/total_fraud], bottom=[tp_block/total_fraud],
           label="Caught by REVIEW", color="orange")
    ax.set_ylim(0,1.0); ax.set_ylabel("Recall (fraction of fraud caught)")
    ax.set_title("Recall Decomposition: BLOCK vs REVIEW")
    ax.legend()
    for v,b in [(tp_block/total_fraud, "BLOCK"), (tp_review/total_fraud, "REVIEW")]:
        pass
    plt.tight_layout()
    fig2 = FIGS/"rl_recall_decomposition.png"
    plt.savefig(fig2, dpi=200); plt.close()

    print(f"Saved figures:\n  - {fig1}\n  - {fig2}")

if __name__ == "__main__":
    main()
