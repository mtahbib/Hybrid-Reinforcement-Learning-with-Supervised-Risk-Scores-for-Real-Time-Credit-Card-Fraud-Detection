import time, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)
from xgboost import Booster, DMatrix
import joblib, torch

# Custom project imports
from src.data_loader import chrono_split
from src.features import add_derived
from src.rl_env import CreditFraudEnv   

# Paths
ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data" / "creditcard.csv"
MODELS = ROOT / "models"
FIGS   = MODELS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

#helpers 
def time_it(fn, N, warmup=20):
    """
    Benchmark function latency & throughput.

    Args:
        fn: callable function (one single-transaction inference).
        N: number of times to call fn().
        warmup: extra calls before measuring (avoid cold start bias).

    Returns:
        stats dict with latency mean, p50, p95, p99, and throughput (TPS).
    """
    for _ in range(warmup): fn()   # warm-up run
    lat = []
    t0 = time.perf_counter()
    for _ in range(N):
        t1 = time.perf_counter()
        fn()   # call model
        lat.append((time.perf_counter() - t1) * 1000.0)  # ms latency
    total = time.perf_counter() - t0
    tps = N / total if total > 0 else float("nan")
    return {
        "count": N,
        "mean_ms": float(np.mean(lat)),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "tps": float(tps),
    }, lat

def threshold_at_fpr(y_true, scores, target_fpr=0.01):
    """
    Find decision threshold where FPR ≈ target (default 1%).
    Used for baselines to ensure fair comparison with RL.
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    i = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(thr[i])

def metrics(ytrue, yhat, proba):
    """
    Compute AUROC, AUPRC, Precision, Recall, and F1.
    For RL (no probabilities), AUROC/AUPRC are set to NaN.
    """
    return dict(
        AUROC=float(roc_auc_score(ytrue, proba)) if proba is not None else float("nan"),
        AUPRC=float(average_precision_score(ytrue, proba)) if proba is not None else float("nan"),
        Precision=float(precision_score(ytrue, yhat, zero_division=0)),
        Recall=float(recall_score(ytrue, yhat, zero_division=0)),
        F1=float(f1_score(ytrue, yhat, zero_division=0)),
    )

# load data/models 
def load_data():
    """
    Load credit card fraud dataset, chronological split,
    and preprocess features (normalize + add derived).
    """
    df = pd.read_csv(DATA)
    sp = chrono_split(df)
    Xtr = add_derived(sp.X_train)
    Xva = add_derived(sp.X_val)
    Xte = add_derived(sp.X_test).reset_index(drop=True)
    yva = sp.y_val.values.astype(int)
    yte = sp.y_test.values.astype(int)
    return Xtr, Xva, Xte, yva, yte

def load_artifacts():
    """
    Load saved models: Logistic Regression, Random Forest, XGBoost,
    and RL policy (TorchScript .ts or PyTorch .pt fallback).
    """
    lr = joblib.load(MODELS / "lr.joblib")
    rf = joblib.load(MODELS / "rf.joblib")
    booster = Booster(); booster.load_model(str(MODELS / "xgb_base.json"))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ts = MODELS / "dqn_policy.ts"
    pt = MODELS / "dqn_policy.pt"
    if ts.exists():
        policy = torch.jit.load(str(ts), map_location=device).eval(); scripted=True
    elif pt.exists():
        from src.rl_train_dqn import DuelingQ
        policy=("pt_fallback", device); scripted=False
    else:
        raise FileNotFoundError("Missing RL policy (models/dqn_policy.ts or .pt)")
    return lr, rf, booster, policy, scripted, device

# ---------------- per-model single-tx callables ----------------
def build_payloads(X, booster, K=5000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(K, len(X)), replace=False)
    rows = []
    for i in idx:
        row = X.iloc[i:i+1]
        xgb_s = float(booster.predict(DMatrix(row))[0])
        rows.append({"row": row, "xgb": xgb_s})
    return rows

def make_lr_fn(lr, payloads):
    """One-transaction callable for Logistic Regression."""
    it = {"i":0, "n":len(payloads)}
    def fn():
        p = payloads[it["i"]]; it["i"]=(it["i"]+1)%it["n"]
        _ = lr.predict_proba(p["row"])[:,1][0]
    return fn

def make_rf_fn(rf, payloads):
    """One-transaction callable for Random Forest."""
    it = {"i":0, "n":len(payloads)}
    def fn():
        p = payloads[it["i"]]; it["i"]=(it["i"]+1)%it["n"]
        _ = rf.predict_proba(p["row"])[:,1][0]
    return fn

def make_xgb_fn(booster, payloads):
    """One-transaction callable for XGBoost."""
    it = {"i":0, "n":len(payloads)}
    def fn():
        p = payloads[it["i"]]; it["i"]=(it["i"]+1)%it["n"]
        _ = booster.predict(DMatrix(p["row"]))[0]
    return fn

def make_rl_fn(policy, scripted, device, payloads):
    """
    One-transaction callable for RL policy.
    Creates observation vector = features + XGB score + budget.
    Only used for speed tests.
    """
    f = payloads[0]["row"].values.astype(np.float32).squeeze()
    obs_dim = f.shape[0] + 2
    if isinstance(policy, tuple):  # .pt fallback
        from src.rl_train_dqn import DuelingQ
        device = policy[1]
        net = DuelingQ(obs_dim, 3).to(device).eval()
        net.load_state_dict(torch.load(MODELS / "dqn_policy.pt", map_location=device))
    else:
        net = policy
    it = {"i":0, "n":len(payloads)}
    def fn():
        p = payloads[it["i"]]; it["i"]=(it["i"]+1)%it["n"]
        obs = np.concatenate([
            p["row"].values.astype(np.float32).squeeze(),
            np.array([p["xgb"]], dtype=np.float32),
            np.array([1.0], dtype=np.float32)   # budget feature
        ], axis=0)
        with torch.inference_mode():
            q = net(torch.as_tensor(obs, device=device).unsqueeze(0))
            _ = int(q.argmax(1).item())
    return fn

# ---------------- RL accuracy via ENV ROLLOUT ----------------
def rl_block_predictions_env(policy, scripted, device, Xte, yte, booster):
    """
    Evaluate RL accuracy by rolling out the trained policy
    inside CreditFraudEnv (consistent with training).
    Returns array: 1 if BLOCK action taken, 0 otherwise.
    """
    xgb_scores_te = booster.predict(DMatrix(Xte)).astype(np.float32)
    env = CreditFraudEnv(
        X=Xte, y=pd.Series(yte),
        xgb_scores=xgb_scores_te,
        review_rate=0.05,            # same budget as training
        episode_len=len(Xte),
        seed=999
    )
    # reload policy if only .pt is available
    if isinstance(policy, tuple):
        from src.rl_train_dqn import DuelingQ
        device = policy[1]
        obs_dim = env.observation_space.shape[0]
        net = DuelingQ(obs_dim, env.action_space.n).to(device).eval()
        net.load_state_dict(torch.load(MODELS / "dqn_policy.pt", map_location=device))
    else:
        net = policy

    acts = []
    obs,_ = env.reset()
    with torch.inference_mode():
        for _ in range(len(Xte)):
            q = net(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            a = int(q.argmax(1).item())   # pick action
            acts.append(a)
            obs, _, done, _, _ = env.step(a)
            if done: break
    acts = np.array(acts, dtype=int)
    return (acts == 2).astype(int)  # only BLOCK = fraud positive

# ---------------- main ----------------
def main():
    # Load data & models
    Xtr, Xva, Xte, yva, yte = load_data()
    lr, rf, booster, policy, scripted, device = load_artifacts()

    # Build payloads (subset of test data for timing)
    payloads = build_payloads(Xte, booster, K=3000)
    fn_lr, fn_rf = make_lr_fn(lr, payloads), make_rf_fn(rf, payloads)
    fn_xgb, fn_rl = make_xgb_fn(booster, payloads), make_rl_fn(policy, scripted, device, payloads)

    # ---- SYSTEM METRICS: TPS & Latency ----
    N = 5000
    results = {}
    print(f"Benchmarking {N} single-tx calls per model (local, no HTTP)...\n")
    for name, fn in [("LR", fn_lr), ("RF", fn_rf), ("XGB", fn_xgb), ("RL", fn_rl)]:
        stats,_ = time_it(fn, N=N, warmup=50)
        results[name] = stats
        print(f"{name:3s}  TPS={stats['tps']:.1f}  p95={stats['p95_ms']:.3f}ms  "
              f"p99={stats['p99_ms']:.3f}ms  mean={stats['mean_ms']:.3f}ms")
    (MODELS/"latency_tps.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(results).T.to_csv(MODELS/"latency_tps.csv")

    # ---- ACCURACY: Baselines @ ~1% FPR + RL via ENV ----
    # Baseline thresholds chosen by validation FPR ≈ 1%
    lr_va, lr_te = lr.predict_proba(Xva)[:,1], lr.predict_proba(Xte)[:,1]
    rf_va, rf_te = rf.predict_proba(Xva)[:,1], rf.predict_proba(Xte)[:,1]
    xg_va, xg_te = booster.predict(DMatrix(Xva)), booster.predict(DMatrix(Xte))
    thr_lr  = threshold_at_fpr(yva, lr_va, 0.01)
    thr_rf  = threshold_at_fpr(yva, rf_va, 0.01)
    thr_xgb = threshold_at_fpr(yva, xg_va, 0.01)
    yhat_lr  = (lr_te  >= thr_lr ).astype(int)
    yhat_rf  = (rf_te  >= thr_rf ).astype(int)
    yhat_xgb = (xg_te >= thr_xgb).astype(int)

    # RL predictions from env rollout
    yhat_rl_block = rl_block_predictions_env(policy, scripted, device, Xte, yte, booster)

    # Final metrics
    acc = {
        "LR":  metrics(yte, yhat_lr,  lr_te),
        "RF":  metrics(yte, yhat_rf,  rf_te),
        "XGB": metrics(yte, yhat_xgb, xg_te),
        "RL":  metrics(yte[:len(yhat_rl_block)], yhat_rl_block, None),
    }
    (MODELS/"latency_accuracy.json").write_text(json.dumps(acc, indent=2))
    pd.DataFrame(acc).T.to_csv(MODELS/"latency_accuracy.csv")

    # ---- VISUALS ----
    labels = ["LogReg","RandomForest","XGBoost","RL (BLOCK)"]

    # Latency p95
    p95_vals = [results["LR"]["p95_ms"], results["RF"]["p95_ms"], results["XGB"]["p95_ms"], results["RL"]["p95_ms"]]
    plt.figure(figsize=(7.2,4.2))
    bars = plt.bar(["LR","RF","XGB","RL"], p95_vals, color=["#888"]*3+["#0aa6b7"])
    plt.ylabel("Latency p95 (ms)"); plt.title("Latency (p95) — single-transaction inference")
    for b,v in zip(bars,p95_vals): plt.text(b.get_x()+b.get_width()/2, v+0.02*max(p95_vals), f"{v:.2f}", ha="center")
    plt.tight_layout(); plt.savefig(FIGS/"latency_p95_bars.png", dpi=200); plt.close()

    # Latency p99
    p99_vals = [results["LR"]["p99_ms"], results["RF"]["p99_ms"], results["XGB"]["p99_ms"], results["RL"]["p99_ms"]]
    plt.figure(figsize=(7.2,4.2))
    bars = plt.bar(["LR","RF","XGB","RL"], p99_vals, color=["#888"]*3+["#0aa6b7"])
    plt.ylabel("Latency p99 (ms)"); plt.title("Latency (p99) — single-transaction inference")
    for b,v in zip(bars,p99_vals): plt.text(b.get_x()+b.get_width()/2, v+0.02*max(p99_vals), f"{v:.2f}", ha="center")
    plt.tight_layout(); plt.savefig(FIGS/"latency_p99_bars.png", dpi=200); plt.close()

    # Throughput TPS
    tps_vals = [results["LR"]["tps"], results["RF"]["tps"], results["XGB"]["tps"], results["RL"]["tps"]]
    plt.figure(figsize=(7.2,4.2))
    bars = plt.bar(["LR","RF","XGB","RL"], tps_vals, color=["#888"]*3+["#0aa6b7"])
    plt.ylabel("Transactions per second"); plt.title("Throughput (TPS) — single-transaction inference")
    for b,v in zip(bars,tps_vals): plt.text(b.get_x()+b.get_width()/2, v+0.02*max(tps_vals), f"{v:.0f}", ha="center")
    plt.tight_layout(); plt.savefig(FIGS/"throughput_tps_bars.png", dpi=200); plt.close()

    # Accuracy (F1)
    f1_vals = [acc["LR"]["F1"], acc["RF"]["F1"], acc["XGB"]["F1"], acc["RL"]["F1"]]
    plt.figure(figsize=(7.6,4.6))
    bars = plt.bar(labels, f1_vals, color=["#888888","#888888","#888888","#0aa6b7"])
    plt.ylabel("F1 score"); plt.title("F1 comparison: Baselines (~1% FPR) vs RL (BLOCK)")
    for b,v in zip(bars, f1_vals): plt.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}", ha="center")
    plt.ylim(0, max(0.1, max(f1_vals)+0.15)); plt.tight_layout()
    plt.savefig(FIGS/"accuracy_f1_bars.png", dpi=200); plt.close()

    print("\nSaved figures in:", FIGS.resolve())
    print("Saved numbers in:", MODELS.resolve())

if __name__ == "__main__":
    main()
