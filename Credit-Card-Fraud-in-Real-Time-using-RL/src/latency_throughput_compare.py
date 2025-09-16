# src/latency_throughput_compare.py
import time, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import Booster, DMatrix
import joblib, torch

# local imports (chronological split & features)
from src.data_loader import chrono_split
from src.features import add_derived

ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data" / "creditcard.csv"
MODELS = ROOT / "models"
FIGS   = MODELS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# ---------------- helpers ----------------
def pcts(arr, ps=(50,95,99)):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return {p: float("nan") for p in ps}
    return {p: float(np.percentile(arr, p)) for p in ps}

def time_it(fn, N, warmup=20):
    # warmup
    for _ in range(warmup):
        fn()
    lat = []
    t0 = time.perf_counter()
    for _ in range(N):
        t1 = time.perf_counter()
        fn()
        lat.append((time.perf_counter() - t1) * 1000.0)  # ms
    total = time.perf_counter() - t0
    tps = N / total if total > 0 else float("nan")
    stats = {
        "count": N,
        "mean_ms": float(np.mean(lat)),
        "p50_ms": pcts(lat)[50],
        "p95_ms": pcts(lat)[95],
        "p99_ms": pcts(lat)[99],
        "tps": float(tps),
    }
    return stats, lat

# ---------------- load data/models ----------------
def load_data():
    df = pd.read_csv(DATA)
    sp = chrono_split(df)
    Xtr = add_derived(sp.X_train)
    Xva = add_derived(sp.X_val)
    Xte = add_derived(sp.X_test).reset_index(drop=True)
    return Xtr, Xva, Xte

def load_artifacts():
    lr = joblib.load(MODELS / "lr.joblib")
    rf = joblib.load(MODELS / "rf.joblib")
    booster = Booster(); booster.load_model(str(MODELS / "xgb_base.json"))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # TorchScript preferred
    ts = MODELS / "dqn_policy.ts"
    pt = MODELS / "dqn_policy.pt"
    if ts.exists():
        policy = torch.jit.load(str(ts), map_location=device).eval()
        scripted = True
    elif pt.exists():
        from src.rl_train_dqn import DuelingQ
        # We'll lazily re-instantiate after we infer obs_dim
        policy = ("pt_fallback", device)
        scripted = False
    else:
        raise FileNotFoundError("Missing RL policy (dqn_policy.ts or .pt)")
    return lr, rf, booster, policy, scripted, device

# ---------------- per-model single-tx callables ----------------
def build_payloads(X, booster, K=5000, seed=42):
    """Return a list of dict payloads: {row_df, xgb_score} to avoid recompute overhead in the loop."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(K, len(X)), replace=False)
    rows = []
    for i in idx:
        row = X.iloc[i:i+1]
        xgb_s = float(booster.predict(DMatrix(row))[0])
        rows.append({"row": row, "xgb": xgb_s})
    return rows

def make_lr_fn(lr, payloads):
    it = {"i":0, "n":len(payloads)}
    def fn():
        p = payloads[it["i"]]; it["i"] = (it["i"]+1) % it["n"]
        _ = lr.predict_proba(p["row"])[:,1][0]
    return fn

def make_rf_fn(rf, payloads):
    it = {"i":0, "n":len(payloads)}
    def fn():
        p = payloads[it["i"]]; it["i"] = (it["i"]+1) % it["n"]
        _ = rf.predict_proba(p["row"])[:,1][0]
    return fn

def make_xgb_fn(booster, payloads):
    it = {"i":0, "n":len(payloads)}
    def fn():
        p = payloads[it["i"]]; it["i"] = (it["i"]+1) % it["n"]
        _ = booster.predict(DMatrix(p["row"]))[0]
    return fn

def make_rl_fn(policy, scripted, device, X, payloads):
    # infer obs_dim from a sample; obs = [features..., xgb_score, budget_ratio=1.0]
    sample = payloads[0]
    f = sample["row"].values.astype(np.float32).squeeze()
    obs_dim = f.shape[0] + 2
    if isinstance(policy, tuple):  # pt fallback
        from src.rl_train_dqn import DuelingQ
        net = DuelingQ(obs_dim, 3).to(device).eval()
        net.load_state_dict(torch.load(MODELS / "dqn_policy.pt", map_location=device))
    else:
        net = policy
    it = {"i":0, "n":len(payloads)}
    def fn():
        p = payloads[it["i"]]; it["i"] = (it["i"]+1) % it["n"]
        obs = np.concatenate([p["row"].values.astype(np.float32).squeeze(),
                              np.array([p["xgb"]], dtype=np.float32),
                              np.array([1.0], dtype=np.float32)], axis=0)
        with torch.inference_mode():
            q = net(torch.as_tensor(obs, device=device).unsqueeze(0))
            _ = int(q.argmax(1).item())  # 0/1/2
    return fn

# ---------------- main ----------------
def main():
    Xtr, Xva, Xte = load_data()
    lr, rf, booster, policy, scripted, device = load_artifacts()
    # pre-build single-tx payloads (rotate over K rows)
    payloads = build_payloads(Xte, booster, K=3000)

    # callables
    fn_lr  = make_lr_fn(lr, payloads)
    fn_rf  = make_rf_fn(rf, payloads)
    fn_xgb = make_xgb_fn(booster, payloads)
    fn_rl  = make_rl_fn(policy, scripted, device, Xte, payloads)

    # measure
    N = 5000  # number of single tx calls per model
    results = {}
    print(f"Benchmarking {N} single-tx calls per model (local, no HTTP)...\n")

    for name, fn in [("LR", fn_lr), ("RF", fn_rf), ("XGB", fn_xgb), ("RL", fn_rl)]:
        stats, lat = time_it(fn, N=N, warmup=50)
        results[name] = stats
        print(f"{name:3s}  TPS={stats['tps']:.1f}  p50={stats['p50_ms']:.2f}ms  "
              f"p95={stats['p95_ms']:.2f}ms  p99={stats['p99_ms']:.2f}ms  mean={stats['mean_ms']:.2f}ms")

    # save numerics
    (MODELS/"latency_tps.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(results).T.to_csv(MODELS/"latency_tps.csv")

    # ----- charts -----
    labels = ["LR","RF","XGB","RL"]
    p95_vals = [results[k]["p95_ms"] for k in labels]
    tps_vals = [results[k]["tps"] for k in labels]

    # Latency (p95) bars — lower is better
    plt.figure(figsize=(6.6,4.0))
    colors = ["#888","#888","#888","#0aa6b7"]
    bars = plt.bar(labels, p95_vals, color=colors)
    plt.ylabel("Latency p95 (ms)")
    plt.title("Latency @ p95 (single-tx inference)")
    for b,v in zip(bars,p95_vals):
        plt.text(b.get_x()+b.get_width()/2, v+0.02*max(p95_vals), f"{v:.2f}", ha="center")
    plt.tight_layout()
    out1 = FIGS/"latency_p95_bars.png"
    plt.savefig(out1, dpi=200); plt.close()

    # Throughput bars — higher is better
    plt.figure(figsize=(6.6,4.0))
    bars = plt.bar(labels, tps_vals, color=colors)
    plt.ylabel("Transactions per second (single-tx)")
    plt.title("Throughput (achieved TPS)")
    for b,v in zip(bars,tps_vals):
        plt.text(b.get_x()+b.get_width()/2, v+0.02*max(tps_vals), f"{v:.0f}", ha="center")
    plt.tight_layout()
    out2 = FIGS/"throughput_tps_bars.png"
    plt.savefig(out2, dpi=200); plt.close()

    print(f"\nSaved:\n  {out1}\n  {out2}\n  {MODELS/'latency_tps.csv'}\n  {MODELS/'latency_tps.json'}")

if __name__ == "__main__":
    main()
