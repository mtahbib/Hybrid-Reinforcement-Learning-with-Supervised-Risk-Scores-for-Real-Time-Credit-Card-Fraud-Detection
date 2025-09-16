# src/eval_compare_debug.py
import json, sys, traceback
from pathlib import Path

def p(msg): 
    print(msg, flush=True)

try:
    p("[1/9] Imports ...")
    import numpy as np, pandas as pd, joblib, torch
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_fscore_support
    from xgboost import Booster, DMatrix

    p("[2/9] Local imports ...")
    from .data_loader import chrono_split
    from .features import add_derived
    from .rl_env import CreditFraudEnv
    MODELS = Path("models"); MODELS.mkdir(exist_ok=True, parents=True)

    def recall_at_fpr(y_true, scores, target_fpr=0.01):
        fpr, tpr, thr = roc_curve(y_true, scores)
        idx = int(np.argmin(np.abs(fpr - target_fpr)))
        return float(tpr[idx]), float(thr[idx])

    def eval_supervised(name, y_test, proba_test, y_val=None, proba_val=None):
        out = {"AUROC": float(roc_auc_score(y_test, proba_test)),
               "AUPRC": float(average_precision_score(y_test, proba_test))}
        if (y_val is not None) and (proba_val is not None):
            _, thr = recall_at_fpr(y_val, proba_val, 0.01)
            y_pred_test = (proba_test >= thr).astype(int)
            p_, r_, f1_, _ = precision_recall_fscore_support(y_test, y_pred_test, average='binary', zero_division=0)
            out["ValThr@1%FPR"] = {"thr": float(thr), "Precision": float(p_), "Recall": float(r_), "F1": float(f1_)}
        return name, out

    def rl_rollout_actions(X, y, xgb_scores, policy_pt=None, policy_ts=None, review_rate=0.05):
        env = CreditFraudEnv(X, y, xgb_scores, review_rate=review_rate, episode_len=len(X), seed=999)
        obs, _ = env.reset()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Prefer TorchScript
        if policy_ts and Path(policy_ts).exists():
            q = torch.jit.load(policy_ts, map_location=device).eval()
            scripted = True
        else:
            from .rl_train_dqn import DuelingQ
            obs_dim = env.observation_space.shape[0]
            q = DuelingQ(obs_dim, env.action_space.n).to(device).eval()
            q.load_state_dict(torch.load(policy_pt, map_location=device))
            scripted = False

        actions=[]
        with torch.no_grad():
            for _ in range(len(X)):
                t_obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                qv = q(t_obs) if scripted else q(t_obs)
                a = int(qv.argmax(dim=1).item())
                actions.append(a)
                obs, _, done, _, _ = env.step(a)
                if done: break
        return np.array(actions), env

    def rl_cost_per_1k(actions, y_true, rc):
        cost=0.0
        for a, label in zip(actions, y_true):
            if a==2:   # BLOCK
                cost += rc['tp_block'] if label==1 else rc['fp_block']
            elif a==0: # APPROVE
                cost += rc['tn'] if label==0 else rc['fn']
            else:      # REVIEW
                cost += rc['review_cost'] + (rc['review_catch'] if label==1 else 0.0)
        return float(1000.0 * cost / len(y_true))

    def alert_metrics(y_true, actions, mode="block"):
        if mode == "block":
            y_pred_alert = (actions==2).astype(int)
        else:
            y_pred_alert = np.isin(actions, [1,2]).astype(int)
        p_, r_, f1_, _ = precision_recall_fscore_support(y_true, y_pred_alert, average='binary', zero_division=0)
        return {"Precision": float(p_), "Recall": float(r_), "F1": float(f1_)}

    # ---------- pipeline ----------
    p("[3/9] Load data ...")
    df = pd.read_csv("data/creditcard.csv")
    sp = chrono_split(df)
    Xtr = add_derived(sp.X_train); Xva = add_derived(sp.X_val); Xte = add_derived(sp.X_test)

    p("[4/9] Load baselines ...")
    lr_path = MODELS/"lr.joblib"; rf_path = MODELS/"rf.joblib"; xgb_path = MODELS/"xgb_base.json"
    assert lr_path.exists() and rf_path.exists() and xgb_path.exists(), "Missing baseline model files in models/"
    lr = joblib.load(lr_path); rf = joblib.load(rf_path)

    p("[5/9] Predict baselines ...")
    lr_va = lr.predict_proba(Xva)[:,1]; lr_te = lr.predict_proba(Xte)[:,1]
    rf_va = rf.predict_proba(Xva)[:,1]; rf_te = rf.predict_proba(Xte)[:,1]

    booster = Booster(); booster.load_model(str(xgb_path))
    d_va, d_te = DMatrix(Xva), DMatrix(Xte)
    xgb_va = booster.predict(d_va); xgb_te = booster.predict(d_te)

    results = {}
    p("[6/9] Score metrics for supervised ...")
    for name, out in [
        eval_supervised("LogReg", sp.y_test, lr_te, sp.y_val, lr_va),
        eval_supervised("RandomForest", sp.y_test, rf_te, sp.y_val, rf_va),
        eval_supervised("XGBoost", sp.y_test, xgb_te, sp.y_val, xgb_va),
    ]:
        results[name] = out

    p("[7/9] Rollout RL policy ...")
    dqn_pt = MODELS/"dqn_policy.pt"; dqn_ts = MODELS/"dqn_policy.ts"
    assert dqn_pt.exists() or dqn_ts.exists(), "Missing RL policy: dqn_policy.pt or dqn_policy.ts"
    actions, env = rl_rollout_actions(Xte, sp.y_test, xgb_te.astype(np.float32),
                                      policy_pt=str(dqn_pt), policy_ts=str(dqn_ts), review_rate=0.05)

    y_true = env.y[env.idx]
    rc = dict(tp_block=5.0, fp_block=-2.0, fn=-10.0, tn=0.2, review_cost=-0.2, review_catch=3.0, over_budget=-5.0)
    cost = rl_cost_per_1k(actions, y_true, rc)

    rl_summary = {
        "Cost_per_1k": cost,
        "Rates": {
            "Approve_rate": float(np.mean(actions==0)),
            "Review_rate": float(np.mean(actions==1)),
            "Block_rate": float(np.mean(actions==2)),
        },
        "Alerts_BLOCK_only": alert_metrics(y_true, actions, mode="block"),
        "Alerts_BLOCK_or_REVIEW": alert_metrics(y_true, actions, mode="block_or_review"),
    }
    results["RL_DQN"] = rl_summary

    p("[8/9] Save & print ...")
    outp = MODELS/"compare_results.json"
    outp.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    p(f"[9/9] Done. Saved -> {outp}")

except Exception as e:
    print("\n[ERROR] An exception occurred:", repr(e), flush=True)
    traceback.print_exc()
    sys.exit(1)
