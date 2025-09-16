# serve.py — minimal FastAPI for RL policy (TorchScript)
from fastapi import FastAPI
from pydantic import BaseModel
import torch, numpy as np
from xgboost import Booster, DMatrix
import joblib

app = FastAPI(title="RL Fraud Scorer")

# load artifacts once
policy = torch.jit.load("models/dqn_policy.ts", map_location="cpu").eval()
booster = Booster(); booster.load_model("models/xgb_base.json")
scaler = joblib.load("models/scaler.joblib")  # from your add_derived()

class Tx(BaseModel):
    # send a full feature row BEFORE add_derived() scaling:
    Time: float
    Amount: float
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float

@app.post("/score")
def score(tx: Tx):
    # 1) build dataframe row
    import pandas as pd
    row = pd.DataFrame([tx.dict()])

    # 2) replicate add_derived(): scale Time/Amount, add Amount_log1p
    row[["Time","Amount"]] = scaler.transform(row[["Time","Amount"]])
    row["Amount_log1p"] = np.log1p(np.abs(row["Amount"]))

    # 3) teacher feature (xgb_score) and budget_ratio=1.0
    xgb_score = float(booster.predict(DMatrix(row))[0])
    obs = np.concatenate([row.values.astype(np.float32).squeeze(),
                          np.array([xgb_score, 1.0], dtype=np.float32)], axis=0)

    # 4) fast policy inference
    with torch.inference_mode():
        q = policy(torch.from_numpy(obs).unsqueeze(0))
        act = int(q.argmax(1).item())  # 0=APPROVE, 1=REVIEW, 2=BLOCK

    return {"action": act, "xgb_score": xgb_score, "qvals": q.squeeze(0).tolist()}
