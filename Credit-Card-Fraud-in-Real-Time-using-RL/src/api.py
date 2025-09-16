# api.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict

app = FastAPI(title="RL Fraud Scorer", version="0.1.0")

# very simple schema: accepts arbitrary PCA features & amount/time
class Tx(BaseModel):
    Time: float = 0
    Amount: float = 0
    # accept V1..V28 without enumerating them one by one
    V: Dict[str, float] = Field(default_factory=dict, description="Keys like V1..V28")

@app.post("/score")
def score(tx: Tx):
    # dummy logic just to prove the API runs
    # in your real file, load models and compute decision here
    amount = tx.Amount
    v_sum = sum(tx.V.values()) if tx.V else 0.0
    decision = "BLOCK" if (amount > 200 and v_sum < 0) else "APPROVE"
    return {"decision": decision, "debug": {"amount": amount, "v_sum": v_sum}}
