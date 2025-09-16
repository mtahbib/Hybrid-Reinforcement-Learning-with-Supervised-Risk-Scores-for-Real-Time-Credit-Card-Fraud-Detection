# src/run_rl_env.py
import pandas as pd
import numpy as np
from xgboost import Booster, DMatrix

from .data_loader import chrono_split
from .features import add_derived
from .rl_env import CreditFraudEnv

def main():
    # --- load dataset & chronological split ---
    df = pd.read_csv("data/creditcard.csv")
    sp = chrono_split(df)

    # --- test features ---
    Xte = add_derived(sp.X_test)

    # --- load XGBoost booster directly and get probabilities ---
    booster = Booster()
    booster.load_model("models/xgb_base.json")
    # Optional: if CUDA is available for prediction, uncomment next line
    # booster.set_param({"device": "cuda"})
    dte = DMatrix(Xte)
    xgb_scores = booster.predict(dte).astype(np.float32)  # prob of class 1

    # --- build RL environment (approve/review/block with review budget) ---
    env = CreditFraudEnv(
        X=Xte,
        y=sp.y_test,
        xgb_scores=xgb_scores,
        review_rate=0.05,   # 5% review budget
        episode_len=200,    # short smoke test
        seed=42
    )

    # --- quick rollout with random actions to verify everything works ---
    obs, _ = env.reset()
    print("Initial obs shape:", obs.shape)   # should be 32

    for t in range(5):
        action = env.action_space.sample()   # 0=APPROVE, 1=REVIEW, 2=BLOCK
        obs, reward, done, trunc, info = env.step(action)
        print(f"Step {t+1}: action={action}, reward={reward:.2f}, done={done}")
        if done:
            break

if __name__ == "__main__":
    main()
