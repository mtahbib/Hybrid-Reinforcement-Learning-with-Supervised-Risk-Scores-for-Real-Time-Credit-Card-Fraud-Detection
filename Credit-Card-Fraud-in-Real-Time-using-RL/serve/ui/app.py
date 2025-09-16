import streamlit as st
import requests
import numpy as np
import time

st.set_page_config(page_title="RL Fraud Scorer", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.header("Server")
API_URL = st.sidebar.text_input("API URL", "http://127.0.0.1:8000/score")

# ---------------- Helper ----------------
def make_tx_payload(time_val, amount_val):
    """Builds transaction payload with Time, Amount, and auto-generated V1..V28."""
    features = {f"V{i}": float(np.random.normal(0, 1)) for i in range(1, 29)}
    payload = {"Time": time_val, "Amount": amount_val}
    payload.update(features)
    return payload

# ---------------- Main UI ----------------
st.title("Fraud Detection UI")

st.subheader("Transaction features")
col1, col2 = st.columns(2)
with col1:
    time_val = st.number_input("Time", min_value=0, value=123456, step=1)
with col2:
    amount_val = st.number_input("Amount", min_value=0.0, value=250.75, step=0.01)

# Action buttons
colA, colB = st.columns(2)
with colA:
    if st.button("Randomize"):
        amount_val = float(np.random.uniform(1, 1000))
        time_val = int(np.random.randint(100000, 999999))
with colB:
    run_score = st.button("Score")

# ---------------- API Call ----------------
if run_score:
    payload = make_tx_payload(time_val, amount_val)
    try:
        t0 = time.perf_counter()
        response = requests.post(API_URL, json=payload)
        latency = (time.perf_counter() - t0) * 1000  # ms

        if response.status_code == 200:
            result = response.json()
            st.success(f"Decision: **{result['decision']}**")
            st.metric("Latency (ms)", f"{latency:.2f}")
            with st.expander("Raw response"):
                st.json(result)
        else:
            st.error(f"Error {response.status_code}: {response.text}")

    except Exception as e:
        st.error(f"Failed to reach API: {e}")
