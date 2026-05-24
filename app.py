import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def get_metrics(data_path, model, le):
    df = pd.read_csv(data_path)
    df["region_enc"] = le.transform(df["region"])
    df = df.drop(columns=["region"])
    X = df.drop(columns=["churn"])
    y = df["churn"]
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1  = f1_score(y, y_pred, zero_division=0)
    return acc, f1

# Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/encoders.pkl", "rb") as f:
    le = pickle.load(f)

st.title("🤖 Model Monitoring Dashboard")
st.markdown("Track model performance before and after data drift.")

# Metrics
acc_orig, f1_orig       = get_metrics("data/raw.csv", model, le)
acc_drift, f1_drift     = get_metrics("data/drifted.csv", model, le)

# Score cards
st.subheader("📊 Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy — Original", f"{acc_orig:.2%}")
    st.metric("F1 Score — Original", f"{f1_orig:.2%}")
with col2:
    st.metric("Accuracy — Drifted", f"{acc_drift:.2%}", delta=f"{acc_drift - acc_orig:.2%}")
    st.metric("F1 Score — Drifted", f"{f1_drift:.2%}", delta=f"{f1_drift - f1_orig:.2%}")

# Chart
st.subheader("📈 Accuracy Comparison")
fig, ax = plt.subplots()
ax.bar(["Original", "Drifted"], [acc_orig, acc_drift], color=["green", "red"])
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
st.pyplot(fig)

# Drift alert
st.subheader("⚠️ Drift Status")
if acc_drift < 0.80:
    st.error("Drift detected! Accuracy below 80% threshold.")
else:
    st.success("Model is performing fine. No retraining needed.")

st.markdown("---")
st.caption("Built with Streamlit · Model Monitoring Project")