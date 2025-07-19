import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import tensorflow as tf
from pathlib import Path

# ---------- CONFIG ----------------------------------------------------
WINDOW, HORIZON, EPOCHS, BATCH = 5, 1, 150, 8
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
# ----------------------------------------------------------------------

# ---------- DATA LOADERS ---------------------------------------------
@st.cache_data
def load_data():
    hist = pd.read_csv("jp_veg_features.csv")
    risk = pd.read_csv("veg_risk_scores.csv")
    return hist, risk

# helper to build sequences
def make_sequences(arr, window=WINDOW, horizon=HORIZON):
    X, y = [], []
    for i in range(len(arr) - window - horizon + 1):
        X.append(arr[i:i + window])
        y.append(arr[i + window : i + window + horizon])
    return np.array(X)[..., None], np.array(y)

# build tiny LSTM
def build_lstm(window=WINDOW):
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Input(shape=(window, 1)),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# train‑if‑missing
@st.cache_resource
def load_or_train_model(veg: str, series: np.ndarray):
    p = MODEL_DIR / f"lstm_{veg.lower().replace(' ', '_')}.keras"
    if p.exists():
        return tf.keras.models.load_model(p)

    X, y = make_sequences(series)
    if len(X) < 5:      # not enough data points
        return None

    model = build_lstm()
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH, verbose=0)
    model.save(p)
    return model
# ----------------------------------------------------------------------

# ====================== STREAMLIT UI ==================================
df, risk_df = load_data()

st.title("🇯🇵 Japanese Vegetable Market Intelligence 📈")
st.sidebar.header("Controls")

# 1. Historical price trends -------------------------------------------
st.header("1. Historical Price Trends")
items = st.multiselect("Select vegetables", df["Item"].unique())
if items:
    subset = df[df["Item"].isin(items)]
    fig = px.line(subset, x="Year", y="Value_winsor", color="Item")
    st.plotly_chart(fig, use_container_width=True)

# 2. LSTM forecast -----------------------------------------------------
st.header("2. LSTM Price Forecast")
veg = st.selectbox("Choose a vegetable", df["Item"].unique())

series = (
    df[df["Item"] == veg]
      .sort_values("Year")["Value_winsor"]
      .values.astype("float32")
)

if len(series) < WINDOW + 3:
    st.info("Not enough data points to train a model for this vegetable.")
else:
    model = load_or_train_model(veg, series)
    if model is None:
        st.info("Model could not be trained (insufficient data).")
    else:
        next_val = model.predict(series[-WINDOW:].reshape(1, WINDOW, 1))[0, 0]
        st.metric(f"Next‑Period Forecast ({veg})", f"{next_val:,.0f}")

# 3. Credit‑risk dashboard --------------------------------------------
st.header("3. Credit‑Risk Assessment")
top_n = st.slider("Show top N risky vegetables", 5, 20, 10)
top_risk = risk_df.head(top_n)

st.dataframe(top_risk.style.format({"RiskScore": "{:.3f}"}))

fig_risk = px.bar(
    top_risk, x="RiskScore", y="Item",
    color="RiskScore", orientation="h",
    color_continuous_scale="Reds"
)
st.plotly_chart(fig_risk, use_container_width=True)

st.caption("Composite Risk = 0.5×Volatility + 0.3×Exposure + 0.2×SupplyRisk")
