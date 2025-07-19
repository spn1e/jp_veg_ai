import pandas as pd, numpy as np, plotly.express as px, streamlit as st
import tensorflow as tf
from pathlib import Path

# ------------- CONFIG -------------------------------------------------
WINDOW = 5
MODEL_DIR = Path(".")   # adjust if models live elsewhere
# ----------------------------------------------------------------------

@st.cache_data
def load_data():
    df_hist = pd.read_csv("jp_veg_features.csv")
    risks   = pd.read_csv("veg_risk_scores.csv")
    return df_hist, risks

@st.cache_resource
def load_model(veg):
    p = MODEL_DIR / f"lstm_{veg.lower().replace(' ', '_')}.keras"
    return tf.keras.models.load_model(p)

df, risk_df = load_data()

st.title("🇯🇵 Japanese Vegetable Market Intelligence  📈")
st.sidebar.header("Controls")

# ---------- Price trend explorer --------------------------------------
st.header("1. Historical Price Trends")
items = st.multiselect("Select vegetables", df["Item"].unique(), default=None)
if items:
    subset = df[df["Item"].isin(items)]
    fig = px.line(subset, x="Year", y="Value_winsor", color="Item")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Forecast ---------------------------------------------------
st.header("2. LSTM Price Forecast")
veg = st.selectbox("Choose a vegetable for forecast", df["Item"].unique())
series = (
    df[df["Item"] == veg].sort_values("Year")["Value_winsor"]
    .values.astype("float32")
)

if len(series) > WINDOW + 3:
    model = load_model(veg)
    X_last = series[-WINDOW:].reshape(1, WINDOW, 1)
    next_val = model.predict(X_last)[0, 0]

    st.metric(f"Next‑Period Forecast ({veg})", f"{next_val:,.0f}")
else:
    st.info("Not enough data points to forecast this item.")

# ---------- Risk dashboard --------------------------------------------
st.header("3. Credit‑Risk Assessment")
top_n = st.slider("Show top N risky vegetables", 5, 20, 10)
top_risk = risk_df.head(top_n)

st.dataframe(top_risk.style.format({"RiskScore": "{:.3f}"}))

fig_risk = px.bar(top_risk, x="RiskScore", y="Item",
                  color="RiskScore", orientation="h",
                  color_continuous_scale="Reds")
st.plotly_chart(fig_risk, use_container_width=True)

st.caption("Composite Risk = 0.5×Volatility + 0.3×Exposure + 0.2×SupplyRisk")
