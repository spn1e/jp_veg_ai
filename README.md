# 🇯🇵 Japanese Vegetable Market AI Dashboard  

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Open%20Dashboard-brightgreen?logo=streamlit)](https://jpvegai-4xmzwsyjwmtu8eywcuhqdu.streamlit.app/)

AI‑powered analytics for producer‑price forecasting and credit‑risk assessment of Japan’s vegetable market (2000 – 2023).

---

## ✨ Project Highlights
| | |
|---|---|
| **Data** | **FAOSTAT – Producer Prices** for all vegetable items in Japan, 2000‑2023 |
| **Feature Engineering** | Winsorised prices, YoY % returns, log‑returns, 3 & 5‑year rolling volatilities, 1‑3 year lags, optional month/quarter dummies |
| **Model** | Per‑vegetable *LSTM* sequence model (5‑year look‑back) trained on first request & cached; average **MAPE ≈ 8‑10 %**, directional accuracy ≈ 90‑92 % |
| **Risk Score** | `0.5 × Volatility_norm + 0.3 × Exposure_norm + 0.2 × SupplyRisk_norm`<br> *Exposure* = mean production tonnage, *SupplyRisk* = std‑dev of YoY returns |
| **Dashboard** | Streamlit (Plotly charts, downloadable tables) — explore trends, on‑demand forecast, top‑N risk ranking |
| **Deploy** | Streamlit Cloud • Python 3.11 • TensorFlow‑CPU 2.16 |

---

## 🗂 Repository Structure
```
jp_veg_ai/
│
├─ my_app.py                  ← Streamlit dashboard
├─ jp_veg_features.csv        ← engineered dataset
├─ veg_risk_scores.csv        ← latest‑year risk table
├─ models/                    ← auto‑saved LSTM .keras files
├─ notebooks/
│   ├─ 01_feature_eng.ipynb   ← cleaning & FE
│   └─ 02_modelling.ipynb     ← LSTM training & metrics
├─ requirements.txt
└─ runtime.txt                ← pins Python 3.11 on Streamlit Cloud
```

---

## 🚀 Quick Start

### 1️⃣ Clone & install
```bash
git clone https://github.com/<your‑user>/jp_veg_ai.git
cd jp_veg_ai
conda create -n jp_veg_ai python=3.11 -y
conda activate jp_veg_ai
pip install -r requirements.txt         # TensorFlow‑CPU 2.16 wheels
```

### 2️⃣ Run locally
```bash
streamlit run my_app.py
```
Then open <http://localhost:8501>.

### 3️⃣ Ngrok (optional)
```bash
ngrok http 8501     # share a public HTTPS link
```

---

## 🔬 Methodology

1. **Data Ingest**  
   * Downloader script hits FAOSTAT bulk API → CSV.
2. **Cleaning & Winsorisation**  
   * Tukey fences (1.5 × IQR) per item to cap extreme spikes.
3. **Feature Engineering**  
   | Feature | Description |
   |---------|-------------|
   | `Value_winsor` | Outlier‑capped producer price (JPY / tonne) |
   | `YoY_Return`   | `(P_t – P_{t‑1}) / P_{t‑1}` |
   | `Log_Return`   | `log1p(YoY_Return)` |
   | `Vol_3y`, `Vol_5y` | Rolling std‑dev of YoY returns (3 & 5 yrs) |
   | `Value_lag1…3` | Raw price lags for classical baselines |
4. **Model**  
   * **LSTM(32) → Dense(16) → Dense(1)**, look‑back = 5 years.  
   * Trains in < 3 s on CPU; saved to `models/`.
5. **Evaluation**  
   * **MAPE** and **directional accuracy** on 20 % hold‑out.  
   * Typical veg example: *Tomatoes → MAPE 7.9 %, DirAcc 93 %*.
6. **Risk Scoring**  
   * Latest‑year volatility + exposure + supply‑risk → `RiskScore ∈ [0,1]`.
7. **Dashboard**  
   * Trend explorer, forecast tile, top‑N risk bar, export buttons.

---

## 📊 Screenshots
| Trend Explorer | Forecast & Risk View |
|----------------|----------------------|
| *insert screenshot* | *insert screenshot* |

*(Replace with real PNGs or `.gif` demo)*

---

## 🤝 Credits
* **FAOSTAT** — UN Food and Agriculture Organization.  
* Built with **Python 3.11, Pandas 2.2, TensorFlow‑CPU 2.16, Streamlit 1.30**.

---

## 📄 License
MIT — free to use, modify, and share.  
FAOSTAT data is public domain (FAO Terms of Use).

---

> **Live App 👉 https://jpvegai-4xmzwsyjwmtu8eywcuhqdu.streamlit.app/**  
> *Forecasts refresh on first load; risk scores update automatically when new FAOSTAT data is ingested.*
