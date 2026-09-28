# 🚀 OmniSales AI — Universal Sales Prediction & Decision Intelligence Platform

An enterprise-grade, high-accuracy machine learning platform designed to predict sales for **any business** using **arbitrary parameters** (marketing spend, pricing, discounts, footfall, inventory, reviews, consumer sentiment, etc.). It delivers probabilistic forecasting, detects hidden patterns, and prescribes strategic decision steps to grow sales.

---

## ✨ Key Features

### 1. 🌐 Universal Ingestion for ANY Business Data
- **Arbitrary Parameters**: Accepts any CSV dataset with custom columns (e.g., E-Commerce, SaaS, Retail, Restaurants, Tech).
- **Auto-Schema Detection**: Automatically identifies target revenue columns, date/time sequences, and feature parameters.
- **4 Pre-loaded Industry Datasets**:
  - 🛍️ **E-Commerce & Digital Retail** (Ad Spend, Traffic, Discounts, AOV, Ratings -> Sales)
  - 💼 **SaaS & B2B Subscriptions** (Sales Reps, Demos, Churn, Inbound Leads, NPS -> MRR)
  - 🍕 **Restaurant & Hospitality** (Footfall, Weather, Promos, Events, Staff -> Daily Sales)
  - 📱 **Consumer Electronics & Tech** (Digital Ads, Partner Stores, Price, Tech Hype -> Weekly Sales)

### 2. 🏆 Multi-Model ML Ensemble with High Accuracy
- Evaluates candidate architectures:
  - **Gradient Boosting Regressor**
  - **Random Forest Regressor**
  - **Ridge Regression (L2 Regularized)**
  - **ElasticNet (L1 + L2 Regularized)**
  - **Voting Ensemble Blend**
- Auto-selects the **Champion Model** based on 5-fold cross-validated $R^2$ score, MAE, and RMSE.

### 3. 🎯 Probabilistic Forecasting & Goal Tracking (Monte Carlo)
- **Probabilistic Confidence Bands**: Generates $P_{10}$ (Bearish), $P_{50}$ (Expected), and $P_{90}$ (Bullish) future projections across 14 periods.
- **Goal Attainment Probability**: Computes exact statistical likelihood of meeting or exceeding your sales target.
- **Goal Gap Analysis**: Measures the gap and visualizes the outcome probability density (Bell Curve).

### 4. 📈 Market Sentiment Modulation
- Interactive slider from **-50% (Extreme Bearish)** to **+50% (Extreme Bullish)**.
- Models how macroeconomic optimism or recessionary pressures alter buyer responsiveness and price sensitivity in real time.

### 5. 🔍 Autonomous Hidden Pattern Discovery
- **Primary Sales Catalyst**: Pinpoints the single most influential variable and its correlation strength.
- **Diminishing Returns Detection**: Identifies efficiency saturation thresholds (e.g., where ad spend ROI flattens).
- **Revenue Friction Detection**: Flags parameters creating negative drag on total margin (e.g., excessive discounts).
- **Anomaly Detection**: Highlights historical periods with unusual sales spikes or dips (>2.2 $\sigma$).

### 6. 💡 Actionable Next Steps ("What to do next to increase sales")
- **Reverse Goal Solver**: Prescribes the exact quantitative adjustments needed in your top parameters to bridge your revenue gap.
- **Resource Reallocation**: Directs budget towards channels with the highest marginal ROI.
- **Friction Mitigation**: Actionable tactics to prevent revenue leakage.
- **Macro Market Positioning**: Tailored strategies for bullish growth or bearish defense.

### 7. 🎛️ Interactive What-If Scenario Sandbox
- Generates dynamic sliders for **every parameter** in your dataset.
- Real-time simulation computes predicted sales, delta vs historical baseline, and new goal probability instantly.

### 8. 📑 Executive Briefing & Export
- Formats a 1-click executive summary briefing modal ready to print or save as PDF.

---

## 🛠️ Tech Stack & Architecture

- **Backend**: Python 3.14 + FastAPI + Uvicorn + Scikit-Learn + Pandas + NumPy + SciPy
- **Frontend**: Vanilla Modern CSS (Dark Glassmorphic Theme, HSL tailored palette, responsive grid) + HTML5 + JavaScript (ES6)
- **Visuals**: Chart.js 4.4 + Lucide Icons + Google Fonts (*Outfit* and *Inter*)

---

## 🚦 Quick Start Guide

### 1. Launch with One Click (Windows)
Double-click `run.bat` in this folder:
```cmd
run.bat
```
This automatically boots the server and opens your browser at `http://localhost:8000`.

### 2. Manual Terminal Launch
```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Start the server
python app.py
```
Open [http://localhost:8000](http://localhost:8000) in your web browser.

---

## 📂 Project Structure

```
sales-prediction/
├── app.py                   # FastAPI application serving REST API & static dashboard
├── ml_engine.py             # Machine learning engine (Multi-model, Monte Carlo, patterns)
├── requirements.txt         # Dependencies (FastAPI, Scikit-learn, Pandas, etc.)
├── run.bat                  # One-click Windows batch launcher
├── test_api.py              # Automated test suite for all endpoints
├── sample_data/             # 4 Industry sample CSV datasets
│   ├── ecommerce_retail.csv
│   ├── saas_b2b.csv
│   ├── restaurant_hospitality.csv
│   └── electronics_tech.csv
└── static/                  # Responsive web dashboard
    ├── index.html           # UI layout and interactive panes
    ├── style.css            # Dark glassmorphism styling
    └── app.js               # Reactive state & Chart.js controllers
```
