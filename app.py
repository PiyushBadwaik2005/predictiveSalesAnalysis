"""
OmniSales AI - FastAPI Server & Universal Intelligence API
==========================================================
Serves the web application and REST API endpoints for:
- CSV uploads & schema detection
- Sample dataset loading
- High-accuracy ML model training & champion selection
- Goal probability calculation & Monte Carlo forecasting
- Market sentiment modulation
- Hidden pattern detection & strategic next-step recommendations
- Instant What-If scenario sandbox simulation
"""

import os
import io
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ml_engine import SalesMLEngine

app = FastAPI(
    title="OmniSales AI Engine",
    description="Universal Sales Prediction, Goal Probability & Business Decision Intelligence Platform",
    version="1.0.0"
)

# Enable CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared in-memory engine and dataset storage for session
engine = SalesMLEngine()
current_dataset_df: Optional[pd.DataFrame] = None
current_dataset_name: str = "None"

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class TrainRequest(BaseModel):
    target_column: str
    feature_columns: List[str]
    date_column: Optional[str] = None
    sentiment_pct: float = 0.0
    goal_target: Optional[float] = None


class ScenarioRequest(BaseModel):
    custom_inputs: Dict[str, float]
    sentiment_pct: float = 0.0
    goal_target: Optional[float] = None


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "has_data": current_dataset_df is not None,
        "current_dataset": current_dataset_name,
        "is_model_trained": engine.champion_model is not None,
        "champion_model": engine.champion_name
    }


@app.get("/api/sample-datasets")
def list_sample_datasets():
    datasets = [
        {
            "id": "ecommerce_retail",
            "name": "E-Commerce & Digital Retail",
            "icon": "shopping-cart",
            "description": "Daily store transactions tracking Ad Spend, Traffic, Discounts, Order Value, and Ratings (180 days).",
            "filename": "ecommerce_retail.csv",
            "default_goal": 20000.0
        },
        {
            "id": "saas_b2b",
            "name": "SaaS / B2B Subscriptions",
            "icon": "layers",
            "description": "Monthly recurring revenue (MRR) driven by Reps, Demos, Churn, Inbound Leads, and NPS (36 months).",
            "filename": "saas_b2b.csv",
            "default_goal": 135000.0
        },
        {
            "id": "restaurant_hospitality",
            "name": "Restaurant & Hospitality",
            "icon": "coffee",
            "description": "Daily food & beverage sales influenced by Footfall, Weather, Promos, Events, and Staff (120 days).",
            "filename": "restaurant_hospitality.csv",
            "default_goal": 11500.0
        },
        {
            "id": "electronics_tech",
            "name": "Consumer Electronics & Tech",
            "icon": "cpu",
            "description": "Weekly hardware unit sales shaped by Pricing, Tech Hype, Retail Partners, and Ad Spend (52 weeks).",
            "filename": "electronics_tech.csv",
            "default_goal": 125000.0
        }
    ]
    return {"datasets": datasets}


@app.post("/api/load-sample/{sample_id}")
def load_sample_dataset(sample_id: str):
    global current_dataset_df, current_dataset_name
    
    file_map = {
        "ecommerce_retail": "ecommerce_retail.csv",
        "saas_b2b": "saas_b2b.csv",
        "restaurant_hospitality": "restaurant_hospitality.csv",
        "electronics_tech": "electronics_tech.csv"
    }

    if sample_id not in file_map:
        raise HTTPException(status_code=404, detail="Sample dataset not found.")

    filepath = os.path.join(SAMPLE_DIR, file_map[sample_id])
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Dataset file missing on disk.")

    df = pd.read_csv(filepath)
    current_dataset_df = df
    current_dataset_name = sample_id

    inspection = engine.inspect_columns(df)
    
    # Calculate feature summary stats for UI sliders (min, max, mean, step)
    feature_stats = {}
    for col in inspection["suggested_features"]:
        s = df[col].dropna()
        min_v = float(s.min())
        max_v = float(s.max())
        mean_v = float(s.mean())
        span = max_v - min_v
        step = round(span / 100, 2) if span > 1 else 0.01
        if span == 0:
            step = 1.0
        feature_stats[col] = {
            "min": round(min_v, 2),
            "max": round(max_v, 2),
            "mean": round(mean_v, 2),
            "median": round(float(s.median()), 2),
            "step": step
        }

    return {
        "status": "success",
        "dataset_name": sample_id,
        "inspection": inspection,
        "feature_stats": feature_stats
    }


@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    global current_dataset_df, current_dataset_name
    
    if not file.filename.endswith(('.csv', '.txt')):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        if len(df) < 5:
            raise HTTPException(status_code=400, detail="CSV file must have at least 5 rows.")

        current_dataset_df = df
        current_dataset_name = file.filename

        inspection = engine.inspect_columns(df)

        # Feature stats for sliders
        feature_stats = {}
        for col in inspection["suggested_features"]:
            s = df[col].dropna()
            min_v = float(s.min())
            max_v = float(s.max())
            mean_v = float(s.mean())
            span = max_v - min_v
            step = round(span / 100, 2) if span > 1 else 0.01
            if span == 0:
                step = 1.0
            feature_stats[col] = {
                "min": round(min_v, 2),
                "max": round(max_v, 2),
                "mean": round(mean_v, 2),
                "median": round(float(s.median()), 2),
                "step": step
            }

        return {
            "status": "success",
            "filename": file.filename,
            "inspection": inspection,
            "feature_stats": feature_stats
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")


@app.post("/api/train-and-predict")
def train_and_predict(payload: TrainRequest):
    global current_dataset_df
    if current_dataset_df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Please upload a CSV or load a sample dataset first.")

    try:
        results = engine.train(
            df=current_dataset_df,
            target_col=payload.target_column,
            feature_cols=payload.feature_columns,
            date_col=payload.date_column,
            sentiment_pct=payload.sentiment_pct,
            goal_target=payload.goal_target
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training & prediction error: {str(e)}")


@app.post("/api/simulate-scenario")
def simulate_scenario(payload: ScenarioRequest):
    try:
        scenario_results = engine.predict_scenario(
            custom_inputs=payload.custom_inputs,
            sentiment_pct=payload.sentiment_pct,
            goal_target=payload.goal_target
        )
        return scenario_results
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Simulation error: {str(e)}")


@app.get("/api/export-report")
def export_report():
    if engine.champion_model is None:
        raise HTTPException(status_code=400, detail="No model has been trained yet to export.")

    report = {
        "title": "OmniSales AI Executive Sales Forecast & Strategy Report",
        "dataset": current_dataset_name,
        "target_metric": engine.target_col,
        "champion_model": engine.champion_name,
        "accuracy_r2_pct": round(engine.metrics[engine.champion_name]["r2"] * 100, 2),
        "mae": engine.metrics[engine.champion_name]["mae"],
        "rmse": engine.metrics[engine.champion_name]["rmse"],
        "baseline_historical_average": round(engine.baseline_mean, 2),
        "top_drivers": [
            {
                "parameter": k,
                "importance_pct": v["importance_pct"],
                "direction": v["direction"],
                "correlation": v["correlation"]
            }
            for k, v in engine.feature_importances.items()
        ]
    }
    return report


# Serve static web interface
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print(">> OmniSales AI Platform starting at: http://localhost:8000")
    print("="*70 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
