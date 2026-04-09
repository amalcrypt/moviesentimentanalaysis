"""
main.py — FastAPI backend for the Sentiment Analysis service.

Endpoints:
  POST /predict            — Classify review text
  GET  /models             — List available models
  GET  /health             — Health check
  GET  /results/classical  — Classical model results JSON
  GET  /results/transformer — Transformer results JSON
  GET  /results/images/{f} — Evaluation images (ROC, CM)
"""

import os
import sys
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import predict, get_available_models

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


# ---------------------------------------------------------------------------
# Lifespan — warm up models on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load default model on startup for fast first inference."""
    available = get_available_models()
    if available:
        try:
            predict("warm up", available[0]["name"])
            print(f"[API] Warmed up model: {available[0]['name']}")
        except Exception as e:
            print(f"[API] Warm-up failed: {e}")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sentiment Analysis API",
    description="Analyze sentiment of text using classical ML and transformer models.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Review text to analyze")
    model: str = Field(
        default="distilbert",
        description="Model to use: distilbert",
    )


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    label: int
    model: str


class ModelInfo(BaseModel):
    name: str
    type: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """API root."""
    return {"message": "Sentiment Analysis API — visit /docs for API documentation."}


@app.post("/predict", response_model=PredictResponse)
async def predict_sentiment(req: PredictRequest):
    """Predict the sentiment of the given text."""
    available = [m["name"] for m in get_available_models()]

    if not available:
        raise HTTPException(
            status_code=503,
            detail="No trained models available. Run training first.",
        )

    model_name = req.model
    if model_name not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not available. Choose from: {available}",
        )

    try:
        result = predict(req.text, model_name)
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_model=list[ModelInfo])
async def list_models():
    """List all available trained models."""
    return get_available_models()


@app.get("/health")
async def health():
    """Health check."""
    models = get_available_models()
    return {
        "status": "healthy",
        "models_available": len(models),
        "models": [m["name"] for m in models],
    }


# ---------------------------------------------------------------------------
# Results endpoints — serve evaluation data to the frontend
# ---------------------------------------------------------------------------
@app.get("/results/transformer")
async def get_transformer_results():
    """Return transformer model evaluation results."""
    path = os.path.join(RESULTS_DIR, "transformer_results.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Transformer results not found.")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

