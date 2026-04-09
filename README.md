# 🧠 SentimentAI — Deep Learning Sentiment Analysis

An end-to-end sentiment analysis system that classifies text (movie reviews, product feedback, etc.) as **positive** or **negative** using a fine-tuned **DistilBERT** transformer model. The project features a premium glassmorphism Next.js UI separated from a scalable FastAPI backend.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Next.js](https://img.shields.io/badge/Next.js-14+-black?logo=next.js)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Deployment](#-deployment)

---

## ✨ Features

- **Transformer Model** — Highly accurate, fine-tuned DistilBERT transformer model.
- **Full Pipeline** — Data preprocessing → custom HuggingFace fine-tuning → inference API.
- **REST API** — FastAPI backend built for production ML Serving.
- **Beautiful Web UI** — Dark-themed, animated glassmorphism interface built with Next.js.
- **Highlighted Results** — Dynamic text parsing that highlights positive/negative keywords locally in the UI.
- **Docker-Ready** — Packaged Dockerfile configuration ready for dedicated hosts (Render, HuggingFace Spaces).

---

## 🏗️ Architecture

```
┌────────────────┐     ┌──────────────┐     ┌──────────────┐
│  Next.js App   │────▶│   FastAPI    │────▶│ DistilBERT   │
│   (Frontend)   │◀────│   Backend    │◀────│  (PyTorch)   │
└────────────────┘     └──────────────┘     └──────────────┘
```

---

## 🚀 Quick Start

### 1. Setup Backend ML Service

```bash
# Set up Python virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install strictly the API/Transformer requirements
pip install -r api-requirements.txt
```

### 2. Train Transformer (Optional, requires GPU)

```bash
python src/train_transformer.py
```
*Note: The script is pre-configured with optimized hyperparameters (learning_rate=2e-5, warmup=0.1) for IMDb fine-tuning.*

### 3. Launch Backend Server

```bash
uvicorn api.main:app --reload
```
The ML API will spin up at **http://localhost:8000**.

### 4. Launch Next.js Frontend

Open a new terminal session:
```bash
cd frontend
npm install
npm run dev
```
Open **http://localhost:3000** in your browser to interact with the UI.

---

## 📁 Project Structure

```
ml/
├── api-requirements.txt      # Slimmed dependencies for inference
├── backend.Dockerfile        # Production container instruction
├── data/                     # Dataset (auto-downloaded)
├── models/                   # Saved DistilBERT HuggingFace artifacts
├── results/                  # Transformer evaluation metrics (json)
├── src/
│   ├── preprocess.py         # Text cleaning routines
│   ├── train_transformer.py  # Fine-tune DistilBERT script
│   └── predict.py            # Local PyTorch inference utilities
├── api/
│   └── main.py               # FastAPI server router
└── frontend/                 # Next.js Application Source
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify text sentiment |
| `/models` | GET | List available active transformer models |
| `/health` | GET | Ping API Health check |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was an absolute masterpiece!", "model": "distilbert"}'
```

### Response

```json
{
  "sentiment": "positive",
  "confidence": 0.9921,
  "label": 1,
  "model": "distilbert"
}
```

---

## 🌍 Deployment

This repository is designed with a **Split Deployment** scheme to bypass Vercel's 250MB serverless limit (since PyTorch weights are incredibly heavy).

1. **Backend**: Provide `backend.Dockerfile` to a service like **Render** or **Hugging Face Spaces** (Docker Mode) to host the ML API effortlessly.
2. **Frontend**: Deploy the `/frontend` app directly to **Vercel**. Set `NEXT_PUBLIC_API_URL` environment variable in Vercel to point to your new Render/HF backend URL!

---

## 🔮 Future Improvements

- [ ] Add RoBERTa model switching via frontend config
- [ ] Multi-class sentiment scale (positive, neutral, negative)
- [ ] Batch prediction endpoint
- [x] Docker containerization
- [ ] CI/CD pipeline with GitHub Actions

---

## 📄 License

MIT License — feel free to use this project in your portfolio!
