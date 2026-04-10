# 🤖 SentimentAI — Deep Learning Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black?logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end sentiment analysis system that classifies text (movie reviews, product feedback, etc.) as **positive** or **negative** using a fine-tuned **DistilBERT** transformer model. The project features a premium glassmorphism Next.js UI separated from a scalable FastAPI backend.

## ✨ Features

- **Transformer Model**: Highly accurate, fine-tuned DistilBERT transformer model.
- **Full Pipeline**: Data preprocessing → custom HuggingFace fine-tuning → inference API.
- **REST API**: FastAPI backend built for production ML Serving.
- **Beautiful Web UI**: Dark-themed, animated glassmorphism interface built with Next.js.
- **Highlighted Results**: Dynamic text parsing that highlights positive/negative keywords locally in the UI.
- **Docker-Ready**: Packaged Dockerfile configuration ready for dedicated hosts (Render, HuggingFace Spaces).

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Next.js    │ ──▶ │   FastAPI   │ ──▶ │ DistilBERT  │
│  Frontend   │     │   Backend   │     │  (PyTorch)  │
└─────────────┘     └─────────────┘     └─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Git

### 1. Clone the Repository

`ash
git clone https://github.com/yourusername/SentimentAI.git
cd SentimentAI
`

### 2. Setup Backend ML Service

`ash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r api-requirements.txt
`

### 3. Train Transformer (Optional, requires GPU)

`ash
python src/train_transformer.py
`

*Note: The script is pre-configured with optimized hyperparameters (learning_rate=2e-5, warmup=0.1) for IMDb fine-tuning.*

### 4. Launch Backend Server

`ash
uvicorn api.main:app --reload
`

The ML API will be available at **http://localhost:8000**.

### 5. Launch Next.js Frontend

In a new terminal:

`ash
cd frontend
npm install
npm run dev
`

Open **http://localhost:3000** in your browser.

## 📁 Project Structure

`
ml/
├── api-requirements.txt      # Slimmed dependencies for inference
├── Dockerfile                # Production container instructions
├── data/                     # Dataset (auto-downloaded)
├── models/                   # Saved DistilBERT HuggingFace artifacts
├── results/                  # Transformer evaluation metrics (JSON)
├── src/
│   ├── preprocess.py         # Text cleaning routines
│   ├── train_transformer.py  # Fine-tune DistilBERT script
│   └── predict.py            # Local PyTorch inference utilities
├── api/
│   └── main.py               # FastAPI server router
└── frontend/                 # Next.js Application Source
`

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| /predict | POST | Classify text sentiment |
| /models | GET | List available active transformer models |
| /health | GET | API health check |

### Example Request

`ash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was an absolute masterpiece!", "model": "distilbert"}'
`

### Response

`json
{
  "sentiment": "positive",
  "confidence": 0.9921,
  "label": 1,
  "model": "distilbert"
}
`

## 🚢 Deployment

This repository uses a **Split Deployment** scheme to bypass Vercel's 250MB serverless limit.

1. **Backend**: Deploy Dockerfile to Render or Hugging Face Spaces.
2. **Frontend**: Deploy /frontend to Vercel. Set NEXT_PUBLIC_API_URL to your backend URL.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.
