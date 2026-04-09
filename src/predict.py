"""
predict.py -- Inference utilities for sentiment prediction.
Supports transformer (DistilBERT) model exclusively.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "models"))
TRANSFORMER_DIR = os.path.join(MODEL_DIR, "distilbert-sentiment")

_transformer_cache = {}

def predict_transformer(text: str) -> dict:
    if "model" not in _transformer_cache:
        if not os.path.exists(TRANSFORMER_DIR):
            raise FileNotFoundError("Transformer model not found. Run train_transformer.py first.")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        _transformer_cache["tokenizer"] = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
        _transformer_cache["model"] = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
        _transformer_cache["model"].eval()
        _transformer_cache["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _transformer_cache["model"].to(_transformer_cache["device"])

    import torch
    tokenizer = _transformer_cache["tokenizer"]
    model = _transformer_cache["model"]
    device = _transformer_cache["device"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding="max_length").to(device)
    
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    confidence, label = torch.max(probs, dim=-1)
    label = int(label.item())

    return {
        "sentiment": "positive" if label == 1 else "negative",
        "confidence": round(float(confidence.item()), 4),
        "label": label,
        "model": "distilbert"
    }

def predict(text: str, model_name: str = "distilbert") -> dict:
    return predict_transformer(text)

def get_available_models() -> list:
    available = []
    model_ready = any(
        os.path.exists(os.path.join(TRANSFORMER_DIR, f))
        for f in ("model.safetensors", "pytorch_model.bin", "config.json")
    )
    if model_ready:
        available.append({"name": "distilbert", "type": "transformer"})
    return available

if __name__ == "__main__":
    for m in get_available_models():
        print(m)
