"""
train_transformer.py -- Fine-tune DistilBERT
"""
import os
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/distilbert-sentiment"
RESULTS_DIR = "results"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": round(accuracy_score(labels, preds), 4)}

def tokenize_data(dataset, tokenizer):
    def _tok(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)
    dataset = dataset.map(_tok, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset

def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, id2label={0: "negative", 1: "positive"}, label2id={"negative": 0, "positive": 1})
    dataset = load_dataset("imdb")
    train_val = dataset["train"].train_test_split(test_size=0.1, seed=42)
    
    train_dataset = tokenize_data(train_val["train"], tokenizer)
    val_dataset = tokenize_data(train_val["test"], tokenizer)
    test_dataset = tokenize_data(dataset["test"], tokenizer)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=3,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        eval_strategy="steps", eval_steps=500, save_strategy="steps", save_steps=500,
        load_best_model_at_end=True, metric_for_best_model="accuracy", fp16=True,
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=val_dataset,
        compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()
    res = trainer.evaluate(test_dataset)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "transformer_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res

if __name__ == "__main__":
    train()
