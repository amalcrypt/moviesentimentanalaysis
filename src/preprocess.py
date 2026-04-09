"""
preprocess.py -- Text cleaning, preprocessing, and dataset preparation
"""
import re, os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def _download_nltk_data():
    for res in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass

_download_nltk_data()

_lemmatizer = WordNetLemmatizer()
try:
    _stop_words = set(stopwords.words("english"))
except LookupError:
    _stop_words = set()

_RE_HTML = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_NON_ALPHA = re.compile(r"[^a-zA-Z\s]")
_RE_MULTI_SPACE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    text = _RE_HTML.sub("", text)
    text = _RE_URL.sub("", text)
    text = text.lower()
    text = _RE_NON_ALPHA.sub(" ", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    return text.strip()

def tokenize_and_lemmatize(text: str) -> str:
    tokens = word_tokenize(text)
    return " ".join([_lemmatizer.lemmatize(t) for t in tokens if t not in _stop_words and len(t) > 2])

def preprocess_text(text: str) -> str:
    return tokenize_and_lemmatize(clean_text(text))

def load_imdb_data(cache_dir="data"):
    os.makedirs(cache_dir, exist_ok=True)
    csv_path = os.path.join(cache_dir, "imdb_dataset.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    ds = load_dataset("imdb", cache_dir=cache_dir)
    rows = []
    for split in ["train", "test"]:
        for ex in ds[split]:
            rows.append({"review": ex["text"], "label": ex["label"], "sentiment": "positive" if ex["label"] == 1 else "negative"})
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df

def prepare_data(test_size=0.2, random_state=42, cache_dir="data"):
    df = load_imdb_data(cache_dir)
    processed_path = os.path.join(cache_dir, "imdb_processed.csv")
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
    else:
        # Preprocessing without tqdm for simple rewrite
        print("Preprocessing (this may take a bit)...")
        df["review"] = df["review"].apply(preprocess_text)
        df.to_csv(processed_path, index=False)
    
    return train_test_split(df["review"], df["label"], test_size=test_size, random_state=random_state, stratify=df["label"])

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    print("Done")
