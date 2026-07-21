"""
vector_representation.py
------------------------
Build and save TF-IDF vectors and sentence embeddings.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


# ==========================
# Paths
# ==========================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CORPUS_PATH = DATA_DIR / "broiler_corpus.csv"
TFIDF_VECTORIZER_PATH = DATA_DIR / "tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH = DATA_DIR / "tfidf_matrix.pkl"
EMBEDDINGS_PATH = DATA_DIR / "broiler_embeddings.npy"


# ==========================
# Load Corpus
# ==========================

def load_corpus():

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            "Run chunking.py first to create broiler_corpus.csv"
        )

    return pd.read_csv(CORPUS_PATH)


# ==========================
# TF-IDF
# ==========================

def create_tfidf(corpus):

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        corpus["document"]
    )

    return vectorizer, tfidf_matrix


# ==========================
# Embeddings
# ==========================

def create_embeddings(corpus):

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = model.encode(
        corpus["document"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings


# ==========================
# Save Files
# ==========================

def save_files(vectorizer, tfidf_matrix, embeddings):

    with open(TFIDF_VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(TFIDF_MATRIX_PATH, "wb") as f:
        pickle.dump(tfidf_matrix, f)

    np.save(
        EMBEDDINGS_PATH,
        embeddings
    )


# ==========================
# Main
# ==========================

if __name__ == "__main__":

    corpus = load_corpus()

    vectorizer, tfidf_matrix = create_tfidf(corpus)

    embeddings = create_embeddings(corpus)

    save_files(
        vectorizer,
        tfidf_matrix,
        embeddings
    )

    print("=" * 60)
    print("🐥 Chicksy - Vector Representation")
    print("=" * 60)

    print(f"Corpus      : {corpus.shape}")
    print(f"TF-IDF      : {tfidf_matrix.shape}")
    print(f"Embeddings  : {embeddings.shape}")

    print("\n✅ tfidf_vectorizer.pkl saved")
    print("✅ tfidf_matrix.pkl saved")
    print("✅ broiler_embeddings.npy saved")