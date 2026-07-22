"""
retrieve_context.py
-------------------
Retrieve relevant chunks using Hybrid Retrieval.
"""

from pathlib import Path
import pickle
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from vector_representation import load_corpus


# ==========================
# Paths
# ==========================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# ==========================
# Load Saved Objects
# ==========================

corpus = load_corpus()

with open(DATA_DIR / "tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(DATA_DIR / "tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

embeddings = np.load(
    DATA_DIR / "broiler_embeddings.npy"
)

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================
# TF-IDF Retrieval
# ==========================

def retrieve_with_tfidf(query, top_k=5):

    query_vector = vectorizer.transform([query])

    similarity = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    indices = similarity.argsort()[::-1][:top_k]

    results = corpus.iloc[indices].copy()

    results["score"] = similarity[indices]

    return results[
        ["chunk_id", "score", "document"]
    ]


# ==========================
# Embedding Retrieval
# ==========================

def retrieve_with_embeddings(query, top_k=5):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    similarity = cosine_similarity(
        query_embedding,
        embeddings
    ).flatten()

    indices = similarity.argsort()[::-1][:top_k]

    results = corpus.iloc[indices].copy()

    results["score"] = similarity[indices]

    return results[
        ["chunk_id", "score", "document"]
    ]


# ==========================
# Hybrid Retrieval
# ==========================

def retrieve_with_hybrid(query, top_k=5):

    tfidf_results = retrieve_with_tfidf(query, top_k)

    embedding_results = retrieve_with_embeddings(query, top_k)

    combined = {}

    for _, row in tfidf_results.iterrows():
        combined[row["chunk_id"]] = row["score"]

    for _, row in embedding_results.iterrows():
        combined[row["chunk_id"]] = (
            combined.get(row["chunk_id"], 0)
            + row["score"]
        )

    ranked = sorted(
        combined.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    ids = [x[0] for x in ranked]

    results = corpus[
        corpus["chunk_id"].isin(ids)
    ].copy()

    results["score"] = results["chunk_id"].map(dict(ranked))

    return results.sort_values(
        "score",
        ascending=False
    )


# ==========================
# Test
# ==========================

if __name__ == "__main__":

    print(
        retrieve_with_hybrid(
            "What is crop fill?"
        )
    )
