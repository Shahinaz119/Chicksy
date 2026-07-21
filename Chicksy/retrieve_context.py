"""
retrieve_context.py
-------------------
Retrieve relevant chunks using TF-IDF, Embeddings, and Hybrid Retrieval.
"""

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from vector_representation import (
    load_corpus,
    create_tfidf,
    create_embeddings
)


# =====================================
# Load Everything Once
# =====================================

corpus = load_corpus()

vectorizer, tfidf_matrix = create_tfidf(corpus)

embedding_model, embeddings = create_embeddings(corpus)


# =====================================
# TF-IDF Retrieval
# =====================================

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


# =====================================
# Embedding Retrieval
# =====================================

def retrieve_with_embeddings(query, top_k=5):

    query_embedding = embedding_model.encode([query])

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


# =====================================
# Hybrid Retrieval (Weighted RRF)
# =====================================

def retrieve_with_hybrid(
    query,
    top_k=5,
    alpha=0.5,
    k_constant=60
):

    tfidf_ranked = retrieve_with_tfidf(
        query,
        top_k=len(corpus)
    )["chunk_id"].tolist()

    embedding_ranked = retrieve_with_embeddings(
        query,
        top_k=len(corpus)
    )["chunk_id"].tolist()

    scores = {}

    for rank, cid in enumerate(tfidf_ranked, start=1):

        scores[cid] = scores.get(
            cid,
            0
        ) + alpha * (1 / (k_constant + rank))

    for rank, cid in enumerate(
        embedding_ranked,
        start=1
    ):

        scores[cid] = scores.get(
            cid,
            0
        ) + (1 - alpha) * (
            1 / (k_constant + rank)
        )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    ids = [cid for cid, _ in ranked]

    results = corpus[
        corpus["chunk_id"].isin(ids)
    ].copy()

    results["score"] = results[
        "chunk_id"
    ].map(dict(ranked))

    return results.sort_values(
        "score",
        ascending=False
    )


# =====================================
# Test
# =====================================

if __name__ == "__main__":

    query = "What is crop fill?"

    print("=" * 60)

    print("TF-IDF")

    print(
        retrieve_with_tfidf(query)
    )

    print("=" * 60)

    print("Embeddings")

    print(
        retrieve_with_embeddings(query)
    )

    print("=" * 60)

    print("Hybrid")

    print(
        retrieve_with_hybrid(query)
    )