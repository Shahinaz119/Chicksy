"""
create_chroma_store.py
----------------------
Create and populate the Chroma vector database.
"""

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


def create_chroma_db():

    corpus = pd.read_csv("data/broiler_corpus.csv")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = model.encode(
        corpus["document"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True
    )

    client = chromadb.PersistentClient(path="chroma_db")

    collection = client.get_or_create_collection(
        name="broiler_handbook"
    )

    # لو شغلتي الملف أكتر من مرة
    try:
        client.delete_collection("broiler_handbook")
    except:
        pass

    collection = client.get_or_create_collection(
        name="broiler_handbook"
    )

    collection.add(
        ids=corpus["chunk_id"].astype(str).tolist(),
        documents=corpus["document"].tolist(),
        embeddings=embeddings.tolist()
    )

    return collection


if __name__ == "__main__":

    collection = create_chroma_db()

    print("=" * 60)
    print("Chicksy - Chroma Store")
    print("=" * 60)

    print("Collection Created Successfully!")

    print("Total Chunks :", collection.count())