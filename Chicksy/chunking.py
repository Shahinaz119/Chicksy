"""
chunking.py
-----------
Split the cleaned handbook text into chunks
and save them as a corpus.
"""

from pathlib import Path
import pandas as pd

from documents import load_pdf
from preprocessing import preprocess_text


# ==========================
# Paths
# ==========================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# ==========================
# Chunking
# ==========================

def create_chunks(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Split text into overlapping chunks.
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ==========================
# Build Corpus
# ==========================

def build_corpus():

    raw_text = load_pdf()

    clean_text = preprocess_text(raw_text)

    chunks = create_chunks(clean_text)

    corpus = pd.DataFrame(
        {
            "chunk_id": range(len(chunks)),
            "document": chunks
        }
    )

    return corpus


# ==========================
# Test
# ==========================

if __name__ == "__main__":

    corpus = build_corpus()

    output_path = DATA_DIR / "broiler_corpus.csv"

    corpus.to_csv(output_path, index=False)

    print("=" * 60)
    print("🐥 Chicksy - Chunking")
    print("=" * 60)

    print(f"Chunks Created : {len(corpus)}")
    print(f"Corpus Saved   : {output_path}")

    print("\nFirst Chunk:\n")
    print(corpus.iloc[0]["document"][:500])

    print("\n✅ Corpus saved successfully!")
    print("=" * 60)