"""
preprocessing.py
----------------
Clean and preprocess the extracted handbook text.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from documents import load_pdf


# ==========================
# Download NLTK Resources
# ==========================

def download_nltk():
    """Download required NLTK resources."""

    resources = [
        "punkt",
        "punkt_tab",
        "stopwords"
    ]

    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)


download_nltk()

STOP_WORDS = set(stopwords.words("english"))


# ==========================
# Text Preprocessing
# ==========================

def preprocess_text(text: str) -> str:
    """
    Clean handbook text.

    Steps
    -----
    1. Lowercase
    2. Remove line breaks
    3. Remove numbers & punctuation
    4. Remove extra spaces
    5. Tokenize
    6. Remove stopwords
    """

    # Lowercase
    text = text.lower()

    # Remove line breaks
    text = text.replace("\n", " ")

    # Keep only letters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]

    return " ".join(tokens)


# ==========================
# Test
# ==========================

if __name__ == "__main__":

    raw_text = load_pdf()

    clean_text = preprocess_text(raw_text)

    print("=" * 60)
    print("🐥 Chicksy - Text Preprocessing")
    print("=" * 60)

    print(f"Raw Characters   : {len(raw_text):,}")
    print(f"Clean Characters : {len(clean_text):,}")

    print("\nPreview:\n")
    print(clean_text[:500])

    print("\n" + "=" * 60)