"""
documents.py
------------
Load the Broiler Handbook PDF and extract its text.
"""

from pathlib import Path
import fitz  # PyMuPDF


# ==========================
# File Paths
# ==========================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "Aviagen-ROSS-Broiler-Handbook-EN.pdf"


# ==========================
# PDF Loader
# ==========================

def load_pdf(pdf_path: Path = PDF_PATH) -> str:
    """
    Extract all text from the Broiler Handbook PDF.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file.

    Returns
    -------
    str
        Complete extracted text.
    """

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found:\n{pdf_path}"
        )

    document = fitz.open(pdf_path)

    pages = []

    for page in document:
        pages.append(page.get_text())

    document.close()

    return "\n".join(pages)


# ==========================
# Test
# ==========================

if __name__ == "__main__":

    text = load_pdf()

    print("=" * 60)
    print("🐥 Chicksy - Document Loader")
    print("=" * 60)

    print("✅ PDF Loaded Successfully")
    print(f"Characters : {len(text):,}")

    print("\nPreview:\n")
    print(text[:500])

    print("\n" + "=" * 60)