"""
loader.py
---------
Step 1 of the RAG pipeline: turn a PDF into clean text, then
split that text into small overlapping chunks.

Why chunk at all?
  - LLMs (and embedding models) work better on small, focused
    pieces of text than one giant blob.
  - Smaller chunks -> more precise retrieval later (Day 1's job
    is just to get good chunks; Day 2 uses them for search).

Why overlap chunks?
  - If a sentence that matters gets cut in half between two
    chunks, overlap means it still appears whole in at least
    one chunk.
"""

import pdfplumber


def load_pdf_text(pdf_path: str) -> str:
    """Extract all text from a PDF file, page by page."""
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:  # some pages (scanned images, blank pages) return None
                full_text.append(text)
            else:
                print(f"[warn] page {page_num} had no extractable text")
    return "\n".join(full_text)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Split text into overlapping chunks of roughly `chunk_size`
    characters, moving forward by (chunk_size - overlap) each time.

    chunk_size=800, overlap=150 is a reasonable starting point for
    Q&A-style retrieval. Too small -> chunks lose context. Too
    large -> retrieval gets less precise because each chunk covers
    too many unrelated ideas.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    text = text.strip()

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # step forward, but re-cover the overlap

    return chunks


if __name__ == "__main__":
    # Quick manual test: point this at any PDF in data/
    import sys

    if len(sys.argv) < 2:
        print("Usage: python loader.py <path_to_pdf>")
        sys.exit(1)

    text = load_pdf_text(sys.argv[1])
    print(f"Extracted {len(text)} characters")

    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")
    print("\n--- First chunk preview ---")
    print(chunks[0][:300])
