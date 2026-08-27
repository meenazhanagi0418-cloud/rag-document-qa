"""
main.py
-------
Ties the whole pipeline together into one runnable script:
  PDF -> chunks -> embeddings/FAISS -> question -> answer

This is what you'd run from the command line to test the full
system end-to-end before wrapping it in a UI on Day 3.

Usage:
    python src/main.py data/yourfile.pdf
"""

import sys
from loader import load_pdf_text, chunk_text
from vector_store import VectorStore
from qa import answer_question


def build_pipeline(pdf_path: str) -> VectorStore:
    print(f"[main] loading {pdf_path} ...")
    text = load_pdf_text(pdf_path)
    print(f"[main] extracted {len(text)} characters")

    chunks = chunk_text(text)
    print(f"[main] split into {len(chunks)} chunks")

    store = VectorStore()
    store.build(chunks)
    return store


def repl(store: VectorStore):
    """Simple command-line question loop for manual testing."""
    print("\nAsk questions about the document (type 'quit' to exit)\n")
    while True:
        question = input("Q: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        result = answer_question(store, question)
        print(f"\nA: {result['answer']}\n")
        print("Sources:")
        for i, src in enumerate(result["sources"], start=1):
            preview = src[:120].replace("\n", " ")
            print(f"  [{i}] {preview}...")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        sys.exit(1)

    store = build_pipeline(sys.argv[1])
    repl(store)
