"""
vector_store.py
----------------
Step 2 of the RAG pipeline: turn text chunks into embeddings
(numeric vectors that capture meaning) and store them in a
FAISS index so we can search by "meaning similarity" instead
of exact keyword matching.

What's an embedding, concretely?
  - A list of numbers (e.g. 1536 floats) representing a piece
    of text's meaning.
  - Two chunks about similar topics end up with numerically
    similar vectors, even if they don't share exact words.

Why FAISS?
  - It's a local, free, fast library for searching through
    vectors to find the closest matches. No external database
    or server needed -- perfect for a portfolio project.
"""

import os
import numpy as np
import faiss
from openai import OpenAI

# Two embedding backends are supported:
#   "openai" - text-embedding-3-small, needs OPENAI_API_KEY, costs a few cents
#   "local"  - sentence-transformers, runs on your machine, completely free
# Set EMBEDDING_BACKEND=local if you don't have (or don't want to spend)
# OpenAI API credits. Same interface either way, so the rest of the
# pipeline (vector_store, qa, main) doesn't need to know or care which
# one is active.
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "openai")

EMBED_MODEL = "text-embedding-3-small"  # cheap + solid quality

_openai_embed_client = None  # lazy-init so importing this file doesn't require OPENAI_API_KEY to be set


def _get_openai_embed_client():
    global _openai_embed_client
    if _openai_embed_client is None:
        _openai_embed_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_embed_client


_local_model = None  # lazy-loaded so importing this file doesn't force a download


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2: small, fast, free, good enough for a portfolio project
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Turn a list of text chunks into embedding vectors.
    Backend is chosen by the EMBEDDING_BACKEND env var (see above).
    Returns a numpy array of shape (num_chunks, embedding_dim).
    """
    if EMBEDDING_BACKEND == "local":
        vectors = _get_local_model().encode(texts, convert_to_numpy=True)
        return vectors.astype("float32")

    # default: OpenAI API
    response = _get_openai_embed_client().embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


class VectorStore:
    """
    Thin wrapper around a FAISS index that also remembers which
    text chunk each vector came from (FAISS itself only stores
    numbers, not the original text).
    """

    def __init__(self):
        self.index = None
        self.chunks: list[str] = []

    def build(self, chunks: list[str]):
        """Embed all chunks and build the FAISS index from scratch."""
        self.chunks = chunks
        vectors = embed_texts(chunks)
        dimension = vectors.shape[1]

        # IndexFlatL2 = simplest possible FAISS index: exact nearest-
        # neighbor search via Euclidean distance. Fine for a few
        # hundred/thousand chunks (typical portfolio-project scale).
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(vectors)
        print(f"[vector_store] indexed {len(chunks)} chunks (dim={dimension})")

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """
        Embed the query, find the top_k most similar chunks.
        Returns list of (chunk_text, distance) -- smaller distance
        = more similar.
        """
        if self.index is None:
            raise RuntimeError("Call build() before search()")

        query_vector = embed_texts([query])
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 if fewer than top_k matches exist
                continue
            results.append((self.chunks[idx], float(dist)))
        return results


if __name__ == "__main__":
    # Quick manual test with a few made-up chunks
    store = VectorStore()
    sample_chunks = [
        "The Eiffel Tower is located in Paris, France.",
        "Python is a popular programming language for AI and data science.",
        "FAISS is a library for efficient similarity search of vectors.",
    ]
    store.build(sample_chunks)

    results = store.search("What language is used for AI projects?", top_k=2)
    for chunk, dist in results:
        print(f"[dist={dist:.3f}] {chunk}")
