# Document Q&A — RAG Assistant

An AI system that answers questions about any PDF document using
Retrieval-Augmented Generation (RAG): it retrieves the most relevant
chunks of the document for a given question, then uses an LLM to
generate an answer grounded in that retrieved context — with sources
cited, not just a guess from the model's training data.

## Architecture

```
PDF Upload
    │
    ▼
[loader.py]     Extract text → split into overlapping chunks
    │
    ▼
[vector_store.py]   Embed each chunk → store in FAISS index
    │
    ▼
User Question
    │
    ▼
[vector_store.py]   Embed question → retrieve top-k similar chunks
    │
    ▼
[qa.py]         Build prompt (context + question) → LLM generates answer
    │
    ▼
[app.py]        Streamlit UI displays answer + source chunks
```

## Why these design choices

- **Chunking with overlap (800 chars, 150 overlap):** prevents important
  sentences from being cut in half between chunks, at the cost of some
  storage redundancy.
- **FAISS IndexFlatL2:** simplest exact nearest-neighbor search — no
  external database needed, fast enough for documents up to a few
  thousand chunks.
- **Two embedding backends (OpenAI or local sentence-transformers):**
  lets the project run with zero API cost during development, while
  keeping the option for higher-quality embeddings in production.
- **Low temperature (0.2) for generation:** RAG answers should be
  grounded and consistent, not creative — this reduces hallucination.
- **Source citation on every answer:** without this, there's no way to
  verify whether the model actually used the document or made
  something up. This is the single most important design decision in
  the project.

## Running it

```bash
pip install -r requirements.txt

# Option A: OpenAI embeddings (needs OPENAI_API_KEY, costs ~fractions of a cent)
export OPENAI_API_KEY="your-key"

# Option B: free local embeddings instead
export EMBEDDING_BACKEND="local"
export OPENAI_API_KEY="your-key"   # still needed for the answer-generation step

# CLI version
python src/main.py data/yourfile.pdf

# Or the web UI
streamlit run src/app.py
```

## What I'd improve with more time

- Add an evaluation script that scores retrieval accuracy on a labeled
  set of question/answer pairs, rather than testing by hand
- Support multi-document collections instead of one PDF at a time
- Add re-ranking after retrieval for better precision on longer documents
- Handle scanned/image-only PDFs with OCR

## Stack

Python · FAISS · OpenAI API (embeddings + gpt-4o-mini) · sentence-transformers
(optional local embeddings) · Streamlit · pdfplumber
