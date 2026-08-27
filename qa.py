"""
qa.py
-----
Step 3 of the RAG pipeline: the "generation" half.

This is where retrieval and generation actually meet:
  1. Take the user's question
  2. Use VectorStore.search() to find the most relevant chunks
  3. Stuff those chunks into a prompt as "context"
  4. Ask the LLM to answer USING ONLY that context
  5. Return the answer plus which chunks it came from (so you can
     show your sources -- this is what makes a RAG answer
     trustworthy instead of a guess)
"""

import os
from openai import OpenAI
from vector_store import VectorStore

# Two generation backends are supported, same pattern as the embedding
# backend in vector_store.py:
#   "openai" - gpt-4o-mini, needs OPENAI_API_KEY + billing set up
#   "groq"   - free API access to open models (e.g. Llama), no billing needed
# Set GENERATION_BACKEND=groq to run this step completely free.
GENERATION_BACKEND = os.environ.get("GENERATION_BACKEND", "openai")

# Only initialize OpenAI client if actually using it (lazy-init)
_openai_client = None
CHAT_MODEL = "gpt-4o-mini"  # cheap + fast, plenty good for this project

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client

_groq_client = None  # lazy-init so importing this file doesn't require GROQ_API_KEY to be set


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _groq_client


GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")  # mixtral was decommissioned; override via .env if needed

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY \
the provided context from a document. If the answer isn't in the context, say \
"I couldn't find that in the document" instead of guessing. Be concise."""


def _generate_answer(system_prompt: str, user_prompt: str) -> str:
    """Route the chat completion call to whichever backend is configured."""
    if GENERATION_BACKEND == "groq":
        response = _get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content

    # default: OpenAI
    response = _get_openai_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def answer_question(store: VectorStore, question: str, top_k: int = 3) -> dict:
    """
    Full RAG query: retrieve relevant chunks, then generate an
    answer grounded in them.

    Returns a dict with the answer and the source chunks used,
    so the UI can show "here's why I said that."
    """
    # --- Retrieval ---
    retrieved = store.search(question, top_k=top_k)
    context = "\n\n---\n\n".join(chunk for chunk, _dist in retrieved)

    # --- Generation ---
    user_prompt = f"""Context from the document:
{context}

Question: {question}

Answer using only the context above."""

    answer = _generate_answer(SYSTEM_PROMPT, user_prompt)

    return {
        "question": question,
        "answer": answer,
        "sources": [chunk for chunk, _dist in retrieved],
    }


if __name__ == "__main__":
    # End-to-end manual test, same sample chunks as vector_store.py
    store = VectorStore()
    store.build([
        "The Eiffel Tower is located in Paris, France and was completed in 1889.",
        "Python is a popular programming language for AI and data science.",
        "FAISS is a library developed by Meta for efficient similarity search of vectors.",
    ])

    result = answer_question(store, "Who made FAISS and what is it for?")
    print("Q:", result["question"])
    print("A:", result["answer"])
    print("\nSources used:")
    for src in result["sources"]:
        print(" -", src)
