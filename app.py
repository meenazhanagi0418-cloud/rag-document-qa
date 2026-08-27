"""
app.py
------
Day 3: the demoable face of the project.

Streamlit UI: upload a PDF, ask questions, see grounded answers
with their source chunks. Everything underneath (loader,
vector_store, qa) is exactly what we built on Day 1 and Day 2 --
this file just wires it to a browser instead of a terminal.

Run with: streamlit run src/app.py
"""

import sys
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from loader import load_pdf_text, chunk_text
from vector_store import VectorStore
from qa import answer_question

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄")
st.title("📄 Document Q&A — RAG Assistant")
st.caption("Upload a PDF, ask questions, get answers grounded in the document with sources cited.")

# Session state holds the built index across reruns (Streamlit reruns
# the whole script on every interaction, so we don't want to re-embed
# the PDF every time the user asks a new question)
if "store" not in st.session_state:
    st.session_state.store = None
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None and st.session_state.store is None:
    with st.spinner("Reading and indexing document... (this only happens once per upload)"):
        # write to a temp file since pdfplumber needs a real file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            text = load_pdf_text(tmp_path)
            if not text.strip():
                st.error("Couldn't extract any text from this PDF (it may be scanned images).")
            else:
                chunks = chunk_text(text)
                store = VectorStore()
                store.build(chunks)
                st.session_state.store = store
                st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}")
        except Exception as e:
            st.error(f"Something went wrong processing the PDF: {e}")
        finally:
            os.remove(tmp_path)

if st.session_state.store is not None:
    question = st.text_input("Ask a question about the document")

    if st.button("Ask") and question.strip():
        with st.spinner("Searching document and generating answer..."):
            try:
                result = answer_question(st.session_state.store, question)
                st.session_state.history.append(result)
            except Exception as e:
                st.error(f"Couldn't get an answer: {e}")

    # Show conversation history, most recent first
    for result in reversed(st.session_state.history):
        st.markdown(f"**Q: {result['question']}**")
        st.write(result["answer"])
        with st.expander("Sources used"):
            for i, src in enumerate(result["sources"], start=1):
                st.markdown(f"**[{i}]** {src}")
        st.divider()

    if st.button("Start over with a new document"):
        st.session_state.store = None
        st.session_state.history = []
        st.rerun()
else:
    st.info("Upload a PDF above to get started.")
