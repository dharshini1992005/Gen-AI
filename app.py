import os

import streamlit as st

from ingestion.pdf_loader import load_pdf
from ingestion.docx_loader import load_docx
from ingestion.chunker import chunk_documents
from ingestion.vectorstore import load_vectorstore, clear_vectorstore
from ingestion.rag_chain import answer_question

UPLOAD_DIR = os.path.join("data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")
st.title("📄 Document Q&A (local RAG)")


def already_ingested(name):
    db = load_vectorstore()
    found = db.get(where={"source_name": name}, limit=1)
    return len(found["ids"]) > 0


def ingest_file(uploaded_file):
    path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == ".pdf":
        docs = load_pdf(path)
    elif ext == ".docx":
        docs = load_docx(path)
    else:
        raise ValueError("Only .pdf and .docx files are supported.")

    chunks = chunk_documents(docs)
    for c in chunks:
        c.metadata["source_name"] = uploaded_file.name

    if not chunks:
        raise ValueError("No text could be extracted from this file.")

    db = load_vectorstore()
    progress = st.progress(0.0, text="Embedding chunks...")
    batch = 100
    for i in range(0, len(chunks), batch):
        db.add_documents(chunks[i:i + batch])
        done = min(i + batch, len(chunks))
        progress.progress(done / len(chunks), text=f"Embedded {done}/{len(chunks)} chunks")
    progress.empty()
    return len(chunks)


# ---------------- Sidebar: upload and manage ----------------
with st.sidebar:
    st.header("Documents")
    files = st.file_uploader(
        "Upload PDF or Word files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    if st.button("Ingest uploaded files", type="primary", disabled=not files):
        for f in files:
            if already_ingested(f.name):
                st.info(f"{f.name}: already ingested, skipped.")
                continue
            try:
                with st.spinner(f"Processing {f.name}..."):
                    n = ingest_file(f)
                st.success(f"{f.name}: {n} chunks stored.")
            except Exception as e:
                st.error(f"{f.name}: {e}")

    try:
        total = load_vectorstore()._collection.count()
    except Exception:
        total = 0
    st.caption(f"Chunks in database: {total}")

    if st.button("Clear database"):
        clear_vectorstore()
        st.session_state.messages = []
        st.success("Database cleared.")
        st.rerun()

    k = st.slider("Chunks to retrieve (k)", 2, 8, 4)

# ---------------- Chat ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            st.caption("Sources: " + "; ".join(m["sources"]))

question = st.chat_input("Ask a question about your documents")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = answer_question(question, k=k)
            except Exception as e:
                answer, sources = f"Error: {e}", []
        st.markdown(answer)
        if sources:
            st.caption("Sources: " + "; ".join(sources))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )