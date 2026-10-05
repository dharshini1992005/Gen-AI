import os

import ollama

from ingestion.vectorstore import load_vectorstore

MODEL = "ministral-3:3b"


def _source_label(d):
    name = d.metadata.get("source_name") or os.path.basename(
        str(d.metadata.get("source", "document"))
    )
    page = d.metadata.get("page", 0) + 1
    pages_range = d.metadata.get("pages")
    where = f"pages {pages_range}" if pages_range else f"page {page}"
    return f"{name} - {where}"


def answer_question(question, k=4):
    db = load_vectorstore()
    docs = db.similarity_search(question, k=k)

    if not docs:
        return "No documents found in the database. Please upload a document first.", []

    context = "\n\n".join(
        f"[{_source_label(d)}]\n{d.page_content}" for d in docs
    )

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I could not find that in the document."
Be clear and concise. If the context contains code, include a short example.

Context:
{context}

Question: {question}
Answer:"""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    sources = sorted({_source_label(d) for d in docs})
    return response["message"]["content"], sources