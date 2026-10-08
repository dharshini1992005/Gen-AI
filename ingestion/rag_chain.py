import os

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from ingestion.vectorstore import load_vectorstore

MODEL = "ministral-3:3b"

llm = ChatOllama(model=MODEL, temperature=0)

prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.
If the answer is not in the context, say "I could not find that in the document."
Be clear and concise. If the context contains code, include a short example.

Context:
{context}

Question: {question}
Answer:"""
)


def _source_label(d):
    name = d.metadata.get("source_name") or os.path.basename(
        str(d.metadata.get("source", "document"))
    )
    page = d.metadata.get("page", 0) + 1
    pages_range = d.metadata.get("pages")
    if pages_range and "-" in str(pages_range):
        start, end = str(pages_range).split("-")
        where = f"page {start}" if start == end else f"pages {pages_range}"
    else:
        where = f"page {page}"
    return f"{name} - {where}"


def _format_docs(docs):
    return "\n\n".join(f"[{_source_label(d)}]\n{d.page_content}" for d in docs)


def build_chain(k=4):
    retriever = load_vectorstore().as_retriever(search_kwargs={"k": k})

    generate = (
        RunnableLambda(
            lambda x: {"context": _format_docs(x["docs"]), "question": x["question"]}
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Retrieve docs, keep the question, then add the generated answer
    return RunnableParallel(
        docs=retriever, question=RunnablePassthrough()
    ).assign(answer=generate)


def answer_question(question, k=4):
    result = build_chain(k).invoke(question)

    if not result["docs"]:
        return "No documents found in the database. Please upload a document first.", []

    sources = sorted({_source_label(d) for d in result["docs"]})
    return result["answer"], sources