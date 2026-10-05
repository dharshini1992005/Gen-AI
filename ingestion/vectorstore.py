from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = "chroma_db"


@lru_cache(maxsize=1)
def get_embeddings():
    # Cached so the model loads once, not on every question
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def build_vectorstore(chunks):
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=PERSIST_DIR,
    )


def load_vectorstore():
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


def clear_vectorstore():
    """Empty the database without deleting files (avoids Windows file locks)."""
    load_vectorstore().delete_collection()