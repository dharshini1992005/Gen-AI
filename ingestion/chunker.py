from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def merge_short_pages(documents, min_chars=300):
    """Drop empty pages and merge short pages with the pages that follow them."""
    merged = []
    buffer = ""
    pages = []
    base_meta = {}

    for d in documents:
        text = d.page_content.strip()
        if not text:
            continue

        if not buffer:
            base_meta = dict(d.metadata)

        buffer += "\n" + text
        pages.append(d.metadata.get("page", 0) + 1)

        if len(buffer) >= min_chars:
            meta = dict(base_meta)
            meta["page"] = pages[0] - 1
            meta["pages"] = f"{pages[0]}-{pages[-1]}"
            merged.append(Document(page_content=buffer.strip(), metadata=meta))
            buffer = ""
            pages = []

    # Keep any leftover text at the end
    if buffer.strip():
        meta = dict(base_meta)
        meta["page"] = pages[0] - 1
        meta["pages"] = f"{pages[0]}-{pages[-1]}"
        merged.append(Document(page_content=buffer.strip(), metadata=meta))

    return merged


def chunk_documents(documents, chunk_size=800, chunk_overlap=150):
    """Merge short pages, then split into overlapping chunks."""
    documents = merge_short_pages(documents)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)