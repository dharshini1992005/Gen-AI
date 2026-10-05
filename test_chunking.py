from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_documents

docs = load_pdf("data/Document 1.pdf")
chunks = chunk_documents(docs)

print(f"{len(docs)} pages -> {len(chunks)} chunks")
print("\n--- FIRST CHUNK ---")
print(chunks[0].page_content)
print(chunks[0].metadata)
print("\n--- SECOND CHUNK ---")
print(chunks[1].page_content)
print(chunks[1].metadata)