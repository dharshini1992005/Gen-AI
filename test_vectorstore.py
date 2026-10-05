import shutil
import os

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_documents
from ingestion.vectorstore import build_vectorstore, PERSIST_DIR

PDF_PATH = "data/Python_pdf.pdf"

# Step 1: remove any old database so chunks are not duplicated
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    print("Old chroma_db removed")

# Step 2: load and chunk the PDF
docs = load_pdf(PDF_PATH)
print(f"{len(docs)} pages loaded")

chunks = chunk_documents(docs)
print(f"{len(chunks)} chunks ready")

if len(chunks) == 0:
    raise SystemExit("No chunks were created. Check the PDF path and the loader.")

# Step 3: build the vector store
db = build_vectorstore(chunks)
print("Vector store built")
print("Stored chunks:", db._collection.count())

# Step 4: quick retrieval test
query = "What is a list comprehension and how is it used?"
results = db.similarity_search(query, k=4)

print(f"\nTop results for: {query}")
for i, r in enumerate(results, 1):
    page = r.metadata.get("page", 0) + 1
    preview = r.page_content[:200].replace("\n", " ")
    print(f"\n--- RESULT {i} (page {page}) ---")
    print(preview)