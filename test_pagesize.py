from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_documents

docs = load_pdf("data/Python_pdf.pdf")
print("Pages:", len(docs))

lengths = [len(d.page_content) for d in docs]
print("Characters per page:", lengths)
print("Pages over 500 chars:", sum(1 for n in lengths if n > 500))

chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
print("Chunks at 500/100:", len(chunks))

chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=50)
print("Chunks at 300/50:", len(chunks))