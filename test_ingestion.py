from ingestion.pdf_loader import load_pdf


file_path = "data/Python_pdf.pdf"

documents = load_pdf(file_path)


print("Number of pages:", len(documents))

for document in documents:
    print("\n--- PAGE ---")
    print(document.page_content[:500])
    print("Metadata:", document.metadata)   