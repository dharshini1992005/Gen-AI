from ingestion.vectorstore import load_vectorstore
from ingestion.rag_chain import answer_question

QUESTION = "What is a list comprehension and how is it used?"
K = 4

# Part 1: show what the retriever finds
print("=" * 60)
print("RETRIEVED CHUNKS")
print("=" * 60)

db = load_vectorstore()
print("Stored chunks in database:", db._collection.count())

results = db.similarity_search(QUESTION, k=K)

if not results:
    raise SystemExit("Nothing retrieved. Rebuild the database with test_vectorstore.py.")

for i, d in enumerate(results, 1):
    page = d.metadata.get("page", 0) + 1
    preview = d.page_content[:150].replace("\n", " ")
    print(f"{i}. PAGE {page} -> {preview}")

# Part 2: generate the answer with Ollama
print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)

answer, pages = answer_question(QUESTION, k=K)

print("QUESTION:", QUESTION)
print("\nANSWER:", answer)
print("\nSOURCE PAGES:", pages)