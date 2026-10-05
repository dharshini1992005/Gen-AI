# Gen-AI RAG System

A complete Retrieval-Augmented Generation (RAG) system that allows users to upload PDF and DOCX documents and ask questions based on their content.

## Features

- Upload PDF and DOCX documents
- Extract text from documents
- Split documents into meaningful chunks
- Generate embeddings
- Store embeddings in ChromaDB
- Perform semantic similarity search
- Retrieve relevant document chunks
- Generate answers using a local Ollama LLM
- Display retrieved sources
- Streamlit-based user interface
- Supports multiple document-based questions

## RAG Architecture

PDF / DOCX
↓
Text Extraction
↓
Text Cleaning
↓
Chunking
↓
Embeddings
↓
ChromaDB
↓
Similarity Search
↓
Top-K Relevant Chunks
↓
Prompt Construction
↓
Ollama LLM
↓
Final Answer + Sources

## Technologies Used

- Python
- Streamlit
- LangChain
- ChromaDB
- Sentence Transformers
- PyMuPDF
- python-docx
- Ollama

## Project Structure

```text
Gen-AI/
│
├── ingestion/
├── data/
├── uploads/
├── app.py
├── requirements.txt
├── README.md