import json
from sentence_transformers import SentenceTransformer
import PyPDF2
import os
import numpy as np

# --- Config ---
RESUME_FILE = "resume.pdf"    # path to your PDF resume
OUTPUT_FILE = "resume_embeddings.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500  # characters per chunk

# --- 1. Load PDF ---
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

resume_text = extract_text_from_pdf(RESUME_FILE)

# --- 2. Split into chunks ---
def chunk_text(text, chunk_size=CHUNK_SIZE):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks

chunks = chunk_text(resume_text)

# --- 3. Generate embeddings ---
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = [model.encode(chunk).tolist() for chunk in chunks]

# --- 4. Save to JSON ---
data = {"chunks": chunks, "embeddings": embeddings}
with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f)

print(f"Saved {len(chunks)} chunks to {OUTPUT_FILE}")
