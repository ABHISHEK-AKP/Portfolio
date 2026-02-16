from flask import Flask, request, Response
import requests
import json
from sentence_transformers import SentenceTransformer
import chromadb
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])
# Initialize
data = json.load(open("resume_embeddings.json"))
chunks = data["chunks"]
embeddings = np.array(data["embeddings"])
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context(query, top_k=2):
    query_emb = model.encode(query)
    sims = embeddings @ query_emb / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb))
    top_idx = sims.argsort()[-top_k:][::-1]
    return " ".join([chunks[i] for i in top_idx])

@app.route("/stream_chat", methods=["GET"])
def stream_chat():
    query = request.args.get("message", "")
    if not query:
        return "No message provided", 400

    # RAG retrieval & prompt construction...
    context = retrieve_context(query)

    prompt = f"""You are Abhishek Kumar Pandey, a highly skilled software engineer with expertise in full-stack development, AI, cloud computing, and modern web technologies. 
Use the context from my resume below to answer questions about my skills, experience, projects, and achievements.\n Context from resume:{context}\n\nQuestion: {query}/n Rules:
1. Answer as Abhishek would speak: professional, friendly, and concise.
2. Include relevant experience and projects where applicable."""

    # Stream from Ollama
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt},
        stream=True
    )

    def generate():
        for line in res.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(generate(), mimetype='text/event-stream')
if __name__ == "__main__":
    app.run(port=5000)
