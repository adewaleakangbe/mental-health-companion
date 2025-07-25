import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Define your knowledge base (for MVP, use hardcoded examples or text files)
docs = [
    "It's okay to feel overwhelmed sometimes. Talking to someone can really help.",
    "Try to breathe slowly and deeply. You're not alone in this.",
    "When anxiety strikes, grounding exercises like naming 5 things you see can help.",
    "Remember, your emotions are valid and deserve compassion.",
    "Even when it’s hard, reaching out to others is a sign of strength.",
    "Self-care is not selfish. It’s necessary for mental well-being.",
]

# Initialize embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(docs)

# Save FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save embeddings and docs for RAG pipeline
os.makedirs("data", exist_ok=True)
with open("data/embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

with open("data/docs.pkl", "wb") as f:
    pickle.dump(docs, f)

print("✅ Embeddings and documents saved to /data")
