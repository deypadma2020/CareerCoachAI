import os
import json
from langchain_cohere import CohereEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

EMBED_FILE = "embeddings.json"
embedding_model = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=os.getenv("COHERE_API_KEY"))

def embed_text(text):
    return embedding_model.embed_query(text)

def load_embeddings():
    if os.path.exists(EMBED_FILE):
        with open(EMBED_FILE, "r") as f:
            return json.load(f)
    return []

def save_embedding_entry(entry):
    data = load_embeddings()
    data.append(entry)
    with open(EMBED_FILE, "w") as f:
        json.dump(data, f, indent=2)

def find_similar_response(query, threshold=0.85):
    query_vec = embed_text(query)
    history = load_embeddings()

    if not history:
        return None, None

    scores = []
    for idx, item in enumerate(history):
        score = cosine_similarity([query_vec], [item["embedding"]])[0][0]
        scores.append((idx, score))

    if not scores:
        return None, None

    best_idx, best_score = max(scores, key=lambda x: x[1])
    if best_score >= threshold:
        return history[best_idx]["response"], best_score

    return None, None
