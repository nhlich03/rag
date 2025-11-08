from fastapi import FastAPI, Query
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

app = FastAPI()

QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
COLLECTION_NAME = "local_terms"

embedding_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

@app.get("/")
def root():
    return {"message": "RAG search API is running with MiniLM and Qdrant!"}

@app.get("/search")
def search(query: str = Query(..., min_length=1)):
    query_vector = next(embedding_model.embed([query]))

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5
    )

    return [
        {
            "text": r.payload.get("text", ""),
            "score": r.score
        } for r in results
    ]
