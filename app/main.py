from fastapi import FastAPI
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

app = FastAPI()

model = TextEmbedding(model_name="BAAI/bge-small-en")

client = QdrantClient(path="/app/data/qdrant_storage")

COLLECTION = "sign_vectors"

@app.get("/search")
def search(q: str, limit: int = 5):
    vec = list(model.query_embed([q]))[0]

    hits = client.search(
        collection_name=COLLECTION,
        query_vector=vec,
        limit=limit
    )

    return [
        {
            "score": float(hit.score),
            "payload": hit.payload
        }
        for hit in hits
    ]

@app.get("/collection_info")
def info():
    return client.get_collection("sign_vectors")

@app.get("/collections")
def list_collections():
    return client.get_collections()