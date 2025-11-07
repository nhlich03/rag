from fastapi import FastAPI
from fastembed.embedding import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, CollectionStatus
import pandas as pd
import numpy as np
import uuid
import os

app = FastAPI()

# Load data
df = pd.read_json("local_description_final.jsonl", lines=True)
df["region"] = df["id"].str.extract(r"([A-Z])$")
region_map = {"B": "miền Bắc", "T": "miền Trung", "N": "miền Nam"}
df["region"] = df["region"].map(region_map)
df["region"] = df["region"].where(df["region"].notna(), None)


def build_full_description(row):
    desc = row["description"] if pd.notna(row["description"]) else ""
    region = f" ({row['region']})" if pd.notna(row["region"]) else ""
    return f"Ký hiệu của từ '{row['word']}'{region}: {desc}"


df["full_description"] = df.apply(build_full_description, axis=1)

# Embedding
embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = list(embedder.embed(df["full_description"].tolist()))

# Save embeddings
np.save("embeddings.npy", np.array(embeddings, dtype=np.float32))

# Qdrant setup
client = QdrantClient(host="qdrant", port=6333)
COLLECTION_NAME = "local-descriptions"

if COLLECTION_NAME not in client.get_collections().collections:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
    )

points = [
    PointStruct(id=str(uuid.uuid4()), vector=vector, payload=row.to_dict())
    for vector, (_, row) in zip(embeddings, df.iterrows())
]

client.upsert(collection_name=COLLECTION_NAME, points=points)


@app.get("/search")
def search(q: str, top_k: int = 5):
    query_vector = list(embedder.embed([q]))[0]
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    return [
        {
            "score": hit.score,
            "word": hit.payload["word"],
            "description": hit.payload["description"],
            "region": hit.payload["region"]
        }
        for hit in results
    ]


@app.get("/health")
def health():
    status = client.get_collection(COLLECTION_NAME).status
    return {"status": status}
