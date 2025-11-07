import os
import io
import re
import numpy as np
import pandas as pd
from typing import List, Optional, Literal, Dict, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)


# =========================
# Config (env-overridable)
# =========================
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-m3")
QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_storage")  # dùng khi chạy embedded
QDRANT_URL = os.getenv("QDRANT_URL")  # ví dụ: "http://qdrant:6333" khi chạy compose
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # nếu Qdrant bật auth
COLLECTION = os.getenv("COLLECTION_NAME", "sign_vectors")
EMBED_DTYPE = "float32"

CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")

REGION_MAP = {"B": "miền Bắc", "T": "miền Trung", "N": "miền Nam"}


# =========================
# FastAPI app & CORS
# =========================
app = FastAPI(title="Sign RAG API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Globals: model & qdrant
# =========================
dense_model: SentenceTransformer = SentenceTransformer(MODEL_NAME)

if QDRANT_URL:
    client_qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    client_qdrant = QdrantClient(path=QDRANT_PATH)


# =========================
# Utils
# =========================
def ensure_collection(vector_size: int):
    """
    Đảm bảo collection tồn tại với kích thước vector phù hợp.
    Nếu chưa có, tạo mới. Nếu đã tồn tại, không làm gì.
    """
    try:
        client_qdrant.get_collection(COLLECTION)
    except Exception:
        client_qdrant.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def extract_region_from_id(_id: Optional[str]) -> Optional[str]:
    if not _id or not isinstance(_id, str):
        return None
    m = re.search(r"([A-Z])$", _id)
    if not m:
        return None
    return REGION_MAP.get(m.group(1))


def build_full_description(word: Optional[str], desc: Optional[str], region: Optional[str]) -> str:
    desc = desc if desc is not None and not pd.isna(desc) else ""
    region_str = f" ({region})" if region else ""
    return f"Ký hiệu của từ '{word}'{region_str}: {desc}"


def encode_texts(texts: List[str]) -> np.ndarray:
    vecs = dense_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vecs.astype(EMBED_DTYPE)


def make_region_filter(region: Optional[str]) -> Optional[Filter]:
    if not region:
        return None
    return Filter(must=[FieldCondition(key="region", match=MatchValue(value=region))])


def get_current_count() -> int:
    try:
        res = client_qdrant.count(collection_name=COLLECTION)
        return int(res.count or 0)
    except Exception:
        return 0


# =========================
# Schemas
# =========================
class Record(BaseModel):
    id: Optional[str] = Field(None, description="Tuỳ chọn: ký tự A-Z cuối để suy ra region")
    word: str
    description: Optional[str] = None
    video_url: Optional[str] = None
    region: Optional[Literal["miền Bắc", "miền Trung", "miền Nam"]] = None


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    region: Optional[Literal["miền Bắc", "miền Trung", "miền Nam"]] = None
    with_scores: bool = True


class SearchHit(BaseModel):
    id: int
    score: Optional[float] = None
    payload: Dict[str, Any]


class SearchResponse(BaseModel):
    hits: List[SearchHit]


# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "qdrant_mode": "server" if QDRANT_URL else "embedded",
        "qdrant_url": QDRANT_URL,
        "collection": COLLECTION,
    }


@app.get("/status", summary="Kiểm tra số bản ghi đã index trong collection.")
def status():
    return {"collection": COLLECTION, "count": get_current_count()}


@app.post("/recreate", summary="Xoá & tạo lại collection (mất dữ liệu).")
def recreate_collection():
    # Nếu muốn an toàn, có thể bắt buộc truyền query param "confirm=yes"
    dummy_size = 1024
    client_qdrant.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dummy_size, distance=Distance.COSINE),
    )
    return {"status": "recreated", "collection": COLLECTION, "size": dummy_size}


@app.post("/index/records", summary="Index từ JSON body (list records).")
def index_records(records: List[Record]):
    if not records:
        return {"indexed": 0}

    df = pd.DataFrame([r.model_dump() for r in records])

    # Suy region nếu trống
    if "region" not in df.columns:
        df["region"] = None
    df["region"] = df.apply(
        lambda r: r["region"] if pd.notna(r["region"]) and r["region"]
        else extract_region_from_id(r.get("id")), axis=1
    )

    # Build full_description
    df["full_description"] = df.apply(
        lambda r: build_full_description(r.get("word"), r.get("description"), r.get("region")),
        axis=1
    )

    # Encode & ensure collection
    embeddings = encode_texts(df["full_description"].tolist())
    ensure_collection(embeddings.shape[1])

    base_id = get_current_count()
    points = []
    for i, row in df.reset_index(drop=True).iterrows():
        payload = {
            "word": row.get("word"),
            "region": row.get("region"),
            "video": row.get("video_url"),
            "description": row.get("description"),
            "full_description": row.get("full_description"),
        }
        points.append(
            PointStruct(
                id=base_id + i,
                vector=embeddings[i],
                payload=payload
            )
        )

    client_qdrant.upsert(collection_name=COLLECTION, points=points)
    return {"indexed": len(points), "new_total": get_current_count()}


@app.post("/index/jsonl", summary="Index từ file JSONL cùng format notebook.")
async def index_jsonl(file: UploadFile = File(...)):
    raw = await file.read()
    df = pd.read_json(io.BytesIO(raw), lines=True)

    # Region từ id nếu thiếu
    if "region" not in df.columns:
        df["region"] = df["id"].apply(extract_region_from_id)
    else:
        df["region"] = df["region"].where(df["region"].notna(), df["id"].apply(extract_region_from_id))

    df["full_description"] = df.apply(
        lambda r: build_full_description(r.get("word"), r.get("description"), r.get("region")),
        axis=1
    )

    embeddings = encode_texts(df["full_description"].tolist())
    ensure_collection(embeddings.shape[1])

    base_id = get_current_count()
    points = []
    for i in range(len(df)):
        payload = {
            "word": df.iloc[i].get("word"),
            "region": df.iloc[i].get("region"),
            "video": df.iloc[i].get("video_url"),
            "description": df.iloc[i].get("description"),
            "full_description": df.iloc[i].get("full_description"),
        }
        points.append(
            PointStruct(
                id=base_id + i,
                vector=embeddings[i],
                payload=payload
            )
        )

    client_qdrant.upsert(collection_name=COLLECTION, points=points)
    return {"indexed": len(points), "new_total": get_current_count()}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    qvec = encode_texts([req.query])[0]
    flt = make_region_filter(req.region)

    results = client_qdrant.search(
        collection_name=COLLECTION,
        query_vector=qvec,
        limit=req.limit,
        query_filter=flt,
        with_payload=True,
        with_vectors=False,
        score_threshold=None
    )

    hits = [
        SearchHit(
            id=int(r.id),
            score=float(r.score) if req.with_scores else None,
            payload=r.payload or {}
        )
        for r in results
    ]
    return SearchResponse(hits=hits)
