from fastapi import FastAPI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
from dotenv import load_dotenv
import numpy as np

# =====================================================
# 1. Cấu hình Gemini + môi trường
# =====================================================

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")  # Đảm bảo .env có GEMINI_API_KEY

if not api_key:
    raise ValueError("❌ Missing GEMINI_API_KEY in environment variables")

genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# =====================================================
# 2. Cấu hình FastAPI + Qdrant
# =====================================================

app = FastAPI()

# Model nhúng SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")

# Kết nối Qdrant local (đã mount volume /app/data/qdrant_storage)
client = QdrantClient(path="/app/data/qdrant_storage")

COLLECTION = "sign_vectors"


# =====================================================
# 3. Hàm tiện ích encode text
# =====================================================

def get_embedding(text: str):
    embedding = model.encode([text], convert_to_numpy=True).astype("float32")[0]
    return embedding


# =====================================================
# 4. Endpoint search cơ bản
# =====================================================

@app.get("/search")
def search(q: str, limit: int = 5):
    """
    Search theo keyword đơn lẻ.
    """
    vec = get_embedding(q)
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


# =====================================================
# 5. Endpoint semantic_search (Gemini trích keyword, search từng keyword riêng)
# =====================================================

@app.get("/semantic_search")
def semantic_search(text: str, limit: int = 5):
    """
    Trích keyword bằng Gemini, search từng keyword riêng,
    trả kết quả độc lập cho từng keyword.
    """

    # --- 1️⃣ Gọi Gemini trích keyword ---
    intent_prompt = f"""Hãy trích xuất các từ khóa quan trọng mô tả ý định của câu sau, cách nhau bởi dấu phẩy: "{text}" """
    resp = gemini_model.generate_content(intent_prompt)

    keywords_raw = resp.text.strip() if hasattr(resp, "text") else str(resp)
    keywords = [kw.strip() for kw in keywords_raw.split(",") if kw.strip()] or [text]

    # --- 2️⃣ Search từng keyword riêng ---
    results_by_keyword = {}
    for kw in keywords:
        vec = get_embedding(kw)
        hits = client.search(
            collection_name=COLLECTION,
            query_vector=vec,
            limit=limit
        )

        keyword_results = [
            {
                "score": float(h.score),
                "payload": h.payload
            }
            for h in hits
        ]

        results_by_keyword[kw] = {
            "total_results": len(keyword_results),
            "results": keyword_results
        }

    # --- 3️⃣ Trả về kết quả ---
    return {
        "input_text": text,
        "keywords": keywords,
        "search_results": results_by_keyword
    }


# =====================================================
# 6. Root endpoint (test)
# =====================================================

@app.get("/")
def root():
    return {"status": "ok", "message": "RAG API is running"}
