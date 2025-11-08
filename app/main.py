from fastapi import FastAPI
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from google import genai
import os
from dotenv import load_dotenv

# =====================================================
# 1. Cấu hình Gemini + môi trường
# =====================================================

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Khởi tạo Gemini client
client_llm = genai.Client(api_key=api_key)

# =====================================================
# 2. Cấu hình FastAPI + Qdrant
# =====================================================

app = FastAPI()

# Model embedding (giữ nguyên fastembed)
model = TextEmbedding(model_name="BAAI/bge-small-en")

# Kết nối Qdrant
client = QdrantClient(path="/app/data/qdrant_storage")

COLLECTION = "sign_vectors"


# =====================================================
# 3. Endpoint search cơ bản
# =====================================================

@app.get("/search")
def search(q: str, limit: int = 5):
    """
    Search theo keyword đơn lẻ.
    """
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


# =====================================================
# 4. Endpoint semantic_search (trả kết quả riêng từng keyword, KHÔNG lọc trùng)
# =====================================================

@app.get("/semantic_search")
def semantic_search(text: str, limit: int = 5):
    """
    Nhận câu tự nhiên, trích từ khóa bằng Gemini,
    search từng keyword riêng biệt,
    trả kết quả riêng từng keyword.
    """

    # --- 1️⃣ Trích xuất keyword bằng Gemini ---
    intent_prompt = f"""Hãy trích xuất 3–5 từ khóa quan trọng mô tả ý định của câu sau, cách nhau bởi dấu phẩy: "{text}" """
    resp = client_llm.models.generate_content(
        model="gemini-2.0-flash",
        contents=intent_prompt
    )

    keywords_raw = resp.text.strip() if hasattr(resp, "text") else str(resp)
    keywords = [kw.strip() for kw in keywords_raw.split(",") if kw.strip()] or [text]

    # --- 2️⃣ Search từng keyword ---
    results_by_keyword = {}

    for kw in keywords:
        vec = list(model.query_embed([kw]))[0]
        hits = client.search(
            collection_name=COLLECTION,
            query_vector=vec,
            limit=limit
        )

        # Giữ nguyên toàn bộ hits, không lọc
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
# 5. Root endpoint (test)
# =====================================================

@app.get("/")
def root():
    return {"status": "ok", "message": "RAG API is running"}
