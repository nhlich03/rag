# RAG API with FastAPI, Gemini, and Qdrant

## 1. Giới thiệu

Dự án này xây dựng một API đơn giản để tìm kiếm ngữ nghĩa (semantic search) dựa trên:
- Qdrant để lưu trữ và truy vấn vector.
- FastEmbed để tạo embedding từ các từ khóa.
- Google Gemini (gemini-2.0-flash) để trích xuất các từ khóa từ câu nhập tự nhiên.

Luồng hoạt động:
1. Người dùng gửi một câu văn tự nhiên (ví dụ: "Tôi muốn đi ăn nhà hàng cùng bạn bè").
2. Gemini sẽ trích xuất 3–5 từ khóa chính (ví dụ: "nhà hàng", "bạn bè", "ăn uống").
3. Hệ thống sẽ search từng từ khóa riêng biệt trong Qdrant.
4. Trả kết quả riêng cho từng từ khóa, không gộp và không lọc trùng text (để hiển thị các vùng miền khác nhau).

---

## 2. Cấu trúc thư mục

```
project/
├── app/
│   ├── data/
│   │   └── qdrant_storage/        # Dữ liệu Qdrant (tự tạo khi chạy)
│   ├── main.py                    # Code API chính
│   └── requirements.txt
├── Dockerfile
├── docker-compose.yaml
└── README.md
```

---

## 3. Cài đặt và chạy

### a. Thiết lập file `.env`

Tạo file `.env` cùng cấp với `app/main.py` (hoặc ở root) và thêm:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### b. Cài đặt dependencies nếu chạy local

```
pip install -r app/requirements.txt
```

### c. Chạy bằng Docker

```
docker compose up --build
```

Server sẽ chạy ở:
```
http://localhost:8000
```

---

## 4. Các endpoint

### 4.1. Kiểm tra server
```
GET /
```
Trả về trạng thái hoạt động.

---

### 4.2. Search cơ bản
```
GET /search?q=nhà hàng&limit=5
```
Tìm kiếm theo từ khóa đơn, dùng FastEmbed + Qdrant.

**Ví dụ output:**
```json
[
  {
    "score": 0.87,
    "payload": {
      "word": "cửa hàng",
      "region": "miền Nam",
      "video": "https://...",
      "description": "..."
    }
  }
]
```

---

### 4.3. Semantic search (tự động trích xuất keyword)

```
GET /semantic_search?text=tôi muốn đi ăn nhà hàng cùng bạn bè
```

Luồng hoạt động:
1. Gemini trích xuất từ khóa chính.
2. Mỗi từ khóa sẽ được search riêng biệt trong Qdrant.
3. Trả về danh sách kết quả riêng cho từng từ khóa.

**Ví dụ output:**
```json
{
  "input_text": "tôi muốn đi ăn nhà hàng cùng bạn bè",
  "keywords": ["đi ăn", "nhà hàng", "bạn bè"],
  "search_results": {
    "đi ăn": {
      "total_results": 5,
      "results": [...]
    },
    "nhà hàng": {
      "total_results": 5,
      "results": [...]
    },
    "bạn bè": {
      "total_results": 3,
      "results": [...]
    }
  }
}
```

---

## 5. Mô tả file chính

**main.py**
- `/search`: Search cơ bản bằng vector.
- `/semantic_search`: Search nâng cao, có trích xuất từ khóa bằng Gemini, không gộp, không lọc trùng text.
- `model`: Sử dụng `BAAI/bge-small-en` từ FastEmbed.
- `QdrantClient`: Kết nối với thư mục `data/qdrant_storage` (local Qdrant).

---

## 6. Troubleshooting

### a. Nếu server không phản hồi
Kiểm tra log container:
```
docker logs rag_api
```

### b. Nếu Gemini báo lỗi 401
Kiểm tra lại `.env` xem đã điền đúng `GOOGLE_API_KEY`.

### c. Nếu search chỉ ra 1 kết quả
Có thể do dữ liệu Qdrant chưa đủ hoặc các payload bị trùng. 
API hiện đã bỏ lọc trùng nên nếu dữ liệu có nhiều vùng miền sẽ hiển thị đầy đủ.

---

## 7. Gợi ý mở rộng

- Thêm endpoint `/async_semantic_search` để chạy tìm kiếm song song cho các keyword.
- Tích hợp `BAAI/bge-m3` hoặc `paraphrase-multilingual-MiniLM-L12-v2` nếu cần tiếng Việt tốt hơn.
- Kết hợp caching để tránh gọi Gemini nhiều lần cho cùng 1 câu hỏi.

---

## 8. Tác giả
RAG API demo – tích hợp FastAPI, Qdrant, FastEmbed và Gemini.
