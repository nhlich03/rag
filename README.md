# RAG API – FastAPI + Qdrant + FastEmbed

Dự án này triển khai một API đơn giản để tìm kiếm ngữ nghĩa (semantic search) sử dụng FastAPI, FastEmbed, và Qdrant chạy ở chế độ local.

---

## 1. Cấu trúc thư mục

``` 
project_root/
├── app/
│   ├── main.py
│   ├── requirements.txt
│   └── data/
│       └── qdrant_storage/
├── Dockerfile
└── docker-compose.yaml
```

---

## 2. Thiết lập môi trường

### a. Cài Docker và Docker Compose
Đảm bảo máy đã cài sẵn:
- Docker Engine
- Docker Compose (phiên bản >= 2)

### b. Xây dựng và khởi động container

```
docker compose up --build
```

Lần đầu chạy, hệ thống sẽ tự động tải model embedding từ Hugging Face (mất khoảng 1–2 phút).

---

## 3. Kiểm tra hoạt động

Sau khi container khởi động thành công, truy cập:

```
http://localhost:8000/docs
```

hoặc gọi trực tiếp API:

```
GET http://localhost:8000/search?q=trái+cây&limit=5
```

---

## 4. Mount dữ liệu Qdrant

Thư mục dữ liệu Qdrant được lưu cục bộ để không mất dữ liệu sau khi container tắt.

```
volumes:
  - ./app/data/qdrant_storage:/app/data/qdrant_storage
```

Nếu muốn mount cả source code để cập nhật code mà không cần build lại, thêm:

```
volumes:
  - ./app:/app
  - ./app/data/qdrant_storage:/app/data/qdrant_storage
```

---

## 5. Tái tạo dữ liệu (nếu đổi model)

Nếu bạn đổi model trong main.py, cần xóa dữ liệu cũ vì vector size khác nhau:

```
rm -rf app/data/qdrant_storage/*
```

Sau đó index lại dữ liệu mới bằng script index_data.py (tạo riêng).

---

## 6. Lệnh hữu ích

- Dừng container:

```
docker compose down
```

- Xem log:

```
docker logs rag_api
```

- Mở shell trong container:

```
docker exec -it rag_api bash
```

---

## 7. Ghi chú

- Model embedding hiện tại: BAAI/bge-small-en
- Nếu bạn từng dùng model sentence-transformers/all-MiniLM-L6-v2, cần đồng nhất model khi index và search.
- Mặc định API không có route "/", bạn có thể thêm:

```python
@app.get("/")
def root():
    return {"status": "ok", "message": "RAG API is running"}
```

---

## 8. License

Dự án được phát triển cho mục đích học tập và thử nghiệm.
