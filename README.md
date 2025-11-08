# 🧠 RAG App with Qdrant + FastAPI + Sentence Transformers

This is a lightweight Retrieval-Augmented Generation (RAG) prototype that:
- Uses `local_description_final.jsonl` for document context
- Generates embeddings using `all-MiniLM-L6-v2`
- Stores vector embeddings in Qdrant
- Performs keyword-based search & semantic retrieval via FastAPI

---

## 📁 Project Structure

```
rag_project/
├── app/
│   ├── main.py
│   ├── data/
│   │   └── local_description_final.jsonl  ← Your document data
│   └── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🚀 Getting Started

### 1. Requirements
- Docker + Docker Compose installed

### 2. Run with Docker

```bash
docker compose up --build
```

Then visit [http://localhost:8000](http://localhost:8000)

---

## 🧩 API Endpoints

### `GET /`
Check if the app is running.

### `POST /search`
Perform keyword + semantic search.

**Payload:**
```json
{
  "query": "từ khóa cần tìm"
}
```

**Response:**
```json
{
  "results": [
    {
      "word": "xyz",
      "description": "...",
      "score": 0.85
    },
    ...
  ]
}
```

---

## 🧠 Embedding Model

Using: `sentence-transformers/all-MiniLM-L6-v2`  
(via `TextEmbedding` wrapper in `qdrant_client`)

---

## 🗃 Qdrant Configuration

Data is stored in a local Qdrant container (vector DB).  
Indexing and searching are handled automatically in `main.py`.

---

## ⚠ Notes

- Make sure your `local_description_final.jsonl` is UTF-8 encoded.
- Adjust the container name, ports, or volumes in `docker-compose.yml` if needed.

---

## 🧼 Clean up

```bash
docker compose down --volumes --remove-orphans
```

---

Made with ❤️ for demo purposes.
