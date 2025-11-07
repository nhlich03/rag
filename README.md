# RAG Search API with FastEmbed and Qdrant

This project is a simple RAG-style search API built with FastAPI, FastEmbed, and Qdrant.

## How It Works

- Parses a `.jsonl` file containing word descriptions and region metadata.
- Generates dense vector embeddings using `fastembed` (`all-MiniLM-L6-v2` model).
- Stores embeddings and metadata in Qdrant.
- Provides an API to perform semantic search over the data.

## Requirements

- Docker + Docker Compose

## Getting Started

### 1. Prepare Your Data

Place your `local_description_final.jsonl` file in the root directory.

### 2. Build & Run

```bash
docker compose up -d --build
```

### 3. Search Endpoint

```
GET http://localhost:8000/search?q=example
```

### 4. Health Check

```
GET http://localhost:8000/health
```

### 5. Output Format
Each search result returns:

```
{
  "score": 0.85,
  "word": "Example",
  "description": "Some description",
  "region": "miền Bắc"
}
```