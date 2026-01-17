# Vietnam Traffic Law RAG System

Hệ thống hỏi đáp thông minh về Luật Giao thông Việt Nam sử dụng kiến trúc **Hybrid Ensemble Agentic RAG**.

## 🎯 Tính năng

- **Hybrid Search**: Kết hợp Vector Search (Qdrant) và Keyword Search (Elasticsearch)
- **RRF Algorithm**: Thuật toán Reciprocal Rank Fusion để merge kết quả
- **Reranker**: Sử dụng BGE-Reranker-v2-m3 để tối ưu độ chính xác
- **LLM Reasoning**: DeepSeek-R1-7B với kỹ thuật Few-shot Prompting
- **FastAPI Backend**: API server hiệu năng cao
- **Streamlit Frontend**: Giao diện chatbot thân thiện
- **🆕 Langfuse Observability**: Theo dõi và debug toàn bộ RAG pipeline

## 🏗️ Kiến trúc

```
User → Streamlit UI → FastAPI Backend
                      ↓
        ┌─────────────┴──────────────┐
        ↓                            ↓
    Qdrant DB                  Elasticsearch
    (Vector Search)            (Keyword Search)
        ↓                            ↓
        └─────────────┬──────────────┘
                      ↓
              RRF Algorithm (Fusion)
                      ↓
              BGE-Reranker (Top N)
                      ↓
          DeepSeek-R1 (Generate Answer)
```

## 📁 Cấu trúc dự án

```
RAG-agent/
├── .env                    # Biến môi trường
├── docker-compose.yml      # Qdrant + Elasticsearch
├── pyproject.toml          # Dependencies
├── ingest_data.py          # Script nạp dữ liệu
├── src/
│   └── app/
│       ├── main.py         # FastAPI server
│       ├── config.py       # Configuration
│       ├── core/
│       │   ├── retrieval.py    # Hybrid Search + RRF
│       │   ├── rerank.py       # Reranker
│       │   └── generation.py   # LLM Generation
│       ├── db/
│       │   ├── qdrant_conn.py  # Qdrant connection
│       │   └── es_conn.py      # Elasticsearch connection
│       ├── prompts/
│       │   └── templates.py    # Few-shot prompts
│       └── models/
│           └── schemas.py      # Pydantic models
└── ui/
    └── app.py              # Streamlit frontend
```

## 🚀 Cài đặt & Chạy

### 1. Cài đặt UV (Package Manager)

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Cài đặt dependencies

```bash
cd RAG-agent
uv sync
```

### 3. Cấu hình biến môi trường

Chỉnh sửa file `.env` với thông tin API của bạn:

```env
# --- DATABASE CONFIG ---
QDRANT_URL="http://localhost:6333"
ES_URL="http://localhost:9200"
COLLECTION_NAME="traffic_law"

# --- MODEL API CONFIG ---
EMBEDDING_API_URL="https://your-api-host.com/v1/embeddings"
RERANK_API_URL="https://your-api-host.com/v1/rerank"
LLM_API_URL="https://your-api-host.com/v1/chat/completions"

# API Key
API_KEY="sk-your-secure-key"

# --- LANGFUSE OBSERVABILITY CONFIG (Optional) ---
# Đăng ký tại: https://cloud.langfuse.com
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_HOST="https://cloud.langfuse.com"
```

### 4. Khởi động Database (Docker)

```bash
docker-compose up -d
```

**Lưu ý cho Linux/WSL**: Nếu gặp lỗi Elasticsearch, chạy:

```bash
sudo sysctl -w vm.max_map_count=262144
```

### 5. Nạp dữ liệu mẫu

```bash
uv run python ingest_data.py
```

### 6. Chạy Backend API

```bash
uv run uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```

API sẽ chạy tại: http://localhost:8000

### 7. Chạy Frontend UI

Mở terminal mới:

```bash
uv run streamlit run ui/app.py
```

Giao diện sẽ mở tại: http://localhost:8501

## 📚 API Endpoints

- `GET /` - Health check
- `POST /query` - Hỏi đáp (chính)
- `GET /stats` - Thống kê hệ thống

### Ví dụ API Request

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Vượt đèn đỏ xe máy phạt bao nhiêu?",
    "top_k": 20,
    "top_n": 5
  }'
```

## 🎨 Ví dụ câu hỏi

- Vượt đèn đỏ xe máy phạt bao nhiêu?
- Không đội mũ bảo hiểm bị phạt thế nào?
- Điều khiển xe khi say rượu bị xử phạt ra sao?
- Tốc độ tối đa trong khu dân cư là bao nhiêu?

## 🛠️ Tech Stack

| Component        | Technology                   |
| ---------------- | ---------------------------- |
| Package Manager  | UV                           |
| Backend          | FastAPI                      |
| Frontend         | Streamlit                    |
| Vector DB        | Qdrant                       |
| Keyword Search   | Elasticsearch                |
| Fusion Algorithm | RRF (Reciprocal Rank Fusion) |
| Reranker         | BGE-Reranker-v2-m3           |
| LLM              | DeepSeek-R1-Distill-Qwen-7B  |
| Infrastructure   | Docker Compose               |

## 🔍 Luồng xử lý

1. **User Input** → Câu hỏi từ người dùng
2. **Hybrid Search** → Tìm kiếm song song trên Qdrant (semantic) và Elasticsearch (keyword)
3. **RRF Fusion** → Merge và rank kết quả từ 2 nguồn
4. **Reranking** → BGE-Reranker lọc ra top N contexts tốt nhất
5. **Generation** → DeepSeek-R1 suy luận và generate câu trả lời
6. **Response** → Trả về câu trả lời + contexts + reasoning

## 📝 Lưu ý

- Đây là project ở mức đồ án môn học, chưa tối ưu cho production
- Cần cấu hình API URLs và API Keys trong file `.env`
- Dữ liệu mẫu trong `ingest_data.py` chỉ là demo, cần thay bằng dữ liệu thật
- Elasticsearch giới hạn RAM 512MB cho phù hợp máy cá nhân

## 🤝 Contributing

Project này được phát triển cho mục đích học tập. Mọi đóng góp đều được hoan nghênh!

## 📄 License

MIT License

---

**Powered by DeepSeek R1 🚀**
