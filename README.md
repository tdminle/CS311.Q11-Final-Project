# Vietnamese Law RAG System ⚖️

Hệ thống RAG (Retrieval-Augmented Generation) cho luật giao thông đường bộ Việt Nam.

## 🌟 Tính năng

- **Hybrid Retrieval**: Kết hợp Vector Search (Qdrant) và Keyword Search (Elasticsearch)
- **Reranking**: Sử dụng BGE-reranker-v2-m3 để cải thiện độ chính xác
- **LLM Reasoning**: DeepSeek-R1-7B với kỹ thuật Few-shot Prompting
- **Vietnamese Support**: Embedding được tối ưu cho tiếng Việt
- **Streamlit UI**: Giao diện web thân thiện
- **Langfuse Observability**: updating...

# Demo 

![demo](images\demo.png)

## 🏗️ Kiến trúc hệ thống

![System_image](images\system.png)


## 📁 Cấu trúc dự án

```
my_final_rag/
├── data/                      # Folder chứa file PDF gốc
│   └── *.pdf                  # Các file PDF luật
├── output_data/               # Folder chứa JSON đã xử lý (tự động tạo)
│   └── combined_output.json   # File JSON tổng hợp
├── data_preperation/          # Scripts xử lý dữ liệu
│   ├── processing.py          # PDFProcessingService - xử lý PDF
│   └── load_data.py           # Tải dữ liệu vào Qdrant/ES
├── src/                       # Source code chính
│   ├── models/                # Models (embedding, LLM, reranker)
│   ├── services/              # Services (RAG, retrieval, generator)
│   ├── data_storage/          # Qdrant & Elasticsearch services
│   └── utils/                 # Utilities (logger)
├── ui/                        # Streamlit UI
│   └── app.py
├── evaluation/                # Evaluation scripts
├── run_app.py                 # Entry point chính
└── .env                       # Environment variables
```

## 🚀 Hướng dẫn cài đặt

### 1. Yêu cầu hệ thống

- Python 3.11+
- Docker (cho Qdrant và Elasticsearch)
- UV package manager (khuyến nghị) hoặc pip

### 2. Cài đặt dependencies

```bash
# Sử dụng UV (khuyến nghị)
uv sync

# Hoặc sử dụng pip
pip install -r requirements.txt
```

### 3. Thiết lập môi trường

Tạo file `.env` với nội dung:

```bash
# HuggingFace Token
HF_TOKEN=your_huggingface_token_here

# Debug mode (optional)
DEBUG_MODE=false
```

### 4. Khởi động services

```bash
# Khởi động Qdrant và Elasticsearch
docker-compose up -d

# Kiểm tra services đang chạy
docker ps
```

Các services sẽ chạy tại:

- Qdrant: http://localhost:6333
- Elasticsearch: http://localhost:9200

## 📚 Workflow xử lý dữ liệu

### Bước 1: Chuẩn bị file PDF

Đặt tất cả file PDF cần xử lý vào folder `data/`:

```bash
my_final_rag/
└── data/
    ├── law_document_1.pdf
    ├── law_document_2.pdf
    └── law_document_3.pdf
```

### Bước 2: Xử lý PDF thành JSON

```bash
# Chạy PDFProcessingService
python data_preperation/processing.py
```

Script này sẽ:

- ✅ Đọc tất cả file PDF trong folder `data/`
- ✅ Trích xuất text và phân tách theo điều, chương
- ✅ Chia thành các chunks phù hợp (max 800 ký tự)
- ✅ Lưu kết quả vào `output_data/combined_output.json`

**Output:**

```
output_data/
├── combined_output.json      # File tổng hợp tất cả PDF
├── law_document_1.json       # (Tùy chọn) Output riêng từng file
└── law_document_2.json
```

### Bước 3: Tải dữ liệu vào vector stores

```bash
# Tải JSON vào Qdrant và Elasticsearch
python data_preperation/load_data.py
```

Script này sẽ:

- ✅ Tự động tìm file `combined_output.json` trong `output_data/`
- ✅ Tạo embeddings cho từng chunk
- ✅ Tải vào Qdrant collection "Law"
- ✅ Tải vào Elasticsearch index "law_documents"

## 🎯 Chạy ứng dụng

### Cách 1: Sử dụng script chính (Khuyến nghị)

```bash
python run_app.py
```

### Cách 2: Chạy trực tiếp với Streamlit

```bash
streamlit run ui/app.py --server.port 8501
```

Truy cập ứng dụng tại: **http://localhost:8501**

## 💡 Sử dụng PDFProcessingService trong code

### Xử lý một file PDF

```python
from pathlib import Path
from data_preperation.processing import PDFProcessingService

# Khởi tạo service
service = PDFProcessingService(
    extraction_method="fitz",  # hoặc "pypdf2"
    max_chunk_length=800
)

# Xử lý một file
chunks = service.process_single_pdf("data/my_law.pdf")

# Lưu kết quả
service.save_to_json(chunks, "output_data/my_law.json")
```

### Xử lý nhiều file PDF từ folder

```python
from data_preperation.processing import PDFProcessingService

# Khởi tạo service
service = PDFProcessingService(
    extraction_method="fitz",
    max_chunk_length=800
)

# Xử lý tất cả PDF trong folder
stats = service.process_folder(
    input_folder="data",
    output_folder="output_data",
    combine_output=True  # Tạo file combined_output.json
)

print(f"Processed: {stats['processed_files']} files")
print(f"Total chunks: {stats['total_chunks']}")
```

### Tùy chỉnh processing

```python
service = PDFProcessingService(
    extraction_method="fitz",      # "pypdf2" hoặc "fitz"
    max_chunk_length=1000          # Độ dài tối đa mỗi chunk
)

# Xử lý với output riêng lẻ cho từng file
stats = service.process_folder(
    input_folder="data",
    output_folder="output_data",
    combine_output=False  # Không tạo file combined
)
```

## 🔧 Cấu hình

### Streamlit UI Settings

Trong sidebar của ứng dụng, bạn có thể tùy chỉnh:

- **Số documents retrieve**: 3-10 (mặc định: 5)
- **Số documents sau rerank**: 1-5 (mặc định: 3)
- **Hiển thị nguồn**: Bật/tắt hiển thị source documents
- **Debug mode**: Xem thông tin chi tiết

### Environment Variables

- `HF_TOKEN`: HuggingFace API token (bắt buộc)
- `DEBUG_MODE`: Enable debug logging (true/false)

## 🔍 Models sử dụng

| Component  | Model                            | Purpose                       |
| ---------- | -------------------------------- | ----------------------------- |
| Embeddings | dangvantuan/vietnamese-embedding | Vector hóa văn bản tiếng Việt |
| LLM        | Qwen/Qwen2.5-7B-Instruct         | Sinh câu trả lời              |
| Reranker   | BAAI/bge-reranker-v2-m3          | Xếp hạng lại kết quả          |

## 📊 Evaluation

```bash
# Chạy tất cả evaluations
cd evaluation
python run_all.py

# Hoặc chạy từng loại
python eval_basic.py         # Basic Q&A evaluation
python eval_retrieval.py     # Retrieval quality
python eval_ragas.py         # RAGAS metrics
```

## 🛠️ Troubleshooting

### Lỗi: No JSON files found

```bash
# Chạy processing trước
python data_preperation/processing.py
```

### Lỗi: Connection refused (Qdrant/ES)

```bash
# Kiểm tra Docker services
docker-compose ps

# Khởi động lại
docker-compose restart
```

### Lỗi: HuggingFace token

Đảm bảo `.env` có `HF_TOKEN` hợp lệ:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

## 📝 Development

### Cấu trúc Service Pattern

```python
from src.services.rag import RAGService

# Initialize
rag = RAGService(
    qdrant_host="localhost",
    qdrant_port=6333,
    collection_name="Law",
    es_index_name="law_documents",
    top_k=5,
    use_rerank=True
)

# Sync usage
result = rag.generate_response_sync("Câu hỏi của bạn?")
print(result["answer"])
print(result["source_documents"])

# Async usage
import asyncio
answer = await rag.generate_response("Câu hỏi của bạn?")
```

### Logging

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Info message")
logger.debug("Debug message")  # Chỉ hiện khi DEBUG_MODE=true
```

## 📄 License

MIT License

## 👥 Contributors

Hệ thống RAG cho luật giao thông Việt Nam

## 🔗 Links

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Elasticsearch Guide](https://www.elastic.co/guide/index.html)
- [LangChain Docs](https://python.langchain.com/)
- [Streamlit Docs](https://docs.streamlit.io/)

---

**Lưu ý**: Đảm bảo Qdrant và Elasticsearch đang chạy trước khi start ứng dụng!
