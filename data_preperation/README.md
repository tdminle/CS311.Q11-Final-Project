# Data Preparation Guide 📚

Hướng dẫn chi tiết về cách xử lý và tải dữ liệu PDF vào hệ thống RAG.

## 📁 Cấu trúc Folders

```
my_final_rag/
├── data/                      # INPUT: Đặt file PDF ở đây
│   ├── law_doc_1.pdf
│   ├── law_doc_2.pdf
│   └── law_doc_3.pdf
│
├── output_data/               # OUTPUT: JSON files (auto-generated)
│   ├── combined_output.json  # File tổng hợp tất cả
│   ├── law_doc_1.json        # (Optional) Output riêng
│   └── law_doc_2.json
│
└── data_preperation/          # Scripts xử lý
    ├── processing.py          # PDF → JSON
    └── load_data.py           # JSON → Qdrant/ES
```

## 🚀 Quick Start

### Bước 1: Chuẩn bị PDF files

Đặt tất cả file PDF cần xử lý vào folder `data/`:

```bash
# Copy file PDF vào folder data
cp /path/to/your/*.pdf data/
```

### Bước 2: Xử lý PDF → JSON

```bash
# Cài đặt dependencies (nếu chưa có)
pip install PyPDF2 pymupdf langchain-text-splitters

# Chạy processing service
python data_preperation/processing.py
```

**Output mẫu:**

```
============================================================
🚀 PDF Processing Service
============================================================
Input folder: data
Output folder: output_data
Found 3 PDF file(s)

📄 Processing: data/law_doc_1.pdf
  ✓ Extracted 125000 characters
  ✓ Created 150 initial chunks
  ✓ Final chunks: 180

📄 Processing: data/law_doc_2.pdf
  ✓ Extracted 98000 characters
  ✓ Created 120 initial chunks
  ✓ Final chunks: 145

✅ Combined output saved to: output_data/combined_output.json

============================================================
📊 Processing Summary:
  Processed: 3/3 files
  Total chunks: 325
============================================================
```

### Bước 3: Tải vào Vector Stores

```bash
# Đảm bảo Qdrant và Elasticsearch đang chạy
docker-compose up -d

# Load data
python data_preperation/load_data.py
```

**Output mẫu:**

```
============================================================
🚀 Vietnamese Law Data Loader
============================================================
📁 Found combined output: output_data/combined_output.json
📂 Loading data from output_data/combined_output.json
✅ Loaded 325 items

🔵 Loading into Qdrant collection: Law
  Initializing embeddings...
  Creating collection...
  Generating embeddings and uploading...
  Processed 325/325 documents
✅ Uploaded 325 points to Qdrant

🟡 Loading into Elasticsearch index: law_documents
  Creating index...
  Uploading documents...
  Uploaded 325/325 documents
✅ Uploaded 325 documents to Elasticsearch

============================================================
🔍 Verification:
  Qdrant 'Law': 325 points
  Elasticsearch 'law_documents': 325 documents
============================================================
✅ Done!
```

## 🔧 PDFProcessingService API

### Khởi tạo Service

```python
from data_preperation.processing import PDFProcessingService

service = PDFProcessingService(
    extraction_method="fitz",  # "pypdf2" hoặc "fitz" (PyMuPDF)
    max_chunk_length=800       # Độ dài tối đa mỗi chunk
)
```

### Xử lý một file PDF

```python
# Xử lý một file
chunks = service.process_single_pdf("data/my_law.pdf")
print(f"Created {len(chunks)} chunks")

# Lưu output
service.save_to_json(chunks, "output_data/my_law.json")
```

### Xử lý nhiều PDFs từ folder

```python
# Xử lý tất cả và tạo file combined
stats = service.process_folder(
    input_folder="data",
    output_folder="output_data",
    combine_output=True  # Tạo combined_output.json
)

print(f"Processed: {stats['processed_files']}")
print(f"Total chunks: {stats['total_chunks']}")
```

### Xử lý với output riêng lẻ

```python
# Mỗi PDF → 1 JSON file riêng
stats = service.process_folder(
    input_folder="data",
    output_folder="output_data",
    combine_output=False  # Không tạo combined file
)
```

## 📊 Output Format

### Structure của JSON chunks

```json
[
  {
    "title": "Điều 1. Phạm vi điều chỉnh Chương I NHỮNG QUY ĐỊNH CHUNG",
    "context": "Luật này quy định về bảo đảm trật tự, an toàn giao thông đường bộ..."
  },
  {
    "title": "Điều 2. Đối tượng áp dụng Chương I NHỮNG QUY ĐỊNH CHUNG",
    "context": "Luật này áp dụng đối với tổ chức, cá nhân tham gia giao thông..."
  }
]
```

## ⚙️ Tùy chỉnh Processing

### Thay đổi extraction method

```python
# Sử dụng PyPDF2 (nhanh hơn nhưng kém chính xác)
service = PDFProcessingService(extraction_method="pypdf2")

# Sử dụng PyMuPDF/fitz (chính xác hơn, khuyến nghị)
service = PDFProcessingService(extraction_method="fitz")
```

### Thay đổi chunk size

```python
# Chunks lớn hơn (tối đa 1000 ký tự)
service = PDFProcessingService(max_chunk_length=1000)

# Chunks nhỏ hơn (tối đa 500 ký tự)
service = PDFProcessingService(max_chunk_length=500)
```

## 🔍 Load Data Service

### Tìm file JSON tự động

```python
from data_preperation.load_data import find_latest_json

# Tự động tìm combined_output.json hoặc file mới nhất
json_file = find_latest_json("output_data")
print(f"Found: {json_file}")
```

### Load JSON data

```python
from data_preperation.load_data import load_json_data

data = load_json_data("output_data/combined_output.json")
print(f"Loaded {len(data)} chunks")
```

### Load vào Qdrant

```python
from data_preperation.load_data import load_to_qdrant

data = load_json_data("output_data/combined_output.json")
load_to_qdrant(data, collection_name="Law")
```

### Load vào Elasticsearch

```python
from data_preperation.load_data import load_to_elasticsearch

data = load_json_data("output_data/combined_output.json")
load_to_elasticsearch(data, index_name="law_documents")
```

## 🛠️ Troubleshooting

### Lỗi: No module named 'PyPDF2'

```bash
pip install PyPDF2 pymupdf langchain-text-splitters
```

### Lỗi: No PDF files found

Đảm bảo có file PDF trong folder `data/`:

```bash
ls data/*.pdf
```

### Lỗi: No JSON files found

Chạy processing trước:

```bash
python data_preperation/processing.py
```

### Lỗi: Connection refused (Qdrant/ES)

```bash
# Kiểm tra Docker
docker-compose ps

# Khởi động services
docker-compose up -d
```

## 📝 Best Practices

### 1. Tổ chức file PDF

```
data/
├── luật_giao_thông.pdf
├── nghị_định_100.pdf
└── thông_tư_41.pdf
```

### 2. Naming convention

- Sử dụng tên file có ý nghĩa
- Tránh ký tự đặc biệt
- Dùng dấu gạch dưới thay khoảng trắng

### 3. Workflow chuẩn

```bash
# 1. Chuẩn bị data
cp *.pdf data/

# 2. Process PDFs
python data_preperation/processing.py

# 3. Verify output
ls output_data/

# 4. Start services (nếu chưa chạy)
docker-compose up -d

# 5. Load vào database
python data_preperation/load_data.py
```

## 📈 Performance Tips

### Large PDF files

Với PDF lớn (>100 trang):

- Sử dụng `extraction_method="fitz"` (chính xác hơn)
- Tăng `max_chunk_length` lên 1000-1500

### Many PDF files

Với nhiều files (>10 PDFs):

- Sử dụng `combine_output=True` để dễ quản lý
- Xem xét xử lý theo batch nếu quá nhiều

### Memory optimization

```python
# Process theo batch nhỏ
import os
from pathlib import Path

pdf_files = list(Path("data").glob("*.pdf"))
batch_size = 5

for i in range(0, len(pdf_files), batch_size):
    batch = pdf_files[i:i+batch_size]
    # Process batch...
```

## 🔗 Related Documentation

- [Main README](../README.md)
- [Processing Service Code](processing.py)
- [Load Data Service Code](load_data.py)

---

**Need help?** Check the main [README.md](../README.md) or review the source code!
