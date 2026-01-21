# RAG Evaluation System

Đánh giá hiệu suất RAG với 3 metrics chính: Basic, Retrieval, và RAGAS.

## 📊 Metrics

### 1. Basic Metrics (eval_basic.py)

- **Success Rate**: Tỷ lệ trả lời thành công
- **Response Time**: Thời gian phản hồi trung bình
- **Answer Length**: Độ dài câu trả lời
- **Sources Used**: Số tài liệu sử dụng

### 2. Retrieval Metrics (eval_retrieval.py)

Đánh giá combo Elasticsearch + Qdrant + Reranker:

- **Hit Rate (Recall@5)**: Tìm được tài liệu đúng trong top 5?
- **MRR**: Tài liệu đúng ở vị trí nào? (Top 1 = tốt nhất)
- **Context Recall**: Tìm đủ thông tin để trả lời?

### 3. RAGAS Metrics (eval_ragas.py)

Đánh giá chất lượng RAG với open-source models:

- **Faithfulness**: Câu trả lời có trung thực với context?
- **Answer Relevancy**: Câu trả lời có liên quan với câu hỏi?
- **Context Precision**: Retrieved contexts có chính xác?
- **Context Recall**: Contexts có đủ thông tin?

Models: Qwen2.5-3B-Instruct (LLM) + all-MiniLM-L6-v2 (Embeddings)

## 🚀 Sử dụng

```bash
# Cài đặt
uv add ragas datasets

# Chạy từng loại
uv run python evaluation/eval_basic.py        # Basic metrics
uv run python evaluation/eval_retrieval.py    # Retrieval metrics
uv run python evaluation/eval_ragas.py        # RAGAS metrics
```

## 📈 Kết quả

Files lưu trong `evaluation/`:

- `basic_eval_*.json` - Basic metrics
- `retrieval_eval_*.json` - Retrieval metrics
- `ragas_eval_*.json` - RAGAS scores

## 🎯 Đánh giá Scores

| Score   | Đánh giá             |
| ------- | -------------------- |
| ≥ 0.8   | ✅ Excellent         |
| 0.6-0.8 | ⚠️ Good              |
| < 0.6   | ❌ Needs improvement |

## 💡 Cải thiện

- **Low Hit Rate**: Điều chỉnh retrieval weights, tăng top_k
- **Low MRR**: Cải thiện reranking, điều chỉnh ensemble weights
- **Low Faithfulness**: Cải thiện prompt, giảm hallucination
- **Low Relevancy**: Tối ưu system prompt
