"""
Retrieval Module - Hybrid Search với thuật toán RRF (Reciprocal Rank Fusion)
"""
from src.app.db.qdrant_conn import get_qdrant_client
from src.app.db.es_conn import get_es_client
from src.app.config import get_settings
from typing import List, Dict
import asyncio
from langfuse import observe

settings = get_settings()


def reciprocal_rank_fusion(results_dict: Dict[str, List[Dict]], k: int = 60) -> List[Dict]:
    """
    Thuật toán Reciprocal Rank Fusion để merge kết quả từ nhiều nguồn
    
    Args:
        results_dict: Dictionary chứa kết quả từ các nguồn khác nhau
                     Ví dụ: {"vector": [...], "keyword": [...]}
        k: Tham số RRF (thường là 60)
    
    Returns:
        List các documents đã được merge và sắp xếp theo điểm RRF
    """
    fused_scores = {}
    doc_metadata = {}  # Lưu metadata của document
    
    # Duyệt qua từng nguồn dữ liệu (qdrant, elasticsearch)
    for source, docs in results_dict.items():
        for rank, doc in enumerate(docs):
            # Sử dụng content làm key để identify document
            doc_content = doc.get("content", "")
            
            # Nếu chưa có document này, khởi tạo score = 0
            if doc_content not in fused_scores:
                fused_scores[doc_content] = 0
                doc_metadata[doc_content] = doc
            
            # Công thức RRF: score += 1 / (k + rank)
            # rank bắt đầu từ 0, nên rank thứ 1 có rank=0
            fused_scores[doc_content] += 1.0 / (k + rank + 1)
    
    # Sắp xếp documents theo điểm giảm dần
    sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Trả về list documents kèm RRF score
    results = []
    for content, score in sorted_items:
        doc = doc_metadata[content].copy()
        doc["rrf_score"] = score
        results.append(doc)
    
    return results


@observe(name="hybrid_search")
async def hybrid_search(query: str, top_k: int = 20) -> List[Dict]:
    """
    Thực hiện hybrid search: Kết hợp Vector Search (Qdrant) và Keyword Search (Elasticsearch)
    
    Args:
        query: Câu hỏi của người dùng
        top_k: Số lượng kết quả lấy từ mỗi nguồn
    
    Returns:
        List documents sau khi merge bằng RRF
    """
    # Lấy clients
    qdrant_client = get_qdrant_client()
    es_client = get_es_client()
    
    # Gọi song song 2 search engine
    qdrant_task = qdrant_client.search(query, limit=top_k)
    es_task = es_client.search(query, size=top_k)
    
    # Đợi cả 2 kết quả
    qdrant_results, es_results = await asyncio.gather(qdrant_task, es_task)
    
    # Chuẩn bị dữ liệu cho RRF
    results_dict = {
        "vector": qdrant_results,
        "keyword": es_results
    }
    
    # Merge kết quả bằng RRF
    merged_results = reciprocal_rank_fusion(results_dict, k=60)
    
    print(f"🔍 Hybrid Search: Qdrant={len(qdrant_results)}, ES={len(es_results)}, Merged={len(merged_results)}")
    
    return merged_results
