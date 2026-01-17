"""
Rerank Module - Sử dụng BGE-Reranker-v2-m3 để rerank các candidates
"""
import httpx
from src.app.config import get_settings
from typing import List, Dict
from langfuse import observe

settings = get_settings()


@observe(name="rerank_documents")
async def rerank_documents(query: str, documents: List[Dict], top_n: int = 5) -> List[Dict]:
    """
    Rerank các documents dựa trên query sử dụng model BGE-Reranker
    
    Args:
        query: Câu hỏi của người dùng
        documents: List các documents cần rerank
        top_n: Số lượng documents tốt nhất cần trả về
    
    Returns:
        List top_n documents đã được rerank
    """
    if not documents:
        return []
    
    # Chuẩn bị dữ liệu cho API
    # Lấy content từ mỗi document
    texts = [doc.get("content", "") for doc in documents]
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Gọi API Reranker
            response = await client.post(
                settings.RERANK_API_URL,
                json={
                    "query": query,
                    "documents": texts,
                    "top_n": top_n
                },
                headers={"Authorization": f"Bearer {settings.API_KEY}"} if settings.API_KEY else {}
            )
            response.raise_for_status()
            rerank_results = response.json()
        
        # Parse kết quả từ API
        # Format có thể là: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
        # hoặc đơn giản hơn tùy API implementation
        
        if "results" in rerank_results:
            # Lấy indices và scores
            ranked_indices = []
            for item in rerank_results["results"][:top_n]:
                idx = item.get("index", item.get("document_index", 0))
                score = item.get("relevance_score", item.get("score", 0))
                ranked_indices.append((idx, score))
            
            # Sắp xếp documents theo thứ tự rerank
            reranked_docs = []
            for idx, score in ranked_indices:
                if idx < len(documents):
                    doc = documents[idx].copy()
                    doc["rerank_score"] = score
                    reranked_docs.append(doc)
            
            print(f"✨ Rerank: {len(documents)} -> {len(reranked_docs)} documents")
            return reranked_docs
        
        else:
            # Nếu API trả về format khác, fallback về top_n documents đầu
            print("⚠️ Rerank API format không như mong đợi, sử dụng fallback")
            return documents[:top_n]
    
    except Exception as e:
        print(f"❌ Lỗi khi gọi Rerank API: {e}")
        # Fallback: Trả về top_n documents có RRF score cao nhất
        return documents[:top_n]
