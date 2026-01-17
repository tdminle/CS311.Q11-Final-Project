"""
FastAPI Backend - Main application entry point
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from src.app.models.schemas import QueryRequest, QueryResponse, HealthCheck, Context
from src.app.core.retrieval import hybrid_search
from src.app.core.rerank import rerank_documents
from src.app.core.generation import generate_answer
from src.app.config import get_settings
from langfuse import observe, Langfuse
import os

# Khởi tạo settings
settings = get_settings()

# Khởi tạo Langfuse (nếu có config)
if settings.LANGFUSE_SECRET_KEY and settings.LANGFUSE_PUBLIC_KEY:
    os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_HOST
    print("✅ Langfuse observability enabled")
else:
    print("⚠️  Langfuse not configured - observability disabled")

# Tạo FastAPI app
app = FastAPI(
    title="Vietnam Traffic Law RAG API",
    description="Hệ thống hỏi đáp Luật Giao thông Việt Nam sử dụng Hybrid Ensemble Agentic RAG",
    version="1.0.0"
)

# CORS middleware để frontend có thể gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên giới hạn origins cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthCheck)
async def root():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        message="Vietnam Traffic Law RAG API is running"
    )


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Kiểm tra sức khỏe của hệ thống"""
    return HealthCheck(
        status="healthy",
        message="All systems operational"
    )


@app.post("/query", response_model=QueryResponse)
@observe(name="rag_query_endpoint")
async def query_endpoint(request: QueryRequest):
    """
    Endpoint chính để xử lý câu hỏi
    
    Flow:
    1. Hybrid Search (Qdrant + Elasticsearch) -> RRF
    2. Rerank bằng BGE-Reranker
    3. Generate answer bằng DeepSeek-R1
    """
    try:
        # Bước 1: Hybrid Search với RRF
        print(f"\n{'='*50}")
        print(f"📝 Question: {request.question}")
        print(f"{'='*50}")
        
        candidates = await hybrid_search(request.question, top_k=request.top_k)
        
        if not candidates:
            # Không tìm thấy kết quả nào
            return QueryResponse(
                question=request.question,
                answer="Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu luật giao thông.",
                contexts=[],
                reasoning=""
            )
        
        # Bước 2: Rerank để lấy top N context tốt nhất
        reranked_docs = await rerank_documents(
            request.question, 
            candidates, 
            top_n=request.top_n
        )
        
        # Bước 3: Generate answer bằng LLM
        answer, reasoning = await generate_answer(request.question, reranked_docs)
        
        # Chuẩn bị contexts để trả về
        contexts = [
            Context(
                content=doc.get("content", ""),
                score=doc.get("rerank_score", doc.get("rrf_score", 0)),
                source=doc.get("source", "unknown")
            )
            for doc in reranked_docs
        ]
        
        print(f"✅ Query processed successfully")
        print(f"{'='*50}\n")
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            contexts=contexts,
            reasoning=reasoning
        )
    
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/stats")
async def get_stats():
    """
    Lấy thống kê về hệ thống (số lượng documents, v.v.)
    """
    try:
        from src.app.db.qdrant_conn import get_qdrant_client
        
        # Lấy client và thông tin collection từ Qdrant
        qdrant_client = get_qdrant_client()
        collection_info = qdrant_client.client.get_collection(settings.COLLECTION_NAME)
        
        return {
            "collection_name": settings.COLLECTION_NAME,
            "total_vectors": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
            "status": "active"
        }
    except Exception as e:
        return {
            "collection_name": settings.COLLECTION_NAME,
            "error": str(e),
            "status": "error"
        }


@app.on_event("shutdown")
async def shutdown_event():
    """Flush Langfuse traces khi shutdown"""
    if settings.LANGFUSE_SECRET_KEY and settings.LANGFUSE_PUBLIC_KEY:
        Langfuse().flush()
        print("📤 Langfuse traces flushed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
