"""
Pydantic schemas - Định nghĩa cấu trúc dữ liệu request/response
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    """Schema cho request hỏi đáp"""
    question: str = Field(..., min_length=1, description="Câu hỏi của người dùng")
    top_k: int = Field(default=20, ge=1, le=100, description="Số lượng kết quả tìm kiếm")
    top_n: int = Field(default=5, ge=1, le=20, description="Số lượng context sau rerank")


class Context(BaseModel):
    """Schema cho từng context"""
    content: str
    score: Optional[float] = None
    source: Optional[str] = None


class QueryResponse(BaseModel):
    """Schema cho response trả về"""
    question: str
    answer: str
    contexts: List[Context]
    reasoning: Optional[str] = None  # Phần suy luận của model (nếu có)


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    message: str
