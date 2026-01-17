"""
Qdrant Connection - Kết nối và thao tác với Qdrant Vector DB
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.app.config import get_settings
import httpx
from typing import List, Dict
import uuid

settings = get_settings()


class QdrantConnection:
    """Class quản lý kết nối và thao tác với Qdrant"""
    
    def __init__(self):
        """Khởi tạo kết nối Qdrant"""
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.collection_name = settings.COLLECTION_NAME
        
    async def create_collection(self, vector_size: int = 768):
        """
        Tạo collection mới trong Qdrant
        Args:
            vector_size: Kích thước vector embedding (mặc định 768 cho nhiều model Vietnamese)
        """
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"✅ Đã tạo collection: {self.collection_name}")
        except Exception as e:
            print(f"⚠️ Collection có thể đã tồn tại: {e}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Gọi API để lấy embedding vector
        Args:
            text: Text cần embedding
        Returns:
            Vector embedding
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                settings.EMBEDDING_API_URL,
                json={"input": text},
                headers={"Authorization": f"Bearer {settings.API_KEY}"} if settings.API_KEY else {}
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
    
    async def insert_documents(self, documents: List[Dict[str, str]]):
        """
        Chèn documents vào Qdrant
        Args:
            documents: List các document, mỗi dict có 'content' và metadata khác
        """
        points = []
        for doc in documents:
            embedding = await self.get_embedding(doc["content"])
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=doc
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"✅ Đã insert {len(points)} documents vào Qdrant")
    
    async def search(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Tìm kiếm semantic search
        Args:
            query: Câu hỏi/query
            limit: Số lượng kết quả trả về
        Returns:
            List các document match
        """
        query_vector = await self.get_embedding(query)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        # Chuyển đổi results thành format dễ sử dụng
        return [
            {
                "content": hit.payload.get("content", ""),
                "score": hit.score,
                "source": "qdrant",
                **hit.payload  # Thêm các metadata khác
            }
            for hit in results
        ]


# Lazy initialization - tạo instance khi cần
_qdrant_client_instance = None

def get_qdrant_client() -> QdrantConnection:
    """Lấy hoặc tạo instance của QdrantConnection"""
    global _qdrant_client_instance
    if _qdrant_client_instance is None:
        _qdrant_client_instance = QdrantConnection()
    return _qdrant_client_instance

# Backward compatibility
qdrant_client = None  # Sẽ được init sau
