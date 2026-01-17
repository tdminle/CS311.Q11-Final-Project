"""
Elasticsearch Connection - Kết nối và thao tác với Elasticsearch
"""
from elasticsearch import AsyncElasticsearch
from src.app.config import get_settings
from typing import List, Dict

settings = get_settings()


class ElasticsearchConnection:
    """Class quản lý kết nối và thao tác với Elasticsearch"""
    
    def __init__(self):
        """Khởi tạo kết nối Elasticsearch"""
        self.client = AsyncElasticsearch([settings.ES_URL])
        self.index_name = settings.COLLECTION_NAME
    
    async def create_index(self):
        """Tạo index mới trong Elasticsearch"""
        # Định nghĩa mapping cho tiếng Việt
        mapping = {
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "standard"  # Có thể custom analyzer cho tiếng Việt
                    },
                    "metadata": {
                        "type": "object",
                        "enabled": True
                    }
                }
            }
        }
        
        try:
            await self.client.indices.create(index=self.index_name, body=mapping)
            print(f"✅ Đã tạo index: {self.index_name}")
        except Exception as e:
            print(f"⚠️ Index có thể đã tồn tại: {e}")
    
    async def insert_documents(self, documents: List[Dict[str, str]]):
        """
        Chèn documents vào Elasticsearch
        Args:
            documents: List các document với 'content' và metadata
        """
        for doc in documents:
            await self.client.index(
                index=self.index_name,
                document=doc
            )
        
        # Refresh để documents có thể search ngay
        await self.client.indices.refresh(index=self.index_name)
        print(f"✅ Đã insert {len(documents)} documents vào Elasticsearch")
    
    async def search(self, query: str, size: int = 20) -> List[Dict]:
        """
        Tìm kiếm BM25 keyword search
        Args:
            query: Câu hỏi/query
            size: Số lượng kết quả trả về
        Returns:
            List các document match
        """
        # BM25 search
        body = {
            "query": {
                "match": {
                    "content": {
                        "query": query,
                        "fuzziness": "AUTO"  # Cho phép typo tolerance
                    }
                }
            },
            "size": size
        }
        
        response = await self.client.search(index=self.index_name, body=body)
        
        # Chuyển đổi results
        return [
            {
                "content": hit["_source"].get("content", ""),
                "score": hit["_score"],
                "source": "elasticsearch",
                **hit["_source"]
            }
            for hit in response["hits"]["hits"]
        ]
    
    async def close(self):
        """Đóng kết nối"""
        await self.client.close()


# Lazy initialization - tạo instance khi cần
_es_client_instance = None

def get_es_client() -> ElasticsearchConnection:
    """Lấy hoặc tạo instance của ElasticsearchConnection"""
    global _es_client_instance
    if _es_client_instance is None:
        _es_client_instance = ElasticsearchConnection()
    return _es_client_instance

# Backward compatibility
es_client = None  # Sẽ được init sau
