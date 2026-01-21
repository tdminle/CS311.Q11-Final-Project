"""
Qdrant Vector Store Service for semantic search.
"""
import json
from typing import List, Dict, Any, Optional
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document
from src.models.embedding import embeddings, get_vector_size
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class QdrantService:
    """Service for managing Qdrant vector store operations."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "my_documents",
        vector_size: int = None,
    ):
        """
        Initialize Qdrant service.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection
            vector_size: Size of embedding vectors (auto-detect if None)
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size or get_vector_size()
        self.url = f"http://{host}:{port}"
        
        logger.info(f"Initializing Qdrant service: {self.url}")
        logger.debug(f"Collection: {collection_name}, Vector size: {self.vector_size}")
        
        # Initialize Qdrant client
        self.client = QdrantClient(host=host, port=port)
        
        # Initialize embeddings
        self.embeddings = embeddings
        
        # Initialize vector store (will create collection if not exists)
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Create collection if it doesn't exist."""
        logger.debug(f"Checking if collection '{self.collection_name}' exists")
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            logger.info(f"Creating new collection: {self.collection_name}")
            logger.debug(f"Collection config: vector_size={self.vector_size}, distance=COSINE")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"✅ Created collection: {self.collection_name}")
        else:
            logger.info(f"Collection '{self.collection_name}' already exists")
    
    def get_vector_store(self) -> QdrantVectorStore:
        """Get LangChain Qdrant vector store."""
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
    
    def get_retriever(self, k: int = 5):
        """
        Get a LangChain retriever for this vector store.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            LangChain retriever
        """
        vector_store = self.get_vector_store()
        return vector_store.as_retriever(search_kwargs={"k": k})
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents to add")
            return []
        
        logger.debug(f"Adding {len(documents)} documents to collection '{self.collection_name}'")
        vector_store = self.get_vector_store()
        doc_ids = vector_store.add_documents(documents)
        logger.info(f"✅ Added {len(documents)} documents to {self.collection_name}")
        return doc_ids
    
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[str]:
        """
        Add texts with optional metadata to the vector store.
        
        Args:
            texts: List of text strings
            metadatas: List of metadata dictionaries (optional)
            
        Returns:
            List of document IDs
        """
        if not texts:
            logger.warning("No texts to add")
            return []
        
        logger.debug(f"Adding {len(texts)} texts to collection '{self.collection_name}'")
        vector_store = self.get_vector_store()
        doc_ids = vector_store.add_texts(texts=texts, metadatas=metadatas)
        logger.info(f"✅ Added {len(texts)} texts to {self.collection_name}")
        return doc_ids
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of Document objects
        """
        vector_store = self.get_vector_store()
        results = vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
        return results
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5,
        filter: Dict[str, Any] = None
    ) -> List[tuple]:
        """
        Search for similar documents with relevance scores.
        
        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of tuples (Document, score)
        """
        vector_store = self.get_vector_store()
        results = vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
        return results
    
    def delete_collection(self):
        """Delete the collection."""
        self.client.delete_collection(collection_name=self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        info = self.client.get_collection(collection_name=self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }


if __name__ == "__main__":
    # Test Qdrant service
    service = QdrantService(collection_name="test_collection")
    
    # Add sample texts
    texts = [
        "Python là một ngôn ngữ lập trình phổ biến",
        "Machine learning là một nhánh của AI",
        "Vector database giúp tìm kiếm ngữ nghĩa"
    ]
    service.add_texts(texts)
    
    # Search
    results = service.similarity_search("ngôn ngữ lập trình", k=2)
    print("\nSearch results:")
    for doc in results:
        print(f"- {doc.page_content}")
    
    # Get collection info
    print(f"\nCollection info: {service.get_collection_info()}")
