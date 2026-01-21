"""
Retrieval Service using EnsembleRetriever (Qdrant + Elasticsearch).
Combines vector search (semantic) and BM25 (keyword) search with reranking.
"""
import json
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from src.data_storage.qdrant_service import QdrantService
from src.models.reranker import rerank_documents
from src.utils.logger import get_logger

# Try to import EnsembleRetriever
try:
    from langchain.retrievers import EnsembleRetriever
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

# Try to import Elasticsearch service
try:
    from src.data_storage.elasticsearch_service import ElasticsearchService
    ES_SERVICE_AVAILABLE = True
except ImportError:
    ES_SERVICE_AVAILABLE = False

# Initialize logger
logger = get_logger(__name__)


class RetrievalService:
    """
    Service for retrieving documents using EnsembleRetriever.
    Combines Qdrant (vector search) and Elasticsearch (BM25) with optional reranking.
    """
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        es_host: str = "localhost",
        es_port: int = 9200,
        collection_name: str = "my_documents",
        es_index_name: Optional[str] = None,
        top_k: int = 5,
        use_rerank: bool = True,
        qdrant_weight: float = 0.5,
        es_weight: float = 0.5,
    ):
        """
        Initialize Retrieval Service with EnsembleRetriever.
        
        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            es_host: Elasticsearch server host
            es_port: Elasticsearch server port
            collection_name: Name of the Qdrant collection
            es_index_name: Name of the Elasticsearch index (defaults to collection_name)
            top_k: Number of documents to retrieve
            use_rerank: Whether to use reranking
            qdrant_weight: Weight for Qdrant retriever (0-1)
            es_weight: Weight for Elasticsearch retriever (0-1)
        """
        logger.info("="*60)
        logger.info("Initializing Retrieval Service")
        logger.info("="*60)
        
        self.top_k = top_k
        self.use_rerank = use_rerank
        self.collection_name = collection_name
        
        # Initialize Qdrant service
        logger.info("Initializing Qdrant (Vector Search)...")
        self.qdrant_service = QdrantService(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name
        )
        
        # Initialize Elasticsearch service if available
        self.es_service = None
        self.es_available = False
        
        if ES_SERVICE_AVAILABLE:
            logger.info("Initializing Elasticsearch (BM25)...")
            try:
                from src.data_storage.elasticsearch_service import ElasticsearchService
                # Use es_index_name if provided, otherwise use collection_name
                index_name = es_index_name or collection_name
                self.es_service = ElasticsearchService(
                    host=es_host,
                    port=es_port,
                    index_name=index_name
                )
                self.es_available = True
            except Exception as e:
                logger.warning(f"Elasticsearch not available: {e}")
                logger.warning("Falling back to Qdrant-only retrieval")
        else:
            logger.warning("Elasticsearch service module not available")
            logger.warning("Falling back to Qdrant-only retrieval")
        
        # Create EnsembleRetriever
        logger.info("Creating EnsembleRetriever...")
        self._create_ensemble_retriever(qdrant_weight, es_weight)
        
        logger.info(f"✅ Retrieval Service initialized (rerank: {use_rerank})")
        logger.info("="*60)
    
    def _create_ensemble_retriever(self, qdrant_weight: float, es_weight: float):
        """Create the EnsembleRetriever from both retrievers."""
        # Get Qdrant retriever
        qdrant_retriever = self.qdrant_service.get_retriever(k=self.top_k * 2)
        
        if self.es_available and ENSEMBLE_AVAILABLE:
            # Get Elasticsearch retriever
            es_retriever = self.es_service.get_retriever(k=self.top_k * 2)
            
            # Create EnsembleRetriever with both
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[qdrant_retriever, es_retriever],
                weights=[qdrant_weight, es_weight]
            )
            logger.info(f"EnsembleRetriever created: Qdrant({qdrant_weight}) + ES({es_weight})")
        else:
            # Use only Qdrant retriever
            self.ensemble_retriever = qdrant_retriever
            logger.info("Using Qdrant-only retriever")
    
    def add_documents(self, documents: List[Document]) -> Dict[str, List[str]]:
        """
        Add documents to both Qdrant and Elasticsearch.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            Dictionary with document IDs from each service
        """
        logger.info(f"Adding {len(documents)} documents to storage...")
        
        result = {}
        
        # Add to Qdrant
        qdrant_ids = self.qdrant_service.add_documents(documents)
        result["qdrant"] = qdrant_ids
        
        # Add to Elasticsearch if available
        if self.es_available:
            es_ids = self.es_service.add_documents(documents)
            result["elasticsearch"] = es_ids
        
        logger.info(f"✅ Documents added to storage")
        return result
    
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        """
        Add texts to both Qdrant and Elasticsearch.
        
        Args:
            texts: List of text strings
            metadatas: List of metadata dictionaries
            
        Returns:
            Dictionary with document IDs from each service
        """
        logger.info(f"Adding {len(texts)} texts to storage...")
        
        result = {}
        
        # Add to Qdrant
        qdrant_ids = self.qdrant_service.add_texts(texts, metadatas)
        result["qdrant"] = qdrant_ids
        
        # Add to Elasticsearch if available
        if self.es_available:
            es_ids = self.es_service.add_texts(texts, metadatas)
            result["elasticsearch"] = es_ids
        
        logger.info(f"✅ Texts added to storage")
        return result
    
    def retrieve(
        self, 
        query: str, 
        k: int = None,
        use_rerank: bool = None
    ) -> List[Document]:
        """
        Retrieve documents using EnsembleRetriever with optional reranking.
        
        Args:
            query: The search query
            k: Number of documents to return (default: self.top_k)
            use_rerank: Override reranking setting
            
        Returns:
            List of Document objects
        """
        if k is None:
            k = self.top_k
        if use_rerank is None:
            use_rerank = self.use_rerank
        
        logger.info(f"🔍 Retrieving documents for: '{query[:50]}...'")
        logger.debug(f"Config: k={k}, use_rerank={use_rerank}")
        
        # Get documents from EnsembleRetriever
        retrieve_k = k * 2 if use_rerank else k
        documents = self.ensemble_retriever.invoke(query)[:retrieve_k]
        
        logger.info(f"✅ EnsembleRetriever returned {len(documents)} documents")
        
        # Apply reranking if enabled
        if use_rerank and documents:
            logger.info("🔄 Applying reranking...")
            
            # Prepare documents for reranking (content, dummy_score)
            docs_to_rerank = [(doc.page_content, 0.0) for doc in documents]
            
            # Rerank
            reranked = rerank_documents(query, docs_to_rerank, top_k=k)
            
            # Map back to Document objects
            content_to_doc = {doc.page_content: doc for doc in documents}
            reranked_docs = [content_to_doc[content] for content, _ in reranked]
            
            logger.info(f"✅ Reranking completed. Final: {len(reranked_docs)} documents")
            return reranked_docs
        
        return documents[:k]
    
    def retrieve_with_scores(
        self, 
        query: str, 
        k: int = None,
        use_rerank: bool = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with scores.
        
        Args:
            query: The search query
            k: Number of documents to return
            use_rerank: Override reranking setting
            
        Returns:
            List of dictionaries with content, metadata, and score
        """
        if k is None:
            k = self.top_k
        if use_rerank is None:
            use_rerank = self.use_rerank
        
        logger.info(f"🔍 Retrieving documents with scores for: '{query[:50]}...'")
        
        # Get documents from EnsembleRetriever
        retrieve_k = k * 2 if use_rerank else k
        documents = self.ensemble_retriever.invoke(query)[:retrieve_k]
        
        logger.info(f"✅ EnsembleRetriever returned {len(documents)} documents")
        
        # Apply reranking if enabled
        if use_rerank and documents:
            logger.info("🔄 Applying reranking...")
            
            # Prepare documents for reranking
            docs_to_rerank = [(doc.page_content, 0.0) for doc in documents]
            
            # Rerank
            reranked = rerank_documents(query, docs_to_rerank, top_k=k)
            
            # Map back with scores
            content_to_doc = {doc.page_content: doc for doc in documents}
            
            results = [
                {
                    "content": content,
                    "metadata": content_to_doc[content].metadata,
                    "score": float(score),
                    "reranked": True
                }
                for content, score in reranked
            ]
            
            logger.info(f"✅ Reranking completed. Final: {len(results)} documents")
            return results
        
        # Return without reranking
        results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": 0.0,  # No score without reranking
                "reranked": False
            }
            for doc in documents[:k]
        ]
        
        return results
    
    def retrieve_as_json(
        self, 
        query: str, 
        k: int = None,
        use_rerank: bool = None
    ) -> str:
        """
        Retrieve documents and return as JSON string.
        
        Args:
            query: The search query
            k: Number of documents to return
            use_rerank: Override reranking setting
            
        Returns:
            JSON string with retrieved documents
        """
        results = self.retrieve_with_scores(query, k, use_rerank)
        return json.dumps(results, ensure_ascii=False, indent=2)
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the storage backends."""
        info = {
            "qdrant": self.qdrant_service.get_collection_info(),
        }
        
        if self.es_available:
            info["elasticsearch"] = self.es_service.get_index_info()
        
        return info


if __name__ == "__main__":
    # Test retrieval service
    print("Testing Retrieval Service...")
    
    try:
        service = RetrievalService(
            collection_name="test_retrieval",
            top_k=3,
            use_rerank=True
        )
        
        # Add sample texts
        texts = [
            "Python là một ngôn ngữ lập trình phổ biến và dễ học",
            "Machine learning là một nhánh của trí tuệ nhân tạo",
            "RAG kết hợp retrieval và generation để trả lời câu hỏi",
            "Vector database lưu trữ embeddings cho tìm kiếm ngữ nghĩa"
        ]
        service.add_texts(texts)
        
        # Retrieve
        query = "ngôn ngữ lập trình Python"
        results = service.retrieve_with_scores(query, k=2)
        
        print(f"\nQuery: {query}")
        print("\nResults:")
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['score']:.4f}] {r['content'][:50]}...")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
