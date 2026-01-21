"""
Elasticsearch Service for BM25 keyword search.
"""
import json
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document
from langchain_community.retrievers import ElasticSearchBM25Retriever
from src.models.embedding import embeddings
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class ElasticsearchService:
    """Service for managing Elasticsearch operations (BM25 keyword search)."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        index_name: str = "my_documents",
        use_ssl: bool = False,
    ):
        """
        Initialize Elasticsearch service.
        
        Args:
            host: Elasticsearch server host
            port: Elasticsearch server port
            index_name: Name of the index
            use_ssl: Whether to use SSL connection
        """
        self.host = host
        self.port = port
        self.index_name = index_name
        self.url = f"http{'s' if use_ssl else ''}://{host}:{port}"
        
        logger.info(f"Initializing Elasticsearch service: {self.url}")
        logger.debug(f"Index: {index_name}")
        
        # Initialize Elasticsearch client
        self.client = Elasticsearch(
            hosts=[{"host": host, "port": port, "scheme": "https" if use_ssl else "http"}],
        )
        
        # Check connection
        if self.client.ping():
            logger.info("✅ Connected to Elasticsearch")
        else:
            logger.error("❌ Could not connect to Elasticsearch")
            raise ConnectionError("Could not connect to Elasticsearch")
        
        # Initialize index
        self._initialize_index()
    
    def _initialize_index(self):
        """Create index if it doesn't exist."""
        logger.debug(f"Checking if index '{self.index_name}' exists")
        
        if not self.client.indices.exists(index=self.index_name):
            logger.info(f"Creating new index: {self.index_name}")
            
            # Define index settings and mappings
            index_settings = {
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "default": {
                                "type": "standard"
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "metadata": {
                            "type": "object",
                            "enabled": True
                        }
                    }
                }
            }
            
            self.client.indices.create(index=self.index_name, body=index_settings)
            logger.info(f"✅ Created index: {self.index_name}")
        else:
            logger.info(f"Index '{self.index_name}' already exists")
    
    def get_retriever(self, k: int = 5):
        """
        Get a BM25 retriever for keyword search.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            BM25 retriever
        """
        return ElasticSearchBM25Retriever(
            client=self.client,
            index_name=self.index_name,
            k=k
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the index.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents to add")
            return []
        
        logger.debug(f"Adding {len(documents)} documents to index '{self.index_name}'")
        
        doc_ids = []
        for i, doc in enumerate(documents):
            doc_body = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            result = self.client.index(index=self.index_name, body=doc_body)
            doc_ids.append(result["_id"])
        
        # Refresh index to make documents searchable immediately
        self.client.indices.refresh(index=self.index_name)
        
        logger.info(f"✅ Added {len(documents)} documents to {self.index_name}")
        return doc_ids
    
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[str]:
        """
        Add texts with optional metadata to the index.
        
        Args:
            texts: List of text strings
            metadatas: List of metadata dictionaries (optional)
            
        Returns:
            List of document IDs
        """
        if not texts:
            logger.warning("No texts to add")
            return []
        
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]
        
        return self.add_documents(documents)
    
    def search(
        self, 
        query: str, 
        k: int = 5
    ) -> List[Document]:
        """
        Search for documents using BM25.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of Document objects
        """
        logger.debug(f"BM25 search: '{query[:50]}...' (k={k})")
        
        search_body = {
            "query": {
                "match": {
                    "content": {
                        "query": query,
                        "fuzziness": "AUTO"
                    }
                }
            },
            "size": k
        }
        
        response = self.client.search(index=self.index_name, body=search_body)
        
        documents = []
        for hit in response["hits"]["hits"]:
            doc = Document(
                page_content=hit["_source"]["content"],
                metadata=hit["_source"].get("metadata", {})
            )
            documents.append(doc)
        
        logger.info(f"✅ BM25 found {len(documents)} documents")
        return documents
    
    def search_with_score(
        self, 
        query: str, 
        k: int = 5
    ) -> List[tuple]:
        """
        Search for documents with BM25 scores.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of tuples (Document, score)
        """
        logger.debug(f"BM25 search with score: '{query[:50]}...' (k={k})")
        
        search_body = {
            "query": {
                "match": {
                    "content": {
                        "query": query,
                        "fuzziness": "AUTO"
                    }
                }
            },
            "size": k
        }
        
        response = self.client.search(index=self.index_name, body=search_body)
        
        results = []
        for hit in response["hits"]["hits"]:
            doc = Document(
                page_content=hit["_source"]["content"],
                metadata=hit["_source"].get("metadata", {})
            )
            score = hit["_score"]
            results.append((doc, score))
        
        logger.info(f"✅ BM25 found {len(results)} documents")
        return results
    
    def delete_index(self):
        """Delete the index."""
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
            logger.info(f"Deleted index: {self.index_name}")
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the index."""
        if not self.client.indices.exists(index=self.index_name):
            return {"error": "Index does not exist"}
        
        stats = self.client.indices.stats(index=self.index_name)
        return {
            "name": self.index_name,
            "docs_count": stats["indices"][self.index_name]["primaries"]["docs"]["count"],
            "size_in_bytes": stats["indices"][self.index_name]["primaries"]["store"]["size_in_bytes"],
        }


if __name__ == "__main__":
    # Test Elasticsearch service
    try:
        service = ElasticsearchService(index_name="test_index")
        
        # Add sample texts
        texts = [
            "Python là một ngôn ngữ lập trình phổ biến",
            "Machine learning là một nhánh của AI",
            "Elasticsearch giúp tìm kiếm văn bản hiệu quả"
        ]
        service.add_texts(texts)
        
        # Search
        results = service.search("ngôn ngữ lập trình", k=2)
        print("\nSearch results:")
        for doc in results:
            print(f"- {doc.page_content}")
        
        # Get index info
        print(f"\nIndex info: {service.get_index_info()}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Elasticsearch is running on localhost:9200")
