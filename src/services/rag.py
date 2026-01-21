"""
RAG (Retrieval-Augmented Generation) Service.
Combines retrieval and generation for question answering.
"""
import json
from typing import Dict, Any, Optional
from src.services.retrieval import RetrievalService
from src.services.generator import GeneratorService
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class RAGService:
    """
    Complete RAG service combining retrieval and generation.
    Uses EnsembleRetriever (Qdrant + Elasticsearch) with reranking.
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
        system_prompt: str = None,
    ):
        """
        Initialize RAG Service.
        
        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            es_host: Elasticsearch server host
            es_port: Elasticsearch server port
            collection_name: Name of the Qdrant collection
            es_index_name: Name of the Elasticsearch index (defaults to collection_name)
            top_k: Number of documents to retrieve
            use_rerank: Whether to use reranking
            qdrant_weight: Weight for Qdrant retriever
            es_weight: Weight for Elasticsearch retriever
            system_prompt: Custom system prompt for generator
        """
        logger.info("="*60)
        logger.info("Initializing RAG Service")
        logger.info("="*60)
        
        # Initialize retrieval service
        self.retrieval_service = RetrievalService(
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            es_host=es_host,
            es_port=es_port,
            collection_name=collection_name,
            es_index_name=es_index_name,
            top_k=top_k,
            use_rerank=use_rerank,
            qdrant_weight=qdrant_weight,
            es_weight=es_weight,
        )
        
        # Initialize generator service
        self.generator_service = GeneratorService(system_prompt=system_prompt)
        
        logger.info("✅ RAG Service initialized successfully")
        logger.info("="*60)
    
    def add_documents(self, documents) -> Dict[str, Any]:
        """Add documents to the retrieval storage."""
        return self.retrieval_service.add_documents(documents)
    
    def add_texts(self, texts, metadatas=None) -> Dict[str, Any]:
        """Add texts to the retrieval storage."""
        return self.retrieval_service.add_texts(texts, metadatas)
    
    async def generate_response(self, question: str, k: int = None) -> str:
        """
        Generate a response using RAG pipeline (async).
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            Generated response
        """
        logger.info("="*60)
        logger.info(f"🔍 Processing question: {question}")
        logger.info("="*60)
        
        # Step 1: Retrieve relevant documents
        logger.info("📄 Step 1: Retrieving relevant documents...")
        context = self.retrieval_service.retrieve_as_json(question, k)
        
        # Step 2: Generate response
        logger.info("🤖 Step 2: Generating response...")
        response = await self.generator_service.generate(question, context)
        
        logger.info("✅ Response generation completed")
        logger.info("="*60)
        
        return response
    
    def generate_response_sync(self, question: str, k: int = None) -> Dict[str, Any]:
        """
        Generate a response using RAG pipeline (sync).
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            Dictionary containing 'answer' and 'source_documents'
        """
        logger.info("="*60)
        logger.info(f"🔍 Processing question: {question}")
        logger.info("="*60)
        
        # Step 1: Retrieve relevant documents
        logger.info("📄 Step 1: Retrieving relevant documents...")
        documents = self.retrieval_service.retrieve(question, k)
        
        # Get context as JSON string for generation
        context = self.retrieval_service.retrieve_as_json(question, k)
        
        # Step 2: Generate response
        logger.info("🤖 Step 2: Generating response...")
        answer = self.generator_service.generate_sync(question, context)
        
        logger.info("✅ Response generation completed")
        logger.info("="*60)
        
        return {
            "answer": answer,
            "source_documents": documents
        }
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the storage backends."""
        return self.retrieval_service.get_storage_info()


if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("Testing RAG Service...")
        print("="*60)
        
        try:
            # Initialize RAG service
            rag = RAGService(
                collection_name="test_rag",
                top_k=3,
                use_rerank=True
            )
            
            # Add sample data
            texts = [
                "Python là một ngôn ngữ lập trình phổ biến và dễ học. Python được sử dụng rộng rãi trong nhiều lĩnh vực.",
                "Machine learning là một nhánh của trí tuệ nhân tạo, cho phép máy tính học từ dữ liệu.",
                "RAG (Retrieval-Augmented Generation) kết hợp retrieval và generation để trả lời câu hỏi chính xác hơn.",
                "Vector database như Qdrant lưu trữ embeddings để thực hiện tìm kiếm ngữ nghĩa hiệu quả."
            ]
            rag.add_texts(texts)
            
            # Test question
            question = "Python là gì và được sử dụng như thế nào?"
            
            print(f"\n❓ Question: {question}\n")
            
            # Generate response
            response = await rag.generate_response(question)
            
            print(f"\n📝 Answer: {response}")
            
            # Show storage info
            print(f"\n📊 Storage Info: {rag.get_storage_info()}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(main())
