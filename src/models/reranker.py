"""
Reranker model for improving retrieval results.
Uses BAAI/bge-reranker-v2-m3 for cross-encoder reranking.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Model configuration
MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH = 512

logger.info(f"Loading reranker model: {MODEL_NAME}")

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()  # Set to evaluation mode

logger.info("✅ Reranker model loaded successfully")


def rerank_documents(
    query: str,
    documents: List[Tuple[str, float]],
    top_k: int = None
) -> List[Tuple[str, float]]:
    """
    Rerank documents using BGE reranker model.
    
    Args:
        query: The search query
        documents: List of tuples (document_text, original_score)
        top_k: Number of top documents to return after reranking (None = return all)
        
    Returns:
        List of tuples (document_text, rerank_score) sorted by rerank score descending
    """
    if not documents:
        logger.warning("No documents to rerank")
        return []
    
    logger.info(f"🔄 Reranking {len(documents)} documents")
    logger.debug(f"Query: '{query[:50]}...'")
    
    # Prepare pairs for reranking
    pairs = [[query, doc[0]] for doc in documents]
    
    logger.debug(f"Tokenizing {len(pairs)} query-document pairs")
    
    # Tokenize and compute scores
    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=MAX_LENGTH
        )
        
        logger.debug("Computing reranking scores")
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        scores = scores.cpu().numpy().tolist()
    
    # Combine documents with new scores
    reranked = list(zip([doc[0] for doc in documents], scores))
    
    # Sort by rerank score (descending)
    reranked.sort(key=lambda x: x[1], reverse=True)
    
    # Apply top_k if specified
    if top_k is not None:
        reranked = reranked[:top_k]
    
    logger.info(f"✅ Reranking completed. Top score: {reranked[0][1]:.4f}")
    logger.debug(f"Score range: [{min(s[1] for s in reranked):.4f}, {max(s[1] for s in reranked):.4f}]")
    
    return reranked


if __name__ == "__main__":
    # Test reranking
    test_query = "Python là gì?"
    test_docs = [
        ("Python là một ngôn ngữ lập trình phổ biến", 0.85),
        ("Java là ngôn ngữ lập trình hướng đối tượng", 0.60),
        ("Python được sử dụng trong machine learning", 0.75),
    ]
    
    print("\nOriginal documents:")
    for i, (doc, score) in enumerate(test_docs, 1):
        print(f"{i}. [{score:.2f}] {doc}")
    
    reranked = rerank_documents(test_query, test_docs)
    
    print("\nReranked documents:")
    for i, (doc, score) in enumerate(reranked, 1):
        print(f"{i}. [{score:.4f}] {doc}")
