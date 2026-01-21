"""
Embedding model initialization module.
Uses Vietnamese embedding model via HuggingFace.
"""
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from src.utils.logger import get_logger

# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger(__name__)

# Vietnamese embedding model configuration
MODEL_NAME = "dangvantuan/vietnamese-embedding"
VECTOR_SIZE = 768  # Dimension of the embedding vectors

logger.info(f"Initializing HuggingFace embeddings with model: {MODEL_NAME}")
logger.debug(f"Using HF_TOKEN: {'***' if os.getenv('HF_TOKEN') else 'NOT SET'}")
logger.debug(f"Vector size: {VECTOR_SIZE}")

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEndpointEmbeddings(
    model=MODEL_NAME,
    task="feature-extraction",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)

logger.info("✅ Embeddings initialized successfully")


def get_vector_size() -> int:
    """Get the vector dimension size."""
    return VECTOR_SIZE


if __name__ == "__main__":
    # Test embedding
    text = "Python là một ngôn ngữ lập trình phổ biến"
    result = embeddings.embed_query(text)
    print(f"Text: {text}")
    print(f"Embedding dimension: {len(result)}")
    print(f"First 5 values: {result[:5]}")
