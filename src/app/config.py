"""
Config module - Load cấu hình từ biến môi trường
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Cấu hình ứng dụng từ file .env"""
    
    # Database Config
    QDRANT_URL: str = "http://localhost:6333"
    ES_URL: str = "http://localhost:9200"
    COLLECTION_NAME: str = "traffic_law"
    
    # Model API Config
    EMBEDDING_API_URL: str
    RERANK_API_URL: str
    LLM_API_URL: str
    
    # API Key
    API_KEY: str = ""
    
    # Langfuse Observability Config
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Singleton pattern để load settings 1 lần duy nhất"""
    return Settings()
