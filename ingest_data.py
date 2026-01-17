"""
Data Ingestion Script - Nạp dữ liệu vào Qdrant và Elasticsearch

Script này được sử dụng để:
1. Đọc dữ liệu từ file (PDF, TXT, etc.)
2. Chunking (cắt nhỏ văn bản)
3. Lưu vào Qdrant (với embeddings)
4. Lưu vào Elasticsearch (text thuần)
"""
import asyncio
from pathlib import Path
from typing import List, Dict
import sys
sys.path.append(str(Path(__file__).parent))

from src.app.db.qdrant_conn import get_qdrant_client
from src.app.db.es_conn import get_es_client


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Cắt văn bản thành các chunks nhỏ
    
    Args:
        text: Văn bản cần cắt
        chunk_size: Kích thước mỗi chunk (số ký tự)
        overlap: Số ký tự overlap giữa các chunks
    
    Returns:
        List các chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Cố gắng cắt ở dấu câu để không cắt ngang câu
        if end < len(text):
            # Tìm dấu câu gần nhất
            last_period = chunk.rfind('.')
            last_question = chunk.rfind('?')
            last_exclamation = chunk.rfind('!')
            
            split_point = max(last_period, last_question, last_exclamation)
            if split_point > chunk_size * 0.5:  # Chỉ cắt nếu không quá ngắn
                chunk = chunk[:split_point + 1]
                end = start + split_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks


async def load_sample_data():
    """
    Load sample data về Luật Giao thông Việt Nam
    Đây là dữ liệu mẫu, bạn cần thay bằng dữ liệu thật từ file PDF
    """
    sample_documents = [
        {
            "content": "Nghị định 100/2019/NĐ-CP, Điều 6, Khoản 4, Điểm e: Phạt tiền từ 600.000 đồng đến 1.000.000 đồng đối với người điều khiển xe thực hiện hành vi: Không chấp hành hiệu lệnh của đèn tín hiệu giao thông.",
            "metadata": {
                "source": "Nghị định 100/2019/NĐ-CP",
                "article": "Điều 6",
                "clause": "Khoản 4, Điểm e"
            }
        },
        {
            "content": "Nghị định 100/2019/NĐ-CP, Điều 6, Khoản 9: Phạt tiền từ 100.000 đồng đến 200.000 đồng đối với người điều khiển xe mô tô, xe gắn máy không đội mũ bảo hiểm hoặc đội mũ bảo hiểm không cài quai đúng quy cách.",
            "metadata": {
                "source": "Nghị định 100/2019/NĐ-CP",
                "article": "Điều 6",
                "clause": "Khoản 9"
            }
        },
        {
            "content": "Nghị định 100/2019/NĐ-CP, Điều 7, Khoản 7: Phạt tiền từ 30.000.000 đồng đến 40.000.000 đồng đối với người điều khiển xe ô tô có nồng độ cồn trong máu hoặc hơi thở vượt quá 80 miligam/100 mililít máu hoặc vượt quá 0,4 miligam/1 lít khí thở.",
            "metadata": {
                "source": "Nghị định 100/2019/NĐ-CP",
                "article": "Điều 7",
                "clause": "Khoản 7"
            }
        },
        {
            "content": "Luật Giao thông đường bộ 2008, Điều 29: Tốc độ tối đa cho phép đối với xe mô tô, xe gắn máy trong khu dân cư là 50 km/h, ngoài khu dân cư là 60 km/h.",
            "metadata": {
                "source": "Luật Giao thông đường bộ 2008",
                "article": "Điều 29",
                "clause": ""
            }
        },
        {
            "content": "Nghị định 100/2019/NĐ-CP, Điều 6, Khoản 3: Phạt tiền từ 400.000 đồng đến 600.000 đồng đối với người điều khiển xe mô tô, xe gắn máy đi vào đường cấm.",
            "metadata": {
                "source": "Nghị định 100/2019/NĐ-CP",
                "article": "Điều 6",
                "clause": "Khoản 3"
            }
        }
    ]
    
    return sample_documents


async def ingest_data(documents: List[Dict]):
    """
    Nạp dữ liệu vào cả Qdrant và Elasticsearch
    
    Args:
        documents: List các documents cần nạp
    """
    print("🚀 Bắt đầu ingest dữ liệu...")
    
    # Lấy clients
    qdrant_client = get_qdrant_client()
    es_client = get_es_client()
    
    # Tạo collections/indices
    print("📦 Tạo collections/indices...")
    await qdrant_client.create_collection(vector_size=768)
    await es_client.create_index()
    
    # Nạp vào Qdrant (với embeddings)
    print("🔵 Đang nạp dữ liệu vào Qdrant...")
    await qdrant_client.insert_documents(documents)
    
    # Nạp vào Elasticsearch (text thuần)
    print("🟢 Đang nạp dữ liệu vào Elasticsearch...")
    await es_client.insert_documents(documents)
    
    print("✅ Hoàn thành ingest dữ liệu!")


async def main():
    """Main function"""
    print("=" * 60)
    print("DATA INGESTION - Vietnam Traffic Law RAG System")
    print("=" * 60)
    
    # Load sample data
    documents = await load_sample_data()
    print(f"\n📄 Đã load {len(documents)} documents")
    
    # Ingest data
    await ingest_data(documents)
    
    # Close connections
    await es_client.close()
    
    print("\n" + "=" * 60)
    print("✨ Ingest hoàn tất! Bạn có thể chạy API server.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
