"""
Data loader service for loading processed JSON data into Qdrant and Elasticsearch.
Reads from output_data folder.
"""
import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Batch
from elasticsearch import Elasticsearch
from langchain_huggingface import HuggingFaceEmbeddings


def load_json_data(file_path: str) -> list[dict]:
    """Load data from JSON file."""
    print(f"📂 Loading data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} items")
    return data


def find_latest_json(output_folder: str = "output_data") -> str:
    """
    Find the latest or combined JSON file in output folder.
    
    Args:
        output_folder: Path to output folder
        
    Returns:
        Path to JSON file to load
    """
    output_path = Path(output_folder)
    
    # First check for combined output
    combined_file = output_path / "combined_output.json"
    if combined_file.exists():
        print(f"📁 Found combined output: {combined_file}")
        return str(combined_file)
    
    # Otherwise, find all JSON files
    json_files = list(output_path.glob("*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {output_folder}")
    
    # Return the most recent file
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"📁 Found latest JSON: {latest_file}")
    return str(latest_file)

def load_to_qdrant(data: list[dict], collection_name: str = "Law"):
    """Load data into Qdrant."""
    print(f"\n🔵 Loading into Qdrant collection: {collection_name}")
    
    # Initialize client
    client = QdrantClient(host="localhost", port=6333)
    
    # Initialize embeddings
    print("  Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="dangvantuan/vietnamese-embedding",
        model_kwargs={"device": "cpu"}
    )
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if collection_name in collection_names:
        collection_info = client.get_collection(collection_name)
        if collection_info.points_count > 0:
            print(f"⚠️  Collection already has {collection_info.points_count} points")
            response = input("  Delete and reload? (yes/no): ")
            if response.lower() == 'yes':
                client.delete_collection(collection_name)
                print("  Deleted old collection")
            else:
                print("  Skipping Qdrant")
                return
    
    # Create collection
    if collection_name not in [c.name for c in client.get_collections().collections]:
        print("  Creating collection...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
    
    # Prepare and upload data
    print("  Generating embeddings and uploading...")
    texts = [item.get("context", "") for item in data]
    metadatas = [{"title": item.get("title", "")} for item in data]
    
    # Generate embeddings in batches
    batch_size = 32
    all_points = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        
        # Generate embeddings
        batch_embeddings = embeddings.embed_documents(batch_texts)
        
        # Create points
        for j, (text, embedding, metadata) in enumerate(zip(batch_texts, batch_embeddings, batch_metadatas)):
            point_id = i + j
            all_points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "page_content": text,
                        "metadata": metadata
                    }
                )
            )
        
        print(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)} documents")
    
    # Upload all points
    client.upsert(
        collection_name=collection_name,
        points=all_points
    )
    
    print(f"✅ Uploaded {len(all_points)} points to Qdrant")

def load_to_elasticsearch(data: list[dict], index_name: str = "law_documents"):
    """Load data into Elasticsearch."""
    print(f"\n🟡 Loading into Elasticsearch index: {index_name}")
    
    # Initialize client
    es = Elasticsearch(
        ["http://localhost:9200"],
        verify_certs=False,
        ssl_show_warn=False
    )
    
    # Check if index exists
    if es.indices.exists(index=index_name):
        count = es.count(index=index_name)
        if count['count'] > 0:
            print(f"⚠️  Index already has {count['count']} documents")
            response = input("  Delete and reload? (yes/no): ")
            if response.lower() == 'yes':
                es.indices.delete(index=index_name)
                print("  Deleted old index")
            else:
                print("  Skipping Elasticsearch")
                return
    
    # Create index with Vietnamese analyzer
    if not es.indices.exists(index=index_name):
        print("  Creating index...")
        index_settings = {
            "mappings": {
                "properties": {
                    "page_content": {"type": "text"},
                    "metadata": {
                        "properties": {
                            "title": {"type": "text"}
                        }
                    }
                }
            }
        }
        es.indices.create(index=index_name, body=index_settings)
    
    # Upload documents
    print("  Uploading documents...")
    for i, item in enumerate(data):
        doc = {
            "page_content": item.get("context", ""),
            "metadata": {
                "title": item.get("title", "")
            }
        }
        es.index(index=index_name, id=i, document=doc)
        
        if (i + 1) % 50 == 0:
            print(f"  Uploaded {i + 1}/{len(data)} documents")
    
    # Refresh index
    es.indices.refresh(index=index_name)
    
    print(f"✅ Uploaded {len(data)} documents to Elasticsearch")


def main():
    """Main function."""
    print("=" * 60)
    print("🚀 Vietnamese Law Data Loader")
    print("=" * 60)
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    output_folder = project_root / "output_data"
    
    # Check if output_data folder exists
    if not output_folder.exists():
        print(f"❌ Output folder not found: {output_folder}")
        print(f"ℹ️  Please run 'python data_preperation/processing.py' first to process PDFs")
        return
    
    # Find JSON file to load
    try:
        json_file = find_latest_json(str(output_folder))
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print(f"ℹ️  Please run 'python data_preperation/processing.py' first to process PDFs")
        return
    
    # Load data
    data = load_json_data(json_file)
    
    # Load to Qdrant
    try:
        load_to_qdrant(data, collection_name="Law")
    except Exception as e:
        print(f"❌ Error loading to Qdrant: {e}")
    
    # Load to Elasticsearch
    try:
        load_to_elasticsearch(data, index_name="law_documents")
    except Exception as e:
        print(f"❌ Error loading to Elasticsearch: {e}")
    
    # Verification
    print("\n" + "=" * 60)
    print("🔍 Verification:")
    try:
        client = QdrantClient(host="localhost", port=6333)
        info = client.get_collection("Law")
        print(f"  Qdrant 'Law': {info.points_count} points")
    except:
        pass
    
    try:
        es = Elasticsearch(["http://localhost:9200"], verify_certs=False, ssl_show_warn=False)
        count = es.count(index="law_documents")
        print(f"  Elasticsearch 'law_documents': {count['count']} documents")
    except:
        pass
    
    print("=" * 60)
    print("✅ Done!")


if __name__ == "__main__":
    main()
