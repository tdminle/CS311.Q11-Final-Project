"""
Run All Evaluations - Chạy đầy đủ 3 loại đánh giá
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.rag import RAGService
from src.services.retrieval import RetrievalService
import json

# Import evaluation modules
from eval_basic import evaluate_basic
from eval_retrieval import evaluate_retrieval

try:
    from eval_ragas import evaluate_with_ragas
    RAGAS_AVAILABLE = True
except:
    RAGAS_AVAILABLE = False
    print("⚠️  RAGAS not available - install with: uv add ragas datasets")


def main():
    """Run all evaluations."""
    print("\n" + "="*60)
    print("🎯 COMPLETE RAG EVALUATION")
    print("="*60)
    
    # Check test file (đúng path từ thư mục gốc)
    test_file = "evaluation/test_questions.json"
    
    if not Path(test_file).exists():
        print(f"\n❌ Test file not found: {test_file}")
        print("Create evaluation/test_questions.json first")
        return
    
    # Load test data
    print(f"\n📂 Loading test questions from {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"✅ Loaded {len(test_data)} questions\n")
    
    # Initialize services
    print("📦 Initializing services...")
    rag_service = RAGService(
        collection_name="Law",
        es_index_name="law_documents",
        top_k=5,
        use_rerank=True
    )
    
    retrieval_service = RetrievalService(
        collection_name="Law",
        es_index_name="law_documents",
        top_k=5,
        use_rerank=True
    )
    print("✅ Services ready!\n")
    
    # 1. Basic Evaluation
    print("\n" + "🔹" * 30)
    print("PART 1/3: BASIC EVALUATION")
    print("🔹" * 30)
    basic_results = evaluate_basic(rag_service, test_data)
    
    # 2. Retrieval Evaluation
    print("\n" + "🔹" * 30)
    print("PART 2/3: RETRIEVAL EVALUATION")
    print("🔹" * 30)
    retrieval_results = evaluate_retrieval(retrieval_service, test_data)
    
    # 3. RAGAS Evaluation
    if RAGAS_AVAILABLE:
        print("\n" + "🔹" * 30)
        print("PART 3/3: RAGAS EVALUATION")
        print("🔹" * 30)
        ragas_results = evaluate_with_ragas(rag_service, test_data)
    else:
        print("\n⚠️  Skipping RAGAS evaluation (not installed)")
        ragas_results = None
    
    # Summary
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    
    print("\n⚡ BASIC:")
    print(f"  Success Rate: {basic_results['metrics']['success_rate']}")
    print(f"  Avg Response Time: {basic_results['metrics']['avg_response_time']}")
    
    print("\n🔍 RETRIEVAL:")
    print(f"  Hit Rate: {retrieval_results['metrics']['hit_rate']}")
    print(f"  MRR: {retrieval_results['metrics']['mrr']}")
    print(f"  Context Recall: {retrieval_results['metrics']['context_recall']}")
    
    if ragas_results:
        print("\n🎯 RAGAS:")
        for metric, score in ragas_results['scores'].items():
            print(f"  {metric}: {score:.3f}")
    
    print("\n" + "="*60)
    print("✅ ALL EVALUATIONS COMPLETE!")
    print("="*60)
    print(f"\n📁 Results saved in evaluation/ folder")
    print()


if __name__ == "__main__":
    main()
