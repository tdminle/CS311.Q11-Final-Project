"""
Basic Evaluation - Đánh giá cơ bản
Metrics: Response time, Success rate, Answer length
"""
import json
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.rag import RAGService


def evaluate_basic(rag_service: RAGService, test_data: List[Dict]) -> Dict[str, Any]:
    """Đánh giá basic metrics."""
    print("="*60)
    print("⚡ BASIC EVALUATION")
    print("="*60)
    
    results = []
    total_time = 0
    success = 0
    
    for i, item in enumerate(test_data, 1):
        question = item["question"]
        print(f"\n[{i}/{len(test_data)}] {question[:50]}...")
        
        try:
            start = time.time()
            result = rag_service.generate_response_sync(question)
            elapsed = time.time() - start
            
            total_time += elapsed
            success += 1
            
            answer = result["answer"]
            sources = result["source_documents"]
            
            results.append({
                "question": question,
                "answer": answer,
                "response_time": elapsed,
                "answer_length": len(answer),
                "num_sources": len(sources),
                "success": True
            })
            
            print(f"  ✅ {elapsed:.2f}s | {len(answer)} chars | {len(sources)} sources")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "question": question,
                "error": str(e),
                "success": False
            })
    
    # Metrics
    n = len(test_data)
    success_rate = success / n
    avg_time = total_time / success if success > 0 else 0
    
    successful = [r for r in results if r.get("success")]
    avg_length = sum(r["answer_length"] for r in successful) / len(successful) if successful else 0
    avg_sources = sum(r["num_sources"] for r in successful) / len(successful) if successful else 0
    
    print("\n" + "="*60)
    print("📊 BASIC METRICS")
    print("="*60)
    print(f"\n✅ Success Rate: {success_rate:.0%} ({success}/{n})")
    print(f"⏱️  Avg Response Time: {avg_time:.2f}s")
    print(f"📝 Avg Answer Length: {avg_length:.0f} chars")
    print(f"📄 Avg Sources: {avg_sources:.1f}")
    print("="*60)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "metrics": {
            "success_rate": f"{success_rate:.0%}",
            "avg_response_time": f"{avg_time:.2f}s",
            "avg_answer_length": f"{avg_length:.0f}",
            "avg_sources": f"{avg_sources:.1f}"
        },
        "results": results
    }
    
    output_file = f"evaluation/basic_eval_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved to: {output_file}\n")
    
    return output


def main():
    """Main function."""
    # Chuyển về thư mục gốc nếu đang ở evaluation/
    import os
    if os.path.basename(os.getcwd()) == 'evaluation':
        os.chdir('..')
    
    test_file = "evaluation/test_questions.json"
    if not Path(test_file).exists():
        print(f"❌ File not found: {test_file}")
        return
    
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"\n📂 Loaded {len(test_data)} test questions")
    
    print("📦 Initializing RAG Service...")
    rag_service = RAGService(
        collection_name="Law",
        es_index_name="law_documents",
        top_k=5,
        use_rerank=True
    )
    print("✅ Ready!\n")
    
    evaluate_basic(rag_service, test_data)


if __name__ == "__main__":
    main()
