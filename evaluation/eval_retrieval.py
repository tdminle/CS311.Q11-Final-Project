"""
Retrieval Evaluation - Đánh giá chất lượng tìm kiếm
Metrics: Hit Rate, MRR, Context Recall
"""
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.retrieval import RetrievalService


def calculate_hit_rate(retrieved_docs: List, relevant_keywords: List[str]) -> bool:
    """Check if any retrieved doc contains relevant keywords."""
    for doc in retrieved_docs:
        content = doc.page_content.lower()
        if any(keyword.lower() in content for keyword in relevant_keywords):
            return True
    return False


def calculate_mrr(retrieved_docs: List, relevant_keywords: List[str]) -> float:
    """Calculate Mean Reciprocal Rank - vị trí của doc đúng đầu tiên."""
    for i, doc in enumerate(retrieved_docs, 1):
        content = doc.page_content.lower()
        if any(keyword.lower() in content for keyword in relevant_keywords):
            return 1.0 / i
    return 0.0


def calculate_context_recall(retrieved_docs: List, ground_truth: str) -> float:
    """Đánh giá xem context có đủ thông tin để trả lời không."""
    if not ground_truth:
        return 1.0
    
    # Simple approach: kiểm tra các từ quan trọng trong ground_truth có trong docs
    gt_words = set(ground_truth.lower().split())
    gt_words = {w for w in gt_words if len(w) > 3}  # Lọc từ ngắn
    
    if not gt_words:
        return 1.0
    
    # Đếm số từ trong ground_truth xuất hiện trong retrieved docs
    found_words = set()
    for doc in retrieved_docs:
        content_words = set(doc.page_content.lower().split())
        found_words.update(gt_words & content_words)
    
    recall = len(found_words) / len(gt_words)
    return recall


def evaluate_retrieval(retrieval_service: RetrievalService, test_data: List[Dict]) -> Dict[str, Any]:
    """Đánh giá toàn diện hệ thống retrieval."""
    print("="*60)
    print("🔍 RETRIEVAL EVALUATION")
    print("="*60)
    
    results = []
    total_hit = 0
    total_mrr = 0.0
    total_recall = 0.0
    
    for i, item in enumerate(test_data, 1):
        question = item["question"]
        keywords = item.get("contexts", [])
        ground_truth = item.get("ground_truth", "")
        
        print(f"\n[{i}/{len(test_data)}] {question[:50]}...")
        
        try:
            # Retrieve documents
            docs = retrieval_service.retrieve(question, k=5)
            
            # Calculate metrics
            hit = calculate_hit_rate(docs, keywords)
            mrr = calculate_mrr(docs, keywords)
            recall = calculate_context_recall(docs, ground_truth)
            
            total_hit += int(hit)
            total_mrr += mrr
            total_recall += recall
            
            result = {
                "question": question,
                "num_retrieved": len(docs),
                "hit": hit,
                "mrr": mrr,
                "context_recall": recall,
                "top_doc": docs[0].page_content[:100] if docs else ""
            }
            results.append(result)
            
            print(f"  Hit: {'✅' if hit else '❌'} | MRR: {mrr:.3f} | Recall: {recall:.3f}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({"question": question, "error": str(e)})
    
    # Calculate averages
    n = len(test_data)
    hit_rate = total_hit / n
    avg_mrr = total_mrr / n
    avg_recall = total_recall / n
    
    print("\n" + "="*60)
    print("📊 RETRIEVAL METRICS")
    print("="*60)
    print(f"\n✅ Hit Rate (Recall@5): {hit_rate:.2%}")
    print(f"   → {total_hit}/{n} câu hỏi tìm được tài liệu đúng")
    print(f"\n📍 MRR (Mean Reciprocal Rank): {avg_mrr:.3f}")
    print(f"   → Tài liệu đúng trung bình ở vị trí {1/avg_mrr:.1f}" if avg_mrr > 0 else "   → Không tìm thấy tài liệu đúng")
    print(f"\n📄 Context Recall: {avg_recall:.2%}")
    print(f"   → Trung bình {avg_recall:.0%} thông tin cần thiết được tìm thấy")
    print("="*60)
    
    # Đánh giá tổng thể
    print("\n💡 ĐÁNH GIÁ:")
    if hit_rate >= 0.8:
        print("  ✅ Hit Rate xuất sắc - Retrieval rất chính xác")
    elif hit_rate >= 0.6:
        print("  ⚠️  Hit Rate tốt - Có thể cải thiện retrieval")
    else:
        print("  ❌ Hit Rate thấp - Cần cải thiện retrieval")
    
    if avg_mrr >= 0.5:
        print("  ✅ MRR tốt - Tài liệu đúng thường ở top")
    elif avg_mrr >= 0.3:
        print("  ⚠️  MRR trung bình - Tài liệu đúng không luôn ở top")
    else:
        print("  ❌ MRR thấp - Tài liệu đúng thường ở vị trí xa")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "metrics": {
            "hit_rate": f"{hit_rate:.2%}",
            "mrr": f"{avg_mrr:.3f}",
            "context_recall": f"{avg_recall:.2%}"
        },
        "results": results
    }
    
    output_file = f"evaluation/retrieval_eval_{timestamp}.json"
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
    
    # Load test data
    test_file = "evaluation/test_questions.json"
    if not Path(test_file).exists():
        print(f"❌ File not found: {test_file}")
        return
    
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"\n📂 Loaded {len(test_data)} test questions")
    
    # Initialize retrieval service
    print("📦 Initializing Retrieval Service...")
    retrieval_service = RetrievalService(
        collection_name="Law",
        es_index_name="law_documents",
        top_k=5,
        use_rerank=True
    )
    print("✅ Ready!\n")
    
    # Run evaluation
    evaluate_retrieval(retrieval_service, test_data)


if __name__ == "__main__":
    main()
