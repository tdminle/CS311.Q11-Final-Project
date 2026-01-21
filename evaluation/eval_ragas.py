"""
RAGAS Evaluation - Đánh giá chất lượng RAG với open-source models
Sử dụng HuggingFace models thay vì OpenAI
"""
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.rag import RAGService

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets import Dataset
    from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("❌ RAGAS not available. Install: uv add ragas datasets")
    sys.exit(1)


def setup_ragas_models():
    """Setup open-source models for RAGAS via HuggingFace."""
    print("🔧 Setting up RAGAS with HuggingFace models...")
    
    # LLM cho RAGAS - dùng model nhỏ nhưng hiệu quả
    # Qwen2.5-3B-Instruct: nhỏ, nhanh, chất lượng tốt
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-3B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
    )
    
    # Embeddings cho RAGAS
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Nhỏ, nhanh
        model_kwargs={"device": "cpu"}
    )
    
    # Wrap for RAGAS
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    
    print("✅ Models ready:")
    print("  LLM: Qwen/Qwen2.5-3B-Instruct")
    print("  Embeddings: all-MiniLM-L6-v2")
    
    return ragas_llm, ragas_embeddings


def evaluate_with_ragas(rag_service: RAGService, test_data: List[Dict]) -> Dict[str, Any]:
    """Đánh giá RAG với RAGAS framework."""
    print("\n" + "="*60)
    print("🎯 RAGAS EVALUATION")
    print("="*60)
    
    # Setup models
    ragas_llm, ragas_embeddings = setup_ragas_models()
    
    # Collect data
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    
    print(f"\n📝 Processing {len(test_data)} questions...")
    
    for i, item in enumerate(test_data, 1):
        question = item["question"]
        print(f"[{i}/{len(test_data)}] {question[:50]}...")
        
        try:
            result = rag_service.generate_response_sync(question)
            
            questions.append(question)
            answers.append(result["answer"])
            contexts_list.append([doc.page_content for doc in result["source_documents"]])
            ground_truths.append(item.get("ground_truth", ""))
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Create dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    print(f"\n✅ Created dataset with {len(questions)} items")
    
    # Configure metrics với custom models
    print("\n🔍 Running RAGAS metrics...")
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    
    # Set LLM and embeddings for metrics
    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, 'embeddings'):
            metric.embeddings = ragas_embeddings
    
    # Evaluate
    try:
        result = evaluate(
            dataset,
            metrics=metrics,
        )
        
        print("\n" + "="*60)
        print("📊 RAGAS SCORES")
        print("="*60)
        
        scores = {}
        for key, value in result.items():
            if key not in ['question', 'answer', 'contexts', 'ground_truth']:
                score = float(value)
                scores[key] = score
                
                # Đánh giá
                if score >= 0.8:
                    status = "✅ Excellent"
                elif score >= 0.6:
                    status = "⚠️  Good"
                else:
                    status = "❌ Needs improvement"
                
                print(f"\n{key.upper()}:")
                print(f"  Score: {score:.3f} | {status}")
        
        print("="*60)
        
        # Giải thích
        print("\n💡 GIẢI THÍCH:")
        print("  Faithfulness: Câu trả lời có trung thực với context?")
        print("  Answer Relevancy: Câu trả lời có liên quan với câu hỏi?")
        print("  Context Precision: Retrieved contexts có chính xác?")
        print("  Context Recall: Contexts có đủ thông tin cần thiết?")
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {
            "timestamp": timestamp,
            "model": {
                "llm": "Qwen/Qwen2.5-3B-Instruct",
                "embeddings": "all-MiniLM-L6-v2"
            },
            "scores": scores,
            "num_samples": len(questions)
        }
        
        output_file = f"evaluation/ragas_eval_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Saved to: {output_file}\n")
        
        return output
        
    except Exception as e:
        print(f"\n❌ RAGAS evaluation failed: {e}")
        return None


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
    
    # Initialize RAG service
    print("📦 Initializing RAG Service...")
    rag_service = RAGService(
        collection_name="Law",
        es_index_name="law_documents",
        top_k=5,
        use_rerank=True
    )
    print("✅ Ready!\n")
    
    # Run evaluation
    evaluate_with_ragas(rag_service, test_data)


if __name__ == "__main__":
    main()
