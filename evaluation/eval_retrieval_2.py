"""
Retrieval Evaluation for Core Legal QA
Metrics: Hit Rate@K, MRR, Context Recall
"""

import json
import sys
import os
import re
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# =========================
# FIX IMPORT PATH
# =========================
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.retrieval import RetrievalService


# =========================
# UTILS
# =========================

def normalize(text: str) -> str:
    return text.lower().strip()


# =========================
# RELEVANCE CHECK (CORE)
# =========================

def is_relevant_doc(doc, expected_article: str, question: str) -> bool:
    """
    Một document được xem là relevant nếu:
    1. Có chứa số Điều mong đợi
    2. Nội dung có ít nhất 1 từ khóa quan trọng của câu hỏi
    """

    text_sources = []

    # metadata
    if hasattr(doc, "metadata"):
        for v in doc.metadata.values():
            if isinstance(v, str):
                text_sources.append(v.lower())

    # content
    if hasattr(doc, "page_content"):
        text_sources.append(doc.page_content.lower())

    full_text = " ".join(text_sources)

    # --- Match Điều ---
    article_num = re.search(r"\d+", expected_article)
    if article_num:
        if article_num.group() not in full_text:
            return False

    # --- Match nội dung câu hỏi ---
    question_terms = [
        w for w in question.lower().split()
        if len(w) > 4
    ]

    overlap = sum(1 for w in question_terms if w in full_text)
    return overlap >= 1


# =========================
# METRICS
# =========================

def calculate_hit_rate(
    docs: List,
    expected_article: str,
    question: str
) -> bool:
    """Hit@K"""
    return any(
        is_relevant_doc(doc, expected_article, question)
        for doc in docs
    )


def calculate_mrr(
    docs: List,
    expected_article: str,
    question: str
) -> float:
    """Mean Reciprocal Rank"""
    for idx, doc in enumerate(docs, start=1):
        if is_relevant_doc(doc, expected_article, question):
            return 1.0 / idx
    return 0.0


def calculate_context_recall(
    docs: List,
    expected_answer: str
) -> float:
    """
    Context Recall:
    Tỷ lệ từ khóa quan trọng trong expected_answer
    xuất hiện trong tập retrieved documents
    """

    if not expected_answer:
        return 1.0

    gt_words = {
        w for w in normalize(expected_answer).split()
        if len(w) > 3
    }

    if not gt_words:
        return 1.0

    found = set()

    for doc in docs:
        if hasattr(doc, "page_content"):
            content_words = set(
                normalize(doc.page_content).split()
            )
            found |= gt_words & content_words

    return len(found) / len(gt_words)


# =========================
# EVALUATION LOOP
# =========================

def evaluate_retrieval(
    retrieval_service: RetrievalService,
    test_data: List[Dict]
) -> Dict[str, Any]:

    print("=" * 70)
    print("🔍 RETRIEVAL EVALUATION – CORE LEGAL QA")
    print("=" * 70)

    total_hit = 0
    total_mrr = 0.0
    total_recall = 0.0
    results = []

    for i, item in enumerate(test_data, 1):
        question = item["question"]
        expected_article = item.get("expected_article", "")
        expected_answer = item.get("expected_answer", "")

        print(f"\n[{i}/{len(test_data)}] {question}")

        try:
            docs = retrieval_service.retrieve(question, k=5)

            hit = calculate_hit_rate(docs, expected_article, question)
            mrr = calculate_mrr(docs, expected_article, question)
            recall = calculate_context_recall(docs, expected_answer)

            total_hit += int(hit)
            total_mrr += mrr
            total_recall += recall

            results.append({
                "question": question,
                "expected_article": expected_article,
                "hit": hit,
                "mrr": mrr,
                "context_recall": recall,
                "top_doc_preview": (
                    docs[0].page_content[:150]
                    if docs and hasattr(docs[0], "page_content")
                    else ""
                )
            })

            print(
                f"  Hit: {'✅' if hit else '❌'} | "
                f"MRR: {mrr:.3f} | "
                f"Recall: {recall:.3f}"
            )

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "question": question,
                "error": str(e)
            })

    n = len(test_data)
    hit_rate = total_hit / n if n else 0
    avg_mrr = total_mrr / n if n else 0
    avg_recall = total_recall / n if n else 0

    print("\n" + "=" * 70)
    print("📊 FINAL METRICS")
    print("=" * 70)
    print(f"✅ Hit Rate@5     : {hit_rate:.2%}")
    print(f"📍 Mean MRR       : {avg_mrr:.3f}")
    print(f"📄 Context Recall : {avg_recall:.2%}")

    # Save result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"evaluation/retrieval_eval_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "metrics": {
                "hit_rate": hit_rate,
                "mrr": avg_mrr,
                "context_recall": avg_recall
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Saved to: {output_path}")
    return results


# =========================
# MAIN
# =========================

def main():
    if os.path.basename(os.getcwd()) == "evaluation":
        os.chdir("..")

    test_file = "evaluation/traffic_law_eval_dataset.json"
    if not Path(test_file).exists():
        print(f"❌ File not found: {test_file}")
        return

    with open(test_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # ✅ FILTER BY TYPE
    ALLOWED_TYPES = {"core_legal_qa", "user_realistic_qa"}
    test_data = [
        item for item in raw_data
        if item.get("type") in ALLOWED_TYPES
    ]

    print(f"📂 Loaded {len(test_data)} questions")
    print(f"📌 Types used: {ALLOWED_TYPES}")

    retrieval_service = RetrievalService(
        collection_name="Law",
        es_index_name="law_documents",
        top_k=5,
        use_rerank=True
    )

    evaluate_retrieval(retrieval_service, test_data)


if __name__ == "__main__":
    main()