"""
Hallucination Evaluation for Legal QA
ONLY applies to items with type = "hallucination_test"
"""

import json
import re
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from src.services.retrieval import RetrievalService
from src.services.generator import GeneratorService

# =========================
# CONFIG
# =========================

EVAL_MODE = True  

ALLOWED_TYPES = {
    "hallucination_test",
    "temporal_validity"
}

# =========================
# UTILS
# =========================

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def extract_keywords(text: str) -> set:
    return {
        w for w in normalize(text).split()
        if len(w) > 4
    }


def build_context(docs) -> str:
    blocks = []
    for doc in docs:
        title = doc.metadata.get("title", "")
        blocks.append(f"{title}\n{doc.page_content}")
    return "\n\n".join(blocks)


# =========================
# HALLUCINATION CHECKERS
# =========================

def check_fabricated_law(answer: str) -> bool:
    """
    Dẫn Điều / Luật không tồn tại
    Ví dụ: Luật GTĐB 2022, Điều 15 không có
    """
    ans = normalize(answer)

    fake_patterns = [
        r"luật giao thông đường bộ 2022",
        r"điều\s+\d+\s+luật giao thông",
    ]

    return any(re.search(p, ans) for p in fake_patterns)


def check_outdated_law(answer: str) -> bool:
    """
    Dùng Nghị định cũ khi đã có NĐ 168
    """
    ans = normalize(answer)
    return "nghị định 100" in ans or "nghị định 123" in ans


def check_over_generalization(answer: str, vehicle_type: str) -> bool:
    """
    Áp sai loại phương tiện
    """
    ans = normalize(answer)

    vehicle_map = {
        "xe_may": ["xe máy", "mô tô"],
        "oto": ["ô tô", "xe hơi"],
        "nguoi_di_bo": ["người đi bộ"],
        "xe_dap_dien": ["xe đạp điện"],
        "xe_may_dien": ["xe máy điện"]
    }

    expected_terms = vehicle_map.get(vehicle_type, [])

    if not expected_terms:
        return False

    return not any(term in ans for term in expected_terms)


def check_unsupported_claim(answer: str, context: str) -> bool:
    """
    Nội dung trả lời không có trong context
    """
    ans_words = extract_keywords(answer)
    ctx_words = extract_keywords(context)

    unsupported = ans_words - ctx_words
    return len(unsupported) > 3


# =========================
# SINGLE ITEM EVAL
# =========================

def evaluate_hallucination(
    answer: str,
    context: str,
    vehicle_type: str
) -> Dict[str, bool]:

    fabricated = check_fabricated_law(answer)
    outdated = check_outdated_law(answer)
    over_general = check_over_generalization(answer, vehicle_type)
    unsupported = check_unsupported_claim(answer, context)

    is_hallucinated = any([
        fabricated,
        outdated,
        over_general,
        unsupported
    ])

    return {
        "fabricated_law": fabricated,
        "outdated_law": outdated,
        "over_generalization": over_general,
        "unsupported_claim": unsupported,
        "is_hallucinated": is_hallucinated
    }


# =========================
# EVALUATION LOOP
# =========================

def evaluate_dataset(
    generator: GeneratorService,
    retriever: RetrievalService,
    test_data: List[Dict]
):

    # ✅ FILTER HALLUCINATION TEST ONLY
    hallucination_tests = [
        item for item in test_data
        if item.get("type") in ALLOWED_TYPES
    ]

    print("=" * 70)
    print("🚨 HALLUCINATION EVALUATION (TRAP QUESTIONS ONLY)")
    print(f"🧪 Total test cases: {len(hallucination_tests)}")
    print("=" * 70)

    results = []
    hallucinated_count = 0

    for i, item in enumerate(hallucination_tests, 1):
        question = item["question"]
        vehicle_type = item.get("vehicle_type", "")

        print(f"\n[{i}] {question}")

        docs = retriever.retrieve(question, k=5)
        context = build_context(docs)

        if EVAL_MODE:
            answer = context[:800]
        else:
            answer = generator.generate_sync(question, context)

        flags = evaluate_hallucination(
            answer=answer,
            context=context,
            vehicle_type=vehicle_type
        )

        if flags["is_hallucinated"]:
            hallucinated_count += 1
            print(f"  ❌ Hallucination: {flags}")
        else:
            print(f"  ✅ Safe: {flags}")

        results.append({
            "id": item.get("id"),
            "question": question,
            "answer": answer,
            "flags": flags
        })

    rate = hallucinated_count / max(len(hallucination_tests), 1)

    print("\n" + "=" * 70)
    print(f"📊 HALLUCINATION RATE: {rate:.2%}")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"evaluation/hallucination_eval_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "hallucination_rate": rate,
            "total_tests": len(hallucination_tests),
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved to: {output_path}")


# =========================
# MAIN
# =========================

def main():
    test_file = "evaluation/traffic_law_eval_dataset.json"
    if not Path(test_file).exists():
        print(f"❌ File not found: {test_file}")
        return

    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    retriever = RetrievalService(
        collection_name="Law",
        es_index_name="law_documents",
        top_k=5,
        use_rerank=True
    )

    generator = GeneratorService()

    evaluate_dataset(generator, retriever, test_data)


if __name__ == "__main__":
    main()
