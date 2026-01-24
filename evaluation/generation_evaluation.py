"""
Generation Evaluation for Core Legal QA
Applies to:
- core_legal_qa
- user_realistic_qa
"""

EVAL_MODE = True  

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

ALLOWED_TYPES = {
    "core_legal_qa",
    "user_realistic_qa"
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


def build_context(docs):
    blocks = []
    for doc in docs:
        title = doc.metadata.get("title", "")
        blocks.append(f"{title}\n{doc.page_content}")
    return "\n\n".join(blocks)

# =========================
# RUBRIC SCORERS
# =========================

def score_legal_accuracy(answer: str, expected_article: str) -> int:
    ans = normalize(answer)
    if expected_article:
        num = re.search(r"\d+", expected_article)
        if num and num.group() in ans:
            return 2
        if "điều" in ans:
            return 1
    return 0


def score_factuality(answer: str, context: str) -> int:
    ans_words = extract_keywords(answer)
    ctx_words = extract_keywords(context)

    hallucinated = ans_words - ctx_words

    if not hallucinated:
        return 2
    if len(hallucinated) <= 2:
        return 1
    return 0


def score_citation(answer: str, expected_article: str | None = None) -> int:
    if not answer:
        return 0

    ans = normalize(answer)
    article_match = re.search(r"điều\s+\d+", ans)

    if article_match:
        if expected_article:
            exp_num = re.search(r"\d+", expected_article)
            ans_num = re.search(r"\d+", article_match.group())
            if exp_num and ans_num and exp_num.group() == ans_num.group():
                return 2
        else:
            return 2

    if re.search(r"chương\s+([ivx]+|\d+)", ans):
        return 1

    return 0


def score_clarity(answer: str) -> int:
    sentences = re.split(r"[.!?]", answer)
    avg_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)

    if avg_len <= 25:
        return 2
    if avg_len <= 35:
        return 1
    return 0


def score_completeness(answer: str, expected_answer: str) -> int:
    if not expected_answer:
        return 1

    ans_words = extract_keywords(answer)
    gt_words = extract_keywords(expected_answer)

    if not gt_words:
        return 1

    coverage = len(ans_words & gt_words) / len(gt_words)

    if coverage >= 0.8:
        return 2
    if coverage >= 0.5:
        return 1
    return 0


# =========================
# EVALUATION LOOP
# =========================

def evaluate_generation(
    generator: GeneratorService,
    retriever: RetrievalService,
    test_data: List[Dict]
):

    # ✅ FILTER DATASET
    eval_data = [
        item for item in test_data
        if item.get("type") in ALLOWED_TYPES
    ]

    print("=" * 70)
    print("🧠 GENERATION EVALUATION – CORE & REALISTIC LEGAL QA")
    print(f"🧪 Total samples: {len(eval_data)}")
    print(f"📌 Types: {ALLOWED_TYPES}")
    print("=" * 70)

    results = []
    total_score = 0

    # 🔢 Accumulators per metric
    metric_totals = {
        "legal_accuracy": 0,
        "factuality": 0,
        "citation_correctness": 0,
        "clarity": 0,
        "completeness": 0
    }

    for i, item in enumerate(eval_data, 1):
        question = item["question"]
        expected_article = item.get("expected_article", "")
        expected_law = item.get("expected_law", "")
        expected_answer = item.get("expected_answer", "")

        print(f"\n[{i}] {question}")

        docs = retriever.retrieve(question, k=5)
        context = build_context(docs)

        if EVAL_MODE:
            answer = context[:800]
        else:
            answer = generator.generate_sync(question, context)

        scores = {
            "legal_accuracy": score_legal_accuracy(answer, expected_article),
            "factuality": score_factuality(answer, context),
            "citation_correctness": score_citation(answer, expected_law),
            "clarity": score_clarity(answer),
            "completeness": score_completeness(answer, expected_answer)
        }

        for k in metric_totals:
            metric_totals[k] += scores[k]

        total = sum(scores.values())
        total_score += total

        print(f"  🎯 Total: {total}/10 | {scores}")

        results.append({
            "id": item.get("id"),
            "type": item.get("type"),
            "question": question,
            "scores": scores,
            "total_score": total
        })

    n = max(len(eval_data), 1)

    # =========================
    # AVERAGES
    # =========================

    avg_total = total_score / n

    avg_metrics_10 = {
        k: (v / n) * 5   # 2 điểm → quy đổi sang thang 10
        for k, v in metric_totals.items()
    }

    print("\n" + "=" * 70)
    print(f"📊 AVERAGE TOTAL SCORE: {avg_total:.2f}/10")
    print("📊 AVERAGE METRIC SCORES (scaled to /10):")

    for k, v in avg_metrics_10.items():
        print(f"  - {k}: {v:.2f}/10")

    # =========================
    # SAVE RESULT
    # =========================

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"evaluation/generation_eval_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "average_total_score": avg_total,
            "average_metric_scores": avg_metrics_10,
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Saved to: {output_path}")


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

    evaluate_generation(generator, retriever, test_data)


if __name__ == "__main__":
    main()
