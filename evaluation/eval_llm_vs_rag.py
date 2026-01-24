"""
Compare LLM-only vs RAG for Core Legal QA
Metrics:
- Article Exact Match
- Answer Recall
- Hallucination Rate
"""

import json
import os
import sys
import re
from pathlib import Path
from typing import List, Dict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.services.rag import RAGService
from src.models.llm import chat_model 


# =========================
# UTILS
# =========================

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_article_numbers(text: str) -> List[str]:
    """Extract Điều X"""
    return re.findall(r"điều\s*(\d+)", text.lower())


def token_overlap(a: str, b: str) -> float:
    a_set = {w for w in normalize(a).split() if len(w) > 3}
    b_set = {w for w in normalize(b).split() if len(w) > 3}
    if not a_set:
        return 0.0
    return len(a_set & b_set) / len(a_set)


# =========================
# ANSWER FUNCTIONS
# =========================

def llm_only_answer(question: str) -> str:
    """LLM-only (NO RETRIEVAL)"""
    response = chat_model.invoke(question)
    return response.content if hasattr(response, "content") else str(response)


def rag_answer(rag: RAGService, question: str) -> str:
    """RAG Answer"""
    return rag.answer(question)


# =========================
# EVALUATION
# =========================

def evaluate_system(
    name: str,
    answer_fn,
    dataset: List[Dict]
) -> Dict:

    print(f"\n================ {name.upper()} =================")

    exact_match = 0
    recall_sum = 0.0
    hallucination = 0
    results = []

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        expected_article = item.get("expected_article", "")
        expected_answer = item.get("expected_answer", "")

        answer = answer_fn(question)

        pred_articles = extract_article_numbers(answer)
        gt_articles = extract_article_numbers(expected_article)

        # --- Exact Match (Điều luật) ---
        em = any(a in pred_articles for a in gt_articles)
        exact_match += int(em)

        # --- Answer Recall ---
        recall = token_overlap(expected_answer, answer)
        recall_sum += recall

        # --- Hallucination ---
        hallucinated = any(
            a not in gt_articles
            for a in pred_articles
        )
        hallucination += int(hallucinated)

        results.append({
            "question": question,
            "answer": answer,
            "exact_match": em,
            "recall": recall,
            "hallucination": hallucinated
        })

        print(
            f"[{i}/{len(dataset)}] "
            f"EM: {'✅' if em else '❌'} | "
            f"Recall: {recall:.2f} | "
            f"Hallucination: {'⚠️' if hallucinated else 'OK'}"
        )

    n = len(dataset)

    return {
        "system": name,
        "metrics": {
            "exact_match_rate": exact_match / n,
            "avg_recall": recall_sum / n,
            "hallucination_rate": hallucination / n
        },
        "results": results
    }


# =========================
# MAIN
# =========================

def main():
    test_file = "evaluation/traffic_law_eval_dataset.json"
    if not Path(test_file).exists():
        print("❌ Dataset not found")
        return

    with open(test_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"📂 Loaded {len(dataset)} Core Legal QA questions")

    # Init RAG
    rag = RAGService()

    # Evaluate
    llm_result = evaluate_system(
        "LLM-only",
        llm_only_answer,
        dataset
    )

    rag_result = evaluate_system(
        "RAG",
        lambda q: rag_answer(rag, q),
        dataset
    )

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"evaluation/llm_vs_rag_{timestamp}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "llm_only": llm_result,
            "rag": rag_result
        }, f, ensure_ascii=False, indent=2)

    print("\n================ FINAL COMPARISON ================")
    print(f"LLM-only Exact Match   : {llm_result['metrics']['exact_match_rate']:.2%}")
    print(f"RAG Exact Match        : {rag_result['metrics']['exact_match_rate']:.2%}")
    print(f"LLM-only Hallucination : {llm_result['metrics']['hallucination_rate']:.2%}")
    print(f"RAG Hallucination      : {rag_result['metrics']['hallucination_rate']:.2%}")
    print(f"\n💾 Saved to: {out_path}")


if __name__ == "__main__":
    main()
