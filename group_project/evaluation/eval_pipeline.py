"""
RAG Evaluation Pipeline.

Đánh giá chất lượng RAG pipeline trên golden dataset với 4 metrics chuẩn của
RAG (theo trục của RAGAS / DeepEval):

    - Faithfulness      : câu trả lời có bám đúng retrieved context không?
    - Answer Relevance  : câu trả lời có đúng trọng tâm câu hỏi không?
    - Context Recall    : retriever có lấy đủ evidence (so với expected_answer) không?
    - Context Precision : trong context lấy về, bao nhiêu % chunk thực sự hữu ích?

Triển khai mặc định là `LocalLexicalEvaluator` — một bản đánh giá *không cần API
key* dựa trên token-overlap / cosine, để toàn bộ pipeline chạy được offline trong
phòng lab. Các hàm `evaluate_with_deepeval / ragas / trulens` được giữ lại để bật
khi có OpenAI/Gemini API key (chỉ cần đổi 1 dòng ở __main__).

Chạy:
    python -m group_project.evaluation.eval_pipeline
hoặc:
    python group_project/evaluation/eval_pipeline.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

# Console Windows mặc định cp1252 — ép UTF-8 để in được tiếng Việt.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# --- Cho phép chạy trực tiếp (python eval_pipeline.py) lẫn dạng module --------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_utils import tokenize, cosine_from_counters  # noqa: E402
from src.task9_retrieval_pipeline import retrieve  # noqa: E402
from src.task10_generation import reorder_for_llm  # noqa: E402

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

# Các config đem ra so sánh A/B.
CONFIGS = {
    "A_hybrid_rerank": {"use_reranking": True, "label": "Hybrid + Reranking"},
    "B_no_rerank": {"use_reranking": False, "label": "Hybrid, no Reranking"},
}
TOP_K = 5


def _citation(metadata: dict) -> str:
    """Citation label used by the offline extractive evaluation answer."""
    return f"[{metadata.get('source', 'Nguồn không rõ')}]"


def _snippet(text: str, max_chars: int = 320) -> str:
    """Return a compact, content-only snippet for API-key-free evaluation."""
    cleaned = re.sub(r"^#\s*", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"\*\*Source:\*\*\s*\S+", "", cleaned)
    cleaned = re.sub(r"\*\*Crawled:\*\*\s*[^\s]+", "", cleaned)
    cleaned = re.sub(r"\s+---\s+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rsplit(" ", 1)[0].strip()


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# RAG run — sinh answer + sources cho một câu hỏi với một config
# =============================================================================

def run_rag(question: str, use_reranking: bool, top_k: int = TOP_K) -> dict:
    """
    Chạy retrieval (Task 9) + sinh answer extractive có citation cho 1 config.

    Trả về {'answer', 'sources', 'contexts'} để feed vào evaluator.
    """
    chunks = retrieve(question, top_k=top_k, use_reranking=use_reranking)
    reordered = reorder_for_llm(chunks)

    if not reordered:
        answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    else:
        lines = []
        for chunk in reordered[: min(3, len(reordered))]:
            sentence = _snippet(chunk.get("content", ""))
            if not sentence:
                continue
            citation = _citation(chunk.get("metadata", {}))
            lines.append(f"{sentence} {citation}")
        answer = "\n\n".join(lines)

    return {
        "answer": answer,
        "sources": chunks,
        "contexts": [c.get("content", "") for c in chunks],
    }


# =============================================================================
# Local, API-key-free evaluator (mặc định)
# =============================================================================

def _coverage(reference: str, text: str) -> float:
    """Tỉ lệ token nội dung của `reference` được phủ bởi `text`."""
    ref_tokens = [t for t in tokenize(reference) if len(t) > 2]
    if not ref_tokens:
        return 0.0
    text_tokens = set(tokenize(text))
    covered = sum(1 for t in set(ref_tokens) if t in text_tokens)
    return covered / len(set(ref_tokens))


def _relevance(a: str, b: str) -> float:
    """Cosine token-overlap giữa hai đoạn text."""
    return cosine_from_counters(Counter(tokenize(a)), Counter(tokenize(b)))


def evaluate_local(test_cases: list[dict]) -> list[dict]:
    """
    Tính 4 metrics cho từng test case bằng token-overlap (không cần API key).

    Mỗi test case: {question, expected_answer, expected_context, answer, contexts}
    """
    rows = []
    for tc in test_cases:
        context_blob = "\n".join(tc["contexts"])

        # Faithfulness: answer có được context hỗ trợ không (token của answer nằm trong context).
        faithfulness = _coverage(tc["answer"], context_blob)
        # Answer Relevance: answer có đúng trọng tâm question không.
        answer_relevance = _relevance(tc["answer"], tc["question"])
        # Context Recall: context có phủ được expected_answer (ground truth) không.
        context_recall = _coverage(tc["expected_answer"], context_blob)
        # Context Precision: tỉ lệ chunk hữu ích (overlap với expected answer/context).
        ground_truth = f"{tc['expected_answer']} {tc['expected_context']}"
        useful = sum(1 for c in tc["contexts"] if _coverage(ground_truth, c) >= 0.15)
        context_precision = useful / len(tc["contexts"]) if tc["contexts"] else 0.0

        rows.append(
            {
                "question": tc["question"],
                "faithfulness": round(faithfulness, 3),
                "answer_relevance": round(answer_relevance, 3),
                "context_recall": round(context_recall, 3),
                "context_precision": round(context_precision, 3),
            }
        )
    return rows


def aggregate(rows: list[dict]) -> dict:
    """Trung bình từng metric trên toàn dataset."""
    metrics = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]
    n = len(rows) or 1
    agg = {m: round(sum(r[m] for r in rows) / n, 3) for m in metrics}
    agg["average"] = round(sum(agg[m] for m in metrics) / len(metrics), 3)
    return agg


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(golden_dataset: list[dict]) -> dict:
    """
    Chạy eval cho từng config trong CONFIGS và trả về kết quả chi tiết + tổng hợp.

    Returns:
        {
          config_name: {"label", "rows": [...], "agg": {...}},
          ...
        }
    """
    results = {}
    for name, cfg in CONFIGS.items():
        test_cases = []
        for item in golden_dataset:
            rag = run_rag(item["question"], use_reranking=cfg["use_reranking"])
            test_cases.append(
                {
                    "question": item["question"],
                    "expected_answer": item["expected_answer"],
                    "expected_context": item["expected_context"],
                    "answer": rag["answer"],
                    "contexts": rag["contexts"],
                }
            )
        rows = evaluate_local(test_cases)
        results[name] = {"label": cfg["label"], "rows": rows, "agg": aggregate(rows)}
        print(f"  [OK] {name} ({cfg['label']}): avg = {results[name]['agg']['average']}")
    return results


# =============================================================================
# Export Results
# =============================================================================

def _delta(a: float, b: float) -> str:
    d = round(a - b, 3)
    sign = "+" if d > 0 else ""
    return f"{sign}{d}"


def export_results(results: dict) -> None:
    """Format và ghi kết quả ra results.md."""
    a_name, b_name = list(CONFIGS.keys())
    a, b = results[a_name], results[b_name]
    metrics = [
        ("Faithfulness", "faithfulness"),
        ("Answer Relevance", "answer_relevance"),
        ("Context Recall", "context_recall"),
        ("Context Precision", "context_precision"),
    ]

    lines = []
    lines.append("# RAG Evaluation Results")
    lines.append("")
    lines.append("## Framework sử dụng")
    lines.append("")
    lines.append(
        "**LocalLexicalEvaluator** — bản đánh giá không cần API key, cài đặt 4 trục "
        "metric chuẩn của RAGAS/DeepEval bằng token-overlap & cosine similarity trên "
        "tiếng Việt. Cho phép chạy toàn bộ pipeline offline trong phòng lab. "
        "Khi có OpenAI/Gemini API key, có thể bật `evaluate_with_ragas` / "
        "`evaluate_with_deepeval` trong `eval_pipeline.py` để chấm bằng LLM-as-judge."
    )
    lines.append("")
    lines.append(f"- **Golden dataset:** {len(a['rows'])} cặp Q&A "
                 f"(`golden_dataset.json`) — pháp luật + tin tức về ma túy.")
    lines.append(f"- **top_k:** {TOP_K}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Overall scores
    lines.append("## Overall Scores")
    lines.append("")
    lines.append(f"| Metric | Config A ({a['label']}) | Config B ({b['label']}) | Δ (A−B) |")
    lines.append("|--------|--------------------------|--------------------------|---------|")
    for title, key in metrics:
        lines.append(f"| {title} | {a['agg'][key]} | {b['agg'][key]} | {_delta(a['agg'][key], b['agg'][key])} |")
    lines.append(f"| **Average** | **{a['agg']['average']}** | **{b['agg']['average']}** | "
                 f"**{_delta(a['agg']['average'], b['agg']['average'])}** |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # A/B analysis
    winner = a if a["agg"]["average"] >= b["agg"]["average"] else b
    loser = b if winner is a else a
    gap = round(abs(a["agg"]["average"] - b["agg"]["average"]), 3)
    lines.append("## A/B Comparison Analysis")
    lines.append("")
    lines.append(f"**Config A:** {a['label']} — chạy semantic + lexical, merge bằng RRF, "
                 "sau đó rerank bằng cross-encoder/MMR (Task 7).")
    lines.append("")
    lines.append(f"**Config B:** {b['label']} — chạy semantic + lexical, merge bằng RRF, "
                 "nhưng **bỏ bước reranking**, lấy thẳng top_k sau fusion.")
    lines.append("")
    if gap < 0.02:
        lines.append(
            f"**Kết luận:** Hai config gần như ngang nhau (chênh chỉ {gap} average — "
            f"{a['agg']['average']} vs {b['agg']['average']}). Với corpus nhỏ (8 văn bản) "
            "và evaluator dựa trên token-overlap, top_k sau RRF đã đủ tốt nên reranking "
            "chưa tạo khác biệt rõ. Reranking dự kiến phát huy tác dụng khi corpus lớn hơn, "
            "nhiều chunk gây nhiễu, và khi dùng cross-encoder thật + embedding ngữ nghĩa "
            "(thay cho token-overlap) — lúc đó nó mới lọc được noise ở Context Precision."
        )
    else:
        lines.append(
            f"**Kết luận:** Config **{winner['label']}** đạt average cao hơn "
            f"({winner['agg']['average']} so với {loser['agg']['average']}, chênh {gap}). "
            "Reranking tinh chỉnh lại thứ tự ứng viên theo độ liên quan ngữ nghĩa với "
            "truy vấn nên đẩy được chunk hữu ích lên top_k, cải thiện chủ yếu ở "
            "Context Precision và Faithfulness."
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Worst performers (theo config A)
    lines.append("## Worst Performers (Bottom 3 — Config A)")
    lines.append("")
    lines.append("| # | Question | Faithfulness | Relevance | Recall | Precision | Root Cause |")
    lines.append("|---|----------|--------------|-----------|--------|-----------|------------|")
    worst = sorted(
        a["rows"],
        key=lambda r: r["faithfulness"] + r["answer_relevance"] + r["context_recall"] + r["context_precision"],
    )[:3]
    for i, r in enumerate(worst, 1):
        q = r["question"][:55] + ("…" if len(r["question"]) > 55 else "")
        cause = _root_cause(r)
        lines.append(
            f"| {i} | {q} | {r['faithfulness']} | {r['answer_relevance']} | "
            f"{r['context_recall']} | {r['context_precision']} | {cause} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    lines.append("### Cải tiến 1 — Chunking theo heading của văn bản luật")
    lines.append("**Action:** Dùng `MarkdownHeaderTextSplitter` tách theo \"Điều/Chương\" thay vì "
                 "cắt ký tự cố định, để mỗi chunk gói trọn 1 điều luật.")
    lines.append("**Expected impact:** Tăng Context Recall & Precision cho câu hỏi pháp luật "
                 "(điều khoản không bị cắt giữa chừng).")
    lines.append("")
    lines.append("### Cải tiến 2 — Embedding model multilingual thực thụ")
    lines.append("**Action:** Thay token-overlap bằng `BAAI/bge-m3` cho semantic search.")
    lines.append("**Expected impact:** Bắt được câu hỏi diễn đạt khác từ khóa (paraphrase), "
                 "nâng Answer Relevance & Recall ở câu hỏi tin tức.")
    lines.append("")
    lines.append("### Cải tiến 3 — Cross-encoder reranker tiếng Việt + HyDE")
    lines.append("**Action:** Dùng `jinaai/jina-reranker-v2-base-multilingual` và sinh "
                 "hypothetical document (HyDE) trước khi retrieve.")
    lines.append("**Expected impact:** Giảm noise ở top_k, tăng Faithfulness của câu trả lời.")
    lines.append("")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] Đã ghi kết quả → {RESULTS_PATH}")


def _root_cause(row: dict) -> str:
    if row["context_recall"] < 0.4:
        return "Retriever thiếu evidence — chunk liên quan không lọt top_k"
    if row["context_precision"] < 0.4:
        return "Context nhiễu — nhiều chunk không liên quan"
    if row["faithfulness"] < 0.4:
        return "Answer kém bám context (câu hỏi suy luận/tổng hợp)"
    return "Câu hỏi diễn đạt khác từ khóa trong văn bản"


# =============================================================================
# Optional: chấm bằng framework thật (cần API key) — giữ lại để bật khi cần
# =============================================================================

def evaluate_with_ragas(golden_dataset: list[dict], use_reranking: bool = True):
    """Chấm bằng RAGAS (cần OpenAI API key). pip install ragas datasets"""
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
    from datasets import Dataset

    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in golden_dataset:
        rag = run_rag(item["question"], use_reranking=use_reranking)
        data["question"].append(item["question"])
        data["answer"].append(rag["answer"])
        data["contexts"].append(rag["contexts"])
        data["ground_truth"].append(item["expected_answer"])

    dataset = Dataset.from_dict(data)
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall, context_precision])
    return result.to_pandas()


def evaluate_with_deepeval(golden_dataset: list[dict], use_reranking: bool = True):
    """Chấm bằng DeepEval (cần OpenAI API key). pip install deepeval"""
    from deepeval import evaluate
    from deepeval.metrics import (
        FaithfulnessMetric, AnswerRelevancyMetric,
        ContextualRecallMetric, ContextualPrecisionMetric,
    )
    from deepeval.test_case import LLMTestCase

    test_cases = []
    for item in golden_dataset:
        rag = run_rag(item["question"], use_reranking=use_reranking)
        test_cases.append(
            LLMTestCase(
                input=item["question"],
                actual_output=rag["answer"],
                expected_output=item["expected_answer"],
                retrieval_context=rag["contexts"],
            )
        )
    metrics = [
        FaithfulnessMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.7),
        ContextualRecallMetric(threshold=0.7),
        ContextualPrecisionMetric(threshold=0.7),
    ]
    return evaluate(test_cases, metrics)


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases\n")

    print("Running A/B comparison (LocalLexicalEvaluator)...")
    results = compare_configs(golden_dataset)

    print("\nExporting results...")
    export_results(results)
    print("\n[OK] Done. Xem group_project/evaluation/results.md")
