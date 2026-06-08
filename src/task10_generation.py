"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
import re
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve
from .rag_utils import text_similarity


# =============================================================================
# RELEVANCE GATE — tránh trả lời câu ngoài phạm vi tri thức
# =============================================================================

# Nếu độ tương đồng giữa query và chunk liên quan nhất < ngưỡng này ⇒ coi như
# không có evidence → trả về "không thể xác minh" (yêu cầu Task 10).
# Đo thực nghiệm: câu in-scope sim >= 0.41, câu out-of-scope sim <= 0.28 → chọn 0.35.
RELEVANCE_THRESHOLD = 0.35

# Lưới an toàn cho câu ngắn nhiều từ đệm ("Còn vụ X thì sao?"): nếu MỌI từ khóa
# nội dung của câu hỏi đều xuất hiện trong nguồn, cho qua dù sim hơi thấp.
SOFT_SIM_FLOOR = 0.30

# Từ đệm/từ dừng không tính là "từ khóa nội dung".
_GATE_STOPWORDS = {
    "còn", "vụ", "thì", "sao", "là", "gì", "nào", "của", "và", "có", "các",
    "này", "đó", "về", "cho", "ai", "bao", "nhiêu", "như", "thế", "được",
    "vì", "lý", "do", "đã", "sẽ", "khi", "tại", "theo", "một", "những",
}

CANNOT_VERIFY = "Tôi không thể xác minh thông tin này từ nguồn hiện có."


def _content_coverage(query: str, chunks: list[dict]) -> float:
    """Tỉ lệ từ khóa nội dung của câu hỏi xuất hiện trong các chunk truy hồi."""
    from .rag_utils import tokenize

    keywords = {t for t in tokenize(query) if len(t) > 1 and t not in _GATE_STOPWORDS}
    if not keywords:
        return 0.0
    corpus = set()
    for chunk in chunks:
        corpus |= set(tokenize(chunk.get("content", "")))
    return sum(1 for kw in keywords if kw in corpus) / len(keywords)


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    front = [chunks[i] for i in range(0, len(chunks), 2)]
    back = [chunks[i] for i in range(1, len(chunks), 2)]
    return front + list(reversed(back))


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"Source {i}")
        doc_type = metadata.get("type", "unknown")
        score = chunk.get("score", 0.0)
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type} | Score: {score:.3f}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


def _citation(metadata: dict) -> str:
    source = metadata.get("source", "Nguồn không rõ")
    year_match = re.search(r"(20\d{2}|19\d{2})", source)
    if year_match:
        label = f"{source}, {year_match.group(1)}"
    else:
        doc_type = metadata.get("type", "nguồn")
        label = f"{source}, {doc_type}"
    return f"[{label}]"


# Dấu hiệu đoạn biểu mẫu pháp lý cần loại: chuỗi chấm/ellipsis, ô trống, gạch dưới.
_FORM_NOISE = re.compile(r"[.…]{4,}|□|_{4,}")


def _is_meaningful(sentence: str) -> bool:
    """Loại câu rác: heading, biểu mẫu nhiều dấu chấm/ô trống/gạch dưới."""
    if sentence.startswith("#") or _FORM_NOISE.search(sentence):
        return False
    letters = sum(ch.isalpha() for ch in sentence)
    return len(sentence) >= 40 and letters / max(1, len(sentence)) > 0.55


def _snippet(text: str, max_sentences: int = 2, max_chars: int = 300) -> str:
    """Lấy tối đa `max_sentences` câu có nghĩa đầu tiên làm trích đoạn trả lời."""
    text = re.sub(r"^#.*$", "", text, flags=re.MULTILINE).strip()
    sentences = re.split(r"(?<=[.!?。])\s+", text)
    picked: list[str] = []
    for sentence in sentences:
        cleaned = re.sub(r"\s+", " ", sentence).strip()
        if _is_meaningful(cleaned):
            picked.append(cleaned)
        if len(picked) >= max_sentences:
            break
    if picked:
        return " ".join(picked)[:max_chars].strip()
    # Không có câu nào "sạch": chỉ trả về nguyên văn nếu nó không phải biểu mẫu rác.
    fallback = re.sub(r"\s+", " ", text).strip()
    return "" if _FORM_NOISE.search(fallback) else fallback[:max_chars]


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K, gate_query: str | None = None) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi dùng để retrieve (có thể đã được rewrite theo ngữ cảnh)
        gate_query: Câu hỏi gốc của user dùng để chấm relevance gate. Khi câu đã
            được rewrite dài dòng, chấm theo câu gốc ngắn gọn để tránh chặn nhầm.
            Mặc định = query.

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    chunks = retrieve(query, top_k=top_k)
    gate_q = gate_query or query

    # Relevance gate: nếu không chunk nào đủ liên quan tới câu hỏi → không bịa đáp án.
    # Chấm theo câu gốc (gate_q) để câu rewrite dài dòng không làm loãng tín hiệu.
    best_relevance = max(
        (text_similarity(gate_q, c.get("content", "")) for c in chunks),
        default=0.0,
    )
    coverage = _content_coverage(gate_q, chunks)
    # Đủ liên quan khi: sim qua ngưỡng, HOẶC sim sàn mềm + phủ trọn từ khóa nội dung.
    relevant = best_relevance >= RELEVANCE_THRESHOLD or (
        best_relevance >= SOFT_SIM_FLOOR and coverage >= 0.999
    )
    if not chunks or not relevant:
        return {"answer": CANNOT_VERIFY, "sources": [], "retrieval_source": "none"}

    reordered = reorder_for_llm(chunks)
    format_context(reordered)

    lines = []
    for chunk in reordered[: min(3, len(reordered))]:
        sentence = _snippet(chunk.get("content", ""))
        if not sentence:
            continue
        citation = _citation(chunk.get("metadata", {}))
        lines.append(f"{sentence} {citation}")
    answer = "\n\n".join(lines) if lines else CANNOT_VERIFY

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
