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

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime for missing optional dep
    OpenAI = None

from .task9_retrieval_pipeline import retrieve
from .rag_utils import text_similarity, tokenize


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

_ANSWER_CITATION_RE = re.compile(r"\[[^\]]+\]")
_SUPPORT_STOPWORDS = {
    "la", "gi", "cua", "va", "co", "cac", "nhung", "mot", "nay", "do", "ve",
    "cho", "theo", "trong", "ngoai", "ra", "thi", "duoc", "bi", "voi", "tu",
    "den", "de", "khi", "neu", "khong", "dieu", "viec", "nay", "do", "qua",
}


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
# Chọn 0.0 vì: RAG cần bám nguồn, không sáng tạo thêm ngoài context.
TEMPERATURE = 0.0

# Model gọi cho bước generation. Có thể override bằng OPENAI_MODEL trong .env.
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a strict extractive RAG answerer.
Answer in Vietnamese using ONLY the retrieved CONTEXT.

Grounding rules:
- Quote or paraphrase only statements that are explicitly present in CONTEXT.
- Do not infer, generalize, or use outside knowledge.
- Do not add interpretations, lessons, impacts, causes, or audiences unless the
  same idea is explicitly stated in CONTEXT.
- If CONTEXT only partially answers the question, answer only that partial part.
- If CONTEXT does not explicitly support an answer, say exactly:
  'Tôi không thể xác minh thông tin này từ nguồn hiện có.'
- Every factual sentence MUST cite the exact source filename shown as
  "Citation to use" in the same document block.
- Do NOT cite generic labels like [Document 1] or [Source 1].
- Do NOT attach a citation from one document to a claim from another document.

Output rules:
- Be concise.
- Prefer 2-4 short bullet points.
- No introduction, no conclusion, no extra commentary."""


def _call_llm(context: str, query: str) -> str:
    """Generate the final answer from retrieved context using OpenAI."""
    if OpenAI is None:
        raise RuntimeError("openai package is not installed")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")

    client = OpenAI()
    user_prompt = (
        "CONTEXT:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{query}\n\n"
        "Trả lời bằng tiếng Việt. Chỉ dùng nội dung được nói trực tiếp trong "
        "CONTEXT. Không suy luận thêm. Mỗi bullet phải bám một đoạn trong "
        "CONTEXT và cite đúng filename ở dòng 'Citation to use'."
    )
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", LLM_MODEL),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    answer = response.choices[0].message.content
    return (answer or "").strip()


def _support_coverage(claim: str, content: str) -> float:
    claim_terms = {
        token for token in tokenize(_ANSWER_CITATION_RE.sub("", claim))
        if len(token) > 1 and token not in _SUPPORT_STOPWORDS
    }
    if not claim_terms:
        return 0.0
    content_terms = set(tokenize(content))
    return len(claim_terms & content_terms) / len(claim_terms)


def _best_supported_source(claim: str, chunks: list[dict]) -> tuple[str | None, float, float]:
    best_source = None
    best_coverage = 0.0
    best_score = 0.0
    for chunk in chunks:
        content = chunk.get("content", "")
        coverage = _support_coverage(claim, content)
        similarity = text_similarity(claim, content)
        score = 0.7 * coverage + 0.3 * similarity
        if score > best_score:
            best_score = score
            best_coverage = coverage
            best_source = chunk.get("metadata", {}).get("source")
    return best_source, best_coverage, best_score


def _ground_llm_answer(answer: str, chunks: list[dict]) -> str:
    """Keep only claims supported by retrieved chunks and repair citations."""
    if CANNOT_VERIFY in answer:
        return CANNOT_VERIFY

    grounded_lines = []
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        source, coverage, score = _best_supported_source(line, chunks)
        if not source or (coverage < 0.45 and score < 0.35):
            continue

        clean_line = _ANSWER_CITATION_RE.sub("", line).rstrip(" .")
        grounded_lines.append(f"{clean_line}. [{source}]")

    return "\n".join(grounded_lines) if grounded_lines else CANNOT_VERIFY


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
            f"Citation to use: [{source}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


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
    context = format_context(reordered)

    try:
        answer = _call_llm(context, query)
    except Exception as exc:
        answer = (
            "Không thể gọi LLM để sinh câu trả lời RAG. "
            f"Lý do: {exc}. "
            "Các nguồn liên quan vẫn được trả về bên dưới."
        )
    if not answer:
        answer = CANNOT_VERIFY
    elif answer != CANNOT_VERIFY:
        answer = _ground_llm_answer(answer, chunks)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        "llm_model": os.getenv("OPENAI_MODEL", LLM_MODEL),
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
