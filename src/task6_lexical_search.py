"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path
import math
from collections import Counter

from .rag_utils import ensure_chunks, tokenize

CORPUS: list[dict] = []


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized = [tokenize(doc["content"]) for doc in corpus]
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))

    avg_len = sum(len(tokens) for tokens in tokenized) / max(1, len(tokenized))
    return {"tokenized": tokenized, "doc_freq": doc_freq, "avg_len": avg_len, "n_docs": len(corpus)}


def _bm25_score(query_tokens: list[str], doc_tokens: list[str], index: dict) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0

    k1 = 1.5
    b = 0.75
    n_docs = index["n_docs"]
    avg_len = index["avg_len"]
    freqs = Counter(doc_tokens)
    score = 0.0

    for token in query_tokens:
        df = index["doc_freq"].get(token, 0)
        if df == 0:
            continue
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        tf = freqs[token]
        denom = tf + k1 * (1 - b + b * len(doc_tokens) / max(1, avg_len))
        score += idf * (tf * (k1 + 1)) / denom

    return score


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    corpus = CORPUS or ensure_chunks()
    index = build_bm25_index(corpus)
    query_tokens = tokenize(query)

    results = []
    for idx, doc_tokens in enumerate(index["tokenized"]):
        score = _bm25_score(query_tokens, doc_tokens, index)
        if score > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": float(score),
                "metadata": corpus[idx].get("metadata", {}),
            })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
