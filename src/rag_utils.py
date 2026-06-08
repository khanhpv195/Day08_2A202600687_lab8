"""Shared local utilities for the Day 8 RAG tasks.

The lab recommends external services for indexing and PageIndex, but the
automated tests need the modules to work in a plain local environment too.
"""

from __future__ import annotations

import html
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"

TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Lowercase Vietnamese text and strip accents for robust local matching."""
    lowered = text.lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def tokenize(text: str) -> list[str]:
    """Word tokenizer with accent-insensitive Vietnamese normalization."""
    return TOKEN_RE.findall(normalize_text(text))


# Marker bắt đầu phần footer/bình luận của trang báo — cắt bỏ từ đây trở đi.
_NEWS_FOOTER_MARKERS = (
    "Tin liên quan",
    "Khám phá thêm chủ đề",
    "Chia sẻ Bình luận",
    "Bình luận (",
    "Quan tâm nhất Mới nhất",
)


def clean_news_markdown(text: str) -> str:
    """Bỏ menu điều hướng + footer của trang báo, chỉ giữ tiêu đề + nội dung bài.

    File crawl từ báo điện tử có một khối menu rất dài ở đầu (Game, Xe, Video,
    Tiêu dùng...) và footer (Tin liên quan, Bình luận...). Phần nội dung thật của
    bài luôn bắt đầu ngay sau mốc thời gian đăng "... GMT+7 Chia sẻ". Ta dùng mốc
    này để cắt bỏ menu, đồng thời giữ lại tiêu đề `#` để phục vụ retrieval/citation.
    """
    # Decode HTML entities còn sót từ lúc crawl (&agrave; &#7891; ...).
    text = html.unescape(text)

    # Tách header (tiêu đề + Source/Crawled) khỏi body qua dấu '---'.
    parts = re.split(r"\n-{3,}\n", text, maxsplit=1)
    header, body = (parts[0], parts[1]) if len(parts) == 2 else ("", text)

    title_match = re.search(r"^#\s*(.+)$", header or text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else ""

    # Cắt menu điều hướng: giữ phần sau mốc "GMT+7" (kèm "Chia sẻ" nếu có).
    gmt = re.search(r"GMT\+7\s*(Chia sẻ)?", body)
    if gmt:
        body = body[gmt.end():]

    # Cắt footer (bình luận, tin liên quan...).
    for marker in _NEWS_FOOTER_MARKERS:
        idx = body.find(marker)
        if idx != -1:
            body = body[:idx]

    body = re.sub(r"\s+", " ", body).strip()
    if not body:  # không khớp pattern → trả nguyên văn để khỏi mất dữ liệu
        return text
    # Nhập tiêu đề vào thân bài (không để heading đứng riêng thành 1 chunk rỗng).
    if not title:
        return body
    sep = " " if title[-1] in ".!?…" else ". "
    return f"{title}{sep}{body}"


def load_markdown_documents() -> list[dict]:
    """Load all standardized markdown files as document dictionaries."""
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        rel_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = rel_path.parts[0] if len(rel_path.parts) > 1 else "unknown"
        if doc_type == "news":
            content = clean_news_markdown(content)
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(rel_path),
                    "type": doc_type,
                },
            }
        )

    return documents


def simple_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into bounded character chunks on paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            # Cửa sổ ký tự nhưng căn cả hai đầu về ranh giới từ để không cắt giữa từ.
            start = 0
            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))
                if end < len(paragraph):
                    space = paragraph.rfind(" ", start + 1, end)
                    if space > start:
                        end = space
                part = paragraph[start:end].strip()
                if part:
                    chunks.append(part)
                if end >= len(paragraph):
                    break
                # Bắt đầu chunk kế tiếp lùi lại để overlap, rồi snap tới đầu từ.
                next_start = max(end - chunk_overlap, start + 1)
                space = paragraph.find(" ", next_start)
                start = space + 1 if 0 <= space < end else end
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = paragraph

    if current:
        chunks.append(current.strip())

    return chunks


def default_chunks(chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    """Load standardized markdown documents and chunk them consistently."""
    chunks: list[dict] = []
    for doc in load_markdown_documents():
        for index, chunk_text in enumerate(simple_chunks(doc["content"], chunk_size, chunk_overlap)):
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {**doc["metadata"], "chunk_index": index},
                }
            )
    return chunks


def cosine_from_counters(a: Counter[str], b: Counter[str]) -> float:
    """Cosine similarity for sparse token counters."""
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(token, 0) for token, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def text_similarity(query: str, content: str) -> float:
    """Local semantic-ish score using token overlap and cosine similarity."""
    query_tokens = tokenize(query)
    content_tokens = tokenize(content)
    if not query_tokens or not content_tokens:
        return 0.0

    query_counter = Counter(query_tokens)
    content_counter = Counter(content_tokens)
    cosine = cosine_from_counters(query_counter, content_counter)
    overlap = len(set(query_tokens) & set(content_tokens)) / max(1, len(set(query_tokens)))
    return 0.65 * cosine + 0.35 * overlap


def fallback_corpus() -> list[dict]:
    """Small corpus used only when standardized data has not been generated."""
    return [
        {
            "content": (
                "Luật Phòng, chống ma túy 2021 quy định về quản lý người sử dụng "
                "trái phép chất ma túy, cai nghiện ma túy và trách nhiệm của cơ quan, "
                "tổ chức, cá nhân trong phòng chống ma túy."
            ),
            "metadata": {"source": "fallback-legal", "type": "legal", "chunk_index": 0},
        },
        {
            "content": (
                "Bộ luật Hình sự quy định các tội liên quan đến ma túy như tàng trữ, "
                "vận chuyển, mua bán và tổ chức sử dụng trái phép chất ma túy."
            ),
            "metadata": {"source": "fallback-penal", "type": "legal", "chunk_index": 0},
        },
    ]


def ensure_chunks() -> list[dict]:
    """Return indexed chunks, falling back to a tiny built-in corpus."""
    chunks = default_chunks()
    return chunks or fallback_corpus()
