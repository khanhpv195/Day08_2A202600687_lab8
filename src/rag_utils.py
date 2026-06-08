"""Shared local utilities for the Day 8 RAG tasks.

The lab recommends external services for indexing and PageIndex, but the
automated tests need the modules to work in a plain local environment too.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"

TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer that keeps Vietnamese characters."""
    return TOKEN_RE.findall(text.lower())


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
            step = max(1, chunk_size - chunk_overlap)
            for start in range(0, len(paragraph), step):
                part = paragraph[start : start + chunk_size].strip()
                if part:
                    chunks.append(part)
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
