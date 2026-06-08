"""
Bonus — Conversation Memory (multi-turn chat).

Chatbot RAG mặc định là stateless: mỗi câu hỏi được retrieve độc lập, nên các câu
follow-up kiểu "còn vụ đó thì sao?", "anh ấy bị phạt bao nhiêu?" sẽ mất ngữ cảnh.

Module này:
    1. Lưu lịch sử hội thoại theo `session_id` (in-memory).
    2. `condense_query()` — viết lại câu follow-up thành câu hỏi độc lập bằng cách
       ghép thêm từ khoá ngữ cảnh từ các lượt trước, để retrieve đúng tài liệu.

Heuristic offline (không cần API key): nếu câu hỏi mới ngắn hoặc chứa "đại từ
follow-up" (vụ đó, anh ấy, cô ấy, họ, còn..., vậy..., trường hợp này...), ta ghép
thêm danh từ khoá của lượt user gần nhất. Nếu có OPENAI_API_KEY có thể thay bằng
LLM-based query rewriting.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

from .rag_utils import tokenize

# session_id -> list[{"role": "user"|"assistant", "content": str}]
_SESSIONS: dict[str, list[dict]] = defaultdict(list)

# Số lượt hội thoại gần nhất giữ lại làm ngữ cảnh.
MAX_HISTORY_TURNS = 6

# Dấu hiệu cho biết câu hỏi là follow-up, cần ghép ngữ cảnh.
_FOLLOWUP_MARKERS = [
    "vụ đó", "vụ này", "vụ việc đó", "trường hợp này", "trường hợp đó",
    "anh ấy", "cô ấy", "ông ấy", "bà ấy", "họ", "người đó", "người này",
    "còn", "thế còn", "vậy còn", "vậy", "thì sao", "ra sao", "như thế nào",
    "điều đó", "việc đó", "cái đó", "nó",
]

# Từ dừng — không dùng làm từ khoá ngữ cảnh.
_STOPWORDS = {
    "là", "gì", "nào", "sao", "thì", "của", "và", "có", "được", "cho", "về",
    "theo", "khi", "này", "đó", "các", "những", "một", "với", "trong", "bị",
    "vì", "ai", "bao", "nhiêu", "tại", "ra", "không", "đã", "sẽ", "còn", "vậy",
}


def get_history(session_id: str) -> list[dict]:
    """Trả về lịch sử hội thoại của một phiên."""
    return _SESSIONS[session_id]


def add_turn(session_id: str, role: str, content: str) -> None:
    """Ghi một lượt vào lịch sử (cắt bớt nếu vượt MAX_HISTORY_TURNS)."""
    history = _SESSIONS[session_id]
    history.append({"role": role, "content": content})
    if len(history) > MAX_HISTORY_TURNS:
        del history[: len(history) - MAX_HISTORY_TURNS]


def reset_session(session_id: str) -> None:
    """Xoá lịch sử một phiên (bắt đầu hội thoại mới)."""
    _SESSIONS.pop(session_id, None)


def _is_followup(message: str) -> bool:
    lowered = message.lower()
    if any(marker in lowered for marker in _FOLLOWUP_MARKERS):
        return True
    # Câu rất ngắn (ít từ nội dung) cũng coi là follow-up.
    content_tokens = [t for t in tokenize(message) if t not in _STOPWORDS and len(t) > 2]
    return len(content_tokens) <= 2


def _keywords(text: str, limit: int = 6) -> list[str]:
    """Lấy các từ khoá nội dung (bỏ stopword) giữ thứ tự xuất hiện."""
    seen: list[str] = []
    for token in tokenize(text):
        if token in _STOPWORDS or len(token) <= 2:
            continue
        if token not in seen:
            seen.append(token)
    return seen[:limit]


def _llm_condense(history: list[dict], message: str) -> str | None:
    """Viết lại câu hỏi bằng LLM nếu có API key; None nếu không khả dụng."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI

        client = OpenAI()
        convo = "\n".join(f"{t['role']}: {t['content']}" for t in history[-4:])
        prompt = (
            "Dựa vào hội thoại sau, viết lại câu hỏi cuối của user thành một câu hỏi "
            "ĐỘC LẬP, đầy đủ ngữ cảnh, bằng tiếng Việt. Chỉ trả về câu hỏi.\n\n"
            f"{convo}\nuser: {message}\n\nCâu hỏi độc lập:"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=120,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


def condense_query(session_id: str, message: str) -> str:
    """
    Biến câu hỏi follow-up thành câu hỏi độc lập dựa trên lịch sử phiên.

    Returns:
        Câu truy vấn đã được bổ sung ngữ cảnh (để feed vào retrieve()).
    """
    history = _SESSIONS[session_id]
    if not history or not _is_followup(message):
        return message

    llm_rewrite = _llm_condense(history, message)
    if llm_rewrite:
        return llm_rewrite

    # Offline: ghép từ khoá ngữ cảnh từ lượt USER gần nhất.
    last_user = next(
        (t["content"] for t in reversed(history) if t["role"] == "user"), ""
    )
    context_kw = [kw for kw in _keywords(last_user) if kw not in tokenize(message)]
    if not context_kw:
        return message
    return f"{message} ({' '.join(context_kw)})"


if __name__ == "__main__":
    sid = "demo"
    add_turn(sid, "user", "DJ Thái Hoàng bị bắt vì lý do gì?")
    add_turn(sid, "assistant", "DJ Thái Hoàng bị bắt quả tang vì tàng trữ trái phép chất ma túy.")
    follow = "Còn anh ấy bị xử lý ở đâu?"
    print("Follow-up :", follow)
    print("Condensed :", condense_query(sid, follow))
