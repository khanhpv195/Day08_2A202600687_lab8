"""
Bonus — HyDE (Hypothetical Document Embeddings).

Ý tưởng (Gao et al., 2022): thay vì embed thẳng câu hỏi (thường ngắn, ít từ khoá,
khó match với văn bản), ta yêu cầu LLM *sinh ra một câu trả lời giả định* cho câu
hỏi rồi embed/đối sánh chính tài liệu giả định đó với corpus. Tài liệu giả định
giàu từ vựng cùng miền nên thu hẹp khoảng cách lexical/ngữ nghĩa giữa query và
chunk → tăng recall, đặc biệt cho câu hỏi diễn đạt khác từ khoá trong văn bản.

    query ──▶ generate_hypothetical_document() ──▶ pseudo-doc
    semantic_search(query + pseudo-doc) ──▶ kết quả

Triển khai:
    - Nếu có OPENAI_API_KEY / GEMINI_API_KEY → gọi LLM sinh pseudo-doc thật.
    - Nếu không có key → fallback offline: mở rộng query bằng từ điển từ khoá
      cùng miền (luật + tin tức ma tuý) để mô phỏng pseudo-doc.
"""

from __future__ import annotations

import os

from .task5_semantic_search import semantic_search

# Từ điển mở rộng theo miền — dùng cho fallback offline khi không có LLM API key.
# Mỗi khoá là 1 chủ đề; nếu query chạm chủ đề đó, ta thêm các từ ngữ pháp lý/báo chí
# thường xuất hiện trong văn bản gốc để pseudo-doc "giống" tài liệu thật hơn.
_DOMAIN_EXPANSIONS: dict[str, str] = {
    "cai nghiện": (
        "cai nghiện ma túy tự nguyện tại gia đình cộng đồng cơ sở cai nghiện "
        "bắt buộc công lập quy trình tiếp nhận phân loại cắt cơn giải độc phục hồi "
        "lao động trị liệu tái hòa nhập cộng đồng Điều 28 Điều 29 Điều 30"
    ),
    "hình phạt": (
        "phạt tù phạt tiền khung hình phạt tội phạm tàng trữ vận chuyển mua bán "
        "tổ chức sử dụng trái phép chất ma túy Bộ luật Hình sự cải tạo không giam giữ"
    ),
    "tàng trữ": (
        "tàng trữ trái phép chất ma túy vận chuyển mua bán heroin ma túy phạt tù "
        "Bộ luật Hình sự bắt quả tang Công an khởi tố điều tra"
    ),
    "chất ma túy": (
        "chất gây nghiện chất hướng thần tiền chất danh mục Chính phủ ban hành "
        "kích thích ức chế thần kinh ảo giác nghiện Điều 2"
    ),
    "nghiêm cấm": (
        "hành vi bị nghiêm cấm trồng cây chứa chất ma túy sản xuất tàng trữ vận chuyển "
        "mua bán sử dụng tổ chức sử dụng cưỡng bức lôi kéo kỳ thị người cai nghiện Điều 5"
    ),
    "nghệ sĩ": (
        "ca sĩ DJ nghệ sĩ showbiz bị bắt tạm giữ sử dụng tàng trữ trái phép chất ma túy "
        "Công an xét nghiệm dương tính phong sát hình ảnh sự nghiệp"
    ),
    "bị bắt": (
        "bị bắt tạm giữ bắt quả tang Công an khởi tố điều tra hành vi tàng trữ "
        "sử dụng tổ chức sử dụng trái phép chất ma túy dương tính"
    ),
    "nghị định": (
        "Nghị định Chính phủ quy định chi tiết hướng dẫn thi hành Luật Phòng chống ma túy "
        "phối hợp cơ quan chuyên trách kiểm soát hoạt động hợp pháp"
    ),
}


def _local_hypothetical_document(query: str) -> str:
    """Sinh pseudo-document offline bằng cách mở rộng từ khoá cùng miền."""
    lowered = query.lower()
    expansions = [text for key, text in _DOMAIN_EXPANSIONS.items() if key in lowered]
    if not expansions:
        # Không khớp chủ đề nào — fallback chung cho miền luật/ma túy.
        expansions.append(
            "Luật Phòng chống ma túy 2021 Bộ luật Hình sự chất ma túy cai nghiện "
            "tàng trữ sử dụng trái phép xử lý theo quy định pháp luật"
        )
    return f"{query}. {' '.join(expansions)}"


def _llm_hypothetical_document(query: str) -> str | None:
    """Sinh pseudo-document bằng LLM nếu có API key; trả về None nếu không khả dụng."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            client = OpenAI()
            prompt = (
                "Viết một đoạn văn bản pháp luật/báo chí tiếng Việt ngắn (3-4 câu) "
                "trả lời giả định cho câu hỏi sau, dùng đúng thuật ngữ chuyên ngành. "
                f"Câu hỏi: {query}"
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return None
    return None


def generate_hypothetical_document(query: str) -> str:
    """
    Sinh một "tài liệu giả định" cho query (LLM nếu có key, ngược lại offline).

    Returns:
        Pseudo-document string (luôn bao gồm query gốc để giữ tín hiệu).
    """
    llm_doc = _llm_hypothetical_document(query)
    if llm_doc:
        return f"{query}. {llm_doc}"
    return _local_hypothetical_document(query)


def hyde_search(query: str, top_k: int = 10) -> list[dict]:
    """
    HyDE retrieval: sinh pseudo-document rồi semantic search trên đó.

    Args:
        query: Câu truy vấn gốc
        top_k: Số kết quả trả về

    Returns:
        List of {'content', 'score', 'metadata'} sorted descending.
    """
    pseudo_doc = generate_hypothetical_document(query)
    return semantic_search(pseudo_doc, top_k=top_k)


if __name__ == "__main__":
    demo_queries = [
        "Nghệ sĩ nào bị xử lý vì ma túy?",
        "Quy trình cai nghiện gồm những bước gì?",
    ]
    for q in demo_queries:
        print(f"\nQuery: {q}")
        print(f"HyDE doc: {generate_hypothetical_document(q)[:140]}...")
        print("-" * 60)
        for i, r in enumerate(hyde_search(q, top_k=3), 1):
            print(f"  {i}. [{r['score']:.3f}] {r['metadata'].get('source')} | {r['content'][:70]}...")
