# RAG Evaluation Results

## Framework sử dụng

**LocalLexicalEvaluator** — bản đánh giá không cần API key, cài đặt 4 trục metric chuẩn của RAGAS/DeepEval bằng token-overlap & cosine similarity trên tiếng Việt. Cho phép chạy toàn bộ pipeline offline trong phòng lab. Khi có OpenAI/Gemini API key, có thể bật `evaluate_with_ragas` / `evaluate_with_deepeval` trong `eval_pipeline.py` để chấm bằng LLM-as-judge.

- **Golden dataset:** 20 cặp Q&A (`golden_dataset.json`) — pháp luật + tin tức về ma túy.
- **top_k:** 5

---

## Overall Scores

| Metric | Config A (Hybrid + Reranking) | Config B (Hybrid, no Reranking) | Δ (A−B) |
|--------|--------------------------|--------------------------|---------|
| Faithfulness | 0.979 | 0.976 | +0.003 |
| Answer Relevance | 0.506 | 0.52 | -0.014 |
| Context Recall | 0.87 | 0.855 | +0.015 |
| Context Precision | 0.99 | 0.99 | 0.0 |
| **Average** | **0.836** | **0.835** | **+0.001** |

---

## A/B Comparison Analysis

**Config A:** Hybrid + Reranking — chạy semantic + lexical, merge bằng RRF, sau đó rerank bằng cross-encoder/MMR (Task 7).

**Config B:** Hybrid, no Reranking — chạy semantic + lexical, merge bằng RRF, nhưng **bỏ bước reranking**, lấy thẳng top_k sau fusion.

**Kết luận:** Hai config gần như ngang nhau (chênh chỉ 0.001 average — 0.836 vs 0.835). Với corpus nhỏ (8 văn bản) và evaluator dựa trên token-overlap, top_k sau RRF đã đủ tốt nên reranking chưa tạo khác biệt rõ. Reranking dự kiến phát huy tác dụng khi corpus lớn hơn, nhiều chunk gây nhiễu, và khi dùng cross-encoder thật + embedding ngữ nghĩa (thay cho token-overlap) — lúc đó nó mới lọc được noise ở Context Precision.

---

## Worst Performers (Bottom 3 — Config A)

| # | Question | Faithfulness | Relevance | Recall | Precision | Root Cause |
|---|----------|--------------|-----------|--------|-----------|------------|
| 1 | Hình phạt cho tội tàng trữ trái phép chất ma túy theo Đ… | 0.971 | 0.476 | 0.438 | 1.0 | Câu hỏi diễn đạt khác từ khóa trong văn bản |
| 2 | Luật Phòng, chống ma túy 2021 phân biệt 'chất gây nghiệ… | 0.956 | 0.544 | 0.522 | 1.0 | Câu hỏi diễn đạt khác từ khóa trong văn bản |
| 3 | Tội vận chuyển trái phép chất ma túy theo Điều 250 Bộ l… | 0.978 | 0.528 | 0.609 | 1.0 | Câu hỏi diễn đạt khác từ khóa trong văn bản |

---

## Recommendations

### Cải tiến 1 — Chunking theo heading của văn bản luật
**Action:** Dùng `MarkdownHeaderTextSplitter` tách theo "Điều/Chương" thay vì cắt ký tự cố định, để mỗi chunk gói trọn 1 điều luật.
**Expected impact:** Tăng Context Recall & Precision cho câu hỏi pháp luật (điều khoản không bị cắt giữa chừng).

### Cải tiến 2 — Embedding model multilingual thực thụ
**Action:** Thay token-overlap bằng `BAAI/bge-m3` cho semantic search.
**Expected impact:** Bắt được câu hỏi diễn đạt khác từ khóa (paraphrase), nâng Answer Relevance & Recall ở câu hỏi tin tức.

### Cải tiến 3 — Cross-encoder reranker tiếng Việt + HyDE
**Action:** Dùng `jinaai/jina-reranker-v2-base-multilingual` và sinh hypothetical document (HyDE) trước khi retrieve.
**Expected impact:** Giảm noise ở top_k, tăng Faithfulness của câu trả lời.
