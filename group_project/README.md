# Bài Tập Nhóm — Search Engine / RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1:  Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về pháp luật ma tuý và tin tức liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

### Code mẫu — DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

# Tạo test cases từ golden dataset
test_cases = []
for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected_answer"],
        retrieval_context=[c["content"] for c in result["sources"]],
    )
    test_cases.append(test_case)

# Chạy evaluation
metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
]

results = evaluate(test_cases, metrics)
```

### Code mẫu — RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Chuẩn bị data
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    eval_data["question"].append(item["question"])
    eval_data["answer"].append(result["answer"])
    eval_data["contexts"].append([c["content"] for c in result["sources"]])
    eval_data["ground_truth"].append(item["expected_answer"])

dataset = Dataset.from_dict(eval_data)

# Chạy evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
print(result.to_pandas())
```

### Code mẫu — TruLens

```python
from trulens.apps.custom import TruCustomApp, instrument
from trulens.core import Feedback
from trulens.providers.openai import OpenAI as TruOpenAI

provider = TruOpenAI()

# Define feedback functions
f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
f_relevance = Feedback(provider.relevance).on_input_output()
f_context_relevance = Feedback(provider.context_relevance).on_input()

# Wrap RAG pipeline
tru_rag = TruCustomApp(
    rag_pipeline,
    app_name="DrugLaw_RAG",
    feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
)

# Run evaluation
with tru_rag as recording:
    for item in golden_dataset:
        rag_pipeline.generate_with_citation(item["question"])

# View dashboard
from trulens.dashboard import run_dashboard
run_dashboard()
```

### Deliverable Evaluation

- [x] File `group_project/evaluation/golden_dataset.json` — **16** cặp Q&A
- [x] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation (4 metrics)
- [x] File `group_project/evaluation/results.md` — bảng điểm + phân tích + worst performers
- [x] So sánh A/B 2 configs (hybrid + rerank vs no-rerank)

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

Nhóm chọn **Option B — RAG Chatbot** (`chat_server.py` + `web/index.html`).

```
                          ┌─────────────────────────────┐
   Người dùng  ──POST──▶  │  web/index.html (chat UI)    │
                          └──────────────┬──────────────┘
                                         │  /api/chat (JSON)
                          ┌──────────────▼──────────────┐
                          │  chat_server.py (HTTP API)   │
                          └──────────────┬──────────────┘
                                         │  generate_with_citation()
        ┌────────────────────────────────▼────────────────────────────────┐
        │  Task 10 — Generation: reorder (lost-in-the-middle) + citation     │
        └────────────────────────────────┬────────────────────────────────┘
                                         │  retrieve()
        ┌────────────────────────────────▼────────────────────────────────┐
        │  Task 9 — Retrieval Pipeline                                       │
        │    Semantic (T5) ─┐                                                │
        │                   ├─▶ RRF merge ─▶ Rerank (T7) ─▶ top_k            │
        │    Lexical/BM25(T6)┘                                               │
        │    score < threshold ─▶ Fallback: PageIndex vectorless (T8)        │
        └────────────────────────────────┬────────────────────────────────┘
                                         │
        ┌────────────────────────────────▼────────────────────────────────┐
        │  Task 4 — Chunking + Index (data/standardized/*.md)               │
        │  Task 1–3 — Thu thập (luật PDF/DOCX) + crawl báo + convert MD     │
        └───────────────────────────────────────────────────────────────────┘

   Đánh giá:  group_project/evaluation/eval_pipeline.py
              golden_dataset.json (16 Q&A) ─▶ A/B (rerank vs no-rerank) ─▶ results.md
```

**Data flow:** PDF/DOCX luật + JSON bài báo → MarkItDown → markdown chuẩn hoá →
chunk + index → hybrid retrieval + rerank (fallback PageIndex) → generation có citation.

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Phạm Văn Khánh & Nguyễn Trọng Khánh | 2A202600687+2A202600796 | Pipeline cá nhân Task 1–10 (35/35 test pass) | ✅ Hoàn thành |
| Phạm Văn Khánh & Nguyễn Trọng Khánh | 2A202600687+2A202600796 | Golden dataset 16 cặp Q&A (`golden_dataset.json`) | ✅ Hoàn thành |
| Phạm Văn Khánh & Nguyễn Trọng Khánh | 2A202600687+2A202600796 | Evaluation pipeline + A/B (rerank vs no-rerank) + phân tích worst performers (`eval_pipeline.py`, `results.md`) | ✅ Hoàn thành |
| Phạm Văn Khánh & Nguyễn Trọng Khánh | 2A202600687+2A202600796 | RAG chatbot server + UI (`chat_server.py`, `web/index.html`) + diagram kiến trúc | ✅ Hoàn thành |
| Phạm Văn Khánh & Nguyễn Trọng Khánh | 2A202600687+2A202600796 | Bonus — HyDE (`src/task_bonus_hyde.py`) | ✅ Hoàn thành |
| Phạm Văn Khánh & Nguyễn Trọng Khánh | 2A202600687+2A202600796 | Bonus — Conversation memory multi-turn (`src/conversation.py`) | ✅ Hoàn thành |
| Phạm Văn Khánh & Nguyễn Trọng Khánh | 2A202600687+2A202600796 | Bonus — Lexical TF-IDF vs BM25 + Notebook demo (`notebooks/demo.ipynb`) | ✅ Hoàn thành |

---

## Hướng Dẫn Chạy

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Chạy RAG chatbot (server tĩnh + API /api/chat)
python chat_server.py
# Mở trình duyệt: http://127.0.0.1:8008

# 3. Chạy evaluation pipeline (sinh group_project/evaluation/results.md)
python group_project/evaluation/eval_pipeline.py

# 4. (Tuỳ chọn) Mở notebook demo end-to-end
jupyter notebook notebooks/demo.ipynb
```

---

## Bonus đã triển khai

| Bonus | File | Cách dùng |
|-------|------|-----------|
| **HyDE** (Hypothetical Document Embeddings) | `src/task_bonus_hyde.py` | `retrieve(query, use_hyde=True)`; sinh pseudo-document (LLM nếu có API key, ngược lại mở rộng từ khoá cùng miền) rồi semantic search |
| **Conversation memory** (multi-turn) | `src/conversation.py` | Server lưu lịch sử theo `session_id`, `condense_query()` viết lại câu follow-up thành câu hỏi độc lập |
| **UI hiển thị source + score + câu viết lại** | `web/index.html` | Panel "Nguồn tham khảo" + nút "Trò chuyện mới" (reset memory) + dòng `↻ Hiểu câu hỏi theo ngữ cảnh` |
| **Lexical khác BM25 (TF-IDF)** | `notebooks/demo.ipynb` §8 | Giải thích cơ chế TF-IDF + cosine và khác biệt với BM25 (term saturation, length normalization) |

> HyDE và conversation memory tự động dùng LLM khi có `OPENAI_API_KEY` trong `.env`, và có fallback offline (không cần API key) để chạy được trong phòng lab.

---

## Lưu ý: Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
