"""
Day 8 v2 — RAG Pipeline
Automated Test Suite cho bài cá nhân.

Chạy:
    pytest tests/test_individual.py -v

Mỗi task được test riêng. Tổng: 50 điểm.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Project root
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SRC_DIR = PROJECT_DIR / "src"

# Add src to path
sys.path.insert(0, str(PROJECT_DIR))


# ===========================================================================
# Task 1 — Thu thập văn bản pháp luật (3 điểm)
# ===========================================================================

class TestTask1(unittest.TestCase):
    """Task 1: Thu thập ≥3 văn bản pháp luật vào data/landing/legal/"""

    def test_landing_legal_dir_exists(self):
        """data/landing/legal/ tồn tại."""
        legal_dir = DATA_DIR / "landing" / "legal"
        self.assertTrue(legal_dir.exists(), f"Thư mục không tồn tại: {legal_dir}")

    def test_minimum_3_legal_files(self):
        """Có tối thiểu 3 file PDF/DOCX trong data/landing/legal/"""
        legal_dir = DATA_DIR / "landing" / "legal"
        if not legal_dir.exists():
            self.skipTest("data/landing/legal/ chưa tồn tại")

        valid_extensions = {".pdf", ".docx", ".doc"}
        files = [f for f in legal_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in valid_extensions]
        self.assertGreaterEqual(
            len(files), 3,
            f"Cần tối thiểu 3 file pháp luật, hiện có {len(files)}: {[f.name for f in files]}"
        )

    def test_files_not_empty(self):
        """Các file pháp luật không rỗng (>1KB)."""
        legal_dir = DATA_DIR / "landing" / "legal"
        if not legal_dir.exists():
            self.skipTest("data/landing/legal/ chưa tồn tại")

        valid_extensions = {".pdf", ".docx", ".doc"}
        files = [f for f in legal_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in valid_extensions]
        for f in files:
            self.assertGreater(
                f.stat().st_size, 1024,
                f"File {f.name} quá nhỏ ({f.stat().st_size} bytes), có thể bị lỗi"
            )


# ===========================================================================
# Task 2 — Crawl bài báo (3 điểm)
# ===========================================================================

class TestTask2(unittest.TestCase):
    """Task 2: Crawl ≥5 bài báo vào data/landing/news/"""

    def test_landing_news_dir_exists(self):
        """data/landing/news/ tồn tại."""
        news_dir = DATA_DIR / "landing" / "news"
        self.assertTrue(news_dir.exists(), f"Thư mục không tồn tại: {news_dir}")

    def test_minimum_5_news_files(self):
        """Có tối thiểu 5 file trong data/landing/news/"""
        news_dir = DATA_DIR / "landing" / "news"
        if not news_dir.exists():
            self.skipTest("data/landing/news/ chưa tồn tại")

        valid_extensions = {".json", ".html", ".md", ".txt"}
        files = [f for f in news_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in valid_extensions]
        self.assertGreaterEqual(
            len(files), 5,
            f"Cần tối thiểu 5 bài báo, hiện có {len(files)}"
        )

    def test_news_files_have_content(self):
        """Mỗi file bài báo có nội dung (>500 bytes)."""
        news_dir = DATA_DIR / "landing" / "news"
        if not news_dir.exists():
            self.skipTest("data/landing/news/ chưa tồn tại")

        files = [f for f in news_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
        for f in files[:5]:  # Check first 5
            self.assertGreater(
                f.stat().st_size, 500,
                f"File {f.name} quá nhỏ, có thể crawl bị lỗi"
            )

    def test_json_files_have_metadata(self):
        """File JSON có các trường metadata cần thiết."""
        news_dir = DATA_DIR / "landing" / "news"
        if not news_dir.exists():
            self.skipTest("data/landing/news/ chưa tồn tại")

        json_files = [f for f in news_dir.iterdir() if f.suffix == ".json"]
        if not json_files:
            self.skipTest("Không có file JSON (có thể dùng format khác)")

        for f in json_files[:3]:
            data = json.loads(f.read_text(encoding="utf-8"))
            self.assertIn("url", data, f"{f.name} thiếu trường 'url'")


# ===========================================================================
# Task 3 — Convert markdown (4 điểm)
# ===========================================================================

class TestTask3(unittest.TestCase):
    """Task 3: Convert toàn bộ files sang markdown trong data/standardized/"""

    def test_standardized_dir_exists(self):
        """data/standardized/ tồn tại."""
        self.assertTrue(
            (DATA_DIR / "standardized").exists(),
            "Thư mục data/standardized/ chưa tồn tại"
        )

    def test_has_markdown_files(self):
        """Có ít nhất 1 file .md trong data/standardized/"""
        std_dir = DATA_DIR / "standardized"
        if not std_dir.exists():
            self.skipTest("data/standardized/ chưa tồn tại")

        md_files = list(std_dir.rglob("*.md"))
        self.assertGreater(len(md_files), 0, "Không tìm thấy file .md nào")

    def test_converted_files_have_content(self):
        """File markdown đã convert có nội dung (>200 chars)."""
        std_dir = DATA_DIR / "standardized"
        if not std_dir.exists():
            self.skipTest("data/standardized/ chưa tồn tại")

        md_files = list(std_dir.rglob("*.md"))
        if not md_files:
            self.skipTest("Chưa có file markdown")

        for f in md_files[:5]:
            content = f.read_text(encoding="utf-8")
            self.assertGreater(
                len(content), 200,
                f"{f.name} quá ngắn ({len(content)} chars), convert có thể bị lỗi"
            )

    def test_legal_and_news_both_converted(self):
        """Cả legal và news đều được convert."""
        std_dir = DATA_DIR / "standardized"
        if not std_dir.exists():
            self.skipTest("data/standardized/ chưa tồn tại")

        has_legal = (std_dir / "legal").exists() and list((std_dir / "legal").rglob("*.md"))
        has_news = (std_dir / "news").exists() and list((std_dir / "news").rglob("*.md"))
        self.assertTrue(
            has_legal or has_news,
            "Cần ít nhất 1 trong 2 thư mục legal/ hoặc news/ có file .md"
        )


# ===========================================================================
# Task 4 — Chunking & Indexing (7 điểm)
# ===========================================================================

class TestTask4(unittest.TestCase):
    """Task 4: Chunking + Indexing hoạt động."""

    def _import_task4(self):
        try:
            from src.task4_chunking_indexing import (
                load_documents, chunk_documents, CHUNK_SIZE, CHUNK_OVERLAP
            )
            return load_documents, chunk_documents, CHUNK_SIZE, CHUNK_OVERLAP
        except (ImportError, NotImplementedError) as e:
            self.skipTest(f"Task 4 chưa implement: {e}")

    def test_config_documented(self):
        """CHUNK_SIZE và CHUNK_OVERLAP được cấu hình."""
        _, _, chunk_size, chunk_overlap = self._import_task4()
        self.assertGreater(chunk_size, 0, "CHUNK_SIZE phải > 0")
        self.assertGreater(chunk_overlap, 0, "CHUNK_OVERLAP phải > 0")
        self.assertLess(chunk_overlap, chunk_size, "OVERLAP phải < SIZE")

    def test_load_documents_returns_list(self):
        """load_documents() trả về list of dicts."""
        load_documents, _, _, _ = self._import_task4()
        try:
            docs = load_documents()
            self.assertIsInstance(docs, list)
            if docs:
                self.assertIn("content", docs[0])
        except NotImplementedError:
            self.skipTest("load_documents chưa implement")

    def test_chunk_documents_produces_chunks(self):
        """chunk_documents() tạo ra chunks từ documents."""
        load_documents, chunk_documents, _, _ = self._import_task4()
        try:
            docs = load_documents()
            if not docs:
                self.skipTest("Không có documents để chunk")
            chunks = chunk_documents(docs[:1])  # Test with 1 doc
            self.assertIsInstance(chunks, list)
            self.assertGreater(len(chunks), 0, "Không tạo được chunk nào")
            self.assertIn("content", chunks[0])
        except NotImplementedError:
            self.skipTest("chunk_documents chưa implement")

    def test_chunks_respect_size_limit(self):
        """Mỗi chunk không vượt quá CHUNK_SIZE (+ tolerance 10%)."""
        load_documents, chunk_documents, chunk_size, _ = self._import_task4()
        try:
            docs = load_documents()
            if not docs:
                self.skipTest("Không có documents")
            chunks = chunk_documents(docs[:1])
            max_allowed = int(chunk_size * 1.1)
            for i, c in enumerate(chunks[:20]):
                self.assertLessEqual(
                    len(c["content"]), max_allowed,
                    f"Chunk {i} vượt quá size limit: {len(c['content'])} > {max_allowed}"
                )
        except NotImplementedError:
            self.skipTest("Chưa implement")


# ===========================================================================
# Task 5 — Semantic Search (6 điểm)
# ===========================================================================

class TestTask5(unittest.TestCase):
    """Task 5: Semantic search module."""

    def _import_task5(self):
        try:
            from src.task5_semantic_search import semantic_search
            return semantic_search
        except (ImportError, NotImplementedError) as e:
            self.skipTest(f"Task 5 chưa implement: {e}")

    def test_returns_list(self):
        """semantic_search() trả về list."""
        search = self._import_task5()
        try:
            results = search("ma tuý", top_k=3)
            self.assertIsInstance(results, list)
        except NotImplementedError:
            self.skipTest("semantic_search chưa implement")

    def test_results_have_required_keys(self):
        """Mỗi result có 'content', 'score', 'metadata'."""
        search = self._import_task5()
        try:
            results = search("hình phạt ma tuý", top_k=3)
            if not results:
                self.skipTest("Không có kết quả (có thể chưa index)")
            for r in results:
                self.assertIn("content", r)
                self.assertIn("score", r)
        except NotImplementedError:
            self.skipTest("Chưa implement")

    def test_results_sorted_descending(self):
        """Kết quả sorted theo score descending."""
        search = self._import_task5()
        try:
            results = search("pháp luật ma tuý", top_k=5)
            if len(results) < 2:
                self.skipTest("Không đủ kết quả để test sort")
            scores = [r["score"] for r in results]
            self.assertEqual(scores, sorted(scores, reverse=True))
        except NotImplementedError:
            self.skipTest("Chưa implement")

    def test_respects_top_k(self):
        """Không trả về nhiều hơn top_k results."""
        search = self._import_task5()
        try:
            results = search("test query", top_k=2)
            self.assertLessEqual(len(results), 2)
        except NotImplementedError:
            self.skipTest("Chưa implement")


# ===========================================================================
# Task 6 — Lexical Search / BM25 (6 điểm)
# ===========================================================================

class TestTask6(unittest.TestCase):
    """Task 6: Lexical search (BM25)."""

    def _import_task6(self):
        try:
            from src.task6_lexical_search import lexical_search
            return lexical_search
        except (ImportError, NotImplementedError) as e:
            self.skipTest(f"Task 6 chưa implement: {e}")

    def test_returns_list(self):
        """lexical_search() trả về list."""
        search = self._import_task6()
        try:
            results = search("Điều 248 ma tuý", top_k=3)
            self.assertIsInstance(results, list)
        except NotImplementedError:
            self.skipTest("lexical_search chưa implement")

    def test_results_have_required_keys(self):
        """Mỗi result có 'content', 'score'."""
        search = self._import_task6()
        try:
            results = search("tàng trữ trái phép", top_k=3)
            if not results:
                self.skipTest("Không có kết quả")
            for r in results:
                self.assertIn("content", r)
                self.assertIn("score", r)
        except NotImplementedError:
            self.skipTest("Chưa implement")

    def test_results_sorted_descending(self):
        """Kết quả sorted theo BM25 score descending."""
        search = self._import_task6()
        try:
            results = search("ma tuý chất cấm", top_k=5)
            if len(results) < 2:
                self.skipTest("Không đủ kết quả")
            scores = [r["score"] for r in results]
            self.assertEqual(scores, sorted(scores, reverse=True))
        except NotImplementedError:
            self.skipTest("Chưa implement")

    def test_keyword_match_scores_higher(self):
        """Query có keyword match phải có score > 0."""
        search = self._import_task6()
        try:
            results = search("ma tuý", top_k=3)
            if not results:
                self.skipTest("Không có kết quả")
            # Ít nhất 1 result phải có score > 0
            max_score = max(r["score"] for r in results)
            self.assertGreater(max_score, 0, "Tất cả score = 0, BM25 có thể bị lỗi")
        except NotImplementedError:
            self.skipTest("Chưa implement")


# ===========================================================================
# Task 7 — Reranking (6 điểm)
# ===========================================================================

class TestTask7(unittest.TestCase):
    """Task 7: Reranking module."""

    def _import_task7(self):
        try:
            from src.task7_reranking import rerank
            return rerank
        except (ImportError, NotImplementedError) as e:
            self.skipTest(f"Task 7 chưa implement: {e}")

    def test_rerank_returns_list(self):
        """rerank() trả về list."""
        rerank_fn = self._import_task7()
        candidates = [
            {"content": "Tội tàng trữ ma tuý", "score": 0.8, "metadata": {}},
            {"content": "Nghệ sĩ bị bắt vì ma tuý", "score": 0.6, "metadata": {}},
            {"content": "Python programming", "score": 0.4, "metadata": {}},
        ]
        try:
            results = rerank_fn("hình phạt ma tuý", candidates, top_k=2)
            self.assertIsInstance(results, list)
        except NotImplementedError:
            self.skipTest("rerank chưa implement")

    def test_rerank_respects_top_k(self):
        """Rerank trả về đúng top_k results."""
        rerank_fn = self._import_task7()
        candidates = [
            {"content": f"Document {i}", "score": 0.9 - i * 0.1, "metadata": {}}
            for i in range(10)
        ]
        try:
            results = rerank_fn("test query", candidates, top_k=3)
            self.assertLessEqual(len(results), 3)
        except NotImplementedError:
            self.skipTest("Chưa implement")

    def test_rerank_has_score(self):
        """Kết quả rerank có trường 'score'."""
        rerank_fn = self._import_task7()
        candidates = [
            {"content": "Luật phòng chống ma tuý", "score": 0.7, "metadata": {}},
            {"content": "Hình phạt tù 2-7 năm", "score": 0.5, "metadata": {}},
        ]
        try:
            results = rerank_fn("hình phạt", candidates, top_k=2)
            if results:
                self.assertIn("score", results[0])
        except NotImplementedError:
            self.skipTest("Chưa implement")


# ===========================================================================
# Task 8 — PageIndex Vectorless (4 điểm)
# ===========================================================================

class TestTask8(unittest.TestCase):
    """Task 8: PageIndex vectorless RAG."""

    def _import_task8(self):
        try:
            from src.task8_pageindex_vectorless import pageindex_search
            return pageindex_search
        except (ImportError, NotImplementedError) as e:
            self.skipTest(f"Task 8 chưa implement: {e}")

    def test_function_exists(self):
        """pageindex_search() function tồn tại."""
        search = self._import_task8()
        self.assertTrue(callable(search))

    def test_returns_list_with_source_marker(self):
        """Kết quả có 'source': 'pageindex'."""
        search = self._import_task8()
        try:
            results = search("ma tuý", top_k=2)
            self.assertIsInstance(results, list)
            if results:
                self.assertEqual(results[0].get("source"), "pageindex")
        except (NotImplementedError, Exception) as e:
            self.skipTest(f"PageIndex chưa sẵn sàng: {e}")


# ===========================================================================
# Task 9 — Retrieval Pipeline (7 điểm)
# ===========================================================================

class TestTask9(unittest.TestCase):
    """Task 9: Retrieval pipeline hoàn chỉnh."""

    def _import_task9(self):
        try:
            from src.task9_retrieval_pipeline import retrieve
            return retrieve
        except (ImportError, NotImplementedError) as e:
            self.skipTest(f"Task 9 chưa implement: {e}")

    def test_retrieve_returns_list(self):
        """retrieve() trả về list of dicts."""
        retrieve_fn = self._import_task9()
        try:
            results = retrieve_fn("hình phạt ma tuý", top_k=3)
            self.assertIsInstance(results, list)
        except NotImplementedError:
            self.skipTest("retrieve chưa implement")

    def test_results_have_required_keys(self):
        """Kết quả có 'content', 'score', 'source'."""
        retrieve_fn = self._import_task9()
        try:
            results = retrieve_fn("luật phòng chống ma tuý", top_k=3)
            if not results:
                self.skipTest("Không có kết quả")
            for r in results:
                self.assertIn("content", r)
                self.assertIn("score", r)
                self.assertIn("source", r)
                self.assertIn(r["source"], ["hybrid", "pageindex"])
        except NotImplementedError:
            self.skipTest("Chưa implement")

    def test_respects_top_k(self):
        """Không trả về nhiều hơn top_k."""
        retrieve_fn = self._import_task9()
        try:
            results = retrieve_fn("test", top_k=2)
            self.assertLessEqual(len(results), 2)
        except NotImplementedError:
            self.skipTest("Chưa implement")

    def test_fallback_logic_exists(self):
        """Pipeline có fallback logic (không crash khi hybrid trả rỗng)."""
        retrieve_fn = self._import_task9()
        try:
            # Query rất obscure → hybrid có thể không tìm thấy → fallback
            results = retrieve_fn("xyzabc123nonsense", top_k=3, score_threshold=0.99)
            # Không crash = pass
            self.assertIsInstance(results, list)
        except NotImplementedError:
            self.skipTest("Chưa implement")

    def test_unaccented_legal_query_retrieves_law(self):
        """Query không dấu vẫn retrieve được Luật Phòng, chống ma túy."""
        retrieve_fn = self._import_task9()
        results = retrieve_fn("cau luat phong chong ma tuy la gi", top_k=5)
        sources = [r.get("metadata", {}).get("source", "") for r in results]
        self.assertIn("73_2021_QH14_m_445185.md", sources)

    def test_artist_responsibility_query_retrieves_news(self):
        """Query về trách nhiệm người nổi tiếng phải lấy nguồn báo chí liên quan."""
        retrieve_fn = self._import_task9()
        query = (
            "Qua các vụ việc nghệ sĩ liên quan đến ma túy, báo chí đặt ra vấn đề gì "
            "về trách nhiệm của người nổi tiếng?"
        )
        results = retrieve_fn(query, top_k=5)
        sources = [r.get("metadata", {}).get("source", "") for r in results]
        self.assertIn("article_03.md", sources)


# ===========================================================================
# Task 10 — Generation có Citation (4 điểm)
# ===========================================================================

class TestTask10(unittest.TestCase):
    """Task 10: Generation có citation + document reordering."""

    def _import_task10(self):
        try:
            from src.task10_generation import (
                generate_with_citation, reorder_for_llm, format_context
            )
            return generate_with_citation, reorder_for_llm, format_context
        except (ImportError, NotImplementedError) as e:
            self.skipTest(f"Task 10 chưa implement: {e}")

    def test_reorder_function_exists(self):
        """reorder_for_llm() function hoạt động."""
        _, reorder, _ = self._import_task10()
        chunks = [
            {"content": f"Chunk {i}", "score": 1.0 - i * 0.1}
            for i in range(5)
        ]
        try:
            reordered = reorder(chunks)
            self.assertEqual(len(reordered), 5, "Reorder phải giữ nguyên số lượng chunks")
            # Chunk đầu tiên (important nhất) vẫn ở đầu
            self.assertEqual(reordered[0]["content"], "Chunk 0")
        except NotImplementedError:
            self.skipTest("reorder_for_llm chưa implement")

    def test_format_context_includes_source(self):
        """format_context() có thông tin source cho citation."""
        _, _, format_ctx = self._import_task10()
        chunks = [
            {"content": "Nội dung pháp luật", "score": 0.9,
             "metadata": {"source": "luat-phong-chong-ma-tuy.pdf", "type": "legal"}}
        ]
        try:
            ctx = format_ctx(chunks)
            self.assertIn("luat-phong-chong-ma-tuy", ctx)
        except NotImplementedError:
            self.skipTest("format_context chưa implement")

    def test_format_context_declares_exact_citation_label(self):
        """Context phải chỉ rõ citation label để LLM không cite kiểu Document 1."""
        _, _, format_ctx = self._import_task10()
        chunks = [
            {
                "content": "Luật quy định về cai nghiện ma túy.",
                "score": 0.9,
                "metadata": {"source": "73_2021_QH14_m_445185.md", "type": "legal"},
            }
        ]
        ctx = format_ctx(chunks)
        self.assertIn("Citation to use: [73_2021_QH14_m_445185.md]", ctx)

    def test_generate_returns_dict_with_answer(self):
        """generate_with_citation() trả về dict có 'answer'."""
        generate, _, _ = self._import_task10()
        try:
            result = generate("Hình phạt tàng trữ ma tuý?")
            self.assertIsInstance(result, dict)
            self.assertIn("answer", result)
            self.assertIsInstance(result["answer"], str)
            self.assertGreater(len(result["answer"]), 0)
        except NotImplementedError:
            self.skipTest("generate_with_citation chưa implement")
        except Exception as e:
            # API key missing, etc — still check structure exists
            self.skipTest(f"Generation error (có thể thiếu API key): {e}")

    def test_generate_calls_llm_with_retrieved_context(self):
        """generate_with_citation() phải gọi LLM với context đã retrieve."""
        from src import task10_generation as generation

        fake_chunks = [
            {
                "content": (
                    "Luật Phòng, chống ma túy 2021 quy định về cai nghiện ma túy "
                    "tự nguyện, cai nghiện bắt buộc và trách nhiệm hỗ trợ."
                ),
                "score": 0.92,
                "metadata": {"source": "73_2021_QH14_m_445185.md", "type": "legal"},
                "source": "hybrid",
            }
        ]
        fake_response = Mock()
        fake_response.choices = [
            Mock(
                message=Mock(
                    content=(
                        "Luật Phòng, chống ma túy 2021 quy định về cai nghiện ma túy "
                        "tự nguyện và trách nhiệm hỗ trợ. [73_2021_QH14_m_445185.md]"
                    )
                )
            )
        ]
        fake_client = Mock()
        fake_client.chat.completions.create.return_value = fake_response

        with (
            patch.object(generation, "retrieve", return_value=fake_chunks),
            patch.object(generation, "OpenAI", return_value=fake_client, create=True),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
        ):
            result = generation.generate_with_citation("cai nghiện ma túy", top_k=1)

        fake_client.chat.completions.create.assert_called_once()
        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        joined_messages = "\n".join(message["content"] for message in call_kwargs["messages"])
        self.assertIn("Luật Phòng, chống ma túy 2021", joined_messages)
        self.assertIn("cai nghiện ma túy", joined_messages)
        self.assertIn("cai nghiện ma túy tự nguyện", result["answer"])
        self.assertIn("[73_2021_QH14_m_445185.md]", result["answer"])

    def test_llm_prompt_forbids_inference_beyond_context(self):
        """Prompt phải cấm LLM suy luận hoặc thêm ý ngoài context."""
        from src import task10_generation as generation

        prompt = generation.SYSTEM_PROMPT.lower()
        self.assertIn("do not infer", prompt)
        self.assertIn("do not add interpretations", prompt)
        self.assertIn("quote or paraphrase only", prompt)

    def test_generate_accepts_unaccented_legal_query(self):
        """Câu hỏi không dấu vẫn qua relevance gate và gọi LLM."""
        from src import task10_generation as generation

        fake_response = Mock()
        fake_response.choices = [
            Mock(message=Mock(content="Luật Phòng, chống ma túy là... [73_2021_QH14_m_445185.md]"))
        ]
        fake_client = Mock()
        fake_client.chat.completions.create.return_value = fake_response

        with (
            patch.object(generation, "OpenAI", return_value=fake_client, create=True),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
        ):
            result = generation.generate_with_citation("cau luat phong chong ma tuy la gi", top_k=3)

        fake_client.chat.completions.create.assert_called_once()
        self.assertNotEqual(result["answer"], generation.CANNOT_VERIFY)
        self.assertGreater(len(result["sources"]), 0)

    def test_generate_repairs_wrong_citation_and_drops_unsupported_claims(self):
        """Post-check phải sửa citation sai và loại claim không có trong context."""
        from src import task10_generation as generation

        fake_chunks = [
            {
                "content": (
                    "Việc một nghệ sĩ nổi tiếng bị nhắc tên trong vụ việc nghiêm trọng "
                    "như vậy đủ gióng lên hồi chuông cảnh báo về trách nhiệm và ý thức "
                    "giữ gìn hình ảnh của người làm nghệ thuật."
                ),
                "score": 0.91,
                "metadata": {"source": "article_03.md", "type": "news"},
                "source": "hybrid",
            },
            {
                "content": "DJ Thái Hoàng là gương mặt có tiếng trong giới DJ.",
                "score": 0.7,
                "metadata": {"source": "article_02.md", "type": "news"},
                "source": "hybrid",
            },
        ]
        fake_response = Mock()
        fake_response.choices = [
            Mock(
                message=Mock(
                    content=(
                        "- Việc một nghệ sĩ nổi tiếng bị nhắc tên cảnh báo về trách nhiệm "
                        "giữ gìn hình ảnh của người làm nghệ thuật. [article_02.md]\n"
                        "- Điều này tác động đến cộng đồng và người hâm mộ. [article_03.md]"
                    )
                )
            )
        ]
        fake_client = Mock()
        fake_client.chat.completions.create.return_value = fake_response

        with (
            patch.object(generation, "retrieve", return_value=fake_chunks),
            patch.object(generation, "OpenAI", return_value=fake_client, create=True),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
        ):
            result = generation.generate_with_citation("trách nhiệm người nổi tiếng", top_k=2)

        self.assertIn("[article_03.md]", result["answer"])
        self.assertNotIn("[article_02.md]", result["answer"])
        self.assertNotIn("người hâm mộ", result["answer"])


class TestChatUI(unittest.TestCase):
    """UI chatbot hiển thị source documents nhưng không lộ raw retrieval/debug data."""

    def test_chat_api_source_summary_hides_raw_debug_data(self):
        import chat_server

        summary = chat_server._source_summary(
            {
                "content": "Nội dung chunk không nên gửi về UI",
                "score": 0.87,
                "metadata": {"source": "article_03.md", "type": "news"},
            }
        )

        self.assertEqual(summary, {"source": "article_03.md", "type": "news"})
        self.assertNotIn("preview", summary)
        self.assertNotIn("score", summary)

    def test_ui_renders_source_names_without_raw_debug_data(self):
        html = (PROJECT_DIR / "web" / "index.html").read_text(encoding="utf-8")
        self.assertIn("source-list", html)
        self.assertIn("source.source", html)
        self.assertNotIn("source.preview", html)
        self.assertNotIn("source.score", html)
        self.assertNotIn("rewrittenQuery", html)


class TestGroupEvaluation(unittest.TestCase):
    """Group evaluation deliverable phải import và chạy được."""

    def test_eval_pipeline_run_rag_smoke(self):
        from group_project.evaluation import eval_pipeline

        result = eval_pipeline.run_rag("Luật phòng chống ma túy là gì?", use_reranking=True, top_k=2)
        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertIn("contexts", result)
        self.assertGreater(len(result["contexts"]), 0)


# ===========================================================================
# Summary
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
