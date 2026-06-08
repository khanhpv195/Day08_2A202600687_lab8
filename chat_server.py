"""Small local web server for testing the RAG pipeline from a chat UI."""

from __future__ import annotations

import json
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from src.task10_generation import generate_with_citation
from src.conversation import add_turn, condense_query, reset_session

ROOT_DIR = Path(__file__).parent
WEB_DIR = ROOT_DIR / "web"
HOST = "127.0.0.1"
PORT = 8008


def _source_summary(source: dict) -> dict:
    metadata = source.get("metadata", {})
    content = source.get("content", "")
    return {
        "source": metadata.get("source", "unknown"),
        "type": metadata.get("type", "unknown"),
        "score": round(float(source.get("score", 0.0)), 4),
        "preview": " ".join(content.split())[:260],
    }


def _clean_answer(answer: str) -> str:
    """Remove crawler metadata so the chat response reads like an answer."""
    cleaned = re.sub(r"^#\s*", "", answer, flags=re.MULTILINE)
    cleaned = re.sub(r"\*\*Source:\*\*\s*\S+", "", cleaned)
    cleaned = re.sub(r"\*\*Crawled:\*\*\s*[^\s]+", "", cleaned)
    cleaned = re.sub(r"\s+---\s+", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


class ChatHandler(SimpleHTTPRequestHandler):
    """Serve the static chat page and a JSON RAG endpoint."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format: str, *args) -> None:
        print(f"[chat] {self.address_string()} - {format % args}")

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(200, {"ok": True})
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/reset":
            self._reset()
            return
        if path != "/api/chat":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            payload = json.loads(raw_body.decode("utf-8") or "{}")
            message = str(payload.get("message", "")).strip()
            top_k = int(payload.get("top_k", 5))
            session_id = str(payload.get("session_id", "default")) or "default"
            use_memory = bool(payload.get("use_memory", True))
        except Exception:
            self._send_json(400, {"error": "Invalid JSON request"})
            return

        if not message:
            self._send_json(400, {"error": "Message is required"})
            return

        try:
            # Conversation memory: viết lại câu follow-up thành câu hỏi độc lập.
            query = condense_query(session_id, message) if use_memory else message
            # Retrieve theo câu đã rewrite, nhưng chấm relevance theo câu gốc của user
            # (câu rewrite có thể dài dòng làm loãng tín hiệu relevance).
            result = generate_with_citation(
                query, top_k=max(1, min(top_k, 8)), gate_query=message
            )
            answer = _clean_answer(result["answer"])

            if use_memory:
                add_turn(session_id, "user", message)
                add_turn(session_id, "assistant", answer)

            self._send_json(
                200,
                {
                    "answer": answer,
                    "rewritten_query": query if query != message else None,
                    "sources": [_source_summary(item) for item in result.get("sources", [])],
                },
            )
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _reset(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            reset_session(str(payload.get("session_id", "default")) or "default")
        except Exception:
            pass
        self._send_json(200, {"ok": True})


def run(host: str = HOST, port: int = PORT) -> None:
    server = ThreadingHTTPServer((host, port), ChatHandler)
    print(f"RAG chat running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
