"""Integration tests for web chat interface and API routes."""
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from backend.app import create_app


class MockService:
    def info(self):
        return {
            "parameters": 1000,
            "vocab_size": 65,
            "context_length": 128,
            "device": "cpu",
            "checkpoint_step": 100,
            "best_val_loss": 1.5,
        }

    def complete(self, prompt, max_new_tokens, temperature, top_k, seed=None):
        return {"text": " completion", "ignored_characters": []}

    def chat(self, messages, max_new_tokens, temperature, top_k, seed=None):
        return {"reply": "Hi there!", "ignored_characters": []}

    def stream_chat(self, messages, max_new_tokens, temperature, top_k, seed=None, repetition_penalty=1.1):
        yield "Hi"
        yield " there"
        yield "!"


@pytest.fixture
def mock_service():
    return MockService()


def test_frontend_serving_index_html(tmp_path, mock_service):
    # Create a dummy index.html in tmp_path
    html_content = "<!DOCTYPE html><html><body><div id='chat'>mini-gpt companion</div></body></html>"
    (tmp_path / "index.html").write_text(html_content, encoding="utf-8")

    app = create_app(service=mock_service, frontend_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "mini-gpt companion" in response.text
    assert "text/html" in response.headers.get("content-type", "")


def test_root_fallback_when_frontend_missing(tmp_path, mock_service):
    empty_dir = tmp_path / "empty_frontend"
    empty_dir.mkdir()

    app = create_app(service=mock_service, frontend_dir=empty_dir)
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "No frontend index.html found" in data["ui"]


def test_streaming_endpoint_sse(mock_service, tmp_path):
    app = create_app(service=mock_service, frontend_dir=tmp_path)
    client = TestClient(app)

    payload = {"messages": [{"role": "user", "content": "Hello!"}]}
    response = client.post("/api/chat/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    body = response.text
    assert "data: [DONE]" in body
    assert '"delta": "Hi"' in body


def test_chat_compat_endpoint(mock_service, tmp_path):
    app = create_app(service=mock_service, frontend_dir=tmp_path)
    client = TestClient(app)

    res = client.post("/chat", json={"message": "Hey"})
    assert res.status_code == 200
    assert res.json()["reply"] == "Hi there!"

    res2 = client.post("/chat", json={"prompt": "Hey prompt"})
    assert res2.status_code == 200

    bad = client.post("/chat", json={})
    assert bad.status_code == 422

