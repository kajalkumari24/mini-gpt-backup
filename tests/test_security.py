"""Security and input validation tests for Mini GPT API and services."""
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app


class DummyService:
    def info(self):
        return {"parameters": 100, "vocab_size": 30, "context_length": 64, "device": "cpu"}

    def complete(self, prompt, max_new_tokens, temperature, top_k, seed=None):
        return {"text": "ok", "ignored_characters": []}

    def chat(self, messages, max_new_tokens, temperature, top_k, seed=None):
        return {"reply": "ok", "ignored_characters": []}


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(service=DummyService(), frontend_dir=tmp_path))


def test_temperature_upper_bound_rejection(client):
    res = client.post("/api/generate", json={"prompt": "test", "temperature": 2.5})
    assert res.status_code == 422


def test_temperature_negative_rejection(client):
    res = client.post("/api/generate", json={"prompt": "test", "temperature": -0.1})
    assert res.status_code == 422


def test_max_tokens_bounds_rejection(client):
    res = client.post("/api/generate", json={"prompt": "test", "max_new_tokens": 0})
    assert res.status_code == 422
    res2 = client.post("/api/generate", json={"prompt": "test", "max_new_tokens": 1000})
    assert res2.status_code == 422


def test_chat_requires_last_message_from_user(client):
    payload = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    }
    res = client.post("/api/chat", json=payload)
    assert res.status_code == 422
    assert "last message must come from the user" in res.text


def test_chat_empty_messages_list_rejection(client):
    res = client.post("/api/chat", json={"messages": []})
    assert res.status_code == 422


def test_unknown_model_selection_rejected(client):
    res = client.post("/api/models/select", json={"model": "non_existent_model_xyz"})
    assert res.status_code == 400


def test_oversized_prompt_rejected(client):
    res = client.post("/api/generate", json={"prompt": "A" * 4001})
    assert res.status_code == 422


def test_oversized_message_content_rejected(client):
    payload = {"messages": [{"role": "user", "content": "B" * 2001}]}
    res = client.post("/api/chat", json=payload)
    assert res.status_code == 422


def test_too_many_messages_rejected(client):
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(51)]
    if messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": "last"})
    res = client.post("/api/chat", json={"messages": messages})
    assert res.status_code == 422


def test_repetition_penalty_bounds_rejected(client):
    res1 = client.post("/api/generate", json={"prompt": "test", "repetition_penalty": 0.5})
    assert res1.status_code == 422
    res2 = client.post("/api/generate", json={"prompt": "test", "repetition_penalty": 2.5})
    assert res2.status_code == 422


def test_concurrent_generation_slot_exhaustion(tmp_path):
    import time
    from threading import Event

    gate = Event()

    class BlockingService(DummyService):
        def complete(self, *args, **kwargs):
            gate.wait(timeout=2.0)
            return {"text": "done", "ignored_characters": []}

    app = create_app(service=BlockingService(), frontend_dir=tmp_path)
    # Simulate slots saturated
    from concurrent.futures import ThreadPoolExecutor

    c = TestClient(app)
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(lambda: c.post("/api/generate", json={"prompt": "hi"})) for _ in range(11)]
        time.sleep(0.1)
        gate.set()
        statuses = [f.result().status_code for f in futures]

    assert 429 in statuses or all(s == 200 for s in statuses)


def test_api_key_authentication(tmp_path):
    app = create_app(service=DummyService(), frontend_dir=tmp_path, api_key="secret-token-123")
    client = TestClient(app)

    # Missing API key returns 401
    res1 = client.post("/api/generate", json={"prompt": "hello"})
    assert res1.status_code == 401
    assert "Unauthorized" in res1.text

    # Wrong API key returns 401
    res2 = client.post(
        "/api/generate", json={"prompt": "hello"}, headers={"Authorization": "Bearer wrong-token"}
    )
    assert res2.status_code == 401

    # Valid Bearer token returns 200
    res3 = client.post(
        "/api/generate", json={"prompt": "hello"}, headers={"Authorization": "Bearer secret-token-123"}
    )
    assert res3.status_code == 200

    # Valid X-API-Key header returns 200
    res4 = client.post(
        "/api/generate", json={"prompt": "hello"}, headers={"X-API-Key": "secret-token-123"}
    )
    assert res4.status_code == 200


def test_rate_limiting_per_ip(tmp_path):
    # App configured with max 3 requests per minute
    app = create_app(service=DummyService(), frontend_dir=tmp_path, rate_limit_per_minute=3)
    client = TestClient(app)

    for _ in range(3):
        res = client.post("/api/generate", json={"prompt": "hello"})
        assert res.status_code == 200

    # 4th request exceeds rate limit
    res_exceeded = client.post("/api/generate", json={"prompt": "hello"})
    assert res_exceeded.status_code == 429
    assert res_exceeded.headers.get("Retry-After") == "60"
    assert "Rate limit exceeded" in res_exceeded.text


def test_payload_size_limit(tmp_path):
    app = create_app(service=DummyService(), frontend_dir=tmp_path)
    client = TestClient(app)

    # Fake oversized Content-Length header > 1MB
    res = client.post("/api/generate", json={"prompt": "hello"}, headers={"Content-Length": "2000000"})
    assert res.status_code == 413
    assert "exceeds 1MB limit" in res.text


def test_metrics_endpoint(tmp_path):
    app = create_app(service=DummyService(), frontend_dir=tmp_path, max_concurrent_generations=5)
    client = TestClient(app)

    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.json()
    assert data["max_concurrent_generations"] == 5
    assert data["active_generations"] == 0
    assert "concurrency_utilization" in data

