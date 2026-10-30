import pytest
from fastapi.testclient import TestClient

from backend.app import create_app


class StubService:
    """Stands in for the real model so the HTTP layer can be tested on its own."""

    def __init__(self):
        self.calls = []

    def info(self):
        return {"parameters": 1234, "vocab_size": 65, "context_length": 128,
                "device": "cpu", "checkpoint_step": 1000, "best_val_loss": 2.49}

    def complete(self, prompt, max_new_tokens, temperature, top_k, seed=None):
        self.calls.append(("complete", prompt, max_new_tokens, temperature, top_k, seed))
        return {"text": " and so on", "ignored_characters": []}

    def chat(self, messages, max_new_tokens, temperature, top_k, seed=None):
        self.calls.append(("chat", messages, max_new_tokens, temperature, top_k, seed))
        return {"reply": "Good morrow", "ignored_characters": ["é"]}


@pytest.fixture
def stub():
    return StubService()


@pytest.fixture
def client(stub, tmp_path):
    return TestClient(create_app(service=stub, frontend_dir=tmp_path / "no_frontend"))


def valid_chat(**overrides):
    body = {"messages": [{"role": "user", "content": "Hello"}]}
    body.update(overrides)
    return body


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_info_includes_model_defaults_and_limits(client):
    data = client.get("/api/info").json()
    assert data["parameters"] == 1234
    assert data["defaults"]["temperature"] == 0.8
    assert data["limits"]["top_k"] == [1, 65]
    assert data["limits"]["max_new_tokens"] == [1, 500]


def test_generate_passes_settings_through(client, stub):
    response = client.post("/api/generate", json={
        "prompt": "ROMEO:", "max_new_tokens": 50, "temperature": 0.5, "top_k": 10, "seed": 3})
    assert response.status_code == 200
    assert response.json()["text"] == " and so on"
    assert stub.calls[-1] == ("complete", "ROMEO:", 50, 0.5, 10, 3)


def test_chat_returns_reply_and_ignored_characters(client, stub):
    response = client.post("/api/chat", json=valid_chat())
    assert response.status_code == 200
    assert response.json() == {"reply": "Good morrow", "ignored_characters": ["é"]}
    kind, messages, *_ = stub.calls[-1]
    assert kind == "chat" and messages == [{"role": "user", "content": "Hello"}]


def test_defaults_are_applied_and_top_k_can_be_turned_off(client, stub):
    client.post("/api/chat", json=valid_chat())
    assert stub.calls[-1][2:] == (200, 0.8, 40, None)
    client.post("/api/chat", json=valid_chat(top_k=None))
    assert stub.calls[-1][4] is None


@pytest.mark.parametrize("settings", [
    {"temperature": 5},
    {"temperature": -0.1},
    {"max_new_tokens": 0},
    {"max_new_tokens": 501},
    {"top_k": 0},
    {"seed": -1},
])
def test_out_of_range_settings_are_rejected(client, settings):
    assert client.post("/api/chat", json=valid_chat(**settings)).status_code == 422


@pytest.mark.parametrize("body", [
    {"messages": []},
    {"messages": [{"role": "user", "content": ""}]},
    {"messages": [{"role": "robot", "content": "hi"}]},
    {"messages": [{"role": "user", "content": "x" * 2001}]},
    {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]},
    {},
])
def test_bad_chat_messages_are_rejected(client, body):
    assert client.post("/api/chat", json=body).status_code == 422


def test_empty_prompt_is_rejected(client):
    assert client.post("/api/generate", json={"prompt": ""}).status_code == 422


def test_missing_checkpoint_gives_a_clear_503(tmp_path):
    app = create_app(checkpoint_path=tmp_path / "missing.pt", frontend_dir=tmp_path)
    with TestClient(app) as client:            # also runs the startup step, which must not crash
        assert client.get("/api/health").json() == {"status": "ok", "model_loaded": False}
        response = client.get("/api/info")
        assert response.status_code == 503
        assert "Train it first" in response.json()["detail"]


def test_chat_page_is_served_and_api_still_wins(stub, tmp_path):
    (tmp_path / "index.html").write_text("<h1>chat page</h1>", encoding="utf-8")
    client = TestClient(create_app(service=stub, frontend_dir=tmp_path))
    assert "chat page" in client.get("/").text
    assert client.get("/api/health").status_code == 200


def test_cors_allows_the_local_dev_server_only(client):
    ok = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    other = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_atomic_model_switching_preserves_active_model_on_failure(client):
    info_before = client.get("/api/info").json()
    active_before = info_before["active_model"]

    # Attempt to switch to an invalid model
    res_bad = client.post("/api/models/select", json={"model": "non_existent"})
    assert res_bad.status_code == 400

    # Ensure active model remains completely unchanged
    info_after = client.get("/api/info").json()
    assert info_after["active_model"] == active_before
