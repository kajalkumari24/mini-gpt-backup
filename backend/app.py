"""FastAPI backend for the Mini GPT chatbot.

Run from the project folder:
    python -m uvicorn backend.app:create_app --factory --port 8000
Then open http://127.0.0.1:8000  (the chat page is served by this same app).
"""
import collections
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from model.configuration import load_generation_config
    _gen_cfg = load_generation_config()
    DEFAULTS = {
        "max_new_tokens": _gen_cfg.max_new_tokens,
        "temperature": _gen_cfg.temperature,
        "top_k": _gen_cfg.top_k,
        "repetition_penalty": _gen_cfg.repetition_penalty,
    }
except Exception:
    DEFAULTS = {"max_new_tokens": 200, "temperature": 0.8, "top_k": 40, "repetition_penalty": 1.1}

LIMITS = {"max_new_tokens": [1, 500], "temperature": [0.0, 2.0], "repetition_penalty": [1.0, 2.0]}

AVAILABLE_MODELS = {
    "chat": {
        "id": "chat",
        "name": "Chat Model (Mini-GPT v3)",
        "description": "Trained on conversational dialogue with [EOS] stop tokens (3.21M params)",
        "checkpoint": PROJECT_ROOT / "checkpoints" / "v3" / "best.pt",
        "tokenizer": PROJECT_ROOT / "checkpoints" / "tokenizer.json",
    },
    "shakespeare": {
        "id": "shakespeare",
        "name": "Shakespeare Model",
        "description": "Trained on Shakespeare plays and poetry (818K params)",
        "checkpoint": PROJECT_ROOT / "checkpoints" / "best.pt",
        "tokenizer": PROJECT_ROOT / "checkpoints" / "tokenizer.json",
    },
}


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class Settings(BaseModel):
    max_new_tokens: int = Field(DEFAULTS["max_new_tokens"], ge=1, le=500)
    temperature: float = Field(DEFAULTS["temperature"], ge=0.0, le=2.0)   # 0 = greedy
    top_k: Optional[int] = Field(DEFAULTS["top_k"], ge=1, le=1000)         # None = off
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1)                  # None = random
    repetition_penalty: Optional[float] = Field(DEFAULTS["repetition_penalty"], ge=1.0, le=2.0)


class GenerateRequest(Settings):
    prompt: str = Field(min_length=1, max_length=4000)


class ChatRequest(Settings):
    messages: List[Message] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def last_message_must_be_from_the_user(self):
        if self.messages[-1].role != "user":
            raise ValueError("The last message must come from the user")
        return self


class ModelSelectRequest(BaseModel):
    model: str


def create_app(
    service=None,
    frontend_dir=None,
    checkpoint_path=None,
    tokenizer_path=None,
    max_concurrent_generations=None,
    api_key=None,
    rate_limit_per_minute=None,
):
    """Build the app. Pass `service` to inject a model (used by the tests)."""
    frontend_dir = Path(frontend_dir or PROJECT_ROOT / "frontend")

    state = {
        "injected_service": service,
        "active_model": "chat" if AVAILABLE_MODELS["chat"]["checkpoint"].is_file() else "shakespeare",
        "services": {},
    }
    load_lock = threading.Lock()

    # Concurrency and CPU resource protection
    env_concurrency = int(os.environ.get("MAX_CONCURRENT_GENERATIONS", "4"))
    MAX_CONCURRENT_GENERATIONS = max_concurrent_generations or env_concurrency
    active_generations = 0
    gen_lock = threading.Lock()

    # Authentication
    API_KEY = api_key if api_key is not None else os.environ.get("MINI_GPT_API_KEY")

    # Rate limiting
    env_rate_limit = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "0"))  # 0 = disabled by default in dev
    RATE_LIMIT_PER_MINUTE = rate_limit_per_minute if rate_limit_per_minute is not None else env_rate_limit
    ip_request_timestamps = collections.defaultdict(list)
    ip_lock = threading.Lock()

    def check_rate_limit(request: Request):
        if RATE_LIMIT_PER_MINUTE <= 0:
            return
        client_ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()
        with ip_lock:
            cutoff = now - 60.0
            timestamps = [t for t in ip_request_timestamps[client_ip] if t > cutoff]
            if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please wait a moment before sending more requests.",
                    headers={"Retry-After": "60"},
                )
            timestamps.append(now)
            ip_request_timestamps[client_ip] = timestamps

    def verify_auth(request: Request):
        if not API_KEY:
            return
        auth_header = request.headers.get("Authorization")
        api_key_header = request.headers.get("X-API-Key")
        token = None
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        elif api_key_header:
            token = api_key_header.strip()
        if token != API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: invalid or missing API key.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def check_request_guards(request: Request):
        # 1. Payload size protection (max 1 MB)
        content_length = request.headers.get("Content-Length")
        if content_length and int(content_length) > 1_048_576:
            raise HTTPException(status_code=413, detail="Request payload exceeds 1MB limit.")
        # 2. Rate limiting
        check_rate_limit(request)
        # 3. Authentication
        verify_auth(request)

    def acquire_generation_slot():
        nonlocal active_generations
        with gen_lock:
            if active_generations >= MAX_CONCURRENT_GENERATIONS:
                raise HTTPException(
                    status_code=429,
                    detail="Too many concurrent generation requests. Please try again shortly.",
                    headers={"Retry-After": "2"},
                )
            active_generations += 1

    def release_generation_slot():
        nonlocal active_generations
        with gen_lock:
            active_generations = max(0, active_generations - 1)

    def get_service(model_id=None):
        if state["injected_service"] is not None:
            return state["injected_service"]

        target_id = model_id or state["active_model"]
        if target_id not in AVAILABLE_MODELS:
            target_id = "chat" if AVAILABLE_MODELS["chat"]["checkpoint"].is_file() else "shakespeare"

        if target_id not in state["services"]:
            with load_lock:
                if target_id not in state["services"]:
                    cfg = AVAILABLE_MODELS[target_id]
                    ckpt = checkpoint_path or cfg["checkpoint"]
                    tok = tokenizer_path or cfg["tokenizer"]
                    if not Path(ckpt).is_file():
                        raise HTTPException(
                            status_code=503,
                            detail=f"No trained model found at {ckpt}. Train it first using scripts/train.py.",
                        )
                    if not Path(tok).is_file():
                        raise HTTPException(
                            status_code=503,
                            detail=f"Tokenizer file not found at {tok}. Run scripts/train.py to generate tokenizer.json.",
                        )
                    from backend.service import ModelService
                    state["services"][target_id] = ModelService.from_checkpoint(ckpt, tok)
        return state["services"][target_id]

    @asynccontextmanager
    async def lifespan(app):
        try:
            get_service()                 # load the active model now so the first request is fast
        except Exception as error:
            print(f"[backend] Model not loaded at startup: {error}")
        yield

    app = FastAPI(title="Mini GPT API", version="1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5500",
            "http://127.0.0.1:5500",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "null",
        ],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    @app.get("/health")
    def health():
        is_loaded = state["injected_service"] is not None or len(state["services"]) > 0
        return {"status": "ok", "model_loaded": is_loaded}

    @app.get("/api/metrics")
    def metrics():
        svc = state["injected_service"] or state["services"].get(state["active_model"])
        device_str = str(getattr(svc, "device", "cpu")) if svc else "unknown"
        return {
            "active_generations": active_generations,
            "max_concurrent_generations": MAX_CONCURRENT_GENERATIONS,
            "concurrency_utilization": active_generations / max(1, MAX_CONCURRENT_GENERATIONS),
            "active_model": state["active_model"],
            "loaded_models": list(state["services"].keys()),
            "device": device_str,
        }

    @app.get("/api/models")
    def list_models():
        models_list = []
        for m_id, m_data in AVAILABLE_MODELS.items():
            if m_data["checkpoint"].is_file():
                models_list.append({
                    "id": m_id,
                    "name": m_data["name"],
                    "description": m_data["description"],
                    "active": m_id == state["active_model"],
                })
        return {"models": models_list, "active_model": state["active_model"]}

    @app.post("/api/models/select")
    def select_model(req: ModelSelectRequest, request: Request):
        check_request_guards(request)
        if req.model not in AVAILABLE_MODELS:
            raise HTTPException(status_code=400, detail=f"Model '{req.model}' is not available.")
        if not AVAILABLE_MODELS[req.model]["checkpoint"].is_file():
            raise HTTPException(status_code=404, detail=f"Checkpoint for '{req.model}' not found.")

        # Two-phase atomic model switch: only update active_model after successful load
        with load_lock:
            if req.model not in state["services"]:
                cfg = AVAILABLE_MODELS[req.model]
                from backend.service import ModelService
                new_service = ModelService.from_checkpoint(cfg["checkpoint"], cfg["tokenizer"])
                state["services"][req.model] = new_service
            state["active_model"] = req.model
        return {"status": "ok", "active_model": state["active_model"]}

    @app.get("/api/info")
    def info():
        model_info = get_service().info()
        limits = {**LIMITS, "top_k": [1, int(model_info.get("vocab_size", 1000))]}
        return {
            **model_info,
            "defaults": DEFAULTS,
            "limits": limits,
            "active_model": state["active_model"],
            "models": [
                {"id": m_id, "name": m["name"], "active": m_id == state["active_model"]}
                for m_id, m in AVAILABLE_MODELS.items()
                if m["checkpoint"].is_file()
            ],
        }

    @app.post("/api/generate")
    def generate_text(request_body: GenerateRequest, request: Request):
        check_request_guards(request)
        acquire_generation_slot()
        svc = get_service()
        try:
            return svc.complete(
                request_body.prompt,
                request_body.max_new_tokens,
                request_body.temperature,
                request_body.top_k,
                request_body.seed,
            )
        finally:
            release_generation_slot()

    @app.post("/api/chat")
    def chat(request_body: ChatRequest, request: Request):
        check_request_guards(request)
        acquire_generation_slot()
        svc = get_service()
        try:
            messages = [m.model_dump() for m in request_body.messages]
            return svc.chat(
                messages,
                request_body.max_new_tokens,
                request_body.temperature,
                request_body.top_k,
                request_body.seed,
            )
        finally:
            release_generation_slot()

    @app.post("/api/chat/stream")
    @app.post("/api/stream_chat")
    @app.post("/stream_chat")
    def chat_stream(request_body: ChatRequest, request: Request):
        check_request_guards(request)
        acquire_generation_slot()
        svc = get_service()
        messages = [m.model_dump() for m in request_body.messages]

        def event_stream():
            try:
                if hasattr(svc, "stream_chat"):
                    for chunk in svc.stream_chat(
                        messages,
                        request_body.max_new_tokens,
                        request_body.temperature,
                        request_body.top_k,
                        request_body.seed,
                        request_body.repetition_penalty or 1.1,
                    ):
                        yield f"data: {json.dumps({'delta': chunk})}\n\n"
                else:
                    res = svc.chat(
                        messages,
                        request_body.max_new_tokens,
                        request_body.temperature,
                        request_body.top_k,
                        request_body.seed,
                    )
                    reply = res.get("reply", "")
                    for char in reply:
                        yield f"data: {json.dumps({'delta': char})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                release_generation_slot()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @app.post("/chat")
    def chat_compat(request_body: dict, request: Request):
        check_request_guards(request)
        acquire_generation_slot()
        svc = get_service()
        try:
            text = request_body.get("message") or request_body.get("prompt") or request_body.get("text") or ""
            messages = request_body.get("messages")
            if not messages and text:
                messages = [{"role": "user", "content": text}]
            if not messages:
                raise HTTPException(status_code=422, detail="No message or messages provided")
            return svc.chat(messages, 200, 0.8, 40, None)
        finally:
            release_generation_slot()

    # The chat page. Mounted last so the /api routes above take priority.
    if (frontend_dir / "index.html").is_file():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    else:
        @app.get("/")
        def root_message():
            return {"status": "ok", "message": "Mini GPT API is running.", "ui": "No frontend index.html found"}

    return app


# Default app instance for standard ASGI runners (e.g. uvicorn backend.app:app)
app = create_app()
