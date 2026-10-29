# Mini-GPT: Architecture, Training, Evaluation & Web Chatbot

Mini-GPT is a clean, dependency-minimal Generative Pre-trained Transformer (GPT) language model built with PyTorch and served with FastAPI. It includes autoregressive generation, multi-turn conversational dialogue formatting, tokenizers (Character-level with atomic special tokens and Byte-level), evaluation metrics, benchmarking utilities, and an interactive single-page web UI featuring an animated SVG companion character.

---

## 1. Project Directory Structure

```text
mini-gpt/
├── backend/
│   ├── app.py                # FastAPI application, routing, SSE streaming, static mounting
│   ├── prompt.py             # Script dialogue formatting, clean reply extraction & stop markers
│   ├── service.py            # Thread-safe model serving, text completion, chat & streaming prefix buffering
│   └── requirements.txt      # Backend dependencies declaration
├── checkpoints/
│   ├── best.pt               # Shakespeare model checkpoint (818K params, 65-char vocab)
│   ├── last.pt               # Shakespeare model last step checkpoint
│   ├── tokenizer.json        # 65-character Shakespeare vocabulary
│   └── v3/
│       ├── best.pt           # Chat v3 checkpoint (3.21M params, conversational weights)
│       └── last.pt           # Chat v3 last step checkpoint
├── configs/
│   ├── model.yaml            # Context length, embedding dimensions, heads, layers, dropout
│   ├── training.yaml         # Learning rate, batch size, steps, warmup, weight decay
│   └── generation.yaml       # Default generation parameters (tokens, temperature, top-k, penalty)
├── data/
│   ├── mini_gpt_conversations.txt  # Dialogue dataset with YOU / BOT turns and [EOS] tokens
│   └── input_shakespeare.txt       # The complete works of Shakespeare corpus
├── evaluation/
│   ├── evaluate.py           # Full evaluation loss, unigram baseline, perplexity, bits/char
│   ├── metrics.py            # Diversity metrics (distinct n-grams), repetition rate, loss metrics
│   ├── compare_models.py     # Multi-model side-by-side comparison harness
│   ├── performance.py        # Throughput (tokens/sec) and latency benchmark utilities
│   ├── generate_samples.py   # Qualitative benchmark sample generator
│   └── report.py             # Markdown summary table formatter
├── frontend/
│   └── index.html            # Single-page chat interface with animated SVG companion (16 moods)
├── model/
│   ├── attention.py          # Multi-head causal self-attention with scaled dot-product
│   ├── configuration.py      # Dataclasses and YAML config loaders (Model, Training, Generation)
│   ├── embeddings.py         # Token and positional embedding layers
│   ├── generation.py         # Autoregressive generation (greedy, temperature, top-k, repetition penalty)
│   ├── gpt.py                # Full GPT transformer architecture with weight tying
│   ├── position.py           # Learnable positional embeddings
│   ├── tokenizer.py          # CharTokenizer (atomic special tokens) & ByteTokenizer (UTF-8 bytes)
│   └── transformer_block.py  # Transformer block (LayerNorm, CausalSelfAttention, FeedForward, residuals)
├── scripts/
│   ├── train.py              # CLI training script with dataset validation and step/device overrides
│   ├── evaluate.py           # CLI checkpoint evaluation against datasets with unigram comparison
│   ├── benchmark.py          # CLI latency, throughput, and memory benchmarking
│   └── generate.py           # CLI single-prompt text generation
├── tests/                    # 133 automated unit and integration tests (Pytest)
├── pyproject.toml            # Package metadata and build configuration
└── requirements.txt          # Root Python dependencies
```

---

## 2. Quickstart & Installation

### Requirements
- Python 3.9+
- PyTorch (`torch`)
- FastAPI (`fastapi`)
- Uvicorn (`uvicorn`)
- PyYAML (`pyyaml`)
- Pydantic (`pydantic`)
- Pytest & HTTPX (for running test suites)

Install all dependencies:
```bash
pip install -r requirements.txt
pip install pytest httpx
```

---

## 3. Training the Model

Train the model using `scripts/train.py`. The script validates that the dataset file exists and is not empty. By default, it targets `data/mini_gpt_conversations.txt`.

### Basic Training Commands

```bash
# Train using default dataset (mini_gpt_conversations.txt)
python scripts/train.py

# Train on conversation dataset with custom max steps and CPU device
python scripts/train.py --data data/mini_gpt_conversations.txt --max-steps 1000 --device cpu

# Train on Shakespeare corpus
python scripts/train.py --data data/input_shakespeare.txt --max-steps 3000 --device auto

# Resume training from an existing checkpoint
python scripts/train.py --resume checkpoints/last.pt --max-steps 4000
```

### Supported CLI Arguments
- `--data`: Path to training text file (default: `data/mini_gpt_conversations.txt`).
- `--checkpoint_dir`: Output directory for checkpoints and `tokenizer.json` (default: `checkpoints`).
- `--max-steps`, `--max_steps`: Total training steps (overrides `configs/training.yaml`).
- `--device`: Target device (`auto`, `cpu`, `cuda`).
- `--batch-size`: Batch size per step.
- `--lr`, `--learning-rate`: Peak learning rate for AdamW optimizer.
- `--eval-interval`: Interval for evaluating loss and saving best checkpoint.
- `--resume`: Path to checkpoint `.pt` to resume state from.

---

## 4. Evaluation and Benchmarking

### Model Evaluation
Evaluate a trained checkpoint on full cross-entropy loss, perplexity, bits per character, comparison against a zero-context unigram baseline, and generation speed:

```bash
# Evaluate Chat Model v3 on conversation dataset
python scripts/evaluate.py --checkpoint checkpoints/v3/best.pt --tokenizer checkpoints/tokenizer.json --data data/mini_gpt_conversations.txt

# Evaluate Shakespeare Model
python scripts/evaluate.py --checkpoint checkpoints/best.pt --tokenizer checkpoints/tokenizer.json --data data/input_shakespeare.txt
```

### Performance Benchmarking
Benchmark token generation latency, throughput (tokens per second), and peak GPU memory across multiple runs:

```bash
python scripts/benchmark.py --checkpoint checkpoints/v3/best.pt --num-tokens 200 --runs 3
```

---

## 5. Tokenizer & Stop Token Architecture

### Tokenizers (`model/tokenizer.py`)
1. **`CharTokenizer`**:
   - Character-level tokenization mapping unique text characters to integer IDs.
   - Supports atomic special tokens (e.g. `[EOS]`), ensuring markers are treated as discrete tokens rather than individual bracket and letter characters.
   - Preserves 100% backward compatibility with legacy 65-character vocabulary checkpoints (`checkpoints/tokenizer.json`).
2. **`ByteTokenizer`**:
   - Byte-level tokenization mapping raw UTF-8 bytes (0–255) to token IDs, plus optional special tokens.
   - Eliminates out-of-vocabulary character issues for arbitrary code symbols, parentheses, underscores, and unicode.

### Stopping Condition & End-of-Answer Detection
- **Root Cause Resolution**: The Shakespeare checkpoint vocabulary (65 characters) does not contain `[` or `]`. When trained on dialogues, the model emitted `EOS`, blank lines (`\n\n`), or next-speaker labels (`\nYOU:`) rather than `[EOS]`.
- **Unified Stop Conditions**: Both batch (`service.chat`) and streaming (`service.stream_chat`) recognize:
  1. `[EOS]` (bracketed marker)
  2. `EOS` (standalone marker at word or line boundaries)
  3. `\n\n` (double newline turn separator)
  4. Speaker labels matching `\n[A-Z][A-Z ]{0,30}:` (e.g., `\nYOU:`, `\nBOT:`)
- **Streaming Prefix Buffering**: `service.stream_chat` uses a rolling prefix buffer that holds characters that could form the prefix of a stop sequence. Characters are only yielded once confirmed safe, preventing character dropping or premature stop token leakage.

---

## 6. Running the Web Chatbot

Launch the FastAPI application:
```bash
python -m uvicorn backend.app:create_app --factory --port 8000
```
Then navigate in your web browser to:
```text
http://127.0.0.1:8000
```

### Features
- **Interactive SVG Companion**: Animated character with 16 distinct emotional states (happy, thinking, laugh, love, cry, etc.) responding dynamically to chat interactions.
- **Model Switching**: Switch between Chat Model v3 and Shakespeare Model via `/api/models/select`.
- **Real-Time Streaming**: Token-by-token streaming via Server-Sent Events (SSE) at `/api/chat/stream`.
- **Parameter Controls**: Adjustable temperature, top-k sampling, maximum tokens, and repetition penalty configured in `configs/generation.yaml`.

### API Endpoints
- `GET /`: Serves `frontend/index.html` (or status JSON if headless).
- `GET /api/health`: System status and model loading state.
- `GET /api/info`: Model metadata, layer configurations, parameter counts, and parameter limits.
- `GET /api/models`: List of registered checkpoints and currently active model.
- `POST /api/models/select`: Hot-swap active model checkpoint.
- `POST /api/generate`: Single prompt completion.
- `POST /api/chat`: Multi-turn conversational dialogue completion.
- `POST /api/chat/stream`: Real-time SSE streaming response.
- `POST /chat`: Compatibility endpoint for simple message queries.

---

## 7. Running the Automated Tests

The test suite contains **133 automated tests** covering attention, embeddings, transformer blocks, gpt architecture, autoregressive generation, character & byte tokenizers, training loops, evaluation metrics, backend endpoints, and security boundaries.

Run all tests:
```bash
pytest
```
To run tests with detailed verbosity:
```bash
pytest -v
```
