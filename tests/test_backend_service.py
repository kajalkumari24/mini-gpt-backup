import pytest
import torch

from backend.service import ModelService
from model.configuration import ModelConfig, TrainingConfig
from model.gpt import GPT
from model.tokenizer import CharTokenizer
from training.checkpoint import save_checkpoint

ALPHABET = "YOUBT:\n abcdefgh"


def make_service():
    tokenizer = CharTokenizer.from_text(ALPHABET)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, context_length=32, embedding_dim=32,
                      num_heads=4, num_layers=2, dropout=0.0)
    return ModelService(GPT(cfg), tokenizer, torch.device("cpu"), {"checkpoint_step": 5})


def test_complete_returns_only_the_new_text():
    service = make_service()
    result = service.complete("abc", max_new_tokens=12, temperature=1.0, top_k=5)
    assert len(result["text"]) == 12
    assert result["ignored_characters"] == []


def test_unknown_characters_are_reported_not_fatal():
    service = make_service()
    result = service.complete("ab#é", max_new_tokens=3, temperature=1.0, top_k=None)
    assert result["ignored_characters"] == ["#", "é"]
    assert len(result["text"]) == 3


def test_same_seed_gives_the_same_text():
    service = make_service()
    a = service.complete("abc", 20, 1.0, None, seed=11)["text"]
    b = service.complete("abc", 20, 1.0, None, seed=11)["text"]
    assert a == b


def test_greedy_is_repeatable_without_a_seed():
    service = make_service()
    assert (service.complete("abc", 15, 0.0, None)["text"]
            == service.complete("abc", 15, 0.0, None)["text"])


def test_chat_reply_is_a_single_speech():
    service = make_service()
    messages = [{"role": "user", "content": "abc é"}]
    result = service.chat(messages, max_new_tokens=40, temperature=1.0, top_k=None, seed=1)
    assert isinstance(result["reply"], str)
    assert "\n\n" not in result["reply"]
    assert result["ignored_characters"] == ["é"]


def test_prompt_with_no_known_characters_still_generates():
    service = make_service()
    assert len(service.complete("###", 4, 1.0, None)["text"]) == 4


def test_info_reports_model_facts():
    service = make_service()
    info = service.info()
    assert info["parameters"] == service.model.num_parameters()
    assert info["context_length"] == 32 and info["checkpoint_step"] == 5


def test_mismatched_tokenizer_and_model_rejected():
    tokenizer = CharTokenizer.from_text("abc")
    cfg = ModelConfig(vocab_size=10, context_length=16, embedding_dim=32,
                      num_heads=4, num_layers=1, dropout=0.0)
    with pytest.raises(ValueError):
        ModelService(GPT(cfg), tokenizer, torch.device("cpu"))


def test_loading_from_a_real_checkpoint(tmp_path):
    tokenizer = CharTokenizer.from_text(ALPHABET)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, context_length=16, embedding_dim=32,
                      num_heads=4, num_layers=1, dropout=0.0)
    model = GPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters())
    tcfg = TrainingConfig(learning_rate=0.001, batch_size=2, max_steps=10, eval_interval=5,
                          random_seed=0, train_split=0.9)
    save_checkpoint(tmp_path / "best.pt", model, optimizer, 42, 1.25, cfg, tcfg, [])
    tokenizer.save(tmp_path / "tokenizer.json")

    service = ModelService.from_checkpoint(tmp_path / "best.pt", tmp_path / "tokenizer.json", "cpu")
    info = service.info()
    assert info["checkpoint_step"] == 42 and info["best_val_loss"] == 1.25
    assert info["parameters"] == model.num_parameters()


def test_stream_chat_yields_chunks_and_stops():
    service = make_service()
    messages = [{"role": "user", "content": "abc"}]
    chunks = list(service.stream_chat(messages, max_new_tokens=20, temperature=1.0, top_k=None, seed=42))
    assert len(chunks) > 0
    full_stream = "".join(chunks)
    assert "[EOS]" not in full_stream
    assert "\n\n" not in full_stream


def test_stream_chat_matches_chat_greedy():
    service = make_service()
    messages = [{"role": "user", "content": "abc"}]
    chat_reply = service.chat(messages, max_new_tokens=25, temperature=0.0, top_k=None)["reply"]
    stream_chunks = list(service.stream_chat(messages, max_new_tokens=25, temperature=0.0, top_k=None))
    stream_reply = "".join(stream_chunks)
    assert stream_reply == chat_reply


def test_stream_chat_client_disconnect_releases_lock():
    service = make_service()
    messages = [{"role": "user", "content": "abc"}]
    gen = service.stream_chat(messages, max_new_tokens=30, temperature=1.0, top_k=None)
    first_chunk = next(gen)
    assert len(first_chunk) > 0
    # Client disconnect triggers close()
    gen.close()
    assert not service._lock.locked()
    # Next call can acquire lock without deadlock
    res = service.complete("abc", 5, 0.0, None)
    assert len(res["text"]) == 5


def test_eos_detection_relies_on_special_tokens_not_brackets():
    """Check 1: Vocabulary containing '[' and ']' without special token must NOT trigger EOS mode."""
    # Tokenizer has brackets in character text but no atomic special tokens
    tok_brackets_only = CharTokenizer.from_text("abcdef[]EOS")
    cfg = ModelConfig(vocab_size=tok_brackets_only.vocab_size, context_length=16, embedding_dim=16, num_heads=2, num_layers=1, dropout=0.0)
    service_no_eos = ModelService(GPT(cfg), tok_brackets_only, torch.device("cpu"))
    assert not service_no_eos.supports_eos(), "Brackets alone must not enable EOS support"

    # Tokenizer explicitly defines atomic special token [EOS]
    tok_with_special = CharTokenizer.from_text("abcdef", special_tokens=["[EOS]"])
    cfg2 = ModelConfig(vocab_size=tok_with_special.vocab_size, context_length=16, embedding_dim=16, num_heads=2, num_layers=1, dropout=0.0)
    service_with_eos = ModelService(GPT(cfg2), tok_with_special, torch.device("cpu"))
    assert service_with_eos.supports_eos(), "Explicit [EOS] special token must enable EOS support"

    # ByteTokenizer with [EOS]
    from model.tokenizer import ByteTokenizer
    byte_tok = ByteTokenizer(special_tokens=["[EOS]"])
    cfg3 = ModelConfig(vocab_size=byte_tok.vocab_size, context_length=16, embedding_dim=16, num_heads=2, num_layers=1, dropout=0.0)
    service_byte = ModelService(GPT(cfg3), byte_tok, torch.device("cpu"))
    assert service_byte.supports_eos(), "ByteTokenizer with [EOS] must enable EOS support"


def test_user_mandated_streaming_chunk_split_stop_markers(monkeypatch):
    """Requirement 3: Split a stop marker across several chunks and verify that the streamed result matches the non-streaming result."""
    from backend.prompt import extract_reply
    import backend.service

    # Create service with explicit [EOS] special token support
    tok = CharTokenizer.from_text(ALPHABET, special_tokens=["[EOS]"])
    cfg = ModelConfig(
        vocab_size=tok.vocab_size, context_length=32, embedding_dim=32, num_heads=4, num_layers=2, dropout=0.0
    )
    service = ModelService(GPT(cfg), tok, torch.device("cpu"), {"checkpoint_step": 1})
    assert service.supports_eos()

    # Test Case 3A: [EOS] split across chunk boundaries: ["[E", "OS]"]
    eos_chunks = ["First paragraph.\n\n", "Second paragraph.", "[E", "OS]"]
    full_text = "".join(eos_chunks)
    non_streaming_result = extract_reply(full_text, eos_mode=True)

    # Monkeypatch generate_stream to yield our specific chunks
    monkeypatch.setattr(
        backend.service,
        "generate_stream",
        lambda *args, **kwargs: iter(eos_chunks),
    )
    streamed_chunks = list(
        service.stream_chat([{"role": "user", "content": "abc"}], max_new_tokens=100, temperature=0.0, top_k=None)
    )
    streamed_result = "".join(streamed_chunks)
    assert streamed_result == non_streaming_result
    assert streamed_result == "First paragraph.\n\nSecond paragraph."

    # Test Case 3B: General speaker label split across chunk boundaries: ["JU", "LIET: Hi."]
    speaker_chunks = ["Hello.\n\n", "JU", "LIET: Hi."]
    full_speaker_text = "ROMEO: " + "".join(speaker_chunks)
    non_streaming_speaker = extract_reply(full_speaker_text, eos_mode=True)

    monkeypatch.setattr(
        backend.service,
        "generate_stream",
        lambda *args, **kwargs: iter(speaker_chunks),
    )
    streamed_speaker_chunks = list(
        service.stream_chat([{"role": "user", "content": "abc"}], max_new_tokens=100, temperature=0.0, top_k=None)
    )
    streamed_speaker_result = "".join(streamed_speaker_chunks)
    assert streamed_speaker_result == non_streaming_speaker
    assert streamed_speaker_result == "Hello."

    # Test Case 3C: Character-by-character streaming of [EOS]
    char_by_char = list("First paragraph.\n\nSecond paragraph.[EOS]")
    monkeypatch.setattr(
        backend.service,
        "generate_stream",
        lambda *args, **kwargs: iter(char_by_char),
    )
    streamed_char_chunks = list(
        service.stream_chat([{"role": "user", "content": "abc"}], max_new_tokens=100, temperature=0.0, top_k=None)
    )
    assert "".join(streamed_char_chunks) == non_streaming_result


def test_streaming_parity_every_possible_chunk_split_both_modes(monkeypatch):
    """Check 3: Verify streaming parity end-to-end in both checkpoint modes at EVERY possible chunk split."""
    from backend.prompt import extract_reply
    import backend.service

    # --- Mode 1: Conversation / EOS mode ---
    tok_eos = CharTokenizer.from_text(ALPHABET, special_tokens=["[EOS]"])
    cfg_eos = ModelConfig(vocab_size=tok_eos.vocab_size, context_length=32, embedding_dim=32, num_heads=4, num_layers=2, dropout=0.0)
    service_eos = ModelService(GPT(cfg_eos), tok_eos, torch.device("cpu"))
    assert service_eos.supports_eos()

    body = "First paragraph.\n\nSecond paragraph."
    eos_marker = "[EOS]"
    expected_eos_reply = extract_reply(body + eos_marker, eos_mode=True)
    assert expected_eos_reply == "First paragraph.\n\nSecond paragraph."

    # Test [EOS] split at EVERY possible boundary index k: 1 to len(eos_marker)-1
    for k in range(1, len(eos_marker)):
        chunk1 = body + eos_marker[:k]
        chunk2 = eos_marker[k:]
        gen_fn = (lambda c1, c2: lambda *args, **kwargs: iter([c1, c2]))(chunk1, chunk2)
        monkeypatch.setattr(backend.service, "generate_stream", gen_fn)
        streamed = "".join(service_eos.stream_chat([{"role": "user", "content": "hi"}], max_new_tokens=50, temperature=0.0, top_k=None))
        assert streamed == expected_eos_reply, f"Mismatch on EOS split index {k}: '{streamed}' vs '{expected_eos_reply}'"

    # Test speaker label split at EVERY possible boundary index in Conversation mode
    speaker_prefix = "Hello.\n\n"
    speaker_label = "JULIET: Good day."
    expected_speaker_reply = extract_reply("ROMEO: " + speaker_prefix + speaker_label, eos_mode=True)
    assert expected_speaker_reply == "Hello."

    for k in range(1, len(speaker_label)):
        chunk1 = speaker_prefix + speaker_label[:k]
        chunk2 = speaker_label[k:]
        gen_fn = (lambda c1, c2: lambda *args, **kwargs: iter([c1, c2]))(chunk1, chunk2)
        monkeypatch.setattr(backend.service, "generate_stream", gen_fn)
        streamed = "".join(service_eos.stream_chat([{"role": "user", "content": "hi"}], max_new_tokens=50, temperature=0.0, top_k=None))
        assert streamed == expected_speaker_reply, f"Mismatch on speaker split index {k}: '{streamed}' vs '{expected_speaker_reply}'"

    # --- Mode 2: Shakespeare / Script mode (no EOS) ---
    tok_no_eos = CharTokenizer.from_text(ALPHABET + "[]")
    cfg_no_eos = ModelConfig(vocab_size=tok_no_eos.vocab_size, context_length=32, embedding_dim=32, num_heads=4, num_layers=2, dropout=0.0)
    service_no_eos = ModelService(GPT(cfg_no_eos), tok_no_eos, torch.device("cpu"))
    assert not service_no_eos.supports_eos()

    sh_body = "Good sir, well met."
    sh_turn_end = "\n\nMore speech follows."
    expected_sh_reply = extract_reply(sh_body + sh_turn_end, eos_mode=False)
    assert expected_sh_reply == "Good sir, well met."

    # Test blank line split at every possible boundary in Shakespeare mode
    for k in range(1, len("\n\n")):
        chunk1 = sh_body + "\n\n"[:k]
        chunk2 = "\n\n"[k:] + "More speech follows."
        gen_fn = (lambda c1, c2: lambda *args, **kwargs: iter([c1, c2]))(chunk1, chunk2)
        monkeypatch.setattr(backend.service, "generate_stream", gen_fn)
        streamed = "".join(service_no_eos.stream_chat([{"role": "user", "content": "hi"}], max_new_tokens=50, temperature=0.0, top_k=None))
        assert streamed == expected_sh_reply, f"Mismatch on Shakespeare split index {k}: '{streamed}' vs '{expected_sh_reply}'"
