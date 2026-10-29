import pytest

from model.tokenizer import ByteTokenizer, CharTokenizer


def test_vocab_is_sorted_and_sized():
    tok = CharTokenizer.from_text("banana")
    assert tok.vocab_size == 3
    assert list(tok.stoi) == ["a", "b", "n"]


def test_round_trip():
    text = "Plants need water"
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text


def test_empty_text_rejected():
    with pytest.raises(ValueError):
        CharTokenizer().build_vocab("")


def test_encode_before_build_rejected():
    with pytest.raises(ValueError):
        CharTokenizer().encode("abc")


def test_unknown_character_strict_and_ignore():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError):
        tok.encode("abz")
    assert tok.decode(tok.encode("abz", errors="ignore")) == "ab"


def test_unknown_token_id_rejected():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError):
        tok.decode([99])


def test_save_and_load(tmp_path):
    tok = CharTokenizer.from_text("héllo wörld")
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.stoi == tok.stoi
    assert loaded.decode(loaded.encode("héllo")) == "héllo"


def test_special_tokens_atomic(tmp_path):
    tok = CharTokenizer.from_text("abc", special_tokens=["[EOS]"])
    assert "[EOS]" in tok.stoi
    encoded = tok.encode("a[EOS]b")
    assert len(encoded) == 3
    assert tok.decode(encoded) == "a[EOS]b"

    path = tmp_path / "tok_special.json"
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.decode(loaded.encode("c[EOS]a")) == "c[EOS]a"


def test_byte_tokenizer_roundtrip(tmp_path):
    tok = ByteTokenizer(special_tokens=["[EOS]"])
    text = "def add(x, y): return x + y  # 🚀"
    encoded = tok.encode(text)
    assert tok.decode(encoded) == text

    with_eos = text + "[EOS]"
    encoded_eos = tok.encode(with_eos)
    assert tok.decode(encoded_eos) == with_eos

    path = tmp_path / "byte_tok.json"
    tok.save(path)
    loaded = ByteTokenizer.load(path)
    assert loaded.decode(loaded.encode(with_eos)) == with_eos


def test_legacy_checkpoint_tokenizer_compatibility():
    from pathlib import Path
    import torch
    from training.checkpoint import load_model_from_checkpoint
    from model.generation import generate

    root = Path(__file__).resolve().parent.parent
    tok_path = root / "checkpoints" / "tokenizer.json"
    if not tok_path.is_file():
        pytest.skip("Legacy tokenizer.json not found")

    tok = CharTokenizer.load(tok_path)
    assert tok.vocab_size == 65
    assert not tok.special_tokens

    # Verify best.pt if present
    best_pt = root / "checkpoints" / "best.pt"
    if best_pt.is_file():
        model, ckpt = load_model_from_checkpoint(best_pt, torch.device("cpu"))
        assert model.config.vocab_size == 65
        assert model.input_embedding.token_embedding.embedding.weight.shape == (65, model.config.embedding_dim)
        assert model.lm_head.weight.shape == (65, model.config.embedding_dim)
        out = generate(model, torch.tensor([[0, 1]]), max_new_tokens=5, temperature=0.0)
        assert out.shape == (1, 7)
        assert out.max().item() < 65

    # Verify v3/best.pt if present
    v3_pt = root / "checkpoints" / "v3" / "best.pt"
    if v3_pt.is_file():
        model_v3, ckpt_v3 = load_model_from_checkpoint(v3_pt, torch.device("cpu"))
        assert model_v3.config.vocab_size == 65
        assert model_v3.input_embedding.token_embedding.embedding.weight.shape == (65, model_v3.config.embedding_dim)
        out_v3 = generate(model_v3, torch.tensor([[0, 1]]), max_new_tokens=5, temperature=0.0)
        assert out_v3.shape == (1, 7)
        assert out_v3.max().item() < 65
