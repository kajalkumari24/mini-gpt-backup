from types import SimpleNamespace

import pytest
import torch

from model.configuration import ModelConfig
from model.generation import generate, generate_text
from model.gpt import GPT
from model.tokenizer import CharTokenizer
from training.data import get_batch


def make_model(context_length=16):
    cfg = ModelConfig(vocab_size=8, context_length=context_length, embedding_dim=32,
                      num_heads=4, num_layers=2, dropout=0.0)
    return GPT(cfg)


class FixedLogitsModel(torch.nn.Module):
    """A stand-in 'model' whose scores never change, so sampling can be checked exactly."""

    def __init__(self, logits):
        super().__init__()
        self.config = SimpleNamespace(context_length=8)
        self.register_buffer("fixed", torch.tensor(logits, dtype=torch.float32))

    def forward(self, token_ids, targets=None):
        b, t = token_ids.shape
        return self.fixed.expand(b, t, -1).clone(), None


LOGITS = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0]


def sample_one_token(model, rows=500, **kwargs):
    start = torch.zeros((rows, 1), dtype=torch.long)
    return generate(model, start, 1, **kwargs)[:, 1]


def test_length_and_prompt_are_preserved():
    prompt = torch.randint(0, 8, (2, 5))
    out = generate(make_model(), prompt, 7)
    assert out.shape == (2, 12)
    assert torch.equal(out[:, :5], prompt)


def test_generated_ids_are_valid_tokens():
    out = generate(make_model(), torch.zeros((3, 1), dtype=torch.long), 20)
    assert out.min() >= 0 and out.max() < 8


def test_greedy_is_deterministic():
    model = make_model()
    start = torch.tensor([[1, 2, 3]])
    assert torch.equal(generate(model, start, 10, temperature=0),
                       generate(model, start, 10, temperature=0))


def test_same_seed_gives_same_sample():
    model = make_model()
    start = torch.tensor([[1, 2, 3]])
    torch.manual_seed(1)
    a = generate(model, start, 15, temperature=1.0)
    torch.manual_seed(1)
    b = generate(model, start, 15, temperature=1.0)
    assert torch.equal(a, b)


def test_top_k_one_is_greedy():
    model = make_model()
    start = torch.tensor([[1, 2, 3]])
    greedy = generate(model, start, 10, temperature=0)
    assert torch.equal(generate(model, start, 10, temperature=1.0, top_k=1), greedy)


def test_top_k_only_allows_the_best_k_tokens():
    model = FixedLogitsModel(LOGITS)
    assert set(sample_one_token(model, top_k=2, temperature=1.0).tolist()) <= {0, 1}
    assert set(sample_one_token(model, top_k=None, temperature=1.0).tolist()) - {0, 1}   # others appear


def test_low_temperature_behaves_like_greedy():
    assert (sample_one_token(FixedLogitsModel(LOGITS), temperature=0.01) == 0).all()


def test_high_temperature_spreads_choices():
    seen = set(sample_one_token(FixedLogitsModel(LOGITS), temperature=100.0).tolist())
    assert len(seen) >= 6


def test_can_generate_past_the_context_window():
    out = generate(make_model(context_length=8), torch.tensor([[1, 2, 3]]), 20)
    assert out.shape == (1, 23)


def test_model_mode_is_restored():
    model = make_model().train()
    generate(model, torch.tensor([[1]]), 3)
    assert model.training


def test_bad_arguments_rejected():
    model, start = make_model(), torch.tensor([[1, 2]])
    with pytest.raises(ValueError):
        generate(model, start, 3, temperature=-1)
    with pytest.raises(ValueError):
        generate(model, start, 3, top_k=0)
    with pytest.raises(ValueError):
        generate(model, start, -1)
    with pytest.raises(ValueError):
        generate(model, torch.tensor([1, 2]), 3)
    with pytest.raises(ValueError):
        generate(model, torch.zeros((1, 0), dtype=torch.long), 3)
    with pytest.raises(ValueError):
        generate(model, start, 3, top_p=-0.1)
    with pytest.raises(ValueError):
        generate(model, start, 3, top_p=1.5)


def test_top_p_nucleus_sampling():
    # LOGITS = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0]
    # With a small top_p, only the top token(s) can ever be sampled
    model = FixedLogitsModel(LOGITS)
    samples = sample_one_token(model, rows=300, top_p=0.4, temperature=1.0).tolist()
    assert set(samples) <= {0, 1}
    # With top_p=1.0 or None, low probability tokens appear
    broad = sample_one_token(model, rows=300, top_p=1.0, temperature=1.0).tolist()
    assert len(set(broad)) > 2


def test_eos_token_id_stops_immediately():
    # A model that always predicts token 4
    logits = [0.0] * 8
    logits[4] = 10.0
    model = FixedLogitsModel(logits)
    start = torch.tensor([[0, 1]])
    # If eos_token_id is 4, generation should stop on the very first token
    out = generate(model, start, max_new_tokens=10, temperature=0.0, eos_token_id=4)
    assert out.shape == (1, 3)
    assert out[0, 2].item() == 4


def test_byte_tokenizer_generation_compatibility():
    from model.tokenizer import ByteTokenizer
    tok = ByteTokenizer(special_tokens=["[EOS]"])
    cfg = ModelConfig(
        vocab_size=tok.vocab_size, context_length=32, embedding_dim=32,
        num_heads=4, num_layers=2, dropout=0.0
    )
    model = GPT(cfg)
    # Ensure generate works with ByteTokenizer instance and stop_strings
    start = torch.tensor([[65, 66]])  # 'AB'
    out = generate(model, start, max_new_tokens=5, tokenizer=tok, stop_strings=["[EOS]"])
    assert out.shape[1] >= 3


def test_generate_text_skips_unknown_characters():
    tokenizer = CharTokenizer.from_text("abcdefgh")
    text = generate_text(make_model(), tokenizer, "abz", max_new_tokens=5)
    assert isinstance(text, str) and len(text) == 2 + 5 and text.startswith("ab")


def test_generate_text_with_unusable_prompt_still_works():
    tokenizer = CharTokenizer.from_text("abcdefgh")
    text = generate_text(make_model(), tokenizer, "zzz", max_new_tokens=5)
    assert len(text) == 1 + 5


def test_trained_model_continues_a_pattern():
    torch.manual_seed(0)
    model = make_model()
    data = torch.tensor([i % 8 for i in range(2000)])          # 0,1,2,...,7,0,1,2,...
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(400):
        x, y = get_batch(data, 16, 16)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    out = generate(model, torch.tensor([[0, 1, 2]]), 8, temperature=0)
    assert out[0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2]
