import pytest
import torch

from model.configuration import ModelConfig, TrainingConfig
from model.gpt import GPT
from training.checkpoint import load_model_from_checkpoint, save_checkpoint
from training.data import get_batch, split_data
from training.trainer import Trainer, estimate_loss, get_lr


def small_model_config():
    return ModelConfig(vocab_size=8, context_length=16, embedding_dim=32,
                       num_heads=4, num_layers=2, dropout=0.0)


def small_train_config(**overrides):
    values = dict(learning_rate=0.003, batch_size=8, max_steps=150, eval_interval=50,
                  random_seed=0, train_split=0.9, eval_iters=5)
    values.update(overrides)
    return TrainingConfig(**values)


def pattern_data(n=2000):
    return torch.tensor([i % 8 for i in range(n)], dtype=torch.long)   # 0,1,...,7,0,1,...


def test_split_sizes_and_order():
    train, val = split_data(list(range(100)), 0.9)
    assert len(train) == 90 and len(val) == 10
    assert train[-1].item() == 89 and val[0].item() == 90


def test_batch_targets_are_inputs_shifted_by_one():
    data = torch.arange(200)
    x, y = get_batch(data, batch_size=6, context_length=10)
    assert x.shape == (6, 10) and y.shape == (6, 10)
    assert torch.equal(y, x + 1)


def test_batches_stay_inside_the_data():
    data = torch.arange(17)                       # exactly context_length + 1
    for _ in range(50):
        x, y = get_batch(data, batch_size=4, context_length=16)
        assert y.max().item() <= 16


def test_too_little_data_rejected():
    with pytest.raises(ValueError):
        get_batch(torch.arange(16), batch_size=2, context_length=16)


def test_estimate_loss_restores_the_mode():
    model = GPT(small_model_config())
    data = pattern_data()
    model.train()
    estimate_loss(model, data, 4, 16, 2)
    assert model.training
    model.eval()
    estimate_loss(model, data, 4, 16, 2)
    assert not model.training


def test_learning_rate_schedule():
    base, warmup, total = 1.0, 10, 100
    assert get_lr(0, total, base, warmup) < base                       # starts small
    assert get_lr(warmup, total, base, warmup) == pytest.approx(base)  # peak after warmup
    assert get_lr(total, total, base, warmup) == pytest.approx(0.1 * base)
    later = [get_lr(s, total, base, warmup) for s in range(warmup, total + 1)]
    assert all(a >= b for a, b in zip(later, later[1:]))               # only decreases


def test_checkpoint_round_trip(tmp_path):
    cfg = small_model_config()
    model = GPT(cfg).eval()
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(tmp_path / "c.pt", model, optimizer, 7, 1.5, cfg, small_train_config(), [])
    loaded, ckpt = load_model_from_checkpoint(tmp_path / "c.pt")
    x = torch.randint(0, 8, (2, 10))
    assert ckpt["step"] == 7 and ckpt["best_val_loss"] == 1.5
    assert torch.allclose(model(x)[0], loaded(x)[0], atol=1e-6)


def test_trainer_learns_a_pattern_and_saves_checkpoints(tmp_path):
    torch.manual_seed(0)
    cfg = small_model_config()
    train, val = split_data(pattern_data().tolist(), 0.9)
    trainer = Trainer(GPT(cfg), train, val, small_train_config(), cfg, torch.device("cpu"), tmp_path)
    history = trainer.train(log=lambda *_: None)
    assert history[-1]["val_loss"] < history[0]["val_loss"] * 0.5
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()


def test_training_can_resume(tmp_path):
    torch.manual_seed(0)
    cfg = small_model_config()
    train, val = split_data(pattern_data().tolist(), 0.9)
    tcfg = small_train_config(eval_interval=25)
    first = Trainer(GPT(cfg), train, val, tcfg, cfg, torch.device("cpu"), tmp_path)
    first.train(max_steps=50, log=lambda *_: None)

    second = Trainer(GPT(cfg), train, val, tcfg, cfg, torch.device("cpu"), tmp_path)
    history = second.train(max_steps=100, resume_from=tmp_path / "last.pt", log=lambda *_: None)
    assert [h["step"] for h in history] == [0, 25, 50, 75, 100]


def test_checkpoint_saves_and_verifies_tokenizer(tmp_path):
    from model.tokenizer import CharTokenizer, ByteTokenizer
    cfg = small_model_config()
    model = GPT(cfg).eval()
    optimizer = torch.optim.AdamW(model.parameters())
    tok = CharTokenizer.from_text("abcdefgh")

    ckpt_path = tmp_path / "saved_tok.pt"
    save_checkpoint(ckpt_path, model, optimizer, 10, 1.2, cfg, small_train_config(), [], tokenizer=tok)

    # Valid load with matching tokenizer succeeds
    loaded_model, ckpt = load_model_from_checkpoint(ckpt_path, device="cpu", tokenizer=tok)
    assert ckpt["tokenizer_config"]["vocab_size"] == 8
    assert ckpt["tokenizer_config"]["type"] == "char"

    # Incompatible tokenizer type (ByteTokenizer vs Char) is rejected
    byte_tok = ByteTokenizer()
    with pytest.raises(ValueError):
        load_model_from_checkpoint(ckpt_path, device="cpu", tokenizer=byte_tok)


def test_split_conversations_and_deduplication():
    from training.data import split_conversations, deduplicate_conversations

    raw = (
        "YOU:\nHi\n\nBOT:\nHello![EOS]\n\n"
        "YOU:\nHi\n\nBOT:\nHello![EOS]\n\n"
        "YOU:\nWhat is 2+2?\n\nBOT:\nIt is 4.[EOS]\n\n"
        "YOU:\nWhat is code?\n\nBOT:\nInstructions for computers.[EOS]\n\n"
    )
    deduped = deduplicate_conversations(raw)
    assert deduped.count("YOU:\nHi") == 1

    train_text, val_text = split_conversations(raw, train_split=0.66)
    assert "YOU:\nHi" in train_text
    assert "YOU:\nWhat is code?" in val_text
    # Verify complete conversation preservation (never sliced in the middle)
    assert train_text.endswith("[EOS]\n\n")
    assert val_text.endswith("[EOS]\n\n")


def test_loss_masking_with_ignore_index():
    cfg = small_model_config()
    model = GPT(cfg)
    inputs = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    targets = torch.tensor([[-100, -100, 2, 3]], dtype=torch.long)

    logits, loss = model(inputs, targets)
    assert loss is not None
    assert torch.isfinite(loss)
    assert loss.item() > 0
