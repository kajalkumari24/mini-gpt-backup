"""Tests for multi-turn chat history formatting and context budget management."""
from backend.prompt import build_chat_prompt, extract_reply


def test_multi_turn_history_alternation():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]
    prompt = build_chat_prompt(messages)
    assert prompt.startswith("YOU:\nHello\n\nBOT:\nHi there!\n\nYOU:\nHow are you?\n\nBOT:\n")
    assert prompt.endswith("BOT:\n")


def test_history_truncation_preserves_trailing_prompt():
    messages = [
        {"role": "user", "content": "Turn 1" * 50},
        {"role": "assistant", "content": "Reply 1" * 50},
        {"role": "user", "content": "Turn 2"},
    ]
    prompt = build_chat_prompt(messages, max_chars=120)
    assert len(prompt) <= 120
    assert prompt.endswith("BOT:\n")
    assert "Turn 2" in prompt


def test_consecutive_user_messages_cleaning():
    messages = [
        {"role": "user", "content": "Hello\n\n\nworld\n\n"},
        {"role": "assistant", "content": "Greeting"},
        {"role": "user", "content": "Next question"},
    ]
    prompt = build_chat_prompt(messages)
    # Consecutive blank lines should be collapsed
    assert "\n\n\n" not in prompt
    assert prompt.endswith("BOT:\n")

