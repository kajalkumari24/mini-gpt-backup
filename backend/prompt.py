"""Turn a chat history into text the model can continue, and cut its reply back out.

The Mini GPT learned from Shakespeare, where plays look like this:

    ROMEO:
    Speech text.

    JULIET:
    Speech text.

so a chat is written as a tiny script between YOU and BOT, and the reply is
whatever the model writes for BOT before the speech ends (a blank line).
"""
import re

USER_LABEL = "YOU"
BOT_LABEL = "BOT"

_LABEL_LINE = re.compile(r"[A-Z][A-Z ]{1,30}:")


def _clean(content):
    """Collapse blank lines so a message cannot fake the end of a speech."""
    return re.sub(r"\n{2,}", "\n", content.strip())


def build_chat_prompt(messages, max_chars=2000):
    """messages: list of {"role": "user" | "assistant", "content": str}.

    Returns the script text ending with the BOT label, ready to be continued.
    If it is longer than max_chars, the oldest part is dropped.
    """
    parts = []
    for message in messages:
        label = USER_LABEL if message["role"] == "user" else BOT_LABEL
        parts.append(f"{label}:\n{_clean(message['content'])}\n\n")
    prompt = "".join(parts) + f"{BOT_LABEL}:\n"
    if max_chars and len(prompt) > max_chars:
        prompt = prompt[-max_chars:]
    return prompt


def extract_reply(generated):
    """Keep only the first speech: stop at [EOS], a blank line, or the next speaker label."""
    if "[EOS]" in generated:
        generated = generated.split("[EOS]")[0]
    speech = generated.split("\n\n")[0]
    kept = []
    for line in speech.split("\n"):
        if _LABEL_LINE.fullmatch(line.strip()):
            break
        kept.append(line)
    reply = "\n".join(kept).strip()
    return re.sub(r"\[EOS\]", "", reply).strip()
