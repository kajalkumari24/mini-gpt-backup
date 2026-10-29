"""Turn a chat history into text the model can continue, and cut its reply back out.

The Mini GPT supports both dialogue and play scripts (e.g. Shakespeare):

    ROMEO:
    Speech text.

    JULIET:
    Speech text.

    YOU:
    Message text.

    BOT:
    First paragraph.

    Second paragraph.[EOS]

A chat is written as a script, and the reply is whatever the model writes for the
speaker before the speech ends ([EOS] or the next speaker label).
"""
import re
from typing import Optional

USER_LABEL = "YOU"
BOT_LABEL = "BOT"

# Words that appear in uppercase with colons in prose or instructions that are NOT speaker labels
NON_SPEAKER_HEADERS = {
    "NOTE", "WARNING", "IMPORTANT", "TIP", "CAUTION", "REMINDER",
    "NOTICE", "EXAMPLE", "SUMMARY", "INFO", "HINT", "DETAILS", "OUTPUT", "INPUT"
}
SECTION_HEADER_RE = re.compile(r"^(?:STEP|CHAPTER|SECTION|RULE|PHASE|STAGE|PART)\s+\d+:", re.IGNORECASE)

# General speaker-label detector: matches uppercase speaker names ending with a colon,
# e.g. "ROMEO:", "JULIET:", "YOU:", "BOT:", "FIRST CITIZEN:", "KING LEAR: Hello", "O'CONNOR:", "SOLDIER 1:"
SPEAKER_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9 _'-]{0,30}:(?:\s.*)?$")
LEADING_SPEAKER_RE = re.compile(r"^[A-Z][A-Z0-9 _'-]{0,30}:\s*")


def is_speaker_label(line: str) -> bool:
    """Check if a line represents a character or participant speaker label.

    Distinguishes legitimate speakers (ROMEO:, JULIET:, FIRST CITIZEN:, O'CONNOR:,
    SOLDIER 1:, AGENT 007:, YOU:, BOT:) from normal prose labels (NOTE:, WARNING:,
    STEP 1:, EXAMPLE:, etc.) or ordinary sentences with colons.
    """
    stripped = line.strip()
    if not stripped or ":" not in stripped:
        return False
    if not SPEAKER_LABEL_RE.match(stripped):
        return False
    name = stripped.split(":", 1)[0].strip()
    if name in NON_SPEAKER_HEADERS:
        return False
    if SECTION_HEADER_RE.match(stripped):
        return False
    return True


def _clean(content: str) -> str:
    """Collapse blank lines so a message cannot fake the end of a speech."""
    return re.sub(r"\n{2,}", "\n", content.strip())


def build_chat_prompt(messages, max_chars: int = 2000) -> str:
    """messages: list of {"role": "user" | "assistant", "content": str}.

    Returns the script text ending with the BOT label, ready to be continued.
    If it is longer than max_chars, the oldest part is dropped while keeping the prompt intact.
    """
    parts = []
    for message in messages:
        label = USER_LABEL if message["role"] == "user" else BOT_LABEL
        parts.append(f"{label}:\n{_clean(message['content'])}\n\n")
    prompt = "".join(parts) + f"{BOT_LABEL}:\n"
    if max_chars and len(prompt) > max_chars:
        prompt = prompt[-max_chars:]
    return prompt


def extract_reply(generated: str, eos_mode: Optional[bool] = None) -> str:
    """Keep only the first speaker's speech from generated text.

    Parameters:
    - generated: Raw text output from the model.
    - eos_mode: If True, operates in conversation mode (preserves internal blank lines,
      terminates on [EOS] or next speaker label). If False, operates in Shakespeare/script
      mode (terminates on blank line or next speaker label). If None, detects from presence
      of [EOS] in the generated text.
    """
    if not generated or not generated.strip():
        return ""

    # If generation begins with blank lines before speech, first speech was empty
    if generated.lstrip(" \t").startswith("\n\n"):
        return ""

    # 1. Strip leading speaker prefix if included in generation (e.g. "ROMEO: Hello" -> "Hello")
    text = generated.lstrip()
    first_line = text.split("\n", 1)[0]
    if is_speaker_label(first_line):
        text = LEADING_SPEAKER_RE.sub("", text)

    # 2. Determine mode
    has_eos = eos_mode if eos_mode is not None else ("[EOS]" in text or bool(re.search(r"(?:\bEOS\b|\bEOS$)", text)))

    if "[EOS]" in text:
        text = text.split("[EOS]")[0]

    if has_eos:
        # Multiline mode: speech ends at [EOS] or next speaker label; internal \n\n is preserved
        lines = text.split("\n")
        kept = []
        for line in lines:
            if is_speaker_label(line):
                break
            kept.append(line)
        reply = "\n".join(kept).strip()
    else:
        # Shakespeare / script mode without [EOS]: speech ends at blank line or next speaker label
        speech = text.split("\n\n")[0]
        lines = speech.split("\n")
        kept = []
        for line in lines:
            if is_speaker_label(line):
                break
            kept.append(line)
        reply = "\n".join(kept).strip()

    # 3. Clean any residual [EOS] or standalone trailing EOS
    reply = re.sub(r"\[EOS\]", "", reply).strip()
    reply = re.sub(r"(?:\bEOS\b|\bEOS$)", "", reply).strip()
    return reply
