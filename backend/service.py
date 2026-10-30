"""The model service: loads the trained GPT once and answers generation requests."""
import json
from pathlib import Path
import threading
from typing import Dict, List, Optional, Any

import torch

from model.configuration import get_device
from model.generation import generate, generate_stream
from model.tokenizer import ByteTokenizer, CharTokenizer, verify_compatibility
from training.checkpoint import load_model_from_checkpoint

from .prompt import (
    build_chat_prompt,
    extract_reply,
    is_speaker_label,
    NON_SPEAKER_HEADERS,
    SECTION_HEADER_RE,
)


class ModelService:
    def __init__(self, model, tokenizer, device, checkpoint_info=None):
        checkpoint_info = checkpoint_info or {}
        verify_compatibility(tokenizer, model.config, checkpoint_info.get("tokenizer_config"))

        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint_info = checkpoint_info
        self._lock = threading.Lock()      # one generation at a time on the shared model

    @classmethod
    def from_checkpoint(cls, checkpoint_path, tokenizer_path, device="auto"):
        device = get_device(device)
        tok_path = Path(tokenizer_path)
        tok_data = json.loads(tok_path.read_text(encoding="utf-8"))
        if tok_data.get("type") == "byte":
            tokenizer = ByteTokenizer.load(tok_path)
        else:
            tokenizer = CharTokenizer.load(tok_path)

        model, checkpoint = load_model_from_checkpoint(checkpoint_path, device, tokenizer=tokenizer)
        info = {
            "checkpoint_step": checkpoint.get("step"),
            "best_val_loss": checkpoint.get("best_val_loss"),
            "tokenizer_config": checkpoint.get("tokenizer_config"),
            "tokenizer_type": getattr(tokenizer, "tokenizer_type", "char"),
        }
        return cls(model, tokenizer, device, info)

    def info(self) -> Dict[str, Any]:
        cfg = self.model.config
        return {
            "parameters": self.model.num_parameters(),
            "vocab_size": cfg.vocab_size,
            "context_length": cfg.context_length,
            "embedding_dim": cfg.embedding_dim,
            "num_heads": cfg.num_heads,
            "num_layers": cfg.num_layers,
            "device": str(self.device),
            **self.checkpoint_info,
        }

    def _ignored(self, text: str) -> List[str]:
        """Characters the tokenizer has never seen. They are skipped, not an error."""
        return sorted(set(text) - set(self.tokenizer.stoi))

    def _continue(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        seed: Optional[int],
        repetition_penalty: float = 1.1,
        stop_strings: Optional[List[str]] = None,
    ) -> str:
        ids = self.tokenizer.encode(prompt, errors="ignore")
        if not ids:
            ids = self.tokenizer.encode("\n", errors="ignore") or [0]
        start = torch.tensor([ids], dtype=torch.long, device=self.device)
        with self._lock:
            if seed is not None:
                torch.manual_seed(seed)
            out = generate(
                self.model,
                start,
                max_new_tokens,
                temperature,
                top_k,
                repetition_penalty=repetition_penalty,
                stop_strings=stop_strings,
                tokenizer=self.tokenizer,
            )
        return self.tokenizer.decode(out[0, len(ids) :].tolist())

    def complete(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        seed: Optional[int] = None,
        repetition_penalty: float = 1.1,
    ) -> Dict[str, Any]:
        """Plain text continuation. Returns only the newly generated text."""
        text = self._continue(
            prompt,
            max_new_tokens,
            temperature,
            top_k,
            seed,
            repetition_penalty=repetition_penalty,
        )
        return {"text": text, "ignored_characters": self._ignored(prompt)}

    def supports_eos(self) -> bool:
        """Check if the model and tokenizer explicitly support atomic [EOS] end-of-sequence token.

        Checks actual special-token definitions rather than arbitrary bracket characters in vocabulary.
        """
        special_tokens = getattr(self.tokenizer, "special_tokens", None) or []
        if "[EOS]" in special_tokens or "EOS" in special_tokens:
            return True
        if hasattr(self.tokenizer, "special_to_id") and "[EOS]" in self.tokenizer.special_to_id:
            return True
        if hasattr(self.tokenizer, "stoi") and "[EOS]" in self.tokenizer.stoi:
            return True
        return False

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        seed: Optional[int] = None,
        repetition_penalty: float = 1.1,
    ) -> Dict[str, Any]:
        """Write the history as a script and return BOT's next speech."""
        import re

        prompt = build_chat_prompt(messages)
        speaker_delim_re = re.compile(r"\n(?:\s*\n)*[A-Z][A-Z0-9 _'-]{0,30}:(?:\s|$)")
        has_eos = self.supports_eos()
        if has_eos:
            stop_strings = ["[EOS]", "EOS\n", speaker_delim_re]
        else:
            stop_strings = ["[EOS]", "EOS\n", "\n\n", speaker_delim_re]
        generated = self._continue(
            prompt,
            max_new_tokens,
            temperature,
            top_k,
            seed,
            repetition_penalty=repetition_penalty,
            stop_strings=stop_strings,
        )
        everything = "".join(m["content"] for m in messages)
        return {
            "reply": extract_reply(generated, eos_mode=has_eos),
            "ignored_characters": self._ignored(everything),
        }

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        seed: Optional[int] = None,
        repetition_penalty: float = 1.1,
    ):
        """Yield generated text tokens in real time, stopping cleanly at [EOS], EOS, or speaker labels.

        Preserves normal multiline paragraphs (\\n\\n) within the speech when no speaker transition occurs.
        """
        import re

        prompt = build_chat_prompt(messages)
        ids = self.tokenizer.encode(prompt, errors="ignore")
        if not ids:
            ids = self.tokenizer.encode("\n", errors="ignore") or [0]
        start = torch.tensor([ids], dtype=torch.long, device=self.device)

        has_eos = self.supports_eos()
        if has_eos:
            stop_sequences = ["[EOS]", "EOS\n"]
        else:
            stop_sequences = ["[EOS]", "EOS\n", "\n\n"]

        speaker_regex = re.compile(r"\n(?:\s*\n)*[A-Z][A-Z0-9 _'-]{0,30}:(?:\s|$)")
        speaker_prefix_regex = re.compile(r"^\n[\s]*[A-Z0-9 _'-]{0,30}$")

        with self._lock:
            if seed is not None:
                torch.manual_seed(seed)
            stream_buffer = ""
            has_yielded = False

            for char in generate_stream(
                self.model,
                start,
                max_new_tokens,
                temperature,
                top_k,
                repetition_penalty=repetition_penalty,
                stop_strings=None,
                tokenizer=self.tokenizer,
            ):
                stream_buffer += char

                # If generation begins with blank lines before any speech, speech is empty
                if not has_yielded and stream_buffer.lstrip(" \t").startswith("\n\n"):
                    return

                # 1. Check if any stop sequence is fully present in stream_buffer
                stop_pos = -1
                for s in stop_sequences:
                    pos = stream_buffer.find(s)
                    if pos != -1 and (stop_pos == -1 or pos < stop_pos):
                        stop_pos = pos

                speaker_match = speaker_regex.search(stream_buffer)
                if speaker_match:
                    matched_text = speaker_match.group(0).lstrip("\n \t")
                    matched_name = matched_text.split(":", 1)[0].strip()
                    if matched_name not in NON_SPEAKER_HEADERS and not SECTION_HEADER_RE.match(matched_text):
                        pos = speaker_match.start()
                        if stop_pos == -1 or pos < stop_pos:
                            stop_pos = pos

                if stop_pos != -1:
                    lead = stream_buffer[:stop_pos]
                    lead = re.sub(r"(?:\[EOS\]|\bEOS\b)$", "", lead)
                    if lead:
                        yield lead
                    return

                # 2. Check if trailing stream_buffer is a prefix of any stop sequence or speaker label
                prefix_len = 0
                for k in range(len(stream_buffer), 0, -1):
                    suffix = stream_buffer[-k:]
                    if any(s.startswith(suffix) for s in stop_sequences) or bool(
                        speaker_prefix_regex.match(suffix)
                    ):
                        prefix_len = k
                        break

                if prefix_len > 0:
                    safe = stream_buffer[:-prefix_len]
                    if safe:
                        yield safe
                        has_yielded = True
                        stream_buffer = stream_buffer[-prefix_len:]
                else:
                    yield stream_buffer
                    has_yielded = True
                    stream_buffer = ""

            if stream_buffer:
                clean_remainder = re.sub(r"(?:\[EOS\]|\bEOS\b)$", "", stream_buffer)
                if clean_remainder:
                    yield clean_remainder
