"""Tokenizers for Mini-GPT: CharTokenizer and ByteTokenizer.

- CharTokenizer: Character-level tokenizer with optional atomic special tokens ([EOS]).
- ByteTokenizer: UTF-8 Byte-level tokenizer (256 base vocabulary + special tokens),
  eliminating out-of-vocabulary character issues for code and symbols.
"""
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def verify_compatibility(tokenizer: Any, model_config: Any, checkpoint_tokenizer_info: Optional[Dict[str, Any]] = None):
    """Verify that a tokenizer is 100% compatible with a model configuration and checkpoint metadata.

    Raises ValueError with a detailed explanation if incompatible.
    """
    if checkpoint_tokenizer_info:
        expected_type = checkpoint_tokenizer_info.get("type")
        current_type = getattr(tokenizer, "tokenizer_type", "char" if isinstance(tokenizer, CharTokenizer) else "byte")
        if expected_type and expected_type != current_type:
            raise ValueError(
                f"Tokenizer type mismatch: Checkpoint expects '{expected_type}' tokenizer, "
                f"but received '{current_type}' tokenizer."
            )

    tok_vocab = getattr(tokenizer, "vocab_size", None)
    model_vocab = getattr(model_config, "vocab_size", None)

    if tok_vocab is None:
        raise ValueError("Provided tokenizer does not have a valid vocab_size attribute.")
    if model_vocab is not None and tok_vocab != model_vocab:
        raise ValueError(
            f"Vocabulary size mismatch: Tokenizer has {tok_vocab} tokens, but model requires {model_vocab}. "
            f"The tokenizer and model checkpoint must originate from the same training run."
        )

    if checkpoint_tokenizer_info:
        expected_hash = checkpoint_tokenizer_info.get("vocab_hash")
        if expected_hash and hasattr(tokenizer, "vocab_hash"):
            current_hash = tokenizer.vocab_hash
            if current_hash != expected_hash:
                raise ValueError(
                    f"Tokenizer vocabulary mapping mismatch: Checkpoint expected vocab hash {expected_hash}, "
                    f"but tokenizer has vocab hash {current_hash}. Character or byte IDs do not align."
                )


class CharTokenizer:
    """Convert text to character IDs and character IDs back to text."""

    tokenizer_type = "char"

    def __init__(self, special_tokens: Optional[List[str]] = None):
        self.stoi = {}
        self.itos = {}
        self.special_tokens = list(special_tokens) if special_tokens else []

    @property
    def vocab_hash(self) -> str:
        """Deterministic hash of the vocabulary mapping for checkpoint compatibility verification."""
        if not self.stoi:
            return ""
        content = "".join(f"{k}:{v};" for k, v in sorted(self.stoi.items()))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def get_config(self) -> Dict[str, Any]:
        """Return tokenizer configuration metadata for checkpoint serialization."""
        return {
            "type": "char",
            "vocab_size": self.vocab_size,
            "special_tokens": list(self.special_tokens),
            "vocab_hash": self.vocab_hash,
        }

    def build_vocab(self, text: str, special_tokens: Optional[List[str]] = None):
        """Build a vocabulary from training text, optionally appending special tokens."""
        if not text:
            raise ValueError("Training text cannot be empty.")

        if special_tokens is not None:
            self.special_tokens = list(special_tokens)

        # Base character set
        characters = sorted(set(text))
        self.stoi = {character: index for index, character in enumerate(characters)}

        # Append special tokens if provided
        for token in self.special_tokens:
            if token not in self.stoi:
                new_id = len(self.stoi)
                self.stoi[token] = new_id

        self.itos = {index: token for token, index in self.stoi.items()}

    @classmethod
    def from_text(cls, text: str, special_tokens: Optional[List[str]] = None):
        tokenizer = cls(special_tokens=special_tokens)
        tokenizer.build_vocab(text, special_tokens=special_tokens)
        return tokenizer

    @property
    def vocab_size(self) -> int:
        """Return the number of tokens in the vocabulary."""
        return len(self.stoi)

    def encode(self, text: str, errors: str = "strict") -> List[int]:
        """Convert text into a list of token IDs.

        errors="strict" raises on characters the tokenizer has never seen.
        errors="ignore" skips them (useful for chatbot input).
        """
        if not self.stoi:
            raise ValueError("Build the vocabulary before encoding.")
        if errors not in ("strict", "ignore"):
            raise ValueError("errors must be 'strict' or 'ignore'")

        if not self.special_tokens:
            if errors == "ignore":
                return [self.stoi[c] for c in text if c in self.stoi]
            unknown = set(text) - set(self.stoi)
            if unknown:
                raise ValueError(f"Text contains unknown characters: {sorted(unknown)!r}")
            return [self.stoi[character] for character in text]

        # Tokenize with special tokens preserved
        escaped = [re.escape(t) for t in self.special_tokens]
        pattern = re.compile(f"({'|'.join(escaped)})")
        parts = pattern.split(text)

        ids = []
        for part in parts:
            if not part:
                continue
            if part in self.stoi:
                ids.append(self.stoi[part])
            else:
                for c in part:
                    if c in self.stoi:
                        ids.append(self.stoi[c])
                    elif errors == "strict":
                        raise ValueError(f"Text contains unknown characters: {[c]!r}")
        return ids

    def decode(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs back into text."""
        if not self.itos:
            raise ValueError("Build the vocabulary before decoding.")
        try:
            return "".join(self.itos[int(token_id)] for token_id in token_ids)
        except KeyError as error:
            raise ValueError(f"Unknown token ID: {error.args[0]}") from error

    def save(self, path: Union[str, Path]):
        """Save the vocabulary as JSON so training and the chatbot share it."""
        if not self.stoi:
            raise ValueError("Build the vocabulary before saving.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "char",
            "chars": [self.itos[i] for i in range(len(self.itos))],
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != "char":
            raise ValueError("Not a character tokenizer file.")
        tokenizer = cls()
        tokenizer.stoi = {c: i for i, c in enumerate(data["chars"])}
        tokenizer.itos = {i: c for c, i in tokenizer.stoi.items()}
        tokenizer.special_tokens = data.get("special_tokens", [])
        return tokenizer


class ByteTokenizer:
    """Byte-level tokenizer: maps UTF-8 bytes (0-255) to tokens, plus optional special tokens."""

    tokenizer_type = "byte"

    def __init__(self, special_tokens: Optional[List[str]] = None):
        self.special_tokens = list(special_tokens) if special_tokens else []
        self._build_vocab()

    @property
    def vocab_hash(self) -> str:
        """Deterministic hash of the special tokens added to the standard 256 byte vocabulary."""
        content = "byte256:" + ";".join(self.special_tokens)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def get_config(self) -> Dict[str, Any]:
        """Return tokenizer configuration metadata for checkpoint serialization."""
        return {
            "type": "byte",
            "vocab_size": self.vocab_size,
            "special_tokens": list(self.special_tokens),
            "vocab_hash": self.vocab_hash,
        }

    def _build_vocab(self):
        # 0-255 are standard byte values
        self.byte_to_id = {b: b for b in range(256)}
        self.id_to_byte = {b: b for b in range(256)}
        self.special_to_id = {}
        self.id_to_special = {}

        for i, token in enumerate(self.special_tokens):
            tid = 256 + i
            self.special_to_id[token] = tid
            self.id_to_special[tid] = token

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.special_tokens)

    def encode(self, text: str, errors: str = "strict") -> List[int]:
        if not self.special_tokens:
            return list(text.encode("utf-8"))

        escaped = [re.escape(t) for t in self.special_tokens]
        pattern = re.compile(f"({'|'.join(escaped)})")
        parts = pattern.split(text)

        ids = []
        for part in parts:
            if not part:
                continue
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
            else:
                ids.extend(list(part.encode("utf-8")))
        return ids

    def decode(self, token_ids: List[int]) -> str:
        out_bytes = bytearray()
        out_text = []

        def flush_bytes():
            if out_bytes:
                out_text.append(out_bytes.decode("utf-8", errors="replace"))
                out_bytes.clear()

        for tid in token_ids:
            tid = int(tid)
            if tid < 256:
                out_bytes.append(tid)
            elif tid in self.id_to_special:
                flush_bytes()
                out_text.append(self.id_to_special[tid])
            else:
                raise ValueError(f"Unknown token ID: {tid}")

        flush_bytes()
        return "".join(out_text)

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "byte",
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != "byte":
            raise ValueError("Not a byte tokenizer file.")
        return cls(special_tokens=data.get("special_tokens", []))
