"""Fast tokenizer wrapper supporting direct Rust tokenizers and Hugging Face fallbacks."""
import os


class FastTokenizer:
    def __init__(self, tokenizer, is_hf: bool = False):
        self._t = tokenizer
        self._is_hf = is_hf

    @classmethod
    def from_pretrained(cls, model_dir: str, tokenizer_file: str = "tokenizer.json"):
        path = os.path.join(model_dir, tokenizer_file)
        if os.path.exists(path):
            try:
                from tokenizers import Tokenizer
                return cls(Tokenizer.from_file(path), is_hf=False)
            except Exception:
                pass
        try:
            from transformers import AutoTokenizer
            return cls(AutoTokenizer.from_pretrained(model_dir), is_hf=True)
        except Exception as e:
            raise RuntimeError(f"Could not load tokenizer from {model_dir}: {e}")

    @property
    def eos_token_id(self) -> int:
        if self._is_hf:
            return self._t.eos_token_id
        tid = self._t.token_to_id("<|endoftext|>")
        if tid is None:
            tid = self._t.token_to_id("<|im_end|>")
        return tid or 151643

    @property
    def pad_token_id(self) -> int:
        if self._is_hf:
            return self._t.pad_token_id or self.eos_token_id
        return self.eos_token_id

    def encode(self, text: str) -> list[int]:
        if self._is_hf:
            return self._t.encode(text)
        return self._t.encode(text).ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if self._is_hf:
            return self._t.decode(ids, skip_special_tokens=skip_special_tokens)
        return self._t.decode(list(ids), skip_special_tokens=skip_special_tokens)
