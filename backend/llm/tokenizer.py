"""TwinSimTokenizer: a byte-level BPE tokenizer trained from our own corpus.

No pretrained vocabulary is used. The base alphabet is the 256 byte values, so
any UTF-8 text can be encoded (the <unk> token exists for completeness and for
decoding out-of-range ids). Merges are learned with the classic BPE procedure:
repeatedly merge the most frequent adjacent pair inside pre-tokenised words.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
# Structural tags used by the persona serializer. Treated as atomic tokens.
STRUCTURE_TOKENS = [
    "<persona>", "</persona>", "<scenario>", "</scenario>",
    "<response>", "</response>", "<feedback>", "</feedback>",
]

# GPT-2-style pre-tokenisation, restricted to what Python's `re` supports.
PRETOKENIZE_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""


class TokenizerError(Exception):
    pass


class TwinSimTokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None,
                 special_tokens: list[str] | None = None,
                 pattern: str = PRETOKENIZE_PATTERN):
        self.special_tokens = list(special_tokens or (SPECIAL_TOKENS + STRUCTURE_TOKENS))
        for required in SPECIAL_TOKENS:
            if required not in self.special_tokens:
                raise TokenizerError(f"missing required special token {required}")
        self.pattern = pattern
        self._re = re.compile(pattern)
        self.merges: list[tuple[int, int]] = [tuple(m) for m in (merges or [])]
        self._build()

    # ------------------------------------------------------------------ ids
    # Layout: [special tokens][256 bytes][merged tokens]
    def _build(self) -> None:
        self.n_special = len(self.special_tokens)
        self.special_to_id = {t: i for i, t in enumerate(self.special_tokens)}
        self.byte_offset = self.n_special
        self.vocab: dict[int, bytes] = {}
        for i, t in enumerate(self.special_tokens):
            self.vocab[i] = t.encode("utf-8")
        for b in range(256):
            self.vocab[self.byte_offset + b] = bytes([b])
        self.merge_rank: dict[tuple[int, int], int] = {}
        next_id = self.byte_offset + 256
        for rank, (a, b) in enumerate(self.merges):
            if a not in self.vocab or b not in self.vocab:
                raise TokenizerError(f"invalid merge {a},{b} at rank {rank}")
            self.vocab[next_id] = self.vocab[a] + self.vocab[b]
            self.merge_rank[(a, b)] = rank
            next_id += 1
        self.merge_to_id = {pair: self.byte_offset + 256 + r for pair, r in self.merge_rank.items()}
        self._special_re = re.compile("(" + "|".join(re.escape(t) for t in
                                                      sorted(self.special_tokens, key=len, reverse=True)) + ")")
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.special_to_id["<unk>"]

    def token_id(self, token: str) -> int:
        return self.special_to_id[token]

    # ------------------------------------------------------------- training
    @classmethod
    def train(cls, texts, vocab_size: int, special_tokens: list[str] | None = None,
              min_frequency: int = 2, verbose: bool = False) -> "TwinSimTokenizer":
        tok = cls(merges=[], special_tokens=special_tokens)
        n_merges = vocab_size - tok.n_special - 256
        if n_merges < 0:
            raise TokenizerError(f"vocab_size {vocab_size} is smaller than specials+256 bytes")
        word_counts: Counter[str] = Counter()
        for text in texts:
            for chunk in tok._special_re.split(text):
                if not chunk or chunk in tok.special_to_id:
                    continue
                word_counts.update(tok._re.findall(chunk))
        words = [[tok.byte_offset + b for b in w.encode("utf-8")] for w in word_counts]
        freqs = list(word_counts.values())
        merges: list[tuple[int, int]] = []
        next_id = tok.byte_offset + 256
        for step in range(n_merges):
            pairs: Counter[tuple[int, int]] = Counter()
            for w, f in zip(words, freqs):
                for i in range(len(w) - 1):
                    pairs[(w[i], w[i + 1])] += f
            if not pairs:
                break
            best, count = max(pairs.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if count < min_frequency:
                break
            merges.append(best)
            a, b = best
            for idx, w in enumerate(words):
                if len(w) < 2:
                    continue
                i, out = 0, []
                while i < len(w):
                    if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                        out.append(next_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                words[idx] = out
            next_id += 1
            if verbose and step % 100 == 0:
                print(f"  merge {step}/{n_merges}: count={count}")
        return cls(merges=merges, special_tokens=tok.special_tokens, pattern=tok.pattern)

    # ------------------------------------------------------------- encoding
    def _encode_word(self, word: str) -> list[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = [self.byte_offset + b for b in word.encode("utf-8")]
        while len(ids) >= 2:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = self.merge_rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            pair = (ids[best_i], ids[best_i + 1])
            ids = ids[:best_i] + [self.merge_to_id[pair]] + ids[best_i + 2:]
        if len(self._cache) < 200_000:
            self._cache[word] = ids
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        out: list[int] = [self.bos_id] if add_bos else []
        for chunk in self._special_re.split(text):
            if not chunk:
                continue
            if chunk in self.special_to_id:
                out.append(self.special_to_id[chunk])
                continue
            for word in self._re.findall(chunk):
                out.extend(self._encode_word(word))
        if add_eos:
            out.append(self.eos_id)
        return out

    def decode(self, ids, skip_special: bool = False) -> str:
        parts: list[bytes] = []
        for i in ids:
            i = int(i)
            if i < self.n_special:
                if skip_special:
                    continue
                parts.append(self.special_tokens[i].encode("utf-8"))
            elif i in self.vocab:
                parts.append(self.vocab[i])
            else:
                if not skip_special:
                    parts.append(b"<unk>")
        return b"".join(parts).decode("utf-8", errors="replace")

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "type": "twinsim-byte-bpe",
            "format_version": 1,
            "special_tokens": self.special_tokens,
            "pattern": self.pattern,
            "merges": [list(m) for m in self.merges],
            "vocab_size": self.vocab_size,
        }

    @property
    def version(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return "tok-" + hashlib.sha256(blob).hexdigest()[:10]

    def save(self, path: str | Path) -> None:
        d = self.to_dict()
        d["version"] = self.version
        Path(path).write_text(json.dumps(d))

    @classmethod
    def load(cls, path: str | Path) -> "TwinSimTokenizer":
        try:
            d = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise TokenizerError(f"cannot read tokenizer at {path}: {e}") from e
        if d.get("type") != "twinsim-byte-bpe":
            raise TokenizerError(f"{path} is not a TwinSim tokenizer file")
        tok = cls(merges=[tuple(m) for m in d["merges"]], special_tokens=d["special_tokens"], pattern=d["pattern"])
        if "version" in d and d["version"] != tok.version:
            raise TokenizerError(f"tokenizer checksum mismatch ({d['version']} != {tok.version}); file corrupted?")
        return tok
