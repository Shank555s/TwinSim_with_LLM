"""Twin inference service: persona + scenario -> response from OUR Transformer checkpoint.

There is no code path to any external LLM. If no checkpoint exists the engine raises
`ModelNotTrained` and the API reports that clearly.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

from llm.checkpoint import LoadedCheckpoint, load_checkpoint
from llm.config import GenerationConfig
from llm.generation import GenerationResult, generate

from .persona_serializer import SerializationFormat, serialize_prompt


class ModelNotTrained(Exception):
    pass


class TwinEngine:
    def __init__(self, max_cached: int = 3):
        self._cache: "OrderedDict[str, tuple[LoadedCheckpoint, threading.Lock]]" = OrderedDict()
        self._lock = threading.Lock()
        self.max_cached = max_cached

    def load(self, name: str, path: str | Path) -> tuple[LoadedCheckpoint, threading.Lock]:
        with self._lock:
            if name in self._cache:
                self._cache.move_to_end(name)
                return self._cache[name]
        ckpt = load_checkpoint(path)
        entry = (ckpt, threading.Lock())
        with self._lock:
            self._cache[name] = entry
            while len(self._cache) > self.max_cached:
                self._cache.popitem(last=False)
        return entry

    def evict(self, name: str) -> None:
        with self._lock:
            self._cache.pop(name, None)

    def generate(self, model: dict, persona: dict, scenario: dict, gen: GenerationConfig,
                 on_text=None) -> tuple[GenerationResult, dict]:
        """`on_text(text_so_far, n_tokens)` streams the partial response as tokens are produced."""
        if not model or not model.get("checkpoint_path"):
            raise ModelNotTrained("No TwinSim model has been trained yet. Run the bootstrap training first.")
        from database.repo import checkpoint_dir
        ckpt, lock = self.load(model["name"], checkpoint_dir(model))
        fmt = SerializationFormat.from_dict(ckpt.training_state.get("serialization_format"))
        tok = ckpt.tokenizer
        prompt = serialize_prompt(persona, scenario, fmt)
        ids = tok.encode(prompt, add_bos=True)
        ctx = ckpt.runner.config.context_length
        if len(ids) >= ctx - 8:
            raise ValueError(f"persona + scenario use {len(ids)} tokens but the model context is {ctx}; "
                             "shorten the scenario or the persona notes")
        gen = GenerationConfig(**{**gen.to_dict(), "max_new_tokens": min(gen.max_new_tokens, ctx - len(ids))})
        stop = [tok.token_id("</response>"), tok.eos_id]
        banned = [tok.pad_id, tok.bos_id, tok.unk_id] + [tok.token_id(t) for t in
                                                          ("<persona>", "</persona>", "<scenario>", "</scenario>",
                                                           "<response>", "<feedback>", "</feedback>")]
        with lock:  # one generation at a time per loaded model (KV cache is per runner)
            cb = None if on_text is None else (lambda new: on_text(tok.decode(new, skip_special=True), len(new)))
            res = generate(ckpt.runner, ids, gen, stop_ids=stop, banned_ids=banned,
                           decode=lambda new: tok.decode(new, skip_special=True), on_token=cb)
        res.text = res.text.strip()
        info = {"prompt_tokens": len(ids), "tokenizer_version": tok.version, "backend": ckpt.runner.backend,
                "device": ckpt.runner.device}
        return res, info


engine = TwinEngine()
