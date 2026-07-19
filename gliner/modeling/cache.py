"""Session cache support for decoder-backed span inference."""

from __future__ import annotations

from typing import Any, Iterable
from threading import RLock
from dataclasses import field, replace, dataclass

import torch
from transformers.cache_utils import DynamicCache


def _empty_long() -> torch.Tensor:
    return torch.empty(0, dtype=torch.long)


@dataclass
class CacheState:
    """All reusable state belonging to one streaming-span session."""

    past_key_values: Any = None
    input_ids: torch.Tensor = field(default_factory=_empty_long)
    attention_mask: torch.Tensor = field(default_factory=_empty_long)
    token_word_mask: torch.Tensor = field(default_factory=_empty_long)
    past_word_embeddings: torch.Tensor | None = None
    past_word_mask: torch.Tensor | None = None
    prompts_embedding: torch.Tensor | None = None
    prompts_mask: torch.Tensor | None = None
    cached_length: int = 0
    next_position_id: int | None = None
    session_id: str | None = None
    labels: tuple[str, ...] = ()
    text: str = ""
    tokens: list[str] = field(default_factory=list)
    char_starts: list[int] = field(default_factory=list)
    char_ends: list[int] = field(default_factory=list)
    span_logits: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def next_position(self) -> int:
        """Return the absolute decoder position for the next token."""
        return self.cached_length if self.next_position_id is None else self.next_position_id

    @property
    def word_length(self) -> int:
        """Return the number of valid cached words."""
        if self.past_word_mask is not None:
            return int(self.past_word_mask.long().sum().item())
        return len(self.tokens)

    def to(self, device: torch.device | str) -> CacheState:
        """Move reusable model tensors to ``device`` while keeping score history on CPU."""
        return replace(
            self,
            past_key_values=_move_past_kv(self.past_key_values, device),
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            token_word_mask=self.token_word_mask.to(device),
            past_word_embeddings=(
                self.past_word_embeddings.to(device) if self.past_word_embeddings is not None else None
            ),
            past_word_mask=self.past_word_mask.to(device) if self.past_word_mask is not None else None,
            prompts_embedding=self.prompts_embedding.to(device) if self.prompts_embedding is not None else None,
            prompts_mask=self.prompts_mask.to(device) if self.prompts_mask is not None else None,
            tokens=list(self.tokens),
            char_starts=list(self.char_starts),
            char_ends=list(self.char_ends),
            # Span scores intentionally remain on CPU so long-running sessions
            # do not retain an ever-growing GPU prediction history.
            span_logits=self.span_logits.copy(),
            metadata=self.metadata.copy(),
        )


class SessionCacheManager:
    """Thread-safe ownership and lifecycle management for session states."""

    def __init__(self) -> None:
        self._states: dict[str, CacheState] = {}
        self._lock = RLock()

    def get(self, session_id: str) -> CacheState | None:
        with self._lock:
            return self._states.get(session_id)

    def put(self, state: CacheState) -> None:
        if state.session_id is None:
            raise ValueError("A cached state must have a session_id")
        with self._lock:
            self._states[state.session_id] = state

    def clear(self, session_ids: str | Iterable[str] | None = None) -> None:
        with self._lock:
            if session_ids is None:
                self._states.clear()
                return
            if isinstance(session_ids, str):
                session_ids = [session_ids]
            for session_id in session_ids:
                self._states.pop(session_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._states)


def _cache_layers(past_kv):
    """Yield ``(layer_index, keys, values)`` across cache API generations."""
    layers = getattr(past_kv, "layers", None)
    if layers is not None:
        for layer_idx, layer in enumerate(layers):
            initialized = getattr(layer, "is_initialized", True)
            if callable(initialized):
                initialized = initialized()
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if initialized and keys is not None and values is not None:
                yield layer_idx, keys, values
        return

    # transformers 4.x DynamicCache stores tensors in parallel lists.  Keep
    # support for it because GLiNER's dependency range spans both cache APIs.
    key_cache = getattr(past_kv, "key_cache", None)
    value_cache = getattr(past_kv, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        for layer_idx, (keys, values) in enumerate(zip(key_cache, value_cache)):
            if torch.is_tensor(keys) and torch.is_tensor(values):
                yield layer_idx, keys, values


def _dynamic_cache_from_layers(layer_values) -> DynamicCache:
    cache = DynamicCache()
    for layer_idx, keys, values in layer_values:
        cache.update(keys, values, layer_idx)
    return cache


def _is_dynamic_cache_like(past_kv) -> bool:
    return hasattr(past_kv, "layers") or (
        hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache")
    )


def _move_past_kv(past_kv, device):
    if past_kv is None:
        return None
    if isinstance(past_kv, (tuple, list)):
        return type(past_kv)(_move_past_kv(item, device) for item in past_kv)

    cache_layers = list(_cache_layers(past_kv))
    if cache_layers or _is_dynamic_cache_like(past_kv):
        return _dynamic_cache_from_layers(
            (layer_idx, keys.to(device), values.to(device))
            for layer_idx, keys, values in cache_layers
        )
    if hasattr(past_kv, "to"):
        return past_kv.to(device)
    return past_kv


def copy_past_key_values(past_kv):
    """Clone a cache because transformer cache objects update in place."""
    if past_kv is None:
        return None
    if isinstance(past_kv, (tuple, list)):
        return type(past_kv)(copy_past_key_values(item) for item in past_kv)

    cache_layers = list(_cache_layers(past_kv))
    if cache_layers or _is_dynamic_cache_like(past_kv):
        return _dynamic_cache_from_layers(
            (layer_idx, keys.clone(), values.clone())
            for layer_idx, keys, values in cache_layers
        )
    if hasattr(past_kv, "clone"):
        return past_kv.clone()
    return past_kv
