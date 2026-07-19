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

    def get_many(self, session_ids: Iterable[str]) -> list[CacheState | None]:
        with self._lock:
            return [self._states.get(session_id) for session_id in session_ids]

    def put(self, state: CacheState) -> None:
        if state.session_id is None:
            raise ValueError("A cached state must have a session_id")
        with self._lock:
            self._states[state.session_id] = state

    def put_many(self, states: Iterable[CacheState]) -> None:
        states = list(states)
        if any(state.session_id is None for state in states):
            raise ValueError("Every cached state must have a session_id")
        with self._lock:
            for state in states:
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


def create_empty_cache(session_id: str | None = None, device=None) -> CacheState:
    """Create an empty cache state."""
    state = CacheState(session_id=session_id, next_position_id=0)
    return state.to(device) if device is not None else state


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


# Backwards-compatible private name used by the original draft tests.
_deep_copy_past_kv = copy_past_key_values


class BatchedKVHelper:
    """Stack and unstack sessions with heterogeneous decoder cache lengths."""

    @staticmethod
    def stack_for_update(
        caches: list[CacheState],
        new_input_ids: list[torch.Tensor],
        new_attention_masks: list[torch.Tensor],
        device: torch.device,
        pad_token_id: int = 0,
    ) -> dict[str, Any]:
        if not caches:
            raise ValueError("At least one cache is required")
        if not (len(caches) == len(new_input_ids) == len(new_attention_masks)):
            raise ValueError("Cache, input, and mask batches must have equal lengths")

        cached_lengths = [cache.cached_length for cache in caches]
        new_lengths = [int(mask.long().sum().item()) for mask in new_attention_masks]
        max_cached = max(cached_lengths, default=0)
        max_new = max(new_lengths, default=0)
        batch_size = len(caches)

        padded_ids = torch.full(
            (batch_size, max_new),
            pad_token_id,
            dtype=new_input_ids[0].dtype,
            device=device,
        )
        current_mask = torch.zeros(batch_size, max_new, dtype=torch.long, device=device)
        position_ids = torch.zeros(batch_size, max_new, dtype=torch.long, device=device)

        for row, (ids, mask, length, cache) in enumerate(
            zip(new_input_ids, new_attention_masks, new_lengths, caches)
        ):
            flat_ids = ids.reshape(-1).to(device)
            flat_mask = mask.reshape(-1).bool().to(device)
            valid_ids = flat_ids[flat_mask]
            padded_ids[row, :length] = valid_ids
            current_mask[row, :length] = 1
            position_ids[row, :length] = torch.arange(
                cache.next_position,
                cache.next_position + length,
                device=device,
            )

        full_mask = torch.zeros(batch_size, max_cached + max_new, dtype=torch.long, device=device)
        for row, (cached_length, new_length) in enumerate(zip(cached_lengths, new_lengths)):
            cache_pad = max_cached - cached_length
            full_mask[row, cache_pad:max_cached] = 1
            full_mask[row, max_cached : max_cached + new_length] = 1

        stacked_past = None
        if max_cached:
            stacked_past = BatchedKVHelper._stack_past_kv(caches, max_cached, device)

        return {
            "input_ids": padded_ids,
            "attention_mask": full_mask,
            "current_attention_mask": current_mask,
            "position_ids": position_ids,
            "past_key_values": stacked_past,
            "cached_lengths": cached_lengths,
            "new_lengths": new_lengths,
            "max_cached_len": max_cached,
        }

    @staticmethod
    def unstack_after_update(
        new_past_kv,
        stacked_info: dict[str, Any],
        old_caches: list[CacheState],
        new_input_ids: list[torch.Tensor],
        new_attention_masks: list[torch.Tensor],
    ) -> list[CacheState]:
        results: list[CacheState] = []
        max_cached = stacked_info["max_cached_len"]
        for row, (old, cached_length, new_length, ids, mask) in enumerate(
            zip(
                old_caches,
                stacked_info["cached_lengths"],
                stacked_info["new_lengths"],
                new_input_ids,
                new_attention_masks,
            )
        ):
            start = max_cached - cached_length
            end = max_cached + new_length
            sliced_kv = BatchedKVHelper._slice_past_kv(new_past_kv, row, start, end)
            valid_ids = ids.reshape(-1)[mask.reshape(-1).bool()]
            valid_mask = torch.ones_like(valid_ids)
            results.append(
                replace(
                    old,
                    past_key_values=sliced_kv,
                    input_ids=torch.cat([old.input_ids, valid_ids]),
                    attention_mask=torch.cat([old.attention_mask, valid_mask]),
                    cached_length=cached_length + new_length,
                    next_position_id=old.next_position + new_length,
                    metadata=old.metadata.copy(),
                )
            )
        return results

    @staticmethod
    def _stack_past_kv(caches: list[CacheState], max_cached: int, device: torch.device):
        non_empty = [cache for cache in caches if cache.past_key_values is not None]
        if not non_empty:
            return None

        cache_layers = [
            {idx: (keys, values) for idx, keys, values in _cache_layers(cache.past_key_values)}
            for cache in caches
        ]
        layer_indices = sorted({layer_idx for layers in cache_layers for layer_idx in layers})
        stacked = DynamicCache()

        for layer_idx in layer_indices:
            reference_key, reference_value = next(layers[layer_idx] for layers in cache_layers if layer_idx in layers)
            keys: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            for layers in cache_layers:
                if layer_idx not in layers:
                    key_shape = list(reference_key.shape)
                    value_shape = list(reference_value.shape)
                    key_shape[0] = value_shape[0] = 1
                    key_shape[-2] = value_shape[-2] = max_cached
                    key = reference_key.new_zeros(key_shape, device=device)
                    value = reference_value.new_zeros(value_shape, device=device)
                else:
                    key, value = layers[layer_idx]
                    key = key.to(device)
                    value = value.to(device)
                    pad = max_cached - key.shape[-2]
                    if pad:
                        key_shape = list(key.shape)
                        value_shape = list(value.shape)
                        key_shape[-2] = value_shape[-2] = pad
                        key = torch.cat([key.new_zeros(key_shape), key], dim=-2)
                        value = torch.cat(
                            [value.new_zeros(value_shape), value],
                            dim=-2,
                        )
                keys.append(key)
                values.append(value)
            stacked.update(torch.cat(keys), torch.cat(values), layer_idx)
        return stacked

    @staticmethod
    def _slice_past_kv(past_kv, batch_idx: int, start: int, end: int):
        if past_kv is None:
            return None
        sliced = DynamicCache()
        for layer_idx, keys, values in _cache_layers(past_kv):
            key_slices = [slice(None)] * keys.ndim
            value_slices = [slice(None)] * values.ndim
            key_slices[0] = value_slices[0] = slice(batch_idx, batch_idx + 1)
            key_slices[-2] = value_slices[-2] = slice(start, end)
            sliced.update(
                keys[tuple(key_slices)].clone(),
                values[tuple(value_slices)].clone(),
                layer_idx,
            )
        return sliced
