"""Reuse the token ids of a conversation's history: multi-turn chat resends every earlier message on each turn."""

import hashlib
from array import array
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Tuple


class PrefixTokenCache:
    """Token ids of rendered prompts, keyed by prefixes that end right before ``boundary``.

    ``boundary`` is an added special token that strips no whitespace: the tokenizer splits the text on it before
    normalization and BPE, so ``encode(a + b) == encode(a) + encode(b)`` whenever ``b`` starts with it.
    """

    def __init__(self, *, boundary: str, max_entries: int):
        self._boundary = boundary
        self._max_entries = max_entries
        self._entries: OrderedDict[bytes, array] = OrderedDict()

    def encode(self, *, text: str, encode: Callable[[str], List[int]]) -> List[int]:
        # The generation prompt starts at the last boundary; everything before it is the next turn's prefix.
        tail_start = text.rfind(self._boundary)
        if tail_start <= 0:
            return list(encode(text))
        keys = self._prefix_keys(text=text, end=tail_start)
        cut, head = self._longest_cached(keys=keys)
        prefix = array("I", head)
        if cut < tail_start:
            prefix.extend(encode(text[cut:tail_start]))
        self._store(key=keys[-1][1], ids=prefix)
        return prefix.tolist() + list(encode(text[tail_start:]))

    def _prefix_keys(self, *, text: str, end: int) -> List[Tuple[int, bytes]]:
        """(position, digest of text[:position]) for every boundary in (0, end], in text order."""
        keys = []
        digest = hashlib.blake2b(digest_size=16)
        start = 0
        cut = text.find(self._boundary, 1)
        while 0 < cut <= end:
            digest.update(text[start:cut].encode("utf-8", "surrogatepass"))
            keys.append((cut, digest.copy().digest()))
            start = cut
            cut = text.find(self._boundary, cut + 1)
        return keys

    def _longest_cached(self, *, keys: List[Tuple[int, bytes]]) -> Tuple[int, array]:
        for position, key in reversed(keys):
            ids = self._entries.get(key)
            if ids is not None:
                self._entries.move_to_end(key)
                return position, ids
        return 0, array("I")

    def _store(self, *, key: bytes, ids: array) -> None:
        self._entries[key] = ids
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


def build_prefix_token_cache(
    *, tokenizer: Any, encode_kwargs: dict, max_entries: int
) -> Optional[PrefixTokenCache]:
    """A cache split at the chat template's turn-start token, or None when that split is not lossless here."""
    boundary = _turn_start_token(tokenizer=tokenizer)
    if boundary is None:
        return None
    cache = PrefixTokenCache(boundary=boundary, max_entries=max_entries)
    probe = [
        {"role": "user", "content": "Hello there, how are you?"},
        {"role": "assistant", "content": " Fine.\n\nThanks for asking "},
        {"role": "user", "content": "\nAnd now?  "},
    ]
    try:
        encode = lambda part: tokenizer.encode(part, **encode_kwargs)
        for turns in (1, 3):
            text = tokenizer.apply_chat_template(
                probe[:turns], tokenize=False, add_generation_prompt=True
            )
            if cache.encode(text=text, encode=encode) != list(encode(text)):
                return None
    except Exception:
        return None
    return PrefixTokenCache(boundary=boundary, max_entries=max_entries)


def _turn_start_token(*, tokenizer: Any) -> Optional[str]:
    """The added special token that opens the generation prompt (e.g. <|start|>, <|im_start|>), if any."""
    probe = [{"role": "user", "content": "x"}]
    try:
        plain = tokenizer.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=False
        )
        prompted = tokenizer.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True
        )
        added = tokenizer.added_tokens_decoder.values()
    except Exception:
        return None
    if not prompted.startswith(plain) or len(prompted) == len(plain):
        return None
    generation_prompt = prompted[len(plain) :]
    candidates = [
        token.content
        for token in added
        if token.special
        and not token.lstrip
        and not token.rstrip
        and not token.normalized
        and generation_prompt.startswith(token.content)
    ]
    return max(candidates, key=len) if candidates else None
