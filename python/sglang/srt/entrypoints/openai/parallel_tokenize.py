"""Lossless parallel encode of long prompts (OKC_PARTOK, default on; OKC_PARTOK=0 keeps the serial encode).

A cold first turn renders to ~290 KB, which the tokenizer manager encodes serially in its event loop (~150 ms on the
serving node). Here the text is cut only where the tokenizer's own pre-tokenizer cuts anyway, the pieces are encoded on
a small thread pool (the Rust encode releases the GIL) and the ids are concatenated: exactly the serial ids.

Cuts sit right after a '\\n' that is followed by an ASCII letter, and only for the one tokenizer layout accepted below
(Muse): no normalizer; pre-tokenizer Split(_SPLIT_REGEX, Isolated) then ByteLevel without prefix space or regex; BPE
without dropout. No alternative of that regex can hold a '\\n' followed by a letter, and none looks behind, so the match
holding the '\\n' ends at the cut and the pre-tokens of text[:p] + text[p:] are those of text; BPE runs per pre-token.
Added tokens are split out first and cannot span a cut (checked at build: none holds '\\n' or '\\r', none strips or is
single-word). Everything else keeps the serial encode: another layout, text shorter than OKC_PARTOK_MIN_CHARS (8192),
no cut point, fewer than two workers (OKC_PARTOK_WORKERS, default min(8, usable cpus // 16)), or a build probe that
differs. OKC_PARTOK_CHECK=1 (tests only) encodes every parallel text both ways and raises on any difference.
"""

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

_SPLIT_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
_ASCII_LETTERS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_PROBES = (
    "Hello\nWorld\nfoo bar\nBaz qux\nend",
    "x = 1\nif x:\n    y = (x)\nprint(y)\nreturn {'a': [1, 2]}\n",
    "a)\nb]\nc.\nd,\ne;\nf:\ng/\nh//\ni-\nj'\nk\"\nl",
    "  \n\nabc\n\n\n   def\n\t\nghi \n jkl\r\nmno\r\n\r\npqr\n\rstu",
    "12\n34\nab12\n5x\n678\n9a",
    "中文\nabc\n日本語\nxyz\n한국어\nQ",
    "😀\nabc\n😀😀\nd\u200d\ne\u0301\nf\n\u0301g",
    "don't\nwe'll\nI'M\nyou're\nit's",
    "<|start|>user<|message|>hi\nthere<|eot|>\n<|start|>assistant to=user<|message|>ok\nbye<|eom|>\nx",
)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


def _usable_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _layout_problem(cfg: dict) -> Optional[str]:
    """None when the tokenizer is the layout the cut rule is proven for, else why not."""
    if cfg.get("normalizer") is not None:
        return "has a normalizer"
    pre = cfg.get("pre_tokenizer") or {}
    seq = pre.get("pretokenizers") if pre.get("type") == "Sequence" else None
    if not seq or len(seq) != 2:
        return "pre-tokenizer is not Split + ByteLevel"
    split, byte_level = seq
    if (
        split.get("type") != "Split"
        or (split.get("pattern") or {}).get("Regex") != _SPLIT_REGEX
        or split.get("behavior") != "Isolated"
        or split.get("invert")
    ):
        return "split regex / behavior differs"
    if byte_level.get("type") != "ByteLevel" or byte_level.get("add_prefix_space") or byte_level.get("use_regex"):
        return "byte-level step differs"
    model = cfg.get("model") or {}
    if model.get("type") != "BPE" or model.get("dropout"):
        return "model is not a dropout-free BPE"
    for tok in cfg.get("added_tokens") or []:
        content = tok.get("content") or ""
        if "\n" in content or "\r" in content or tok.get("lstrip") or tok.get("rstrip") or tok.get("single_word"):
            return f"added token {content!r} could span a cut"
    return None


class ParallelEncoder:
    def __init__(self, *, tokenizer: Any, encode_kwargs: dict, workers: int, min_chars: int, check: bool):
        self.tokenizer = tokenizer
        self.encode_kwargs = dict(encode_kwargs)
        self.workers = workers
        self.min_chars = min_chars
        self.check = check
        self._pool: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._checked = False
        self._failed = False

    def pieces(self, text: str, piece_chars: Optional[int] = None) -> List[str]:
        """text cut after every '\\n' + ASCII letter that is at least piece_chars past the previous cut."""
        n = len(text)
        if piece_chars is None:
            if n < self.min_chars:
                return [text]
            # About two pieces per worker, so one slow piece does not hold the whole encode.
            piece_chars = max(self.min_chars // 2, -(-n // (2 * self.workers)))
        cuts, pos = [], 0
        while True:
            i = text.find("\n", pos + max(piece_chars, 1) - 1)
            while 0 <= i < n - 1 and text[i + 1] not in _ASCII_LETTERS:
                i = text.find("\n", i + 1)
            if i < 0 or i >= n - 1:
                break
            cuts.append(i + 1)
            pos = i + 1
        return [text[a:b] for a, b in zip([0] + cuts, cuts + [n])]

    def _executor(self) -> ThreadPoolExecutor:
        if self._pool is None:
            with self._lock:
                if self._pool is None:
                    self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="okc-partok")
        return self._pool

    def encode_pieces(self, encode: Callable[[str], List[int]], pieces: List[str], pool: bool = True) -> List[int]:
        ids: List[int] = []
        for part in self._executor().map(encode, pieces) if pool else map(encode, pieces):
            ids.extend(part)
        return ids

    def wrap(self, encode: Callable[[str], List[int]]) -> Callable[[str], List[int]]:
        """encode, made parallel for long texts; the same ids."""

        def encode_parallel(text: str) -> List[int]:
            pieces = self.pieces(text)
            if len(pieces) < 2 or self._failed:
                return encode(text)
            try:
                ids = self.encode_pieces(encode, pieces)
            except Exception as e:  # a broken pool must not fail the request: serial from now on
                self._failed = True
                logger.warning("OKC partok: pool encode failed (%s: %s); serial from now on", type(e).__name__, e)
                return encode(text)
            if self.check:
                ref = list(encode(text))
                if ids != ref:
                    at = next((i for i, (a, b) in enumerate(zip(ids, ref)) if a != b), min(len(ids), len(ref)))
                    raise RuntimeError(
                        f"OKC partok check: parallel encode differs from the serial one at token {at} "
                        f"({len(ids)} vs {len(ref)} tokens, {len(pieces)} pieces, {len(text)} chars)"
                    )
                if not self._checked:
                    self._checked = True
                    logger.info(
                        "OKC partok check: first parallel encode equals the serial one (%d tokens, %d pieces)",
                        len(ids),
                        len(pieces),
                    )
            return ids

        return encode_parallel


def build_parallel_encoder(*, tokenizer: Any, encode_kwargs: dict) -> Optional[ParallelEncoder]:
    """A ParallelEncoder for this tokenizer and encode kwargs, or None (serial encode) with the reason logged once."""
    if os.environ.get("OKC_PARTOK", "1") == "0":
        logger.info("OKC partok off: OKC_PARTOK=0")
        return None
    cpus = _usable_cpus()
    workers = _int_env("OKC_PARTOK_WORKERS", 0) or min(8, cpus // 16)
    problem = None if workers >= 2 else f"{workers} worker(s) from {cpus} usable cpus"
    if problem is None:
        try:
            backend = getattr(tokenizer, "backend_tokenizer", None) or getattr(tokenizer, "_tokenizer", None)
            problem = _layout_problem(json.loads(backend.to_str()))
        except Exception as e:
            problem = f"tokenizer layout unreadable ({type(e).__name__})"
    enc = None
    if problem is None:
        enc = ParallelEncoder(
            tokenizer=tokenizer,
            encode_kwargs=encode_kwargs,
            workers=workers,
            min_chars=_int_env("OKC_PARTOK_MIN_CHARS", 8192),
            check=os.environ.get("OKC_PARTOK_CHECK", "0") == "1",
        )
        encode = lambda text: tokenizer.encode(text, **encode_kwargs)
        try:
            for probe in _PROBES:
                if enc.encode_pieces(encode, enc.pieces(probe, piece_chars=1), pool=False) != list(encode(probe)):
                    problem = f"probe {probe[:24]!r} encodes differently when cut"
                    break
        except Exception as e:
            problem = f"probe failed ({type(e).__name__}: {e})"
    if problem is not None:
        logger.info("OKC partok off: %s", problem)
        return None
    logger.info(
        "OKC partok on: %d workers, texts >= %d chars%s", enc.workers, enc.min_chars, ", CHECK on" if enc.check else ""
    )
    return enc
