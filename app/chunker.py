"""Cut streaming LLM text into pieces the TTS can start speaking right away.

Thai rarely uses full stops: spaces separate phrases and sentences. So a space
counts as a boundary here, and the first piece is cut as early as possible
(the time to the first audible word matters most), while later pieces are
longer so the prosody stays natural.
"""

from __future__ import annotations

import re
from collections.abc import Callable

_STRONG = ".!?…\n"
_MARKDOWN = re.compile(r"(\*\*|__|`+|^#{1,6}\s*|^\s*[-*+•]\s+|^\s*\d+[.)]\s+)", re.MULTILINE)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"https?://\S+")
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF️‍]+"
)


def clean_for_speech(text: str) -> str:
    """Strip what a TTS would read out literally: markdown, links, emoji."""
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    text = _MARKDOWN.sub("", text)
    text = _EMOJI.sub("", text)
    text = text.replace("*", "").replace("#", "")
    return re.sub(r"[ \t]+", " ", text)


class SpeechChunker:
    """`segment` splits Thai text into words; with it, a long first phrase with no
    space yet is cut at a word boundary instead of waiting for the next space."""

    def __init__(
        self,
        first_min: int = 6,
        next_min: int = 40,
        hard_max: int = 160,
        segment: Callable[[str], list[str]] | None = None,
        first_word_cut: int = 10,
        first_wait: int = 16,
    ) -> None:
        self.first_min = first_min
        self.next_min = next_min
        self.hard_max = hard_max
        self.segment = segment
        self.first_word_cut = first_word_cut
        self.first_wait = first_wait
        self._buf = ""
        self._emitted = 0

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while True:
            piece = self._take()
            if piece is None:
                break
            out.append(piece)
        return out

    def flush(self) -> list[str]:
        piece = clean_for_speech(self._buf).strip()
        self._buf = ""
        return [piece] if _speakable(piece) else []

    def _take(self) -> str | None:
        minimum = self.first_min if self._emitted == 0 else self.next_min
        cut = self._find_cut(minimum)
        if cut is None:
            return None
        head, self._buf = self._buf[:cut], self._buf[cut:]
        piece = clean_for_speech(head).strip()
        if not _speakable(piece):
            return self._take()
        self._emitted += 1
        return piece

    def _find_cut(self, minimum: int) -> int | None:
        buf = self._buf
        # A sentence end is always a good place, once there is enough text.
        for i, ch in enumerate(buf):
            if ch in _STRONG and _is_sentence_end(buf, i) and i + 1 >= minimum // 2:
                return i + 1
        # Otherwise the last space after the minimum length; waiting for the
        # next space keeps a word that is still streaming in from being split.
        if len(buf) >= minimum:
            space = buf.rfind(" ", minimum - 1)
            if space > 0:
                return space + 1
        if self._emitted == 0 and self.segment and len(buf) >= self.first_wait:
            cut = self._word_cut(buf, self.first_word_cut)
            if cut:
                return cut
        if len(buf) >= self.hard_max:
            space = buf.rfind(" ", 0, self.hard_max)
            if space > 0:
                return space + 1
            return (self._word_cut(buf[: self.hard_max], self.hard_max // 2) if self.segment else None) or self.hard_max
        return None

    def _word_cut(self, buf: str, at_least: int) -> int | None:
        """End of the first complete word reaching `at_least` characters.

        The last word is never used: it may still be arriving.
        """
        words = self.segment(buf)
        end = 0
        for word in words[:-1]:
            end += len(word)
            if end >= at_least:
                return end if buf[:end] == "".join(words)[:end] else None
        return None


def thai_word_segmenter() -> Callable[[str], list[str]] | None:
    """PyThaiNLP's dictionary segmenter (well under a millisecond per call), if installed."""
    try:
        from pythainlp.tokenize import word_tokenize
    except ImportError:
        return None
    return lambda text: word_tokenize(text, engine="newmm", keep_whitespace=True)


def _is_sentence_end(buf: str, i: int) -> bool:
    if buf[i] == "\n":
        return True
    if i + 1 >= len(buf):
        return False  # might be a decimal point or "..." still arriving
    nxt = buf[i + 1]
    return nxt.isspace() or (buf[i] in "!?…" and not nxt.isdigit())


def _speakable(text: str) -> bool:
    return any(ch.isalnum() for ch in text)
