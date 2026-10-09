"""Language detection and chunking (FR-B4, FR-B5)."""

from __future__ import annotations

import re

# Arabic script block (covers standard Arabic letters + Arabic-Indic digits).
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")

# Sentence-ish boundaries for both English and Arabic punctuation.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟])\s+")


def detect_language(text: str) -> str:
    """Tag text as en / ar / bilingual by character-script ratio.

    Args:
        text: Raw text to classify.

    Returns:
        "ar" if >85% of scripted characters are Arabic, "en" if <15% are,
        otherwise "bilingual".
    """
    arabic_chars = len(_ARABIC_RE.findall(text))
    latin_chars = len(_LATIN_RE.findall(text))
    total = arabic_chars + latin_chars
    if total == 0:
        return "en"
    arabic_ratio = arabic_chars / total
    if arabic_ratio > 0.85:
        return "ar"
    if arabic_ratio < 0.15:
        return "en"
    return "bilingual"


def _split_long_unit(unit: str, chunk_size: int) -> list[str]:
    """Split one paragraph that's still too long, sentence by sentence,
    falling back to a hard character split for any run-on sentence.

    Args:
        unit: A single paragraph exceeding chunk_size.
        chunk_size: Maximum characters per output piece.

    Returns:
        A list of text pieces, each at most chunk_size characters.
    """
    sentences = _SENTENCE_SPLIT_RE.split(unit)
    out: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                out.append(current)
            if len(sentence) <= chunk_size:
                current = sentence
            else:
                for i in range(0, len(sentence), chunk_size):
                    out.append(sentence[i : i + chunk_size])
                current = ""
    if current:
        out.append(current)
    return out


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Chunk text: paragraph -> sentence -> hard character split, with a
    trailing-overlap window carried into the start of the next chunk.

    Written from scratch rather than pulling in a framework splitter — this
    keeps the ingestion path dependency-light and fully inspectable on a
    30-document corpus where chunk boundaries are easy to eyeball by hand.

    Args:
        text: Full document text to chunk.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Characters of the previous chunk's tail repeated at
            the start of the next chunk, to preserve context across a split.

    Returns:
        Ordered list of chunk texts.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    raw_chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            raw_chunks.append(current)
        if len(para) <= chunk_size:
            current = para
        else:
            sub_chunks = _split_long_unit(para, chunk_size)
            if sub_chunks:
                raw_chunks.extend(sub_chunks[:-1])
                current = sub_chunks[-1]
            else:
                current = ""
    if current:
        raw_chunks.append(current)

    if chunk_overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        tail = raw_chunks[i - 1][-chunk_overlap:]
        overlapped.append(f"{tail} {raw_chunks[i]}".strip())
    return overlapped
