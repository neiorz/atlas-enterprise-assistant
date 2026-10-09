"""
Ingestion orchestration (FR-B).

"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from src.chunking import detect_language, split_text
from src.loaders import load_document, strip_markdown_headings
from src.settings import settings

_POLICY_ID_LABEL_RE = re.compile(
    r"(?:Policy\s*ID|Document\s*ID|رقم\s*السياسة|رقم\s*الوثيقة)"
    r"\**\s*[:：]?\**\s*([A-Z]{2,5}-[A-Z]{2,5}-\d{3})"
)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    domain: str
    source_filename: str
    language: str
    doc_id: str


def iter_corpus_files() -> Iterator[tuple[str, Path]]:
    """Yield (domain, path) for every file under each domain folder.

    Raises:
        FileNotFoundError: if an expected domain folder is missing.
    """
    for domain in settings.domains:
        domain_dir = settings.docs_dir / domain
        if not domain_dir.exists():
            raise FileNotFoundError(f"Expected domain folder missing: {domain_dir}")
        for path in sorted(domain_dir.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                yield domain, path


def build_chunks() -> list[Chunk]:
    """Load, chunk, and tag every document in the corpus.

    Returns:
        All chunks across all domains, in corpus order.
    """
    chunks: list[Chunk] = []
    for domain, path in iter_corpus_files():
        raw_text = load_document(path)
        if path.suffix.lower() == ".md":
            raw_text = strip_markdown_headings(raw_text)

        language = detect_language(raw_text)
        doc_id = path.stem
        pieces = split_text(raw_text, settings.chunk_size, settings.chunk_overlap)

        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{i}",
                    text=piece,
                    domain=domain,
                    source_filename=path.name,
                    language=language,
                    doc_id=doc_id,
                )
            )
    return chunks


def extract_policy_id(text: str) -> str | None:
    """Return the document's self-declared policy/document ID, if present."""
    match = _POLICY_ID_LABEL_RE.search(text)
    return match.group(1) if match else None


def build_policy_index() -> dict[str, dict]:
    """Map each document's self-declared policy ID to its domain/filename.

    Backs the policy_lookup tool with a fast, exact lookup instead of
    relying on the LLM to recall or guess an ID -> file mapping.

    Returns:
        {policy_id: {"domain": ..., "source_filename": ...}}
    """
    index: dict[str, dict] = {}
    for domain, path in iter_corpus_files():
        raw_text = load_document(path)
        policy_id = extract_policy_id(raw_text)
        if policy_id:
            index[policy_id] = {"domain": domain, "source_filename": path.name}
    return index


if __name__ == "__main__":
    import json

    all_chunks = build_chunks()
    print(f"Loaded {len(all_chunks)} chunks from {settings.docs_dir}")

    by_domain: dict[str, int] = {}
    by_lang: dict[str, int] = {}
    for c in all_chunks:
        by_domain[c.domain] = by_domain.get(c.domain, 0) + 1
        by_lang[c.language] = by_lang.get(c.language, 0) + 1
    print("By domain:", by_domain)
    print("By language:", by_lang)

    policy_index = build_policy_index()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    policy_index_path = settings.data_dir / "policy_index.json"
    policy_index_path.write_text(
        json.dumps(policy_index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Policy index: {len(policy_index)} IDs -> {policy_index_path}")
