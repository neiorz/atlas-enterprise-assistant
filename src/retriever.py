"""
Retrieval layer (FR-C).

"""

from __future__ import annotations

from dataclasses import dataclass

from src.settings import settings
from src.vectorstore import search


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source_filename: str
    domain: str
    language: str
    score: float


def retrieve(
    question: str, domain: str, top_k: int | None = None
) -> list[RetrievedChunk]:
    """Domain-filtered retrieval (FR-C1, FR-C2, FR-C3).

    No separate language filter is applied on the query side. A bilingual
    document is tagged domain=<its domain> regardless of language, so it's
    already inside the domain-filtered slice; and because bge-m3 embeds
    English and Arabic into the same semantic space, an Arabic question
    naturally scores highest against Arabic/bilingual chunks and an English
    question against English/bilingual ones — that's what makes FR-C4
    (cross-language retrieval) work without extra logic.
    """
    hits = search(
        query=question, domain=domain, top_k=top_k or settings.retrieval_top_k
    )
    return [
        RetrievedChunk(
            chunk_id=hit.payload["chunk_id"],
            text=hit.payload["text"],
            source_filename=hit.payload["source_filename"],
            domain=hit.payload["domain"],
            language=hit.payload["language"],
            score=hit.score,
        )
        for hit in hits
    ]


def retrieve_multi_domain(
    question: str, domains: list[str], top_k: int | None = None
) -> list[RetrievedChunk]:
    """Fan-out retrieval across several domains, merged and deduped by score.

    Used by the router's low-confidence fallback (FR-D2) — when the router
    isn't sure which single domain applies, search all of them rather than
    guessing wrong and returning nothing useful.
    """
    per_domain_k = top_k or settings.retrieval_top_k
    merged: dict[str, RetrievedChunk] = {}
    for domain in domains:
        for chunk in retrieve(question, domain, top_k=per_domain_k):
            if (
                chunk.chunk_id not in merged
                or chunk.score > merged[chunk.chunk_id].score
            ):
                merged[chunk.chunk_id] = chunk
    return sorted(merged.values(), key=lambda c: c.score, reverse=True)[:per_domain_k]


def unique_sources(chunks: list[RetrievedChunk]) -> list[str]:
    """Order-preserving, de-duplicated filenames for citations (FR-C3, FR-G1, FR-G2)."""
    seen: list[str] = []
    for c in chunks:
        if c.source_filename not in seen:
            seen.append(c.source_filename)
    return seen


def format_context_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a labeled context block for the answer
    prompt, so the model can tie each fact back to a specific filename."""
    blocks = []
    for c in chunks:
        blocks.append(f"[Source: {c.source_filename}]\n{c.text}")
    return "\n\n---\n\n".join(blocks)
