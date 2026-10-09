from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.settings import settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Loaded once per process — the model is a few hundred MB to ~2GB."""
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed chunk texts for indexing. Normalized so cosine similarity
    in Qdrant behaves as expected."""
    if not texts:
        return []
    model = get_embedder()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=16,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
