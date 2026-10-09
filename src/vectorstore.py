"""
Qdrant vector store (FR-B7, FR-B8, FR-C1..C4).

"""

from __future__ import annotations

import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.embeddings import embed_query, embed_texts
from src.ingest import Chunk, build_chunks
from src.settings import settings


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """One client per process, reused across searches.

    Opening a local-path Qdrant client involves storage/lock setup, so doing
    it per query would add avoidable latency to every turn (NFR-B1) and can
    collide with a second client on the same path. Cloud mode uses the same
    caching since a client is cheap to reuse either way.
    """
    if settings.qdrant_url:
        return QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key or None
        )
    return QdrantClient(path=settings.qdrant_local_path)


def collection_exists(client: QdrantClient) -> bool:
    names = [c.name for c in client.get_collections().collections]
    return settings.qdrant_collection in names


def _create_collection(client: QdrantClient) -> None:
    if collection_exists(client):
        client.delete_collection(settings.qdrant_collection)

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=qm.VectorParams(
            size=settings.embedding_dim, distance=qm.Distance.COSINE
        ),
    )

    # Payload indexes only accelerate filtering on a Qdrant *server*; the local
    # embedded mode ignores them and emits a UserWarning if we ask for them
    # (it filters by scanning, which is instant on 122 points). So create them
    # only when we're actually talking to a server via QDRANT_URL.
    if settings.qdrant_url:
        for field in ("domain", "language", "source_filename"):
            client.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name=field,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def build_index(force: bool = False) -> int:
    """Build (or reuse) the persisted index. FR-B7: a re-run loads the
    existing index unless force=True (wired to scripts/reset.sh)."""
    client = get_client()

    if collection_exists(client) and not force:
        existing = client.count(settings.qdrant_collection).count
        print(
            f"Index already exists with {existing} vectors — reusing it. "
            f"Run with --force (or scripts/reset.sh) to rebuild."
        )
        return existing

    _create_collection(client)

    chunks: list[Chunk] = build_chunks()
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)

    points = [
        qm.PointStruct(
            id=_point_id(chunk.chunk_id),
            vector=vector,
            payload={
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "domain": chunk.domain,
                "source_filename": chunk.source_filename,
                "language": chunk.language,
                "doc_id": chunk.doc_id,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    client.upsert(collection_name=settings.qdrant_collection, points=points)
    return len(points)


def search(query: str, domain: str | None = None, top_k: int | None = None):
    """Low-level vector search. Returns raw Qdrant ScoredPoint hits;
    src/retriever.py wraps this into the typed result the graph consumes.

    Uses ``query_points`` — qdrant-client >= 1.14 removed the older
    ``client.search(query_vector=...)`` call entirely (it raised AttributeError
    on 1.19.1), so this is the supported spelling going forward.
    """
    client = get_client()

    # NFR-D4: a missing index must surface as an actionable message, never a
    # raw qdrant traceback on the user's first question.
    if not collection_exists(client):
        raise FileNotFoundError(
            f"Vector index not found at '{settings.qdrant_local_path}'. "
            f"Run `python -m src.ingest && python -m src.vectorstore` "
            f"(or `bash scripts/reset.sh`) to build it first."
        )

    query_vector = embed_query(query)

    query_filter = None
    if domain is not None:
        query_filter = qm.Filter(
            must=[qm.FieldCondition(key="domain", match=qm.MatchValue(value=domain))]
        )

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k or settings.retrieval_top_k,
        with_payload=True,
    )
    return response.points


if __name__ == "__main__":
    import sys

    force_rebuild = "--force" in sys.argv
    n = build_index(force=force_rebuild)
    print(f"Index ready: {n} vectors in collection '{settings.qdrant_collection}'.")
