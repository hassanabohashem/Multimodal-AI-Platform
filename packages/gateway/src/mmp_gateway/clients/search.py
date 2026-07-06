"""Qdrant search operations owned by the gateway."""
from __future__ import annotations

from qdrant_client import AsyncQdrantClient, models

from mmp_common.schemas import SearchFilters, SearchHit

COLLECTION = "images_siglip2_v1"
DIM = 1152  # SigLIP-2 so400m


class SearchStore:
    """Thin wrapper over Qdrant for retrieval and ingestion."""

    def __init__(self, url: str) -> None:
        self._client = AsyncQdrantClient(url=url)

    async def ensure_collection(self) -> None:
        """Create the collection with int8 quantization if it doesn't exist."""
        if await self._client.collection_exists(COLLECTION):
            return
        await self._client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
            ),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
        )
        await self._client.create_payload_index(COLLECTION, "tags", models.PayloadSchemaType.KEYWORD)
        await self._client.create_payload_index(COLLECTION, "source", models.PayloadSchemaType.KEYWORD)

    async def search(
        self, vector: list[float], top_k: int, filters: SearchFilters | None
    ) -> list[SearchHit]:
        """Vector search with optional payload filters."""
        conditions: list[models.FieldCondition] = []
        if filters and filters.tags:
            conditions.append(models.FieldCondition(key="tags", match=models.MatchAny(any=filters.tags)))
        if filters and filters.source:
            conditions.append(models.FieldCondition(key="source", match=models.MatchValue(value=filters.source)))
        res = await self._client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=top_k,
            query_filter=models.Filter(must=conditions) if conditions else None,
            with_payload=True,
        )
        return [
            SearchHit(
                image_id=str(p.id),
                score=float(p.score),
                uri=str(p.payload.get("uri", "")),
                thumb_uri=p.payload.get("thumb_uri"),
                caption=p.payload.get("caption"),
            )
            for p in res.points
        ]
