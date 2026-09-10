from __future__ import annotations

import os
from typing import Any


class KnowledgeRetriever:
    """Best-effort Pinecone context retrieval with a local fallback."""

    def __init__(self) -> None:
        self.namespace = os.getenv("PINECONE_NAMESPACE", "cardiofusion_knowledge")
        self._index: Any = None
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX")
        if api_key and index_name:
            try:
                from pinecone import Pinecone

                self._index = Pinecone(api_key=api_key).Index(index_name)
            except Exception:
                self._index = None

    @property
    def available(self) -> bool:
        return self._index is not None

    def query(self, vector: list[float], top_k: int = 3) -> list[dict[str, Any]]:
        if self._index is None:
            return []
        result = self._index.query(
            vector=vector,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
        )
        matches = result.get("matches", [])
        return [match.to_dict() if hasattr(match, "to_dict") else dict(match) for match in matches]