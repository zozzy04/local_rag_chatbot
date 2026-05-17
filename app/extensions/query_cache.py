"""Estensione §5 #8 — Query Cache con Similarity Threshold.

Mantiene un cache in-memory delle ultime N risposte. Per ogni nuova query,
calcola la cosine similarity tra il suo embedding e quelli in cache.
Se sim > threshold restituisce la risposta cached senza toccare Qdrant/LLM.

Impatto atteso: -latenza drastica sulle query frequenti o semanticamente simili.
Nessun impatto su Recall/MRR (non cambia la qualità, solo la velocità).
"""
import logging
import time
from collections import OrderedDict
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class QueryCache:
    """LRU cache semantica per query RAG."""

    def __init__(self, max_size: int = 100, similarity_threshold: float = 0.95):
        self.max_size = max_size
        self.threshold = similarity_threshold
        # OrderedDict mantiene ordine di inserimento per LRU
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _embed(self, text: str, embedder) -> np.ndarray:
        vec = embedder.embed_query(text)
        return np.array(vec, dtype=np.float32)

    def lookup(self, question: str, embedder) -> dict | None:
        """Cerca nel cache. Ritorna il risultato cached se sim > threshold."""
        if not self._cache:
            self._misses += 1
            return None

        q_emb = self._embed(question, embedder)

        for key, entry in reversed(list(self._cache.items())):
            sim = _cosine(q_emb, entry["embedding"])
            if sim >= self.threshold:
                self._hits += 1
                self._cache.move_to_end(key)
                logger.info(f"Cache HIT (sim={sim:.3f}) per: '{question[:60]}'")
                return entry["result"]

        self._misses += 1
        return None

    def store(self, question: str, result: dict, embedder):
        """Salva la query e il suo risultato nel cache."""
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        q_emb = self._embed(question, embedder)
        self._cache[question] = {"embedding": q_emb, "result": result}
        logger.info(f"Cache STORE per: '{question[:60]}'")

    def stats(self) -> dict:
        return {
            "cache_size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, self._hits + self._misses),
        }


_query_cache = QueryCache()


def get_query_cache() -> QueryCache:
    return _query_cache
