"""Estensione §5 #2 — Hybrid Search: BM25 + Dense con Reciprocal Rank Fusion.

Strategia: per ogni query combina i risultati BM25 (lessicale, exact-match) con
quelli del dense retriever (semantico, Qdrant). La fusione avviene via RRF
(già implementato per la federazione). BM25 non richiede embedding e cattura
bene termini tecnici esatti (nomi di servizi, funzioni, parametri).

Impatto atteso: +Recall (soprattutto su query con termini tecnici esatti),
+MRR (il documento giusto sale in ranking perché doppia evidenza lessicale+semantica).
"""
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BM25Index:
    """Indice BM25 in-memory costruito sui chunk del vector store."""

    def __init__(self):
        self._bm25 = None
        self._docs: list[Any] = []
        self._ready = False

    def build(self, docs: list[Any]):
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [d.page_content.lower().split() for d in docs]
            self._bm25 = BM25Okapi(tokenized)
            self._docs = docs
            self._ready = True
            logger.info(f"BM25 index costruito su {len(docs)} chunk.")
        except ImportError:
            logger.warning("rank_bm25 non installato — hybrid search disabilitato.")

    @property
    def ready(self) -> bool:
        return self._ready

    def search(self, query: str, k: int = 10) -> list[tuple[Any, float]]:
        if not self._ready:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        results = [(self._docs[i], float(scores[i])) for i in top_idx if scores[i] > 0]
        return results


# Singleton — costruito una volta per sessione
_bm25_index = BM25Index()


def get_bm25_index() -> BM25Index:
    return _bm25_index


def load_all_chunks_from_qdrant(client, collection_name: str, embedding_fn) -> list[Any]:
    """Scarica tutti i chunk dal vector store per costruire l'indice BM25."""
    from langchain_qdrant import QdrantVectorStore
    from langchain_core.documents import Document

    docs = []
    try:
        offset = None
        while True:
            records, next_offset = client.scroll(
                collection_name=collection_name,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in records:
                payload = r.payload or {}
                text = payload.get("page_content") or payload.get("text") or ""
                metadata = payload.get("metadata", {})
                if text:
                    docs.append(Document(page_content=text, metadata=metadata))
            if next_offset is None:
                break
            offset = next_offset
        logger.info(f"Caricati {len(docs)} chunk da Qdrant per BM25.")
    except Exception as e:
        logger.warning(f"Errore caricamento chunk per BM25: {e}")
    return docs


def hybrid_retrieve(
    question: str,
    vector_store,
    bm25_index: BM25Index,
    top_k: int = 4,
    qdrant_filter=None,
    k_rrf: int = 60,
) -> list[tuple[Any, float]]:
    """Combina dense + BM25 via RRF."""
    # Dense results
    dense = vector_store.similarity_search_with_score(
        query=question, k=top_k * 2, filter=qdrant_filter
    )

    # BM25 results
    bm25_raw = bm25_index.search(question, k=top_k * 2) if bm25_index.ready else []

    # RRF
    rrf_scores: dict[str, float] = {}
    item_map: dict[str, Any] = {}

    for rank, (doc, _) in enumerate(dense, start=1):
        key = doc.page_content[:80]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k_rrf + rank)
        item_map[key] = doc

    for rank, (doc, _) in enumerate(bm25_raw, start=1):
        key = doc.page_content[:80]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k_rrf + rank)
        item_map[key] = doc

    merged = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
    result = [(item_map[k], rrf_scores[k]) for k in merged[:top_k]]
    logger.info(f"Hybrid search: {len(dense)} dense + {len(bm25_raw)} BM25 → {len(result)} fusi via RRF")
    return result
