"""Estensione §5 #3 — Multi-Query Expansion.

Genera N varianti della query originale tramite LLM, esegue il retrieval
per ciascuna, fonde i risultati deduplicando per hash del testo e
ri-classifica per score massimo tra tutte le query.

Impatto atteso: +Recall (recupera chunk rilevanti che la query originale
phrased diversamente potrebbe non trovare). Costo: N chiamate LLM aggiuntive.
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)

MULTI_QUERY_PROMPT = """Sei un esperto di information retrieval.
Genera {n} riformulazioni alternative della domanda originale, che esprimano la stessa intenzione
con parole diverse (sinonimi, prospettive diverse, decomposizione in sotto-domande).
Scrivi SOLO le domande, una per riga, senza numerazione né prefissi.

DOMANDA ORIGINALE: {question}

RIFORMULAZIONI:"""


def expand_query(question: str, llm, n: int = 3) -> list[str]:
    """Genera N varianti della query con il modello LLM."""
    try:
        prompt = MULTI_QUERY_PROMPT.format(n=n, question=question)
        raw = llm.invoke(prompt).strip()
        variants = [line.strip() for line in raw.split("\n") if line.strip()][:n]
        logger.info(f"Multi-query: generate {len(variants)} varianti da '{question[:60]}'")
        return variants
    except Exception as e:
        logger.warning(f"Multi-query expansion fallita: {e}. Uso query originale.")
        return []


def multi_query_retrieve(
    question: str,
    vector_store,
    llm,
    top_k: int = 4,
    n_variants: int = 3,
    qdrant_filter=None,
) -> list[tuple[Any, float]]:
    """Retrieval multi-query: esegue k ricerche e fonde i risultati.

    Deduplicazione per testo del chunk (prime 100 chars come chiave).
    Score finale = massimo score tra tutte le query per quel chunk.
    """
    queries = [question] + expand_query(question, llm, n=n_variants)
    seen: dict[str, tuple[Any, float]] = {}

    for q in queries:
        try:
            results = vector_store.similarity_search_with_score(
                query=q, k=top_k, filter=qdrant_filter
            )
            for doc, score in results:
                key = doc.page_content[:100]
                if key not in seen or score > seen[key][1]:
                    seen[key] = (doc, score)
        except Exception as e:
            logger.warning(f"Retrieval fallito per variante '{q[:40]}': {e}")

    merged = sorted(seen.values(), key=lambda x: x[1], reverse=True)
    logger.info(f"Multi-query: {len(merged)} chunk unici dopo fusione da {len(queries)} query")
    return merged[:top_k]
