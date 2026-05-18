"""Federated retrieval logic — §7.3 del README.

Strategia di fusion: Reciprocal Rank Fusion (RRF).
Motivazione: RRF non richiede che gli score siano comparabili tra gruppi con
embedding diversi. Il rank è una misura ordinale robusta e non dipende dalla
scala assoluta dei punteggi coseno.
"""
import asyncio
import logging
import os
import time
from typing import Any

import httpx
import yaml

logger = logging.getLogger(__name__)

FEDERATION_TIMEOUT = float(os.getenv("FEDERATION_TIMEOUT_S", "5"))
GROUP_ID = os.getenv("GROUP_ID", "groupA")
API_KEY = os.getenv("API_KEY", "changeme")


def load_peers(peers_file: str = "peers.yaml") -> list[dict]:
    try:
        with open(peers_file, "r") as f:
            data = yaml.safe_load(f)
        return [p for p in data.get("peers", []) if p.get("enabled", True)]
    except FileNotFoundError:
        logger.warning("peers.yaml non trovato — nessun peer remoto configurato.")
        return []
    except Exception as e:
        logger.error(f"Errore lettura peers.yaml: {e}")
        return []


def reciprocal_rank_fusion(
    group_results: dict[str, list[dict]], k_rrf: int = 60
) -> list[dict]:
    """Combina risultati di N gruppi via RRF.

    Ogni item in group_results[group_id] è un dict con almeno: chunk_id, text, source, page, score.
    Ritorna lista ordinata per rrf_score decrescente.
    """
    rrf_scores: dict[str, float] = {}
    item_by_id: dict[str, dict] = {}

    for group_id, chunks in group_results.items():
        for rank, chunk in enumerate(chunks, start=1):
            fallback_id = f"{chunk['source']}_{chunk['page']}_{rank}"
            uid = f"{group_id}::{chunk.get('chunk_id', fallback_id)}"
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + 1.0 / (k_rrf + rank)
            if uid not in item_by_id:
                item_by_id[uid] = {**chunk, "group_id": group_id}

    merged = sorted(item_by_id.keys(), key=lambda uid: rrf_scores[uid], reverse=True)
    return [item_by_id[uid] for uid in merged]


async def _call_peer_retrieve(
    client: httpx.AsyncClient,
    peer: dict,
    question: str,
    k: int,
    source_filter: str | None = None,
) -> tuple[str, list[dict], str | None]:
    """Interroga /retrieve di un singolo peer. Ritorna (peer_id, chunks, error)."""
    peer_id = peer["id"]
    base_url = peer["base_url"].rstrip("/")
    url = f"{base_url}/retrieve"

    headers = {
        "X-API-Key": peer.get("api_key", API_KEY),
        "X-API-Version": "1.0",
        "X-Group-Id": GROUP_ID,
    }
    body: dict[str, Any] = {"question": question, "k": k}
    if source_filter:
        body["source_filter"] = source_filter

    try:
        resp = await client.post(url, json=body, headers=headers, timeout=FEDERATION_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            chunks = data.get("chunks", [])
            logger.info(f"Peer {peer_id}: {len(chunks)} chunks ricevuti")
            return peer_id, chunks, None
        else:
            msg = f"HTTP {resp.status_code}"
            logger.warning(f"Peer {peer_id} ha risposto con {msg}")
            return peer_id, [], msg
    except httpx.TimeoutException:
        msg = f"timeout dopo {FEDERATION_TIMEOUT}s"
        logger.warning(f"Peer {peer_id}: {msg}")
        return peer_id, [], msg
    except Exception as e:
        logger.warning(f"Peer {peer_id}: errore — {e}")
        return peer_id, [], str(e)


async def federated_retrieve_async(
    question: str,
    top_k: int,
    local_chunks: list[dict],
    peers: list[dict] | None = None,
    source_filter: str | None = None,
) -> tuple[list[dict], dict[str, str]]:
    """Recupera chunks da tutti i peer in parallelo e fonde con i chunks locali.

    Ritorna (merged_chunks, errors_by_peer_id).
    """
    if peers is None:
        peers = load_peers()

    group_results: dict[str, list[dict]] = {GROUP_ID: local_chunks}
    errors: dict[str, str] = {}

    if not peers:
        return local_chunks, errors

    async with httpx.AsyncClient() as client:
        tasks = [
            _call_peer_retrieve(client, peer, question, top_k, source_filter)
            for peer in peers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for res in results:
        if isinstance(res, Exception):
            logger.error(f"Eccezione nella chiamata peer: {res}")
            continue
        peer_id, chunks, error = res
        if error:
            errors[peer_id] = error
        else:
            group_results[peer_id] = chunks

    merged = reciprocal_rank_fusion(group_results)
    return merged, errors


def federated_retrieve(
    question: str,
    top_k: int,
    local_chunks: list[dict],
    peers: list[dict] | None = None,
    source_filter: str | None = None,
) -> tuple[list[dict], dict[str, str]]:
    """Wrapper sincrono per federated_retrieve_async."""
    return asyncio.run(
        federated_retrieve_async(question, top_k, local_chunks, peers, source_filter)
    )
