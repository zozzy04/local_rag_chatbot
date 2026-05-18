"""Estensione §5 #7 — Structured JSON Logging.

Ogni interazione RAG (query → retrieval → risposta) viene registrata in
`logs/rag_logs.jsonl` (un JSON per riga). Include: timestamp, question,
chunks recuperati con score, risposta, confidenza, latenza, modello.

Questo log è la base per l'error analysis (§8.3 punto 9) e per costruire
nuovi golden dataset a partire dalla storia operativa reale.

Impatto atteso: nessuno sulle metriche di qualità, ma abilita osservabilità
completa e auditabilità (requisito §2 del README).
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_FILE = LOG_DIR / "rag_logs.jsonl"


def _ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_interaction(
    question: str,
    answer: str,
    sources: list[dict],
    confidence: dict,
    latency_ms: int,
    model_name: str,
    embedding_model: str,
    top_k: int,
    federated: bool = False,
    extra: dict | None = None,
):
    """Scrive una riga JSONL nel file di log strutturato."""
    _ensure_log_dir()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer[:500],
        "is_abstention": "ASTENSIONE:" in answer,
        "sources": [
            {
                "file": s.get("file"),
                "page": s.get("pag"),
                "score": s.get("score"),
                "excerpt": s.get("estratto", "")[:150],
            }
            for s in sources
        ],
        "confidence_level": confidence.get("livello", "N/A"),
        "confidence_reason": confidence.get("motivazione", ""),
        "latency_ms": latency_ms,
        "top_k": top_k,
        "model_name": model_name,
        "embedding_model": embedding_model,
        "federated": federated,
        "n_sources": len(sources),
    }
    if extra:
        record.update(extra)

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Errore scrittura log strutturato: {e}")


def read_logs(last_n: int = 50) -> list[dict]:
    """Legge gli ultimi N record dal log JSONL."""
    if not LOG_FILE.exists():
        return []
    records = []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        logger.warning(f"Errore lettura log strutturato: {e}")
    return records[-last_n:]
