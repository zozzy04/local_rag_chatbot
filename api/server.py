"""FastAPI server conforme a api_spec.yaml — §7.2 del README.

Avvio: python app.py serve [--port 8000]
"""
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GROUP_ID = os.getenv("GROUP_ID", "groupA")
GROUP_NAME = os.getenv("GROUP_NAME", "Team RAG")
GROUP_DESCRIPTION = os.getenv("GROUP_DESCRIPTION", "Corpus su AI, Agents e Cloud")
API_KEY = os.getenv("API_KEY", "changeme")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
DOMAIN_TAGS = os.getenv("DOMAIN_TAGS", "ai,agents,cloud").split(",")

# ---------------------------------------------------------------------------
# Rate limiting (graceful degradation se slowapi non installato)
# ---------------------------------------------------------------------------
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
    SLOWAPI_AVAILABLE = True
except ImportError:
    limiter = None
    SLOWAPI_AVAILABLE = False
    logger.warning("slowapi non installato — rate limiting disabilitato.")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="GenAI RAG — Federated Knowledge API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if SLOWAPI_AVAILABLE:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------
def _error(code: str, message: str, status: int, details: dict | None = None):
    body = {"error_code": code, "message": message, "group_id": GROUP_ID}
    if details:
        body["details"] = details
    return JSONResponse(status_code=status, content=body)


# ---------------------------------------------------------------------------
# Auth dependency (eccetto /health)
# ---------------------------------------------------------------------------
def _check_headers(
    x_api_key: str = Header(..., alias="X-API-Key"),
    x_api_version: str = Header(..., alias="X-API-Version"),
    x_group_id: str = Header(..., alias="X-Group-Id"),
):
    if x_api_version != "1.0":
        raise HTTPException(
            status_code=426,
            detail={"error_code": "VERSION_MISMATCH", "message": f"Versione {x_api_version} non supportata. Accettata: 1.0", "group_id": GROUP_ID},
        )
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail={"error_code": "UNAUTHORIZED", "message": "API key non valida o mancante.", "group_id": GROUP_ID},
        )
    return {"caller_group_id": x_group_id}


# ---------------------------------------------------------------------------
# Pydantic models (request)
# ---------------------------------------------------------------------------
class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=50)
    source_filter: Optional[str] = None
    min_score: Optional[float] = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=50)
    source_filter: Optional[str] = None


# ---------------------------------------------------------------------------
# Utility: convert internal result → API StructuredAnswer
# ---------------------------------------------------------------------------
def _normalize_confidence(raw: str) -> str:
    r = raw.lower()
    if "alta" in r or "high" in r:
        return "alta"
    if "bassa" in r or "low" in r or "astensione" in r:
        return "bassa"
    return "media"


def _to_structured_answer(result: dict, is_abstention: bool = False) -> dict:
    sources = [
        {
            "file": s["file"],
            "page": s["pag"],
            "score": s["score"],
            "excerpt": s["estratto"],
            "chunk_id": f"{s['file']}_p{s['pag']}",
        }
        for s in result.get("fonti", [])
    ]
    conf = result.get("confidenza", {})
    return {
        "answer": result.get("risposta_diretta", ""),
        "sources": sources,
        "confidence": _normalize_confidence(conf.get("livello", "media")),
        "confidence_reason": conf.get("motivazione", ""),
        "is_abstention": is_abstention,
        "abstention_reason": result.get("risposta_diretta", "") if is_abstention else None,
    }


# ---------------------------------------------------------------------------
# Lazy imports (evita crash al load se Qdrant/Ollama non disponibili)
# ---------------------------------------------------------------------------
def _get_rag_functions():
    from app.main import core_rag_ask, ingest_local_documents, active_collections, COLLECTION_NAME
    import qdrant_client
    return core_rag_ask, ingest_local_documents, active_collections, COLLECTION_NAME


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "group_id": GROUP_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/info", tags=["System"])
def info(auth=Depends(_check_headers)):
    try:
        core_rag_ask, _, active_collections, COLLECTION_NAME = _get_rag_functions()
        import qdrant_client as qc
        client = qc.QdrantClient(url=os.getenv("QDRANT_HOST", "http://qdrant:6333"))
        collections = client.get_collections().collections
        col_names = [c.name for c in collections]
        n_chunks = 0
        if COLLECTION_NAME in col_names:
            info_col = client.get_collection(COLLECTION_NAME)
            n_chunks = info_col.points_count or 0
    except Exception:
        n_chunks = 0

    try:
        from db.database import SessionLocal
        from db.models import Document
        session = SessionLocal()
        n_documents = session.query(Document).filter(Document.group_id == GROUP_ID).count()
        session.close()
    except Exception:
        n_documents = 0

    return {
        "group_id": GROUP_ID,
        "group_name": GROUP_NAME,
        "description": GROUP_DESCRIPTION,
        "n_documents": n_documents,
        "n_chunks": n_chunks,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "chunking_config": os.getenv("CHUNK_PRESET", "MEDIUM"),
        "api_version": "1.0",
        "domain_tags": DOMAIN_TAGS,
    }


@app.get("/documents", tags=["Knowledge"])
def list_documents(auth=Depends(_check_headers)):
    try:
        from db.database import SessionLocal
        from db.models import Document, Chunk
        from sqlalchemy import func

        session = SessionLocal()
        docs = session.query(Document).filter(Document.group_id == GROUP_ID).all()
        result = []
        for d in docs:
            n_chunks = session.query(func.count(Chunk.id)).filter(Chunk.document_id == d.id).scalar()
            result.append({
                "filename": d.filename,
                "n_pages": d.n_pages,
                "n_chunks": n_chunks,
                "ingested_at": d.ingested_at.isoformat() if d.ingested_at else None,
                "file_hash": d.file_hash[:16],
            })
        session.close()
        return {"group_id": GROUP_ID, "documents": result}
    except Exception as e:
        logger.error(f"Errore /documents: {e}")
        return _error("SERVICE_UNAVAILABLE", str(e), 503)


@app.post("/retrieve", tags=["Retrieval"])
def retrieve(req: RetrieveRequest, auth=Depends(_check_headers)):
    t0 = time.time()
    try:
        core_rag_ask, _, active_collections, COLLECTION_NAME = _get_rag_functions()
        import qdrant_client as qc
        from qdrant_client.http import models as rest
        from langchain_ollama import OllamaEmbeddings
        from langchain_qdrant import QdrantVectorStore

        QDRANT_HOST = os.getenv("QDRANT_HOST", "http://qdrant:6333")
        OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

        client = qc.QdrantClient(url=QDRANT_HOST)
        embeddings = OllamaEmbeddings(base_url=OLLAMA_HOST, model=EMBEDDING_MODEL)

        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            return _error("SERVICE_UNAVAILABLE", "Vector store non ancora inizializzato.", 503)

        vs = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)

        qdrant_filter = None
        if req.source_filter:
            qdrant_filter = rest.Filter(
                must=[rest.FieldCondition(key="metadata.source", match=rest.MatchValue(value=req.source_filter))]
            )

        results = vs.similarity_search_with_score(query=req.question, k=req.k, filter=qdrant_filter)

        chunks = []
        for i, (doc, score) in enumerate(results):
            if req.min_score is not None and score < req.min_score:
                continue
            chunks.append({
                "chunk_id": f"{doc.metadata.get('source', 'unk')}_p{doc.metadata.get('page', 0)}_{i}",
                "text": doc.page_content,
                "source": doc.metadata.get("source", "N/A"),
                "page": doc.metadata.get("page", 1),
                "score": round(float(score), 4),
                "section": None,
            })

        latency_ms = int((time.time() - t0) * 1000)
        return {
            "group_id": GROUP_ID,
            "question": req.question,
            "chunks": chunks,
            "embedding_model": EMBEDDING_MODEL,
            "latency_ms": latency_ms,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Errore /retrieve: {e}")
        return _error("SERVICE_UNAVAILABLE", str(e), 503)


@app.post("/ask", tags=["Generation"])
def ask(req: AskRequest, auth=Depends(_check_headers)):
    t0 = time.time()
    try:
        core_rag_ask, _, _, _ = _get_rag_functions()
        result = core_rag_ask(question=req.question, top_k=req.k, source_filter=req.source_filter)

        is_abstention = "ASTENSIONE:" in result.get("risposta_diretta", "")
        structured = _to_structured_answer(result, is_abstention=is_abstention)

        latency_ms = int((time.time() - t0) * 1000)
        return {
            "group_id": GROUP_ID,
            "question": req.question,
            "answer": structured,
            "model_name": os.getenv("OLLAMA_MODEL", "mistral"),
            "latency_ms": latency_ms,
            "tokens_in": None,
            "tokens_out": None,
        }
    except HTTPException:
        raise
    except ValueError as e:
        return _error("SERVICE_UNAVAILABLE", str(e), 503)
    except Exception as e:
        logger.error(f"Errore /ask: {e}")
        return _error("GENERATION_FAILED", str(e), 500)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_code": "INTERNAL_ERROR", "message": str(exc.detail), "group_id": GROUP_ID},
    )
