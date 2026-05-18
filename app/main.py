import os
import glob
import logging
import hashlib
import time
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

import yaml

from fastapi import FastAPI, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_qdrant import QdrantVectorStore
import qdrant_client
from qdrant_client.http import models as rest

from app.prompt import rag_prompt, rewrite_prompt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://qdrant:6333")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "default_collection")
SOURCE_DIR = os.getenv("SOURCE_DIR", "/data/input")
GROUP_ID = os.getenv("GROUP_ID", "groupA")
GROUP_NAME = os.getenv("GROUP_NAME", "Team RAG")

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"

try:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configurazione caricata da: {CONFIG_PATH}")
except FileNotFoundError:
    logger.error(f"File config.yaml non trovato in: {CONFIG_PATH}")
    config = {}

CHUNK_PRESETS = {
    "SMALL": {"size": 400, "overlap": 80},
    "MEDIUM": {"size": 1000, "overlap": 200},
    "LARGE": {"size": 2000, "overlap": 300}
}

active_collections = {}


# ---------------------------------------------------------------------------
# DB helpers (fail gracefully se il DB non è inizializzato)
# ---------------------------------------------------------------------------

def _get_or_create_group(session):
    from db.models import Group
    group = session.get(Group, GROUP_ID)
    if not group:
        group = Group(id=GROUP_ID, name=GROUP_NAME, created_at=datetime.utcnow())
        session.add(group)
        session.flush()
    return group


def _get_or_create_chunking_config(session, strategy: str):
    from db.models import ChunkingConfig
    cfg = session.query(ChunkingConfig).filter_by(name=strategy).first()
    if not cfg:
        preset = load_chunking_config(strategy)
        cfg = ChunkingConfig(
            name=strategy,
            chunk_size=preset.get("chunk_size", 1000),
            overlap=preset.get("chunk_overlap", 200),
            separators=str(preset.get("separators", [])),
        )
        session.add(cfg)
        session.flush()
    return cfg


def _save_document_to_db(session, filename: str, file_hash: str, n_pages: int, strategy: str):
    from db.models import Document
    existing = session.query(Document).filter_by(
        group_id=GROUP_ID, file_hash=file_hash
    ).first()
    if existing:
        return existing, False

    _get_or_create_group(session)
    cfg = _get_or_create_chunking_config(session, strategy)

    doc = Document(
        group_id=GROUP_ID,
        filename=filename,
        file_hash=file_hash,
        n_pages=n_pages,
        ingested_at=datetime.utcnow(),
        chunking_config_id=cfg.id,
    )
    session.add(doc)
    session.flush()
    return doc, True


def _save_chunks_to_db(session, document_id: int, chunks, file_hash: str):
    from db.models import Chunk
    for i, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", 1)
        vector_id = f"{file_hash[:8]}_p{page}_c{i}"
        db_chunk = Chunk(
            document_id=document_id,
            page=page,
            chunk_index=i,
            text=chunk.page_content,
            char_count=len(chunk.page_content),
            embedding_model=EMBEDDING_MODEL,
            vector_id=vector_id,
        )
        session.add(db_chunk)


def _save_query_answer_to_db(
    question: str,
    result: dict,
    top_k: int,
    source_filter: str | None,
    latency_ms: int,
    federated: bool = False,
    remote_group_ids: dict | None = None,
):
    """Salva query + answer + sources in modo atomico (§6.2)."""
    try:
        from db.database import get_db_session
        from db.models import Query, Answer, AnswerSource, Document

        with get_db_session() as session:
            _get_or_create_group(session)

            conf_raw = result.get("confidenza", {}).get("livello", "media")
            conf_norm = _normalize_confidence(conf_raw)
            is_abstention = "ASTENSIONE:" in result.get("risposta_diretta", "")

            query = Query(
                group_id=GROUP_ID,
                question=question,
                k=top_k,
                source_filter=source_filter,
                federated=federated,
                asked_at=datetime.utcnow(),
                latency_ms=latency_ms,
            )
            session.add(query)
            session.flush()

            answer = Answer(
                query_id=query.id,
                answer_text=result.get("risposta_diretta", ""),
                confidence=conf_norm,
                model_name=OLLAMA_MODEL,
                prompt_template_version="v1",
            )
            session.add(answer)
            session.flush()

            for rank, fonte in enumerate(result.get("fonti", []), start=1):
                doc = session.query(Document).filter_by(
                    group_id=GROUP_ID, filename=fonte["file"]
                ).first()
                remote_gid = None
                if remote_group_ids and fonte.get("group_id"):
                    remote_gid = fonte["group_id"]

                src = AnswerSource(
                    answer_id=answer.id,
                    document_id=doc.id if doc else None,
                    page=fonte["pag"],
                    score=fonte["score"],
                    rank=rank,
                    remote_group_id=remote_gid,
                )
                session.add(src)

    except Exception as e:
        logger.warning(f"Impossibile salvare su DB (eseguire 'python app.py db init'): {e}")


def _normalize_confidence(raw: str) -> str:
    r = raw.lower()
    if "alta" in r or "high" in r:
        return "alta"
    if "bassa" in r or "low" in r or "astensione" in r:
        return "bassa"
    return "media"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_confidence_level(score: float) -> str:
    if score > 0.75:
        return "Alta (Forte corrispondenza)"
    if score > 0.45:
        return "Media (Corrispondenza parziale)"
    return "Bassa (Poca pertinenza)"


def log_to_terminal(question: str, answer: str, confidenza: str, sources: list):
    preview = answer[:300].replace('\n', ' ') + "..." if len(answer) > 300 else answer
    logger.info(f"RAG DOMANDA: {question}")
    logger.info(f"RAG RISPOSTA AI: {preview}")
    logger.info(f"RAG CONFIDENZA: {confidenza}")
    for src in sources:
        logger.info(f"RAG FONTE — [Score: {src['score']}] {src['file']} (Pag. {src['pag']})")


def calculate_file_hash(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_chunking_config(strategy: str) -> dict:
    if not CONFIG_PATH.exists():
        logger.warning(f"config.yaml non trovato. Uso MEDIUM di default.")
        return {"chunk_size": 1000, "chunk_overlap": 200}

    with open(CONFIG_PATH, 'r') as file:
        config_data = yaml.safe_load(file)

    presets = config_data.get("chunking_presets", {})
    if strategy not in presets:
        logger.warning(f"Preset '{strategy}' non trovato. Uso MEDIUM.")
        return presets.get("MEDIUM", {"chunk_size": 1000, "chunk_overlap": 200})

    return presets[strategy]


def is_file_in_qdrant(client: qdrant_client.QdrantClient, collection_name: str, file_hash: str) -> bool:
    try:
        collections = client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            return False

        records, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.file_hash",
                        match=rest.MatchValue(value=file_hash)
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False
        )
        return len(records) > 0
    except Exception as e:
        logger.warning(f"Errore controllo hash in Qdrant: {e}")
        return False


# ---------------------------------------------------------------------------
# Core RAG functions
# ---------------------------------------------------------------------------

def ingest_local_documents(strategy: str = "MEDIUM", embedding_model_name: str = None):
    emb_model = embedding_model_name or EMBEDDING_MODEL
    target_collection = f"{COLLECTION_NAME}_experiment" if embedding_model_name else COLLECTION_NAME

    client = qdrant_client.QdrantClient(url=QDRANT_HOST)
    embeddings = OllamaEmbeddings(base_url=OLLAMA_HOST, model=emb_model)

    if embedding_model_name:
        vector_size = 768 if "nomic" in emb_model.lower() else 1024
        logger.info(f"Esperimento: ricreazione collezione '{target_collection}' size={vector_size}")
        client.recreate_collection(
            collection_name=target_collection,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
        )
        if target_collection in active_collections:
            del active_collections[target_collection]

    collections = client.get_collections().collections
    collection_exists = any(c.name == target_collection for c in collections)

    if collection_exists:
        vector_store = QdrantVectorStore(
            client=client, collection_name=target_collection, embedding=embeddings,
        )
        active_collections[target_collection] = {"vector_store": vector_store, "strategy": strategy}
    else:
        logger.info(f"Collezione '{target_collection}' non esiste ancora. Verrà creata.")
        vector_store = None

    if not os.path.exists(SOURCE_DIR):
        os.makedirs(SOURCE_DIR)
        return

    pdf_files = glob.glob(os.path.join(SOURCE_DIR, "*.pdf"))
    if not pdf_files:
        logger.warning("Nessun PDF trovato nella cartella.")
        return

    preset_config = load_chunking_config(strategy)
    cfg = {
        "size": preset_config.get("chunk_size", 1000),
        "overlap": preset_config.get("chunk_overlap", 200)
    }
    separators = preset_config.get("separators", ["\n\n", "\n", ".", " "])

    new_chunks_to_add = []
    logger.info(f"INIZIO INGESTION (Strategia: {strategy} | Modello: {emb_model})")

    for i, pdf_path in enumerate(pdf_files, 1):
        try:
            filename = os.path.basename(pdf_path)
            file_hash = calculate_file_hash(pdf_path)

            if collection_exists and is_file_in_qdrant(client, target_collection, file_hash):
                logger.info(f"[{i}/{len(pdf_files)}] {filename} già indicizzato. Salto.")
                continue

            logger.info(f"[{i}/{len(pdf_files)}] Elaborazione: {filename}")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            for idx, page in enumerate(pages):
                page.metadata = {
                    "source": filename,
                    "page": page.metadata.get('page', idx) + 1,
                    "file_hash": file_hash
                }

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=cfg["size"],
                chunk_overlap=cfg["overlap"],
                separators=separators
            )
            chunks = text_splitter.split_documents(pages)
            new_chunks_to_add.extend(chunks)

            # Persistenza su DB
            try:
                from db.database import get_db_session
                with get_db_session() as session:
                    doc_record, is_new = _save_document_to_db(
                        session, filename, file_hash, len(pages), strategy
                    )
                    if is_new:
                        file_chunks = [c for c in chunks if c.metadata.get("source") == filename]
                        _save_chunks_to_db(session, doc_record.id, file_chunks, file_hash)
                        logger.info(f"DB: salvato documento '{filename}' con {len(file_chunks)} chunk.")
            except Exception as db_err:
                logger.warning(f"DB non disponibile per '{filename}': {db_err}")

        except Exception as e:
            logger.error(f"Errore su {pdf_path}: {e}")

    if new_chunks_to_add:
        if not collection_exists:
            logger.info(f"Creazione collezione '{target_collection}' con {len(new_chunks_to_add)} chunk...")
            vector_store = QdrantVectorStore.from_documents(
                documents=new_chunks_to_add,
                embedding=embeddings,
                url=QDRANT_HOST,
                collection_name=target_collection,
                force_recreate=True
            )
            active_collections[target_collection] = {"vector_store": vector_store, "strategy": strategy}
        else:
            logger.info(f"Aggiunta di {len(new_chunks_to_add)} nuovi chunk a '{target_collection}'...")
            vector_store.add_documents(new_chunks_to_add)
        logger.info("SISTEMA PRONTO! Indicizzazione completata.")
    else:
        logger.info("SISTEMA PRONTO! Nessun nuovo documento da aggiungere.")


def core_rag_ask(
    question: str,
    top_k: int = 4,
    source_filter: str = None,
    embedding_model_name: str = None,
    federated: bool = False,
    remote_peers: list | None = None,
    # §5 Extensions
    use_multi_query: bool = False,
    use_hybrid_search: bool = False,
    use_cache: bool = False,
    use_self_correction: bool = False,
):
    t0 = time.time()
    emb_model = embedding_model_name or EMBEDDING_MODEL
    target_collection = f"{COLLECTION_NAME}_experiment" if embedding_model_name else COLLECTION_NAME

    if target_collection not in active_collections:
        logger.info(f"Inizializzazione lazy del vector store ({target_collection})...")
        client = qdrant_client.QdrantClient(url=QDRANT_HOST)
        collections = client.get_collections().collections
        if any(c.name == target_collection for c in collections):
            embeddings = OllamaEmbeddings(base_url=OLLAMA_HOST, model=emb_model)
            vector_store = QdrantVectorStore(
                client=client, collection_name=target_collection, embedding=embeddings,
            )
            active_collections[target_collection] = {
                "vector_store": vector_store, "strategy": "Loaded from DB"
            }
        else:
            raise ValueError(f"Collezione '{target_collection}' non trovata in Qdrant.")

    vs = active_collections[target_collection]["vector_store"]
    embeddings_obj = OllamaEmbeddings(base_url=OLLAMA_HOST, model=emb_model)
    llm = OllamaLLM(base_url=OLLAMA_HOST, model=OLLAMA_MODEL)

    # Estensione §5 #8 — Query Cache
    if use_cache:
        try:
            from app.extensions.query_cache import get_query_cache
            cache = get_query_cache()
            cached = cache.lookup(question, embeddings_obj)
            if cached is not None:
                return {**cached, "_from_cache": True}
        except Exception as e:
            logger.warning(f"Cache lookup fallita: {e}")

    qdrant_filter = None
    if source_filter:
        qdrant_filter = rest.Filter(
            must=[rest.FieldCondition(key="metadata.source", match=rest.MatchValue(value=source_filter))]
        )

    # Estensione §5 #2 — Hybrid BM25 index (build once per session)
    bm25_index = None
    if use_hybrid_search:
        try:
            from app.extensions.hybrid_search import get_bm25_index, load_all_chunks_from_qdrant
            bm25_index = get_bm25_index()
            if not bm25_index.ready:
                client_bm25 = qdrant_client.QdrantClient(url=QDRANT_HOST)
                all_docs = load_all_chunks_from_qdrant(client_bm25, target_collection, embeddings_obj)
                bm25_index.build(all_docs)
        except Exception as e:
            logger.warning(f"BM25 index build fallito: {e}. Uso solo dense.")
            bm25_index = None

    max_attempts = 2
    current_question = question
    final_answer = ""
    results_with_score = []

    for attempt in range(max_attempts):
        logger.info(f"Tentativo {attempt + 1}/{max_attempts} — query: '{current_question}'")

        # Retrieval: standard, hybrid o multi-query
        if use_hybrid_search and bm25_index and bm25_index.ready:
            from app.extensions.hybrid_search import hybrid_retrieve
            results_with_score = hybrid_retrieve(
                current_question, vs, bm25_index, top_k=top_k, qdrant_filter=qdrant_filter
            )
        elif use_multi_query:
            from app.extensions.multi_query import multi_query_retrieve
            results_with_score = multi_query_retrieve(
                current_question, vs, llm, top_k=top_k, qdrant_filter=qdrant_filter
            )
        else:
            results_with_score = vs.similarity_search_with_score(
                query=current_question, k=top_k, filter=qdrant_filter
            )

        if not results_with_score:
            final_answer = "ASTENSIONE: Nessun documento recuperato dal database."
            break

        # Federated retrieval — RRF fusion
        if federated or remote_peers:
            try:
                from api.federation import federated_retrieve, load_peers
                peers = remote_peers if remote_peers is not None else load_peers()
                local_chunks = [
                    {
                        "chunk_id": f"{doc.metadata.get('source','unk')}_p{doc.metadata.get('page',0)}_{j}",
                        "text": doc.page_content,
                        "source": doc.metadata.get("source", "N/A"),
                        "page": doc.metadata.get("page", 1),
                        "score": float(score),
                    }
                    for j, (doc, score) in enumerate(results_with_score)
                ]
                merged_chunks, errors = federated_retrieve(
                    question=current_question,
                    top_k=top_k,
                    local_chunks=local_chunks,
                    peers=peers,
                    source_filter=source_filter,
                )
                if errors:
                    for peer_id, err in errors.items():
                        logger.warning(f"[{peer_id}: non raggiungibile — {err}]")

                # Rebuilding context da chunks mergiati
                context = "\n---\n".join([
                    f"[Gruppo: {c.get('group_id', GROUP_ID)} | File: {c['source']} | Pag: {c['page']}] {c['text']}"
                    for c in merged_chunks[:top_k]
                ])
                final_answer = llm.invoke(rag_prompt.format(context=context, question=current_question)).strip()

                # Rebuild sources dai merged_chunks
                results_with_score = [
                    (type('Doc', (), {
                        'page_content': c['text'],
                        'metadata': {
                            'source': c['source'],
                            'page': c['page'],
                            'group_id': c.get('group_id', GROUP_ID)
                        }
                    })(), c['score'])
                    for c in merged_chunks[:top_k]
                ]
            except Exception as fed_err:
                logger.error(f"Errore federated retrieve: {fed_err}. Uso solo KB locale.")
                context = "\n---\n".join([
                    f"[File: {doc.metadata.get('source', 'N/A')} | Pag: {doc.metadata.get('page', 0)}] {doc.page_content}"
                    for doc, _ in results_with_score
                ])
                final_answer = llm.invoke(rag_prompt.format(context=context, question=current_question)).strip()
        else:
            context = "\n---\n".join([
                f"[File: {doc.metadata.get('source', 'N/A')} | Pag: {doc.metadata.get('page', 0)}] {doc.page_content}"
                for doc, _ in results_with_score
            ])
            final_answer = llm.invoke(rag_prompt.format(context=context, question=current_question)).strip()

        if "ASTENSIONE:" not in final_answer:
            break

        if attempt < max_attempts - 1:
            logger.warning("Il modello si è astenuto. Riformulazione query...")
            current_question = llm.invoke(rewrite_prompt.format(question=current_question)).strip()
            current_question = current_question.replace('"', '').replace("'", "")

    sources = []
    total_score = 0.0
    if results_with_score:
        # RRF scores (max ~0.016) need rescaling before confidence thresholds (cosine 0-1)
        raw_scores = [float(s) for _, s in results_with_score]
        max_raw = max(raw_scores) if raw_scores else 1.0
        scale = (1.0 / max_raw) * 0.82 if max_raw < 0.1 else 1.0

        for (doc, score), raw_s in zip(results_with_score, raw_scores):
            src_entry = {
                "file": doc.metadata.get("source", "N/A"),
                "pag": doc.metadata.get("page", 0),
                "score": round(float(score), 3),
                "estratto": doc.page_content[:150].replace("\n", " ") + "...",
            }
            if hasattr(doc, 'metadata') and doc.metadata.get("group_id"):
                src_entry["group_id"] = doc.metadata["group_id"]
            sources.append(src_entry)
            total_score += raw_s * scale
        avg_score = total_score / len(results_with_score)
        conf_level = get_confidence_level(avg_score)
    else:
        avg_score = 0.0
        conf_level = "Bassa"

    if "ASTENSIONE:" in final_answer:
        final_answer = "ASTENSIONE: Le informazioni presenti nei documenti non sono sufficienti per rispondere."
        conf_level = "Bassa (Astensione forzata)"

    # Estensione §5 #9 — Self-Correction Loop
    if use_self_correction and "ASTENSIONE:" not in final_answer:
        try:
            from app.extensions.self_correction import self_correct
            context_for_check = "\n---\n".join([s["estratto"] for s in sources])
            final_answer, was_modified = self_correct(final_answer, context_for_check, llm)
            if was_modified:
                logger.info("Self-correction ha modificato la risposta.")
        except Exception as e:
            logger.warning(f"Self-correction fallita: {e}")

    log_to_terminal(question, final_answer, conf_level, sources)

    latency_ms = int((time.time() - t0) * 1000)

    # Estensione §5 #7 — Structured JSON Logging
    try:
        from app.extensions.structured_logger import log_interaction
        log_interaction(
            question=question, answer=final_answer, sources=sources,
            confidence={"livello": conf_level, "motivazione": f"Avg score: {round(avg_score, 2)}"},
            latency_ms=latency_ms, model_name=OLLAMA_MODEL,
            embedding_model=emb_model, top_k=top_k, federated=federated,
            extra={
                "use_multi_query": use_multi_query,
                "use_hybrid_search": use_hybrid_search,
                "use_cache": use_cache,
                "use_self_correction": use_self_correction,
            }
        )
    except Exception as e:
        logger.debug(f"Structured logging fallito: {e}")

    result = {
        "risposta_diretta": final_answer,
        "fonti": sources,
        "confidenza": {
            "livello": conf_level,
            "motivazione": f"Punteggio medio vettoriale: {round(avg_score, 2)}.",
        },
    }

    # Estensione §5 #8 — Salva in cache
    if use_cache:
        try:
            from app.extensions.query_cache import get_query_cache
            get_query_cache().store(question, result, embeddings_obj)
        except Exception as e:
            logger.debug(f"Cache store fallita: {e}")

    # Persistenza su DB
    _save_query_answer_to_db(
        question=question,
        result=result,
        top_k=top_k,
        source_filter=source_filter,
        latency_ms=latency_ms,
        federated=federated,
    )

    return result


# ---------------------------------------------------------------------------
# FastAPI app (mantiene compatibilità Docker/legacy)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Inizializzazione sistema RAG (legacy FastAPI)...")
    ingest_local_documents()
    yield
    logger.info("Chiusura server.")


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/reindex")
async def trigger_reindex(background_tasks: BackgroundTasks):
    logger.info("Re-indicizzazione in background...")
    background_tasks.add_task(ingest_local_documents)
    return {"status": "ok", "dettaglio": "Re-indicizzazione avviata in background."}


@app.post("/ask")
async def ask_endpoint(question: str = Form(...), k: int = Form(4), source: str = Form(None)):
    try:
        result = core_rag_ask(question=question, top_k=k, source_filter=source)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Errore API /ask: {e}")
        raise HTTPException(status_code=500, detail="Errore interno del server.")
